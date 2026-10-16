"""
Local Hugging Face Transformers backend (optional).

Runs a pretrained chat model (for example Qwen/Qwen2.5-0.5B-Instruct) inside
the benchmark process instead of asking a server. model_adapter.py sends
requests here when BENCHMARK_BACKEND=transformers.

Design rules:
- torch and transformers are imported only when this backend is actually used,
  so the normal project (Ollama, dashboard, tests) does not need them.
- Models are NEVER downloaded here. Everything is loaded with
  local_files_only=True and Hugging Face offline mode, so the model must already
  be in the local Hugging Face cache (see README for the one-time download).
- The tokenizer and model are loaded once per process and reused for every task.
- Generation is greedy (no sampling) and bounded by a token limit and a time limit.
- Expected failures become HFBackendError (a RuntimeError) or TimeoutError,
  which the benchmark runner already records as a task error.

Not yet verified on this machine: compatibility with specific torch /
transformers versions, Apple MPS behaviour, memory use and speed.
"""

import os
import threading
import time

# Upper bound on generated tokens per answer (this backend has no "thinking" phase).
MAX_NEW_TOKENS = 512

_loaded = {}                 # model_id -> (tokenizer, model, device)
_load_lock = threading.Lock()


class HFBackendError(RuntimeError):
    """The local Transformers backend could not load the model or generate an answer."""


def _import_libraries():
    """Import torch and transformers only now, in offline mode. Returns (torch, transformers)."""
    # Never contact the Hugging Face Hub from this backend (must be set before import).
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    try:
        import torch
        import transformers
    except ImportError as error:
        raise HFBackendError(
            "The transformers backend needs extra packages (torch, transformers). Install "
            "requirements-hf.txt into the separate .venv-hf environment and run the benchmark "
            f"with .venv-hf/bin/python (see README). Import failed: {error}"
        ) from error
    return torch, transformers


def pick_device(torch):
    """Apple GPU (MPS) when available, otherwise CPU."""
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(torch, device):
    """Half precision on the Apple GPU to save memory; full precision on CPU."""
    return torch.float16 if device == "mps" else torch.float32


def load_model(model_id):
    """
    Load the tokenizer and model from the LOCAL cache, once per process.
    Returns (tokenizer, model, device). Raises HFBackendError if that fails.
    """
    with _load_lock:
        if model_id in _loaded:
            return _loaded[model_id]
        torch, transformers = _import_libraries()
        device = pick_device(torch)
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_id, local_files_only=True, dtype=pick_dtype(torch, device),
            )
            model.to(device)
            model.eval()
        except Exception as error:
            raise HFBackendError(
                f"Could not load '{model_id}' from the local Hugging Face cache (device: {device}). "
                "If it hasn't been downloaded yet, download it once first (see README). "
                f"Cause: {type(error).__name__}: {error}"
            ) from error
        _loaded[model_id] = (tokenizer, model, device)
        return _loaded[model_id]


def check_model_available(model_id):
    """
    For the runner's start-up check: load the model now (from the local cache only),
    so a missing model is reported before any task runs and loading time is not
    counted in the first task. Returns None if ready, otherwise a readable message.
    """
    try:
        load_model(model_id)
    except HFBackendError as error:
        return str(error)
    return None


def _prompt_tensors(torch, tokenizer, messages, device):
    """Build the prompt with the model's own chat template. Returns (input_ids, attention_mask)."""
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    # Some transformers versions return the ids directly, others a dict-like object.
    if hasattr(encoded, "keys") and "input_ids" in encoded.keys():
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
    else:
        input_ids, attention_mask = encoded, None
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    return input_ids.to(device), attention_mask.to(device)


def ask_model(config, system_prompt, question):
    """
    Answer one question with the local model.
    Returns (answer_text, reasoning_text, finish_reason), like model_adapter.ask_model().
    Raises HFBackendError (a RuntimeError) on failure, TimeoutError if the time limit is hit.
    """
    tokenizer, model, device = load_model(config.model_id)
    torch, _ = _import_libraries()
    max_new_tokens = min(config.max_tokens, MAX_NEW_TOKENS)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    started = time.perf_counter()
    try:
        input_ids, attention_mask = _prompt_tensors(torch, tokenizer, messages, device)
        with torch.no_grad():
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,                       # greedy: same input, same answer
                max_time=config.task_time_limit,       # stop generating after this many seconds
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id,
            )
        new_tokens = output[0][input_ids.shape[-1]:]   # only what the model generated
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    except Exception as error:
        raise HFBackendError(
            f"Generation with '{config.model_id}' failed (device: {device}): "
            f"{type(error).__name__}: {error}"
        ) from error

    if time.perf_counter() - started >= config.task_time_limit:
        raise TimeoutError(f"Task took longer than {config.task_time_limit} seconds")
    finish_reason = "length" if len(new_tokens) >= max_new_tokens else "stop"
    return answer, "", finish_reason
