"""
Model adapter: everything about TALKING to a model lives here.

The benchmark runner decides what to ask and how to mark the answer.
This file only knows how to send a question to a model server and read
the reply. Keeping the two apart means we can point the benchmark at a
different model without touching the benchmark logic.

It speaks the "OpenAI-compatible" chat format (POST /chat/completions with
streaming). Ollama supports this, and so do many hosted services, but not
every provider behaves identically. Only local Ollama has been tested.

Settings come from environment variables (never from files in the repo):
    BENCHMARK_BASE_URL   server address   (default: local Ollama)
    BENCHMARK_MODEL_ID   model name       (default: qwen3:4b)
    BENCHMARK_API_KEY    private key      (optional; never saved or printed)
"""

import json
import os
import time
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_MODEL_ID = "qwen3:4b"

ENV_BASE_URL = "BENCHMARK_BASE_URL"
ENV_MODEL_ID = "BENCHMARK_MODEL_ID"
ENV_API_KEY = "BENCHMARK_API_KEY"

# Shown when there are too many models to list in an error message.
MAX_MODELS_IN_ERROR = 20


class ModelConfigError(ValueError):
    """A setting such as BENCHMARK_BASE_URL is not usable."""


@dataclass(frozen=True)
class ModelConfig:
    """All the settings needed to talk to one model."""

    base_url: str = DEFAULT_BASE_URL
    model_id: str = DEFAULT_MODEL_ID
    # repr=False hides the key if this config is ever printed.
    api_key: str | None = field(default=None, repr=False)

    # qwen3 "thinks" before answering, and the thinking counts toward max_tokens.
    # With only 300 tokens the model ran out while still thinking and returned
    # an empty answer, so we allow much more room.
    max_tokens: int = 4096

    connect_timeout: int = 10    # seconds to wait for the server to accept the connection
    read_timeout: int = 120      # seconds to wait for the NEXT streamed chunk (covers model loading)
    task_time_limit: int = 300   # seconds allowed for one whole answer, so nothing hangs forever


def build_model_config(base_url, model_id, api_key=None):
    """
    Check the given settings and turn them into a ModelConfig.
    Used by both the runner (values from environment variables) and the
    dashboard (values typed into a form). No network request happens here,
    so a bad address is rejected before anything is sent.
    """
    base_url = (base_url or "").strip().rstrip("/")
    model_id = (model_id or "").strip()
    api_key = (api_key or "").strip() or None

    parts = urlsplit(base_url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        # The value itself is not repeated here, in case a secret was pasted into it.
        raise ModelConfigError(
            "Base URL must be a full address starting with http:// or https://, "
            f"for example {DEFAULT_BASE_URL}"
        )
    if not model_id:
        raise ModelConfigError("Model ID must not be empty.")

    return ModelConfig(base_url=base_url, model_id=model_id, api_key=api_key)


def load_model_config(env=None):
    """
    Build a ModelConfig from environment variables.
    Anything not set falls back to the local Ollama defaults.
    """
    env = os.environ if env is None else env

    base_url = (env.get(ENV_BASE_URL) or "").strip() or DEFAULT_BASE_URL
    model_id = (env.get(ENV_MODEL_ID) or "").strip() or DEFAULT_MODEL_ID
    try:
        return build_model_config(base_url, model_id, env.get(ENV_API_KEY))
    except ModelConfigError as error:
        # Name the variable, so terminal users know which setting to fix.
        raise ModelConfigError(f"{ENV_BASE_URL}: {error}") from None


def safe_base_url(url):
    """
    The address without any login details or ?query part, so it is safe to save.
    Example: https://user:pw@host/v1?key=abc  ->  https://host/v1
    """
    parts = urlsplit(url)
    host = parts.netloc.rpartition("@")[2]   # drop "user:password@" if present
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def redact(text, config):
    """Hide the API key (and any secret parts of the address) in a message."""
    if not text:
        return text
    text = str(text)
    if config.api_key:
        text = text.replace(config.api_key, "***")
    safe_url = safe_base_url(config.base_url)
    if config.base_url != safe_url:
        text = text.replace(config.base_url, safe_url)
    return text


def _headers(config):
    """Only send an Authorization header when a key was actually given."""
    if config.api_key:
        return {"Authorization": f"Bearer {config.api_key}"}
    return {}


def check_model_available(config):
    """
    Ask the server which models it has.
    Returns None if the model is available, otherwise a readable error message.
    """
    where = safe_base_url(config.base_url)
    try:
        response = requests.get(
            f"{config.base_url}/models",
            headers=_headers(config),
            timeout=config.connect_timeout,
        )
    except requests.ConnectionError:
        return (
            f"Could not connect to the model endpoint at {where}.\n"
            "Is the server running? For local Ollama, open the Ollama app or run: ollama serve"
        )
    except requests.RequestException as error:
        return redact(f"Request to {where}/models failed: {error}", config)

    if response.status_code != 200:
        return redact(
            f"The model endpoint at {where}/models returned HTTP {response.status_code}: "
            f"{response.text[:300]}",
            config,
        )

    try:
        model_ids = [model.get("id") for model in response.json().get("data", [])]
    except (ValueError, AttributeError):
        return f"The model endpoint at {where}/models did not return a valid model list."

    if config.model_id not in model_ids:
        shown = ", ".join(str(m) for m in model_ids[:MAX_MODELS_IN_ERROR]) or "(none)"
        if len(model_ids) > MAX_MODELS_IN_ERROR:
            shown += f", ... ({len(model_ids)} in total)"
        return f"Model '{config.model_id}' was not found at {where}.\nAvailable models: {shown}"

    return None


def ask_model(config, system_prompt, question):
    """
    Send one question to the model using streaming.
    Returns (answer_text, reasoning_text, finish_reason).
    Raises an exception with a readable message if anything goes wrong.
    """
    payload = {
        "model": config.model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "max_tokens": config.max_tokens,
        "temperature": 0,
        "stream": True,
    }

    answer_parts = []
    reasoning_parts = []
    finish_reason = None
    deadline = time.perf_counter() + config.task_time_limit

    with requests.post(
        f"{config.base_url}/chat/completions",
        json=payload,
        headers=_headers(config),
        stream=True,
        timeout=(config.connect_timeout, config.read_timeout),
    ) as response:
        if response.status_code != 200:
            raise RuntimeError(redact(
                f"Model endpoint returned HTTP {response.status_code}: {response.text[:300]}",
                config,
            ))

        for line in response.iter_lines(decode_unicode=True):
            if time.perf_counter() > deadline:
                raise TimeoutError(f"Task took longer than {config.task_time_limit} seconds")

            if not line or not line.startswith("data: "):
                continue

            data = line[len("data: "):]
            if data == "[DONE]":
                break

            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                raise ValueError(redact(f"Malformed data from model endpoint: {data[:200]}", config))

            if "error" in chunk:
                raise RuntimeError(redact(f"Model endpoint error: {chunk['error']}", config))

            choices = chunk.get("choices") or []
            if not choices:
                continue

            delta = choices[0].get("delta") or {}
            if delta.get("content"):
                answer_parts.append(delta["content"])
            # Thinking text arrives separately. Ollama calls it "reasoning";
            # some other servers call it "reasoning_content".
            reasoning = delta.get("reasoning") or delta.get("reasoning_content")
            if reasoning:
                reasoning_parts.append(reasoning)
            if choices[0].get("finish_reason"):
                finish_reason = choices[0]["finish_reason"]

    return "".join(answer_parts), "".join(reasoning_parts), finish_reason
