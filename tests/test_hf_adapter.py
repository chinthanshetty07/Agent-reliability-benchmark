"""
Tests for tasks/hf_adapter.py (the optional local Hugging Face backend).
Run from the project root with:
    python3 -m unittest discover -s tests -v

torch and transformers are NOT needed: small fake versions of both are put in
place of the real libraries, so these tests only check how the adapter USES
them (loading once, chat template, greedy generation, decoding only new
tokens, device choice, error handling). Real compatibility, speed and memory
use must still be checked in the dedicated .venv-hf environment.
Any attempt to open a network connection fails the test.
"""

import os
import socket
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "tasks"))

import hf_adapter  # noqa: E402
from model_adapter import ModelConfig  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT_IDS = [101, 102, 103]           # what the fake chat template "encodes"
VOCAB = {101: "<sys>", 102: "<q>", 103: "<assistant>",
         1: "Profit is 3000.", 2: "\nFinal answer: 3000", 0: "<eos>"}


# ---------- Fakes for torch and transformers ----------

class FakeTensor:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.device = "cpu"

    @property
    def shape(self):
        return (len(self.rows), len(self.rows[0]))

    def to(self, device):
        self.device = device
        return self

    def __getitem__(self, index):
        return self.rows[index]


class FakeNoGrad:
    entered = 0

    def __enter__(self):
        FakeNoGrad.entered += 1

    def __exit__(self, *exc):
        return False


def make_fake_torch(mps_available=True, has_mps_attribute=True):
    torch = mock.MagicMock(name="torch")
    torch.float16, torch.float32 = "float16", "float32"
    torch.no_grad = FakeNoGrad
    torch.ones_like = lambda t: FakeTensor([[1] * len(row) for row in t.rows])
    if has_mps_attribute:
        torch.backends.mps.is_available.return_value = mps_available
    else:
        torch.backends = object()                 # an older torch without MPS support
    return torch


class FakeTokenizer:
    pad_token_id = None
    eos_token_id = 0

    def __init__(self, dict_output=False):
        self.dict_output = dict_output
        self.template_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((messages, kwargs))
        ids = FakeTensor([PROMPT_IDS])
        if self.dict_output:                      # newer transformers style
            return {"input_ids": ids, "attention_mask": FakeTensor([[1, 1, 1]])}
        return ids

    def decode(self, ids, skip_special_tokens=False):
        return "".join(VOCAB[i] for i in ids if not (skip_special_tokens and i == 0))


class FakeModel:
    def __init__(self, new_tokens=(1, 2, 0), error=None):
        self.new_tokens = list(new_tokens)
        self.error = error
        self.device = None
        self.eval_called = False
        self.generate_calls = []

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def generate(self, input_ids, **kwargs):
        self.generate_calls.append((input_ids, kwargs))
        if self.error:
            raise self.error
        return FakeTensor([input_ids.rows[0] + self.new_tokens])


def make_fake_transformers(tokenizer, model, load_error=None):
    transformers = mock.MagicMock(name="transformers")
    if load_error:
        transformers.AutoModelForCausalLM.from_pretrained.side_effect = load_error
    else:
        transformers.AutoModelForCausalLM.from_pretrained.return_value = model
    transformers.AutoTokenizer.from_pretrained.return_value = tokenizer
    return transformers


def no_network(*args, **kwargs):
    raise AssertionError("network access attempted")


class HFAdapterTestCase(unittest.TestCase):
    def setUp(self):
        hf_adapter._loaded.clear()
        self.addCleanup(hf_adapter._loaded.clear)
        # Any network connection attempt fails the test.
        mock.patch.object(socket.socket, "connect", side_effect=no_network).start()
        mock.patch.object(socket, "create_connection", side_effect=no_network).start()
        mock.patch.dict(os.environ).start()
        for name in ("HF_HUB_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
            os.environ.pop(name, None)
        self.addCleanup(mock.patch.stopall)
        self.config = ModelConfig(model_id=MODEL_ID, backend="transformers",
                                  base_url="transformers://local")

    def use_fakes(self, tokenizer=None, model=None, torch=None, load_error=None):
        self.tokenizer = tokenizer or FakeTokenizer()
        self.model = model or FakeModel()
        self.torch = torch or make_fake_torch()
        self.transformers = make_fake_transformers(self.tokenizer, self.model, load_error)
        mock.patch.dict(sys.modules, {"torch": self.torch, "transformers": self.transformers}).start()

    def ask(self, question="Revenue 10000, expenses 7000. Profit?"):
        return hf_adapter.ask_model(self.config, "SYSTEM PROMPT", question)


class TestOptionalDependencies(HFAdapterTestCase):
    def test_importing_the_project_does_not_import_torch_or_transformers(self):
        # A fresh Python process, so nothing from this test run is already imported.
        code = ("import sys; sys.path.insert(0, 'tasks'); import model_adapter, hf_adapter, ai_runner; "
                "print('torch' in sys.modules, 'transformers' in sys.modules)")
        out = subprocess.run([sys.executable, "-c", code], cwd=PROJECT, capture_output=True,
                             text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertEqual(out.stdout.strip(), "False False", out.stderr)

    def test_missing_packages_give_a_clear_error(self):
        with mock.patch.dict(sys.modules, {"torch": None, "transformers": None}):
            with self.assertRaises(hf_adapter.HFBackendError) as caught:
                hf_adapter.load_model(MODEL_ID)
        message = str(caught.exception)
        self.assertIn("requirements-hf.txt", message)
        self.assertIn(".venv-hf", message)
        self.assertIsInstance(caught.exception, RuntimeError)       # the runner records RuntimeError

    def test_check_reports_missing_packages_as_a_message(self):
        with mock.patch.dict(sys.modules, {"torch": None, "transformers": None}):
            message = hf_adapter.check_model_available(MODEL_ID)
        self.assertIn("requirements-hf.txt", message)


class TestLoading(HFAdapterTestCase):
    def test_loads_only_from_the_local_cache_in_offline_mode(self):
        self.use_fakes()
        hf_adapter.load_model(MODEL_ID)
        tok_call = self.transformers.AutoTokenizer.from_pretrained.call_args
        model_call = self.transformers.AutoModelForCausalLM.from_pretrained.call_args
        self.assertEqual(tok_call.args, (MODEL_ID,))
        self.assertIs(tok_call.kwargs["local_files_only"], True)
        self.assertEqual(model_call.args, (MODEL_ID,))
        self.assertIs(model_call.kwargs["local_files_only"], True)
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
        self.assertTrue(self.model.eval_called)

    def test_loads_once_per_process_not_once_per_task(self):
        self.use_fakes()
        for _ in range(3):
            self.ask()
        self.assertEqual(self.transformers.AutoTokenizer.from_pretrained.call_count, 1)
        self.assertEqual(self.transformers.AutoModelForCausalLM.from_pretrained.call_count, 1)
        self.assertEqual(len(self.model.generate_calls), 3)

    def test_uses_apple_mps_when_available(self):
        self.use_fakes(torch=make_fake_torch(mps_available=True))
        tokenizer, model, device = hf_adapter.load_model(MODEL_ID)
        self.assertEqual((device, model.device), ("mps", "mps"))
        self.assertEqual(self.transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs["dtype"], "float16")

    def test_uses_cpu_when_mps_is_not_available(self):
        self.use_fakes(torch=make_fake_torch(mps_available=False))
        _, model, device = hf_adapter.load_model(MODEL_ID)
        self.assertEqual((device, model.device), ("cpu", "cpu"))
        self.assertEqual(self.transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs["dtype"], "float32")

    def test_uses_cpu_when_torch_has_no_mps_support(self):
        self.use_fakes(torch=make_fake_torch(has_mps_attribute=False))
        self.assertEqual(hf_adapter.load_model(MODEL_ID)[2], "cpu")

    def test_model_not_in_local_cache_gives_a_clear_error(self):
        self.use_fakes(load_error=OSError(f"{MODEL_ID} is not available in the local cache"))
        message = hf_adapter.check_model_available(MODEL_ID)
        self.assertIn("Could not load", message)
        self.assertIn("local Hugging Face cache", message)
        self.assertIn("download it once first", message)
        with self.assertRaises(hf_adapter.HFBackendError):
            self.ask()

    def test_check_returns_none_and_preloads_when_ready(self):
        self.use_fakes()
        self.assertIsNone(hf_adapter.check_model_available(MODEL_ID))
        self.ask()                                                # reuses the preloaded model
        self.assertEqual(self.transformers.AutoModelForCausalLM.from_pretrained.call_count, 1)


class TestInference(HFAdapterTestCase):
    def test_uses_the_chat_template_with_system_and_user_messages(self):
        self.use_fakes()
        self.ask("What is the profit?")
        messages, kwargs = self.tokenizer.template_calls[0]
        self.assertEqual(messages, [
            {"role": "system", "content": "SYSTEM PROMPT"},
            {"role": "user", "content": "What is the profit?"},
        ])
        self.assertIs(kwargs["add_generation_prompt"], True)

    def test_generation_is_greedy_bounded_and_without_gradients(self):
        self.use_fakes()
        FakeNoGrad.entered = 0
        self.ask()
        input_ids, kwargs = self.model.generate_calls[0]
        self.assertIs(kwargs["do_sample"], False)
        self.assertEqual(kwargs["max_new_tokens"], hf_adapter.MAX_NEW_TOKENS)
        self.assertEqual(kwargs["max_time"], self.config.task_time_limit)
        self.assertEqual(kwargs["pad_token_id"], FakeTokenizer.eos_token_id)
        self.assertEqual(input_ids.device, "mps")                # moved to the model's device
        self.assertEqual(kwargs["attention_mask"].rows, [[1, 1, 1]])
        self.assertEqual(FakeNoGrad.entered, 1)

    def test_token_limit_respects_a_smaller_configured_limit(self):
        self.use_fakes()
        self.config = ModelConfig(model_id=MODEL_ID, backend="transformers", max_tokens=64)
        self.ask()
        self.assertEqual(self.model.generate_calls[0][1]["max_new_tokens"], 64)

    def test_decodes_only_the_newly_generated_tokens(self):
        self.use_fakes()
        answer, reasoning, finish = self.ask()
        self.assertEqual(answer, "Profit is 3000.\nFinal answer: 3000")   # no prompt, no <eos>
        self.assertNotIn("<sys>", answer)
        self.assertEqual((reasoning, finish), ("", "stop"))

    def test_dict_style_chat_template_output_is_supported(self):
        self.use_fakes(tokenizer=FakeTokenizer(dict_output=True))
        answer, _, _ = self.ask()
        self.assertEqual(answer, "Profit is 3000.\nFinal answer: 3000")

    def test_hitting_the_token_limit_is_reported_as_length(self):
        self.use_fakes(model=FakeModel(new_tokens=[1] * hf_adapter.MAX_NEW_TOKENS))
        self.assertEqual(self.ask()[2], "length")

    def test_generation_failure_becomes_a_runtime_error(self):
        self.use_fakes(model=FakeModel(error=NotImplementedError("operator not supported on MPS")))
        with self.assertRaises(hf_adapter.HFBackendError) as caught:
            self.ask()
        self.assertIsInstance(caught.exception, RuntimeError)
        self.assertIn("NotImplementedError", str(caught.exception))
        self.assertIn("mps", str(caught.exception))

    def test_time_limit_is_reported_as_timeout(self):
        self.use_fakes()
        clock = iter([0.0, self.config.task_time_limit + 1.0])
        with mock.patch.object(hf_adapter.time, "perf_counter", side_effect=lambda: next(clock)):
            with self.assertRaises(TimeoutError):
                self.ask()


if __name__ == "__main__":
    unittest.main()
