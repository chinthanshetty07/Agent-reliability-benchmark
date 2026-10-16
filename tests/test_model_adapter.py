"""
Tests for tasks/model_adapter.py.
Run from the project root with:
    python3 -m unittest discover -s tests -v
Every server reply here is FAKE (mocked), so no network, Ollama, or API key is needed.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent / "tasks"))

import ai_runner  # noqa: E402
import model_adapter  # noqa: E402
from model_adapter import ModelConfig, ModelConfigError  # noqa: E402

FAKE_KEY = "sk-test-DO-NOT-LEAK-12345"


class FakeResponse:
    """Pretends to be the object requests.get / requests.post returns."""

    def __init__(self, status_code=200, lines=(), text="", json_data=None):
        self.status_code = status_code
        self._lines = list(lines)
        self.text = text
        self._json_data = json_data

    def iter_lines(self, decode_unicode=True):
        return iter(self._lines)

    def json(self):
        if self._json_data is None:
            raise ValueError("no JSON")
        return self._json_data

    # Lets the fake be used in a "with ... as response:" block.
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def sse(delta, finish_reason=None):
    """One streamed line, in the format the server sends."""
    return "data: " + json.dumps({"choices": [{"delta": delta, "finish_reason": finish_reason}]})


GOOD_STREAM = [
    sse({"reasoning": "Revenue minus "}),
    sse({"reasoning": "expenses."}),
    "",                                   # blank keep-alive lines are skipped
    sse({"content": "Final answer: "}),
    sse({"content": "3000"}),
    sse({}, finish_reason="stop"),
    "data: [DONE]",
]


class TestLoadConfig(unittest.TestCase):
    def test_defaults_are_local_ollama(self):
        config = model_adapter.load_model_config({})
        self.assertEqual(config.base_url, "http://127.0.0.1:11434/v1")
        self.assertEqual(config.model_id, "qwen3:4b")
        self.assertIsNone(config.api_key)

    def test_environment_variables_are_used(self):
        config = model_adapter.load_model_config({
            "BENCHMARK_BASE_URL": "https://example.test/v1/",
            "BENCHMARK_MODEL_ID": "some-model",
            "BENCHMARK_API_KEY": FAKE_KEY,
        })
        self.assertEqual(config.base_url, "https://example.test/v1")   # trailing "/" removed
        self.assertEqual(config.model_id, "some-model")
        self.assertEqual(config.api_key, FAKE_KEY)

    def test_empty_values_fall_back_to_defaults(self):
        config = model_adapter.load_model_config(
            {"BENCHMARK_MODEL_ID": "  ", "BENCHMARK_API_KEY": ""}
        )
        self.assertEqual(config.model_id, "qwen3:4b")
        self.assertIsNone(config.api_key)

    def test_bad_address_gives_clear_error(self):
        with self.assertRaises(ModelConfigError) as caught:
            model_adapter.load_model_config({"BENCHMARK_BASE_URL": "localhost:11434"})
        self.assertIn("http://", str(caught.exception))

    def test_terminal_error_names_the_variable(self):
        with self.assertRaises(ModelConfigError) as caught:
            model_adapter.load_model_config({"BENCHMARK_BASE_URL": "localhost:11434"})
        self.assertIn("BENCHMARK_BASE_URL", str(caught.exception))

    def test_printing_config_does_not_show_key(self):
        config = ModelConfig(api_key=FAKE_KEY)
        self.assertNotIn(FAKE_KEY, repr(config))
        self.assertNotIn(FAKE_KEY, str(config))


class TestBuildConfig(unittest.TestCase):
    """build_model_config() checks values typed into the dashboard form."""

    def test_valid_values(self):
        config = model_adapter.build_model_config(" https://example.test/v1/ ", " m1 ", f" {FAKE_KEY} ")
        self.assertEqual(config.base_url, "https://example.test/v1")
        self.assertEqual(config.model_id, "m1")
        self.assertEqual(config.api_key, FAKE_KEY)

    def test_blank_key_means_no_key(self):
        self.assertIsNone(model_adapter.build_model_config("http://h/v1", "m", "   ").api_key)

    def test_invalid_addresses_are_rejected(self):
        for bad in ["", "   ", "localhost:11434", "ftp://host/v1", "http://", "not a url"]:
            with self.subTest(base_url=bad), self.assertRaises(ModelConfigError):
                model_adapter.build_model_config(bad, "m")

    def test_empty_model_id_is_rejected(self):
        with self.assertRaises(ModelConfigError):
            model_adapter.build_model_config("http://h/v1", "  ")

    def test_error_does_not_repeat_the_typed_value(self):
        # Someone might paste a key into the wrong box.
        with self.assertRaises(ModelConfigError) as caught:
            model_adapter.build_model_config(FAKE_KEY, "m")
        self.assertNotIn(FAKE_KEY, str(caught.exception))

    def test_no_network_request_is_made(self):
        with mock.patch("model_adapter.requests.get") as get, \
             mock.patch("model_adapter.requests.post") as post:
            model_adapter.build_model_config("http://h/v1", "m", FAKE_KEY)
            with self.assertRaises(ModelConfigError):
                model_adapter.build_model_config("bad", "m")
        get.assert_not_called()
        post.assert_not_called()


class TestHeaders(unittest.TestCase):
    def sent_headers(self, config):
        with mock.patch("model_adapter.requests.post", return_value=FakeResponse(lines=GOOD_STREAM)) as post:
            model_adapter.ask_model(config, "system", "question")
        return post.call_args.kwargs["headers"]

    def test_key_is_sent_when_set(self):
        headers = self.sent_headers(ModelConfig(api_key=FAKE_KEY))
        self.assertEqual(headers, {"Authorization": f"Bearer {FAKE_KEY}"})

    def test_no_authorization_header_without_key(self):
        self.assertEqual(self.sent_headers(ModelConfig()), {})


class TestAskModel(unittest.TestCase):
    def ask(self, response, config=None):
        with mock.patch("model_adapter.requests.post", return_value=response) as post:
            reply = model_adapter.ask_model(config or ModelConfig(), "system prompt", "question?")
        return reply, post

    def test_stream_is_assembled(self):
        reply, post = self.ask(FakeResponse(lines=GOOD_STREAM))
        self.assertEqual(reply, ("Final answer: 3000", "Revenue minus expenses.", "stop"))

        # The request itself is what the benchmark has always sent.
        sent = post.call_args
        self.assertEqual(sent.args[0], "http://127.0.0.1:11434/v1/chat/completions")
        self.assertEqual(sent.kwargs["json"]["model"], "qwen3:4b")
        self.assertEqual(sent.kwargs["json"]["max_tokens"], 4096)
        self.assertTrue(sent.kwargs["json"]["stream"])
        self.assertEqual(sent.kwargs["timeout"], (10, 120))

    def test_reasoning_content_field_also_works(self):
        lines = [sse({"reasoning_content": "thinking"}), sse({"content": "42"}, "stop")]
        reply, _ = self.ask(FakeResponse(lines=lines))
        self.assertEqual(reply, ("42", "thinking", "stop"))

    def test_malformed_chunk_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.ask(FakeResponse(lines=["data: {not json"]))

    def test_error_inside_stream_raises(self):
        with self.assertRaises(RuntimeError) as caught:
            self.ask(FakeResponse(lines=['data: {"error": {"message": "model crashed"}}']))
        self.assertIn("model crashed", str(caught.exception))

    def test_http_error_hides_the_key(self):
        # Some servers repeat the key back in their error message.
        response = FakeResponse(status_code=401, text=f"Invalid API key: {FAKE_KEY}")
        with self.assertRaises(RuntimeError) as caught:
            self.ask(response, ModelConfig(api_key=FAKE_KEY))
        message = str(caught.exception)
        self.assertIn("HTTP 401", message)
        self.assertIn("***", message)
        self.assertNotIn(FAKE_KEY, message)


class TestCheckModelAvailable(unittest.TestCase):
    def check(self, response=None, error=None, config=None):
        patch = mock.patch("model_adapter.requests.get", return_value=response, side_effect=error)
        with patch:
            return model_adapter.check_model_available(config or ModelConfig())

    def test_model_found(self):
        response = FakeResponse(json_data={"data": [{"id": "qwen3:4b"}]})
        self.assertIsNone(self.check(response))

    def test_model_missing_lists_available_models(self):
        response = FakeResponse(json_data={"data": [{"id": "llama3:8b"}]})
        message = self.check(response)
        self.assertIn("'qwen3:4b' was not found", message)
        self.assertIn("llama3:8b", message)

    def test_connection_refused_gives_clear_message(self):
        message = self.check(error=model_adapter.requests.ConnectionError("refused"))
        self.assertIn("Could not connect", message)

    def test_unauthorized_hides_the_key(self):
        response = FakeResponse(status_code=401, text=f"bad key {FAKE_KEY}")
        message = self.check(response, config=ModelConfig(api_key=FAKE_KEY))
        self.assertIn("HTTP 401", message)
        self.assertNotIn(FAKE_KEY, message)


class TestSafeBaseUrl(unittest.TestCase):
    def test_login_details_and_query_are_removed(self):
        url = "https://user:password@example.test/v1?key=abc"
        self.assertEqual(model_adapter.safe_base_url(url), "https://example.test/v1")

    def test_normal_address_is_unchanged(self):
        url = "http://127.0.0.1:11434/v1"
        self.assertEqual(model_adapter.safe_base_url(url), url)


class TestNoSecretInSavedResults(unittest.TestCase):
    task = {
        "task_id": "finance_001",
        "category": "financial_calculation",
        "difficulty": "easy",
        "question": "Revenue 10000, expenses 7000. Profit?",
        "expected_answer": "3000",
    }

    def run_and_save(self, response):
        """Run one task through the real runner with a fake server; save to a temp folder."""
        config = ModelConfig(base_url="https://user:pw@example.test/v1", api_key=FAKE_KEY)
        with tempfile.TemporaryDirectory() as folder:
            result_file = Path(folder) / "results.json"
            with mock.patch.object(ai_runner, "MODEL_CONFIG", config), \
                 mock.patch.object(ai_runner, "RESULT_FILE", result_file), \
                 mock.patch("model_adapter.requests.post", return_value=response) as post:
                result = ai_runner.run_task(self.task)
                run_info = {
                    "model": config.model_id,
                    "base_url": model_adapter.safe_base_url(config.base_url),
                }
                ai_runner.save_results(run_info, [result], total_tasks=1)
            saved_text = result_file.read_text()
        # The key really was used for the request, but must not be in the file.
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], f"Bearer {FAKE_KEY}")
        return result, saved_text

    def test_successful_run_saves_no_secret(self):
        result, saved = self.run_and_save(FakeResponse(lines=GOOD_STREAM))
        self.assertEqual(result["status"], "correct")
        self.assertNotIn(FAKE_KEY, saved)
        self.assertNotIn("user:pw", saved)

    def test_failed_run_saves_no_secret(self):
        result, saved = self.run_and_save(FakeResponse(status_code=401, text=f"Invalid key {FAKE_KEY}"))
        self.assertEqual(result["status"], "error")
        self.assertIn("HTTP 401", result["error"])
        self.assertNotIn(FAKE_KEY, saved)

    def test_result_fields_are_unchanged(self):
        # Same per-task fields as results.json had before this change.
        result, _ = self.run_and_save(FakeResponse(lines=GOOD_STREAM))
        self.assertEqual(list(result), [
            "task_id", "category", "difficulty", "question", "expected_answer", "model",
            "model_answer", "model_reasoning", "extracted_answer", "extraction_method",
            "grading_note", "correct", "status", "response_time_seconds", "finish_reason",
            "error", "warning",
        ])


class TestBackendSelection(unittest.TestCase):
    """BENCHMARK_BACKEND chooses who answers; the default is unchanged (Ollama / OpenAI-compatible)."""

    HF_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

    def test_default_backend_is_unchanged(self):
        config = model_adapter.load_model_config({})
        self.assertEqual(config.backend, "openai")
        self.assertEqual(config.base_url, "http://127.0.0.1:11434/v1")
        self.assertEqual(config.model_id, "qwen3:4b")
        self.assertEqual(ModelConfig().backend, "openai")

    def test_explicit_openai_backend_behaves_like_the_default(self):
        config = model_adapter.load_model_config({"BENCHMARK_BACKEND": "openai",
                                                  "BENCHMARK_BASE_URL": "https://example.test/v1/"})
        self.assertEqual((config.backend, config.base_url), ("openai", "https://example.test/v1"))

    def test_transformers_backend_needs_no_address_or_key(self):
        config = model_adapter.load_model_config({
            "BENCHMARK_BACKEND": "Transformers",           # case doesn't matter
            "BENCHMARK_MODEL_ID": self.HF_MODEL,
            "BENCHMARK_BASE_URL": "https://ignored.example/v1",
            "BENCHMARK_API_KEY": FAKE_KEY,
        })
        self.assertEqual(config.backend, "transformers")
        self.assertEqual(config.model_id, self.HF_MODEL)
        self.assertEqual(config.base_url, "transformers://local")
        self.assertIsNone(config.api_key)                   # a local model never receives a key

    def test_local_backend_never_keeps_a_key_even_if_given_directly(self):
        config = model_adapter.build_model_config(None, self.HF_MODEL, FAKE_KEY, backend="transformers")
        self.assertIsNone(config.api_key)
        self.assertEqual(config.base_url, "transformers://local")

    def test_transformers_backend_requires_an_explicit_model(self):
        with self.assertRaises(ModelConfigError) as caught:
            model_adapter.load_model_config({"BENCHMARK_BACKEND": "transformers"})
        self.assertIn("BENCHMARK_MODEL_ID", str(caught.exception))

    def test_unknown_backend_is_rejected(self):
        with self.assertRaises(ModelConfigError) as caught:
            model_adapter.load_model_config({"BENCHMARK_BACKEND": "vllm"})
        self.assertIn("BENCHMARK_BACKEND", str(caught.exception))
        with self.assertRaises(ModelConfigError):
            model_adapter.build_model_config("http://h/v1", "m", backend="vllm")

    def test_dashboard_style_call_is_unchanged(self):
        # dashboard.py and run_launcher.py call build_model_config(base_url, model_id, api_key).
        config = model_adapter.build_model_config("http://h/v1", "m", FAKE_KEY)
        self.assertEqual((config.backend, config.base_url, config.api_key), ("openai", "http://h/v1", FAKE_KEY))
        with self.assertRaises(ModelConfigError):
            model_adapter.build_model_config("transformers://local", "m")   # still http/https only

    def test_transformers_requests_go_to_the_local_adapter_not_the_network(self):
        import hf_adapter
        config = model_adapter.build_model_config(None, self.HF_MODEL, backend="transformers")
        with mock.patch.object(hf_adapter, "ask_model", return_value=("Final answer: 1", "", "stop")) as ask, \
             mock.patch.object(hf_adapter, "check_model_available", return_value=None) as check, \
             mock.patch("model_adapter.requests.post") as post, \
             mock.patch("model_adapter.requests.get") as get:
            self.assertIsNone(model_adapter.check_model_available(config))
            reply = model_adapter.ask_model(config, "system", "question")
        self.assertEqual(reply, ("Final answer: 1", "", "stop"))
        check.assert_called_once_with(self.HF_MODEL)
        ask.assert_called_once_with(config, "system", "question")
        post.assert_not_called()
        get.assert_not_called()

    def test_openai_requests_do_not_touch_the_local_adapter(self):
        import hf_adapter
        with mock.patch.object(hf_adapter, "ask_model") as ask, \
             mock.patch("model_adapter.requests.post", return_value=FakeResponse(lines=GOOD_STREAM)):
            model_adapter.ask_model(ModelConfig(), "system", "question")
        ask.assert_not_called()

    def test_saved_base_url_for_local_runs(self):
        self.assertEqual(model_adapter.safe_base_url("transformers://local"), "transformers://local")

    def test_full_runner_run_with_a_fake_local_model_keeps_the_results_format(self):
        # The real runner, with the local backend faked (no torch, no download),
        # writing only to a temporary folder.
        import hf_adapter
        expected = {t["question"]: t["expected_answer"]
                    for t in json.loads((Path(__file__).parent.parent / "tasks" / "benchmark.json").read_text())}

        def fake_answer(config, system_prompt, question):
            return f"Final answer: {expected[question]}", "", "stop"

        with tempfile.TemporaryDirectory() as folder, \
             mock.patch.dict("os.environ", {"BENCHMARK_BACKEND": "transformers",
                                            "BENCHMARK_MODEL_ID": self.HF_MODEL}), \
             mock.patch.multiple(ai_runner, MODEL_CONFIG=ai_runner.MODEL_CONFIG,
                                 RESULTS_DIR=ai_runner.RESULTS_DIR, RESULT_FILE=ai_runner.RESULT_FILE,
                                 HISTORY_DIR=ai_runner.HISTORY_DIR), \
             mock.patch.object(hf_adapter, "check_model_available", return_value=None), \
             mock.patch.object(hf_adapter, "ask_model", side_effect=fake_answer), \
             mock.patch("model_adapter.requests.post") as post, \
             mock.patch("builtins.print"):
            ai_runner.main(["--results-dir", str(Path(folder) / "hf-run")])
            saved = json.loads((Path(folder) / "hf-run" / "results.json").read_text())
        post.assert_not_called()
        self.assertEqual(saved["model"], self.HF_MODEL)
        self.assertEqual(saved["base_url"], "transformers://local")
        self.assertEqual(list(saved), ["model", "base_url", "task_file", "started_at",
                                       "finished_at", "run_status", "summary", "results"])
        self.assertEqual(saved["run_status"], "finished")
        self.assertEqual(saved["summary"]["correct"], len(expected))


if __name__ == "__main__":
    unittest.main()
