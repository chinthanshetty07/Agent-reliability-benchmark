"""
Tests for the dashboard's "Configure model" / "Test connection" sidebar.
Run from the project root with:
    python3 -m unittest discover -s tests -v

Streamlit's AppTest runs dashboard.py like a browser would, but inside this
test. All network calls are FAKE (mocked): no Ollama, internet, or API key needed.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import requests
from streamlit.testing.v1 import AppTest

PROJECT = Path(__file__).parent.parent
DASHBOARD = PROJECT / "dashboard.py"
FAKE_KEY = "sk-test-DO-NOT-LEAK-67890"


class FakeResponse:
    """Pretends to be what requests.get returns."""

    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    def json(self):
        if self._json_data is None:
            raise ValueError("no JSON")
        return self._json_data


MODEL_LIST = FakeResponse(json_data={"data": [{"id": "qwen3:4b"}, {"id": "llama3.2:3b"}]})


def page_text(at):
    """Every piece of text on the rendered page (main area AND sidebar)."""
    parts = []

    def walk(node):
        proto = getattr(node, "proto", None)
        if proto is not None:
            parts.append(str(proto))
        for child in getattr(node, "children", {}).values():
            walk(child)

    walk(at._tree)
    return "\n".join(parts)


class DashboardTestCase(unittest.TestCase):
    def setUp(self):
        # Fake network: GET /models is controlled per test; POST (chat) must never happen.
        self.get = mock.patch("requests.get").start()
        self.post = mock.patch("requests.post").start()
        # Start every test with no BENCHMARK_* settings from the real terminal.
        mock.patch.dict(os.environ).start()
        for name in list(os.environ):
            if name.startswith("BENCHMARK_"):
                del os.environ[name]
        self.addCleanup(mock.patch.stopall)

    def tearDown(self):
        # No test may ever reach the chat endpoint, which would generate output and cost tokens.
        self.post.assert_not_called()

    def open_dashboard(self, path=None):
        at = AppTest.from_file(str(path or DASHBOARD), default_timeout=30).run()
        self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
        return at

    def submit(self, at, base_url=None, model_id=None, api_key=None):
        inputs = {t.label: t for t in at.sidebar.text_input}
        if base_url is not None:
            inputs["Base URL"].input(base_url)
        if model_id is not None:
            inputs["Model ID"].input(model_id)
        if api_key is not None:
            inputs["API key (optional)"].input(api_key)
        next(b for b in at.button if b.label == "Test connection").click().run()
        self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
        return at


class TestConnectionResults(DashboardTestCase):
    def test_success(self):
        self.get.return_value = MODEL_LIST
        at = self.submit(self.open_dashboard())
        self.assertEqual(
            [s.value for s in at.sidebar.success],
            ["Connected to http://127.0.0.1:11434/v1. Model 'qwen3:4b' is available."],
        )
        self.assertEqual(self.get.call_args.args[0], "http://127.0.0.1:11434/v1/models")

    def test_missing_model(self):
        self.get.return_value = MODEL_LIST
        at = self.submit(self.open_dashboard(), model_id="not-installed:1b")
        message = at.sidebar.error[0].value
        self.assertIn("'not-installed:1b' was not found", message)
        self.assertIn("llama3.2:3b", message)        # lists what IS available

    def test_server_not_running(self):
        self.get.side_effect = requests.ConnectionError("refused")
        at = self.submit(self.open_dashboard())
        self.assertIn("Could not connect", at.sidebar.error[0].value)

    def test_server_error(self):
        self.get.return_value = FakeResponse(status_code=500, text="internal error")
        at = self.submit(self.open_dashboard())
        self.assertIn("HTTP 500", at.sidebar.error[0].value)

    def test_unexpected_error_shows_message_not_traceback(self):
        self.get.side_effect = RuntimeError("something odd")
        at = self.submit(self.open_dashboard())   # submit() also checks no exception/traceback
        self.assertIn("Unexpected error", at.sidebar.error[0].value)

    def test_invalid_url_is_rejected_before_any_request(self):
        for bad in ["localhost:11434", "ftp://example.test/v1", ""]:
            with self.subTest(base_url=bad):
                at = self.submit(self.open_dashboard(), base_url=bad)
                self.assertIn("Base URL must be a full address", at.sidebar.error[0].value)
        self.get.assert_not_called()

    def test_empty_model_id_is_rejected_before_any_request(self):
        at = self.submit(self.open_dashboard(), model_id="  ")
        self.assertIn("Model ID must not be empty", at.sidebar.error[0].value)
        self.get.assert_not_called()

    def test_nothing_is_sent_until_the_button_is_clicked(self):
        self.open_dashboard()
        self.get.assert_not_called()


class TestApiKeySafety(DashboardTestCase):
    def test_key_is_sent_as_header_only_when_given(self):
        self.get.return_value = MODEL_LIST
        at = self.submit(self.open_dashboard())
        self.assertEqual(self.get.call_args.kwargs["headers"], {})
        self.submit(at, api_key=FAKE_KEY)
        self.assertEqual(self.get.call_args.kwargs["headers"], {"Authorization": f"Bearer {FAKE_KEY}"})

    def test_key_echoed_by_server_is_not_displayed(self):
        # Some servers repeat the key back in their error message.
        self.get.return_value = FakeResponse(status_code=401, text=f"Invalid API key: {FAKE_KEY}")
        at = self.submit(self.open_dashboard(), api_key=FAKE_KEY)
        self.assertIn("HTTP 401", at.sidebar.error[0].value)
        self.assertIn("***", at.sidebar.error[0].value)
        self.assertNotIn(FAKE_KEY, page_text(at))

    def test_key_in_unexpected_error_is_not_displayed(self):
        self.get.side_effect = RuntimeError(f"failed with {FAKE_KEY}")
        at = self.submit(self.open_dashboard(), api_key=FAKE_KEY)
        self.assertNotIn(FAKE_KEY, page_text(at))

    def test_key_not_displayed_after_success(self):
        self.get.return_value = MODEL_LIST
        at = self.submit(self.open_dashboard(), api_key=FAKE_KEY)
        self.assertEqual(len(at.sidebar.success), 1)
        self.assertNotIn(FAKE_KEY, page_text(at))

    def test_page_text_check_would_catch_a_leak(self):
        # Proves page_text() really sees sidebar messages, so the tests above mean something.
        self.get.return_value = FakeResponse(status_code=404, text="no such model")
        at = self.submit(self.open_dashboard())
        self.assertIn("no such model", page_text(at))

    def test_dashboard_never_uses_streamlit_cache(self):
        # Streamlit caches are shared by all users, so nothing holding the key may be cached.
        self.assertNotIn("st.cache", DASHBOARD.read_text())


class TestPrefill(DashboardTestCase):
    def inputs(self, at):
        return {t.label: t.value for t in at.sidebar.text_input}

    def test_defaults_without_environment_variables(self):
        values = self.inputs(self.open_dashboard())
        self.assertEqual(values["Base URL"], "http://127.0.0.1:11434/v1")
        self.assertEqual(values["Model ID"], "qwen3:4b")
        self.assertEqual(values["API key (optional)"], "")

    def test_prefilled_from_environment_but_key_never(self):
        os.environ["BENCHMARK_BASE_URL"] = "https://user:pw@example.test/v1"
        os.environ["BENCHMARK_MODEL_ID"] = "provider-model"
        os.environ["BENCHMARK_API_KEY"] = FAKE_KEY
        at = self.open_dashboard()
        values = self.inputs(at)
        self.assertEqual(values["Base URL"], "https://example.test/v1")   # login details removed
        self.assertEqual(values["Model ID"], "provider-model")
        self.assertEqual(values["API key (optional)"], "")                # never pre-filled
        self.assertNotIn(FAKE_KEY, page_text(at))
        self.assertNotIn("user:pw", page_text(at))


class TestExistingDashboardStillWorks(DashboardTestCase):
    def test_results_sections_render(self):
        at = self.open_dashboard()
        labels = [m.label for m in at.metric]
        for label in ["Accuracy", "Correct", "Incorrect", "Errors",
                      "Average response time", "Model", "Run status", "Completed trials"]:
            self.assertIn(label, labels)
        self.assertEqual(len(at.dataframe), 3)   # tasks, trials, consistency
        self.assertEqual(len(at.main.error), 0)

    def test_results_still_render_after_a_connection_test(self):
        self.get.return_value = MODEL_LIST
        at = self.submit(self.open_dashboard())
        self.assertEqual(len(at.dataframe), 3)

    def test_panel_works_when_results_file_is_missing(self):
        # A copy of the app in an empty folder: no results/ at all.
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "tasks").mkdir()
            shutil.copy(DASHBOARD, Path(folder) / "dashboard.py")
            shutil.copy(PROJECT / "tasks" / "model_adapter.py", Path(folder) / "tasks")
            self.get.return_value = MODEL_LIST

            at = self.open_dashboard(Path(folder) / "dashboard.py")
            self.assertIn("results/results.json was not found", at.main.error[0].value)
            self.submit(at)
            self.assertEqual(len(at.sidebar.success), 1)


if __name__ == "__main__":
    unittest.main()
