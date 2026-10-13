"""
Tests for running a benchmark from the dashboard (Run, progress, Stop).
Run from the project root with:
    python3 -m unittest discover -s tests -v

Uses Streamlit's AppTest with FAKE runner scripts (no real runner, no model,
no network) and TEMPORARY run folders. Each test checks that the real
results/ folder is unchanged and that no real results/runs/ folder appears.
Key checks use assertFalse(...) so a failure never prints the key.

Note: AppTest re-runs the whole page where a browser would re-run only the
progress section; that part is checked by hand in a browser.
"""

import gc
import hashlib
import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import streamlit as st
from streamlit.testing.v1 import AppTest

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "tasks"))
sys.path.insert(0, str(PROJECT / "tests"))

import run_launcher  # noqa: E402
from model_adapter import ModelConfig  # noqa: E402
from test_run_launcher import (  # noqa: E402  (fake runner scripts only)
    FAKE_FAIL, FAKE_OK, FAKE_SLOW, FAKE_START, FAKE_STARTUP, FAKE_STUBBORN, FAKE_UNREADABLE,
)

DASHBOARD = PROJECT / "dashboard.py"
REAL_RESULTS = PROJECT / "results"
FAKE_KEY = "sk-test-DO-NOT-LEAK-DASHBOARD-13579"
FAKE_KEY_SHA256 = hashlib.sha256(FAKE_KEY.encode()).hexdigest()
REGISTRY_FIELDS = {"process", "folder", "model_id", "endpoint", "started_at", "stop_result"}

# Every task "fails", yet the run finishes normally with exit code 0.
FAKE_ALL_FAILED = FAKE_START + """
data = {"run_status": "finished", "summary": {"completed_tasks": 3, "total_tasks": 3,
        "correct": 0, "incorrect": 0, "errors": 3}, "results": []}
(folder / "results.json").write_text(json.dumps(data))
"""


def fingerprint(folder):
    return {
        str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def page_text(at):
    """Every piece of text on the rendered page (main area and sidebar)."""
    parts = []

    def walk(node):
        proto = getattr(node, "proto", None)
        if proto is not None:
            parts.append(str(proto))
        for child in getattr(node, "children", {}).values():
            walk(child)

    walk(at._tree)
    return "\n".join(parts)


def find_registry():
    """The dashboard's shared registry dict, found in memory."""
    for obj in gc.get_objects():
        if isinstance(obj, dict) and set(obj) == {"lock", "run", "confirmed_not_running"}:
            return obj
    return None


def deep_contains(obj, predicate, seen=None, depth=0):
    """True if predicate() matches obj or anything reachable inside it."""
    seen = set() if seen is None else seen
    if id(obj) in seen or depth > 8:
        return False
    seen.add(id(obj))
    if predicate(obj):
        return True
    if isinstance(obj, dict):
        children = list(obj.keys()) + list(obj.values())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        children = list(obj)
    elif hasattr(obj, "__dict__") and not isinstance(obj, type):
        children = list(vars(obj).values())
    else:
        children = []
    return any(deep_contains(child, predicate, seen, depth + 1) for child in children)


def holds_key_or_settings(obj):
    if isinstance(obj, ModelConfig):
        return True
    if isinstance(obj, (str, bytes, os.PathLike)):
        return FAKE_KEY in (obj.decode(errors="replace") if isinstance(obj, bytes) else str(obj))
    return False


class DashboardRunTestCase(unittest.TestCase):
    def setUp(self):
        self.real_results_before = fingerprint(REAL_RESULTS)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.temp = Path(temp.name).resolve()
        self.runs_dir = self.temp / "runs"

        # Runs go to a temp folder, and the "runner" is a fake script (never ai_runner.py).
        mock.patch.object(run_launcher, "RUNS_DIR", self.runs_dir).start()
        self.use_runner(FAKE_OK)
        # Record every process the dashboard starts, so none is ever left running.
        self.started = []
        real_start_run = run_launcher.start_run

        def spy_start_run(config, folder, runner=None):
            process = real_start_run(config, folder, runner=runner)
            self.started.append(process)
            return process

        mock.patch.object(run_launcher, "start_run", side_effect=spy_start_run).start()
        # No network at all: these fail the test if the dashboard tries to use them.
        self.get = mock.patch("requests.get").start()
        self.post = mock.patch("requests.post").start()
        mock.patch.dict(os.environ).start()
        for name in list(os.environ):
            if name.startswith("BENCHMARK_"):
                del os.environ[name]
        self.addCleanup(mock.patch.stopall)
        # The registry is shared by the whole program; start each test with an empty one.
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)

    def tearDown(self):
        for process in self.started:
            if process.poll() is None:
                process.kill()
                process.wait()
        self.get.assert_not_called()
        self.post.assert_not_called()
        self.assertEqual(fingerprint(REAL_RESULTS), self.real_results_before, "the real results/ changed")
        self.assertFalse((REAL_RESULTS / "runs").exists(), "a run was written to the real results/runs/")

    def use_runner(self, source):
        path = self.temp / f"fake_runner_{len(list(self.temp.glob('fake_runner_*')))}.py"
        path.write_text(textwrap.dedent(source))
        mock.patch.object(run_launcher, "RUNNER", path).start()

    def open_dashboard(self):
        at = AppTest.from_file(str(DASHBOARD), default_timeout=30).run()
        self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
        return at

    def click(self, at, label):
        next(b for b in at.button if b.label == label).click().run()
        self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
        return at

    def run_benchmark(self, at, base_url=None, model_id=None, api_key=None):
        inputs = {t.label: t for t in at.sidebar.text_input}
        if base_url is not None:
            inputs["Base URL"].input(base_url)
        if model_id is not None:
            inputs["Model ID"].input(model_id)
        if api_key is not None:
            inputs["API key (optional)"].input(api_key)
        return self.click(at, "Run benchmark")

    def wait_until(self, condition, timeout=20):
        import time
        deadline = time.time() + timeout
        while time.time() < deadline:
            if condition():
                return
            time.sleep(0.05)
        self.fail("timed out waiting")

    def run_folders(self):
        return sorted(self.runs_dir.iterdir()) if self.runs_dir.exists() else []

    def folder_of(self, index):
        """
        The run folder of the index-th started run, taken from its own command line
        (the folder follows --results-dir). Folder names have one-second precision,
        so sorting them by name does NOT reliably give the order runs were started.
        """
        return Path(self.started[index].args[-1])

    def messages(self, at, kind):
        return [m.value for m in getattr(at.sidebar, kind)]


class TestRunButton(DashboardRunTestCase):
    def test_run_starts_one_run_in_a_new_folder(self):
        at = self.run_benchmark(self.open_dashboard())
        self.assertEqual(len(self.started), 1)
        self.started[0].wait(timeout=20)
        at.run()
        self.assertIn("Finished: 3/3 correct.", self.messages(at, "success"))
        folders = self.run_folders()
        self.assertEqual(len(folders), 1)
        self.assertEqual(sorted(p.name for p in folders[0].iterdir()),
                         ["env_report.json", "launcher.json", "results.json", "runner.log"])

    def test_second_run_is_refused_while_one_is_active(self):
        self.use_runner(FAKE_SLOW)
        at = self.run_benchmark(self.open_dashboard())
        self.run_benchmark(at)
        self.assertIn("A benchmark is already running. Stop it or wait for it to finish.",
                      self.messages(at, "error"))
        self.assertEqual(len(self.started), 1)
        self.assertEqual(len(self.run_folders()), 1)

    def test_invalid_settings_start_nothing(self):
        for kwargs, expected in [
            ({"base_url": "localhost:11434"}, "Base URL must be a full address"),
            ({"base_url": "ftp://example.test/v1"}, "Base URL must be a full address"),
            ({"model_id": "   "}, "Model ID must not be empty"),
        ]:
            with self.subTest(**kwargs):
                at = self.run_benchmark(self.open_dashboard(), **kwargs)
                self.assertTrue(any(expected in m for m in self.messages(at, "error")))
        self.assertEqual(self.started, [])
        self.assertFalse(self.runs_dir.exists())       # not even a folder was created

    def test_new_run_after_the_previous_finished_gets_a_new_folder(self):
        at = self.run_benchmark(self.open_dashboard())
        self.started[0].wait(timeout=20)
        first = self.folder_of(0)
        first_files = fingerprint(first)
        self.run_benchmark(at.run())
        self.started[1].wait(timeout=20)
        self.assertEqual(len(self.run_folders()), 2)
        self.assertEqual(fingerprint(first), first_files)   # the earlier run was not touched


class TestApiKeySafety(DashboardRunTestCase):
    def test_registry_holds_no_key_and_no_settings_object(self):
        self.use_runner(FAKE_SLOW)
        self.run_benchmark(self.open_dashboard(), api_key=FAKE_KEY)
        registry = find_registry()
        self.assertIsNotNone(registry)
        self.assertEqual(set(registry["run"]), REGISTRY_FIELDS)
        self.assertFalse(deep_contains(registry, holds_key_or_settings),
                         "the registry holds the API key or a model settings object")

    def test_key_reaches_the_run_but_is_never_shown_or_written(self):
        # Check the page at EVERY step: right after clicking Run, while the run is
        # active, and after it has finished (a leak could appear at any one of them).
        self.use_runner(FAKE_SLOW)
        at = self.run_benchmark(self.open_dashboard(), api_key=FAKE_KEY)
        self.assertFalse(FAKE_KEY in page_text(at), "API key shown right after clicking Run")
        folder = self.folder_of(0)
        self.wait_until(lambda: (folder / "results.json").exists())
        at.run()
        self.assertFalse(FAKE_KEY in page_text(at), "API key shown while the run is active")
        self.click(at, "Stop benchmark")
        self.assertFalse(FAKE_KEY in page_text(at), "API key shown after the run ended")

        # The key did reach the run (FAKE_OK records a hash of it, never the key itself).
        self.use_runner(FAKE_OK)
        at = self.run_benchmark(at.run(), api_key=FAKE_KEY)
        self.started[1].wait(timeout=20)
        at.run()
        self.assertFalse(FAKE_KEY in page_text(at), "API key shown after the second run")
        report = json.loads((self.folder_of(1) / "env_report.json").read_text())
        self.assertEqual(report["key_sha256"], FAKE_KEY_SHA256)

        # Neither run had the key in its command line or in any file it wrote.
        for process in self.started:
            self.assertFalse(FAKE_KEY in " ".join(map(str, process.args)), "API key in the command line")
        for run_folder in self.run_folders():
            for path in run_folder.rglob("*"):
                if path.is_file():
                    self.assertFalse(FAKE_KEY in path.read_text(errors="replace"),
                                     f"API key found in {run_folder.name}/{path.name}")


class TestProgressAndStop(DashboardRunTestCase):
    def test_progress_is_shown_while_running(self):
        self.use_runner(FAKE_SLOW)
        at = self.run_benchmark(self.open_dashboard())
        folder = self.folder_of(0)
        self.wait_until(lambda: (run_launcher.read_progress(folder)["completed"] or 0) >= 2)
        at.run()
        self.assertTrue(any(m.startswith("Running: 2 of 3") for m in self.messages(at, "info")))
        self.assertTrue(any(b.label == "Stop benchmark" for b in at.button))

    def test_stop_reports_the_real_saved_state(self):
        self.use_runner(FAKE_SLOW)
        at = self.run_benchmark(self.open_dashboard())
        folder = self.folder_of(0)
        self.wait_until(lambda: (folder / "results.json").exists())
        at.run()
        self.click(at, "Stop benchmark")
        process = self.started[0]
        self.assertIsNotNone(process.poll())                              # really ended
        self.assertEqual(json.loads((folder / "results.json").read_text())["run_status"], "interrupted")
        self.assertIn("Stopped: 2 of 3 tasks saved.", self.messages(at, "warning"))
        self.assertFalse(any(b.label == "Stop benchmark" for b in at.button))

    def test_forced_stop_is_reported_honestly(self):
        self.use_runner(FAKE_STUBBORN)                                   # ignores Ctrl+C
        mock.patch.object(run_launcher, "STOP_GRACE_SECONDS", 1).start()
        at = self.run_benchmark(self.open_dashboard())
        folder = self.folder_of(0)
        self.wait_until(lambda: "ready" in (folder / "runner.log").read_text())
        at.run()
        self.click(at, "Stop benchmark")
        self.assertIsNotNone(self.started[0].poll())
        # It never saved a final status; the dashboard must say so.
        self.assertEqual(json.loads((folder / "results.json").read_text())["run_status"], "running")
        self.assertTrue(any("without saving a final status" in m for m in self.messages(at, "error")))
        self.assertTrue(any("ended forcibly" in c.value for c in at.sidebar.caption))

    def test_stop_during_startup(self):
        self.use_runner(FAKE_STARTUP)
        at = self.run_benchmark(self.open_dashboard())
        folder = self.folder_of(0)
        self.wait_until(lambda: "starting" in (folder / "runner.log").read_text())
        at.run()
        self.click(at, "Stop benchmark")
        self.assertEqual(self.started[0].poll(), -2)
        self.assertFalse((folder / "results.json").exists())
        self.assertTrue(any("Stopped before the first task" in m for m in self.messages(at, "warning")))

    def test_every_task_failed_is_not_shown_as_success(self):
        self.use_runner(FAKE_ALL_FAILED)
        at = self.run_benchmark(self.open_dashboard())
        self.assertEqual(self.started[0].wait(timeout=20), 0)            # exit code 0...
        at.run()
        self.assertTrue(any("every task failed" in m for m in self.messages(at, "error")))
        self.assertEqual(self.messages(at, "success"), [])               # ...but not "success"

    def test_failed_start_shows_the_log(self):
        self.use_runner(FAKE_FAIL)
        at = self.run_benchmark(self.open_dashboard())
        self.started[0].wait(timeout=20)
        at.run()
        self.assertTrue(any("Could not start the benchmark" in m for m in self.messages(at, "error")))
        self.assertTrue(any("Could not connect" in c.value for c in at.sidebar.code))


class TestStartErrors(DashboardRunTestCase):
    """Errors raised by start_run() must appear as a short, redacted message, never a traceback."""

    def run_with_start_error(self, error, api_key=None):
        with mock.patch.object(run_launcher, "start_run", side_effect=error):
            return self.run_benchmark(self.open_dashboard(), api_key=api_key)   # also checks: no exception

    def test_value_error_is_a_friendly_message(self):
        at = self.run_with_start_error(ValueError("embedded null byte"))
        self.assertEqual(len(at.exception), 0)                                    # no traceback on the page
        self.assertIn("Could not start the benchmark (ValueError): embedded null byte",
                      self.messages(at, "error"))
        self.assertIsNone(find_registry()["run"])                                 # no run registered
        self.assertEqual(self.started, [])                                        # no process started

    def test_unexpected_error_containing_the_key_is_redacted(self):
        at = self.run_with_start_error(ValueError(f"bad value {FAKE_KEY}"), api_key=FAKE_KEY)
        errors = self.messages(at, "error")
        self.assertTrue(any("Could not start the benchmark (ValueError): bad value ***" in e for e in errors))
        self.assertFalse(FAKE_KEY in page_text(at), "API key shown on the dashboard")

    def test_real_invalid_setting_end_to_end(self):
        # No mock: a key with an invalid null character makes the real launcher's
        # process start raise ValueError; the launcher cleans up, the page stays friendly.
        at = self.run_benchmark(self.open_dashboard(), api_key="fake\x00key")
        self.assertEqual(len(at.exception), 0)
        self.assertTrue(any("Could not start the benchmark (ValueError)" in e for e in self.messages(at, "error")))
        self.assertIsNone(find_registry()["run"])
        self.assertEqual(run_launcher.runs_without_final_status(), [])           # nothing left blocking

    def test_launch_error_and_os_error_messages_are_unchanged(self):
        for error, expected in [
            (run_launcher.LaunchError("disk full"), "Could not start the benchmark: disk full"),
            (OSError("no such file"), "Could not start the benchmark: no such file"),
        ]:
            with self.subTest(error=type(error).__name__):
                at = self.run_with_start_error(error)
                self.assertIn(expected, self.messages(at, "error"))

    def test_a_new_run_can_start_after_an_unexpected_error(self):
        at = self.run_with_start_error(ValueError("embedded null byte"))
        self.run_benchmark(at)                                                    # real (fake-runner) start now
        self.assertEqual(len(self.started), 1)
        self.started[0].wait(timeout=20)


class TestSharedRegistry(DashboardRunTestCase):
    def test_a_new_session_sees_and_can_stop_the_active_run(self):
        # A browser refresh or a second tab is a new session; the registry is shared.
        self.use_runner(FAKE_SLOW)
        first_tab = self.run_benchmark(self.open_dashboard())
        folder = self.folder_of(0)
        self.wait_until(lambda: (folder / "results.json").exists())

        second_tab = self.open_dashboard()
        self.assertTrue(any(b.label == "Stop benchmark" for b in second_tab.button))
        self.click(second_tab, "Stop benchmark")
        self.assertIsNotNone(self.started[0].poll())
        first_tab.run()
        self.assertIn("Stopped: 2 of 3 tasks saved.", self.messages(first_tab, "warning"))

    def test_a_second_tab_cannot_start_another_run(self):
        self.use_runner(FAKE_SLOW)
        self.run_benchmark(self.open_dashboard())
        second_tab = self.run_benchmark(self.open_dashboard())
        self.assertIn("A benchmark is already running. Stop it or wait for it to finish.",
                      self.messages(second_tab, "error"))
        self.assertEqual(len(self.started), 1)

    def test_run_without_a_live_process_is_shown_as_unknown(self):
        # Simulates a dashboard restart: a folder still saying "running", no process handle.
        orphan = run_launcher.create_run_folder()
        (orphan / "results.json").write_text(json.dumps({
            "run_status": "running",
            "summary": {"completed_tasks": 4, "total_tasks": 16, "correct": 4, "incorrect": 0, "errors": 0},
        }))
        before = fingerprint(orphan)
        at = self.open_dashboard()
        warnings = self.messages(at, "warning")
        self.assertTrue(any(f"Status unknown for `{orphan.name}`" in w for w in warnings))
        self.assertTrue(any("can't be stopped from here" in w for w in warnings))
        self.assertFalse(any(b.label == "Stop benchmark" for b in at.button))
        self.assertEqual(fingerprint(orphan), before)                   # never written to


class TestDashboardRestart(DashboardRunTestCase):
    """
    A dashboard restart is simulated by clearing the shared registry: the run's
    process handle is lost, but its program keeps running, as in real life.
    """

    def restart_dashboard(self):
        st.cache_resource.clear()
        return self.open_dashboard()

    def start_and_wait_for_progress(self, runner):
        self.use_runner(runner)
        self.run_benchmark(self.open_dashboard())
        folder = self.folder_of(0)
        self.wait_until(lambda: (folder / "results.json").exists())
        return self.started[0], folder

    def test_old_run_still_running_blocks_new_runs(self):
        old_process, old_folder = self.start_and_wait_for_progress(FAKE_SLOW)
        at = self.restart_dashboard()
        self.assertTrue(any(f"`{old_folder.name}` is still running" in w for w in self.messages(at, "warning")))
        self.assertFalse(any(b.label == "Stop benchmark" for b in at.button))   # no handle: no Stop

        self.run_benchmark(at)
        self.assertTrue(any("New runs are blocked" in e for e in self.messages(at, "error")))
        self.assertEqual(len(self.started), 1)                                    # nothing new started
        self.assertIsNone(old_process.poll())                                     # old run untouched

    def test_old_run_that_has_actually_ended_does_not_block(self):
        old_process, old_folder = self.start_and_wait_for_progress(FAKE_SLOW)
        old_process.kill()                         # it crashes after the restart: no final status saved
        old_process.wait()
        before = fingerprint(old_folder)

        at = self.restart_dashboard()
        self.assertTrue(any(f"`{old_folder.name}` ended without saving a final status" in m
                            for m in self.messages(at, "info")))
        self.run_benchmark(at)
        self.assertEqual(len(self.started), 2)                                    # new run allowed
        self.assertNotEqual(self.folder_of(1), old_folder)                        # in its own folder
        self.assertEqual(fingerprint(old_folder), before)                         # old files untouched

    def test_old_run_blocks_until_it_ends_then_new_runs_are_allowed(self):
        old_process, old_folder = self.start_and_wait_for_progress(FAKE_SLOW)
        at = self.restart_dashboard()
        self.run_benchmark(at)
        self.assertEqual(len(self.started), 1)
        old_process.send_signal(2)                 # the old run is stopped elsewhere (Ctrl+C)
        old_process.wait(timeout=20)
        self.run_benchmark(at.run())
        self.assertEqual(len(self.started), 2)

    def test_old_run_that_finished_normally_does_not_block_or_warn(self):
        self.use_runner(FAKE_OK)
        self.run_benchmark(self.open_dashboard())
        self.started[0].wait(timeout=20)
        at = self.restart_dashboard()
        self.assertEqual(self.messages(at, "warning"), [])
        self.run_benchmark(at)
        self.assertEqual(len(self.started), 2)

    def test_unverifiable_run_blocks_until_the_user_confirms(self):
        # A "running" folder with no launcher.json: the process can't be checked.
        orphan = run_launcher.create_run_folder()
        (orphan / "results.json").write_text(json.dumps({
            "run_status": "running",
            "summary": {"completed_tasks": 4, "total_tasks": 16, "correct": 4, "incorrect": 0, "errors": 0},
        }))
        before = fingerprint(orphan)

        at = self.run_benchmark(self.open_dashboard())
        self.assertTrue(any("New runs are blocked" in e for e in self.messages(at, "error")))
        self.assertEqual(self.started, [])

        self.click(at, "I've checked: it isn't running. Allow new runs")
        self.assertTrue(any("You confirmed it isn't running" in m for m in self.messages(at, "info")))
        self.run_benchmark(at)
        self.assertEqual(len(self.started), 1)
        self.started[0].wait(timeout=20)
        self.assertEqual(fingerprint(orphan), before)          # confirmation never writes to the folder

        # The confirmation lives in memory only: after a restart it is asked again.
        at = self.restart_dashboard()
        self.run_benchmark(at)
        self.assertTrue(any("New runs are blocked" in e for e in self.messages(at, "error")))
        self.assertEqual(len(self.started), 1)

    # --- Restarts while the old run has no readable results yet ---

    def start_and_wait_for_log(self, runner, word):
        self.use_runner(runner)
        self.run_benchmark(self.open_dashboard())
        folder = self.folder_of(0)
        self.wait_until(lambda: word in (folder / "runner.log").read_text())
        return self.started[0], folder

    def test_restart_during_startup_blocks_new_runs(self):
        old_process, old_folder = self.start_and_wait_for_log(FAKE_STARTUP, "starting")
        self.assertFalse((old_folder / "results.json").exists())        # no results yet
        before = fingerprint(old_folder)

        at = self.restart_dashboard()
        self.assertTrue(any(f"`{old_folder.name}` is still running, but it hasn't saved any results yet" in w
                            for w in self.messages(at, "warning")))
        self.assertFalse(any(b.label == "Stop benchmark" for b in at.button))
        self.run_benchmark(at)
        self.assertTrue(any("New runs are blocked" in e for e in self.messages(at, "error")))
        self.assertEqual(len(self.started), 1)
        self.assertIsNone(old_process.poll())                             # old run untouched
        self.assertEqual(fingerprint(old_folder), before)

    def test_restart_with_unreadable_results_blocks_new_runs(self):
        old_process, old_folder = self.start_and_wait_for_log(FAKE_UNREADABLE, "ready")
        before = fingerprint(old_folder)

        at = self.restart_dashboard()
        self.assertTrue(any(f"`{old_folder.name}` is still running, but its results file can't be read" in w
                            for w in self.messages(at, "warning")))
        self.run_benchmark(at)
        self.assertTrue(any("New runs are blocked" in e for e in self.messages(at, "error")))
        self.assertEqual(len(self.started), 1)
        self.assertEqual(fingerprint(old_folder), before)

    def test_restart_after_a_run_ended_without_results_allows_new_runs_quietly(self):
        old_process, old_folder = self.start_and_wait_for_log(FAKE_STARTUP, "starting")
        old_process.kill()                                                # ended before saving anything
        old_process.wait()
        before = fingerprint(old_folder)

        at = self.restart_dashboard()
        shown = self.messages(at, "warning") + self.messages(at, "info")
        self.assertFalse(any(old_folder.name in m for m in shown))        # not listed (run history is 3B-3)
        self.use_runner(FAKE_OK)
        self.run_benchmark(at)
        self.assertEqual(len(self.started), 2)                            # new run allowed
        self.assertEqual(fingerprint(old_folder), before)

    def test_unknown_run_without_results_blocks_until_confirmed(self):
        orphan = run_launcher.create_run_folder()
        (orphan / "launcher.json").write_text("{not json")               # process can't be checked
        before = fingerprint(orphan)

        at = self.run_benchmark(self.open_dashboard())
        self.assertTrue(any(f"Status unknown for `{orphan.name}`: it hasn't saved any results yet" in w
                            for w in self.messages(at, "warning")))
        self.assertTrue(any("New runs are blocked" in e for e in self.messages(at, "error")))
        self.assertEqual(self.started, [])

        self.click(at, "I've checked: it isn't running. Allow new runs")
        self.run_benchmark(at)
        self.assertEqual(len(self.started), 1)
        self.started[0].wait(timeout=20)
        self.assertEqual(fingerprint(orphan), before)                    # never written to


class TestExistingDashboardUnchanged(DashboardRunTestCase):
    def test_results_and_trial_sections_still_render_with_a_run(self):
        at = self.run_benchmark(self.open_dashboard())
        self.started[0].wait(timeout=20)
        at.run()
        labels = [m.label for m in at.metric]
        for label in ["Accuracy", "Correct", "Incorrect", "Errors",
                      "Average response time", "Model", "Run status", "Completed trials"]:
            self.assertIn(label, labels)
        self.assertEqual(len(at.dataframe), 3)


if __name__ == "__main__":
    unittest.main()
