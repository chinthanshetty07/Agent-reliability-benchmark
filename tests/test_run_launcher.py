"""
Tests for tasks/run_launcher.py.
Run from the project root with:
    python3 -m unittest discover -s tests -v

No real model is used. Tests use:
- small FAKE runner scripts, written into a temporary folder, and
- a FAKE model server on this computer, so the REAL runner can be tested
  end to end without contacting any real model.
Every run folder is inside a temporary folder, and each test checks that the
real results/ folder is unchanged.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "tasks"))

import model_adapter  # noqa: E402
import run_launcher  # noqa: E402
from model_adapter import ModelConfig  # noqa: E402

REAL_RESULTS = PROJECT / "results"
FAKE_KEY = "sk-test-DO-NOT-LEAK-LAUNCH-24680"
FAKE_KEY_SHA256 = hashlib.sha256(FAKE_KEY.encode()).hexdigest()
EXPECTED = {
    task["question"]: task["expected_answer"]
    for task in json.loads((PROJECT / "tasks" / "benchmark.json").read_text())
}


def fingerprint(folder):
    return {
        str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


# ---------- Fake runner scripts (stand-ins for tasks/ai_runner.py) ----------

# Shared start of every fake runner: read --results-dir like the real one.
FAKE_START = """
import hashlib, json, os, signal, sys, time
from pathlib import Path
folder = Path(sys.argv[sys.argv.index("--results-dir") + 1])
def save(status, done, total=3):
    data = {"run_status": status, "summary": {"completed_tasks": done, "total_tasks": total,
            "correct": done, "incorrect": 0, "errors": 0}, "results": []}
    tmp = folder / "results.json.tmp"
    tmp.write_text(json.dumps(data))
    os.replace(tmp, folder / "results.json")
"""

# Reports what it received WITHOUT writing the key itself (only its hash).
FAKE_OK = FAKE_START + """
key = os.environ.get("BENCHMARK_API_KEY")
(folder / "env_report.json").write_text(json.dumps({
    "has_key": key is not None,
    "key_sha256": hashlib.sha256(key.encode()).hexdigest() if key else None,
    "base_url": os.environ.get("BENCHMARK_BASE_URL"),
    "model_id": os.environ.get("BENCHMARK_MODEL_ID"),
    "benchmark_vars": sorted(n for n in os.environ if n.startswith("BENCHMARK_")),
    "argv": sys.argv[1:],
}))
for done in range(1, 4):
    print(f"task {done} done")
    save("running", done)
save("finished", 3)
"""

# Works slowly; saves "interrupted" on Ctrl+C, like the real runner.
FAKE_SLOW = FAKE_START + """
save("running", 0)
try:
    for done in range(1, 1000):
        time.sleep(0.1)
        save("running", min(done, 2))
    save("finished", 3)
except KeyboardInterrupt:
    save("interrupted", 2)
"""

# Ignores Ctrl+C entirely, so it can only be stopped forcibly.
FAKE_STUBBORN = FAKE_START + """
signal.signal(signal.SIGINT, signal.SIG_IGN)
save("running", 1)
print("ready", flush=True)
while True:
    time.sleep(0.1)
"""

# Fails at start-up, before saving any results (like "model not found").
FAKE_FAIL = FAKE_START + """
sys.exit("ERROR: Could not connect to the model endpoint")
"""

# Still "starting up" (no results yet) and, like the real runner at that stage,
# does not catch Ctrl+C, so Ctrl+C ends it with exit code -2.
FAKE_STARTUP = FAKE_START + """
print("starting", flush=True)
time.sleep(60)
"""

# Keeps running, but its results.json can't be read.
FAKE_UNREADABLE = FAKE_START + """
(folder / "results.json").write_text("{not json")
print("ready", flush=True)
time.sleep(60)
"""


# ---------- Fake model server (for tests with the REAL runner) ----------

class FakeModelServer:
    """A tiny OpenAI-style server on this computer. mode: "ok", "reject", or "slow"."""

    def __init__(self, mode="ok"):
        self.mode = mode
        self.auth_headers = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):   # keep test output quiet
                pass

            def do_GET(self):
                server.auth_headers.append(self.headers.get("Authorization"))
                if server.mode == "reject":
                    # A badly behaved server that repeats the key back.
                    body = f"Invalid API key: {self.headers.get('Authorization')}".encode()
                    self.send_response(401)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                body = json.dumps({"data": [{"id": "fake-model"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                server.auth_headers.append(self.headers.get("Authorization"))
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if server.mode == "slow":
                    time.sleep(0.5)
                question = request["messages"][-1]["content"]
                chunk = {"choices": [{"delta": {"content": f"Final answer: {EXPECTED[question]}"},
                                      "finish_reason": "stop"}]}
                body = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass   # expected when a test stops the runner mid-answer

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


# ---------- Test base ----------

class LauncherTestCase(unittest.TestCase):
    def setUp(self):
        self.real_results_before = fingerprint(REAL_RESULTS)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.temp = Path(temp.name).resolve()
        self.runs_dir = self.temp / "runs"
        # Every run folder goes inside the temp folder, never results/runs/.
        mock.patch.object(run_launcher, "RUNS_DIR", self.runs_dir).start()
        self.addCleanup(mock.patch.stopall)
        self.processes = []

    def tearDown(self):
        for process in self.processes:       # never leave a test process running
            if process.poll() is None:
                process.kill()
                process.wait()
        self.assertEqual(fingerprint(REAL_RESULTS), self.real_results_before,
                         "the real results/ folder changed")
        self.assertFalse((REAL_RESULTS / "runs").exists(), "a run was written to the real results/runs/")

    def fake_runner(self, source):
        path = self.temp / f"fake_runner_{len(list(self.temp.glob('fake_runner_*')))}.py"
        path.write_text(textwrap.dedent(source))
        return path

    def start(self, config, runner=None):
        folder = run_launcher.create_run_folder()
        process = run_launcher.start_run(config, folder, runner=runner)
        self.processes.append(process)
        return process, folder

    def wait_until(self, condition, timeout=20):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if condition():
                return True
            time.sleep(0.05)
        self.fail("timed out waiting")

    def assert_key_nowhere(self, process, folder):
        """
        The key must not be in the command line or in ANY file the run wrote.
        assertFalse (not assertNotIn) so that a failure says WHERE the key was
        found without printing the key or the text around it.
        """
        self.assertFalse(FAKE_KEY in " ".join(map(str, process.args)), "API key found in the command line")
        for path in folder.rglob("*"):
            if path.is_file():
                self.assertFalse(FAKE_KEY in path.read_text(errors="replace"), f"API key found in {path.name}")


# ---------- Unique run folders ----------

class TestRunFolders(LauncherTestCase):
    def test_creates_a_new_empty_folder_inside_the_runs_folder(self):
        folder = run_launcher.create_run_folder()
        self.assertTrue(folder.is_dir())
        self.assertEqual(folder.parent, self.runs_dir)
        self.assertEqual(list(folder.iterdir()), [])
        self.assertRegex(folder.name, r"^\d{8}-\d{6}-[0-9a-f]{6}$")

    def test_two_quick_runs_get_different_folders(self):
        folders = {run_launcher.create_run_folder() for _ in range(20)}
        self.assertEqual(len(folders), 20)

    def test_existing_folder_is_never_reused(self):
        # Force every name attempt to clash with a folder that already holds results.
        with mock.patch.object(run_launcher.secrets, "token_hex", return_value="aaaaaa"):
            first = run_launcher.create_run_folder()
            (first / "results.json").write_text("old results")
            with self.assertRaises(run_launcher.LaunchError):
                run_launcher.create_run_folder()
        self.assertEqual((first / "results.json").read_text(), "old results")

    def test_start_refuses_a_folder_that_is_not_empty(self):
        folder = run_launcher.create_run_folder()
        (folder / "results.json").write_text("old results")
        with self.assertRaises(run_launcher.LaunchError):
            run_launcher.start_run(ModelConfig(), folder, runner=self.fake_runner(FAKE_OK))
        self.assertEqual((folder / "results.json").read_text(), "old results")

    def test_start_refuses_a_missing_folder(self):
        with self.assertRaises(run_launcher.LaunchError):
            run_launcher.start_run(ModelConfig(), self.temp / "missing", runner=self.fake_runner(FAKE_OK))
        self.assertFalse((self.temp / "missing").exists())

    def test_runs_folder_is_ignored_by_git(self):
        completed = subprocess.run(
            ["git", "check-ignore", "-q", "results/runs/20261013-000000-abcdef/results.json"],
            cwd=PROJECT, capture_output=True,
        )
        if completed.returncode == 128:
            self.skipTest("not inside a git repository")
        self.assertEqual(completed.returncode, 0, "results/runs/ is not ignored by git")


# ---------- Environment for the runner ----------

class TestRunnerEnvironment(LauncherTestCase):
    def test_settings_are_passed(self):
        env = run_launcher.build_runner_env(
            ModelConfig(base_url="https://example.test/v1", model_id="m1", api_key=FAKE_KEY),
            base_env={"PATH": "/usr/bin"},
        )
        self.assertEqual(env["BENCHMARK_BASE_URL"], "https://example.test/v1")
        self.assertEqual(env["BENCHMARK_MODEL_ID"], "m1")
        self.assertEqual(env["BENCHMARK_API_KEY"], FAKE_KEY)
        self.assertEqual(env["PATH"], "/usr/bin")          # the rest of the environment is kept

    def test_empty_key_does_not_inherit_a_key_from_the_dashboard(self):
        env = run_launcher.build_runner_env(
            ModelConfig(api_key=None),
            base_env={"BENCHMARK_API_KEY": "sk-from-the-dashboard-environment", "PATH": "/usr/bin"},
        )
        self.assertNotIn("BENCHMARK_API_KEY", env)

    def test_other_benchmark_settings_are_not_inherited(self):
        env = run_launcher.build_runner_env(
            ModelConfig(),
            base_env={"BENCHMARK_BASE_URL": "http://elsewhere/v1", "BENCHMARK_SOMETHING": "x"},
        )
        self.assertEqual(env["BENCHMARK_BASE_URL"], ModelConfig().base_url)
        self.assertNotIn("BENCHMARK_SOMETHING", env)


# ---------- Starting a run (fake runner) ----------

class TestStartRun(LauncherTestCase):
    def test_runner_receives_settings_but_key_is_never_written(self):
        config = ModelConfig(base_url="https://example.test/v1", model_id="m1", api_key=FAKE_KEY)
        process, folder = self.start(config, runner=self.fake_runner(FAKE_OK))
        self.assertEqual(process.wait(timeout=20), 0)

        report = json.loads((folder / "env_report.json").read_text())
        self.assertTrue(report["has_key"])
        self.assertEqual(report["key_sha256"], FAKE_KEY_SHA256)   # the right key arrived...
        self.assertEqual(report["base_url"], "https://example.test/v1")
        self.assertEqual(report["model_id"], "m1")
        self.assertEqual(report["argv"], ["--results-dir", str(folder)])
        self.assert_key_nowhere(process, folder)                  # ...but is written nowhere
        self.assertIn("task 3 done", (folder / "runner.log").read_text())

    def test_inherited_key_does_not_reach_the_runner(self):
        with mock.patch.dict(os.environ, {"BENCHMARK_API_KEY": FAKE_KEY}):
            process, folder = self.start(ModelConfig(api_key=None), runner=self.fake_runner(FAKE_OK))
            process.wait(timeout=20)
        report = json.loads((folder / "env_report.json").read_text())
        self.assertFalse(report["has_key"])
        self.assertEqual(report["benchmark_vars"], ["BENCHMARK_BASE_URL", "BENCHMARK_MODEL_ID"])

    def test_start_returns_immediately(self):
        started = time.time()
        process, _ = self.start(ModelConfig(), runner=self.fake_runner(FAKE_SLOW))
        self.assertLess(time.time() - started, 2)
        self.assertIsNone(process.poll())                         # still running in the background
        run_launcher.stop_run(process)


# ---------- Reading progress ----------

class TestProgress(LauncherTestCase):
    def write_results(self, folder, text):
        (folder / "results.json").write_text(text)

    def test_missing_results_means_not_started(self):
        folder = run_launcher.create_run_folder()
        self.assertEqual(run_launcher.read_progress(folder)["status"], "not_started")

    def test_progress_during_and_after_a_run(self):
        folder = run_launcher.create_run_folder()
        summary = {"completed_tasks": 7, "total_tasks": 16, "correct": 6, "incorrect": 1, "errors": 0}
        self.write_results(folder, json.dumps({"run_status": "running", "summary": summary}))
        progress = run_launcher.read_progress(folder)
        self.assertEqual((progress["status"], progress["completed"], progress["total"], progress["correct"]),
                         ("running", 7, 16, 6))

        self.write_results(folder, json.dumps({"run_status": "finished", "summary": {**summary, "completed_tasks": 16}}))
        self.assertEqual(run_launcher.read_progress(folder)["status"], "finished")

    def test_broken_results_file_does_not_crash(self):
        folder = run_launcher.create_run_folder()
        for text in ["{not json", "[]", '{"run_status": "running"}']:
            with self.subTest(text=text):
                self.write_results(folder, text)
                self.assertEqual(run_launcher.read_progress(folder)["status"], "unreadable")

    def test_finished_run_is_described_as_finished(self):
        process, folder = self.start(ModelConfig(), runner=self.fake_runner(FAKE_OK))
        process.wait(timeout=20)
        described = run_launcher.describe_run(process, folder)
        self.assertEqual((described["state"], described["completed"], described["exit_code"]), ("finished", 3, 0))

    def test_failed_start_is_described_as_failed_with_a_log(self):
        process, folder = self.start(ModelConfig(), runner=self.fake_runner(FAKE_FAIL))
        process.wait(timeout=20)
        described = run_launcher.describe_run(process, folder)
        self.assertEqual((described["state"], described["exit_code"]), ("failed", 1))
        self.assertIn("Could not connect", (folder / "runner.log").read_text())


# ---------- Stopping ----------

class TestStopping(LauncherTestCase):
    def test_graceful_stop_saves_interrupted(self):
        process, folder = self.start(ModelConfig(), runner=self.fake_runner(FAKE_SLOW))
        self.wait_until(lambda: (folder / "results.json").exists())
        self.assertEqual(run_launcher.stop_run(process), "interrupted")
        described = run_launcher.describe_run(process, folder)
        self.assertEqual((described["state"], described["completed"]), ("interrupted", 2))

    def test_forced_stop_when_the_run_ignores_the_request(self):
        process, folder = self.start(ModelConfig(), runner=self.fake_runner(FAKE_STUBBORN))
        self.wait_until(lambda: "ready" in (folder / "runner.log").read_text())
        self.assertEqual(run_launcher.stop_run(process, grace_seconds=1), "killed")
        self.assertIsNotNone(process.poll())                      # really ended
        # No final status was saved, and that is reported honestly.
        self.assertEqual(run_launcher.describe_run(process, folder)["state"], "stopped_unexpectedly")

    def test_default_wait_follows_the_setting(self):
        # The dashboard calls stop_run(process) without a wait time, so the
        # default must come from STOP_GRACE_SECONDS at the time of the call.
        mock.patch.object(run_launcher, "STOP_GRACE_SECONDS", 1).start()
        process, folder = self.start(ModelConfig(), runner=self.fake_runner(FAKE_STUBBORN))
        self.wait_until(lambda: "ready" in (folder / "runner.log").read_text())
        started = time.time()
        self.assertEqual(run_launcher.stop_run(process), "killed")
        self.assertLess(time.time() - started, 5)

    def test_stopping_a_finished_run_does_nothing(self):
        process, _ = self.start(ModelConfig(), runner=self.fake_runner(FAKE_OK))
        process.wait(timeout=20)
        self.assertEqual(run_launcher.stop_run(process), "already_stopped")

    def test_stop_during_startup_is_reported_as_stopped_before_start(self):
        process, folder = self.start(ModelConfig(), runner=self.fake_runner(FAKE_STARTUP))
        self.wait_until(lambda: "starting" in (folder / "runner.log").read_text())
        run_launcher.stop_run(process)
        described = run_launcher.describe_run(process, folder)
        # Check the REAL state, not stop_run()'s return value.
        self.assertEqual(described["exit_code"], -2)
        self.assertFalse((folder / "results.json").exists())
        self.assertEqual(described["state"], "stopped_before_start")

    def test_startup_error_is_still_failed_not_stopped(self):
        process, folder = self.start(ModelConfig(), runner=self.fake_runner(FAKE_FAIL))
        process.wait(timeout=20)
        described = run_launcher.describe_run(process, folder)
        self.assertEqual((described["state"], described["exit_code"]), ("failed", 1))


class TestOutcome(unittest.TestCase):
    """outcome() turns a described run into a colour level and a plain message."""

    def described(self, state, completed=None, total=16, correct=None, errors=None, exit_code=0):
        return {"state": state, "exit_code": exit_code, "status": state, "completed": completed,
                "total": total, "correct": correct, "incorrect": None, "errors": errors}

    def test_successful_run(self):
        self.assertEqual(run_launcher.outcome(self.described("finished", 16, correct=16, errors=0)),
                         ("success", "Finished: 16/16 correct."))

    def test_partial_failures(self):
        level, message = run_launcher.outcome(self.described("finished", 16, correct=12, errors=3))
        self.assertEqual(level, "warning")
        self.assertIn("3 error(s)", message)
        self.assertIn("12/16 correct", message)

    def test_every_task_failed_is_an_error_even_with_exit_code_0(self):
        level, message = run_launcher.outcome(self.described("finished", 16, correct=0, errors=16, exit_code=0))
        self.assertEqual(level, "error")
        self.assertIn("every task failed", message)

    def test_no_tasks_ran(self):
        self.assertEqual(run_launcher.outcome(self.described("finished", 0, total=0, correct=0, errors=0)),
                         ("warning", "Finished, but no tasks ran."))

    def test_interrupted_run(self):
        self.assertEqual(run_launcher.outcome(self.described("interrupted", 7, correct=6, errors=0)),
                         ("warning", "Stopped: 7 of 16 tasks saved."))

    def test_startup_interruption(self):
        level, message = run_launcher.outcome(self.described("stopped_before_start", exit_code=-2))
        self.assertEqual(level, "warning")
        self.assertIn("Stopped before the first task", message)

    def test_unexpected_termination(self):
        level, message = run_launcher.outcome(self.described("stopped_unexpectedly", 4, exit_code=-9))
        self.assertEqual(level, "error")
        self.assertIn("without saving a final status", message)

    def test_failed_start(self):
        level, message = run_launcher.outcome(self.described("failed", exit_code=1))
        self.assertEqual(level, "error")
        self.assertIn("runner.log", message)

    def test_running(self):
        self.assertEqual(run_launcher.outcome(self.described("running", None, total=None, exit_code=None))[0], "info")
        self.assertEqual(run_launcher.outcome(self.described("running", 5, exit_code=None)),
                         ("info", "Running: 5 of 16 tasks done."))


class TestUnfinishedRunsAndLog(LauncherTestCase):
    def make_run(self, status=None):
        folder = run_launcher.create_run_folder()
        if status is not None:
            (folder / "results.json").write_text(json.dumps({
                "run_status": status,
                "summary": {"completed_tasks": 1, "total_tasks": 3, "correct": 1, "incorrect": 0, "errors": 0},
            }))
        return folder

    def test_lists_only_runs_still_saying_running(self):
        running = self.make_run("running")
        self.make_run("finished")
        self.make_run("interrupted")
        self.make_run(None)                        # no results at all
        before = fingerprint(self.runs_dir)
        self.assertEqual(run_launcher.unfinished_runs(), [running])
        self.assertEqual(fingerprint(self.runs_dir), before)   # read-only

    def test_excluded_run_is_left_out(self):
        running = self.make_run("running")
        self.assertEqual(run_launcher.unfinished_runs(exclude=[running]), [])

    def test_no_runs_folder_yet(self):
        self.assertFalse(self.runs_dir.exists())
        self.assertEqual(run_launcher.unfinished_runs(), [])

    def test_log_tail(self):
        folder = self.make_run(None)
        self.assertEqual(run_launcher.read_log_tail(folder), "")
        (folder / "runner.log").write_text("\n".join(f"line {n}" for n in range(1, 31)))
        tail = run_launcher.read_log_tail(folder, max_lines=3)
        self.assertEqual(tail, "line 28\nline 29\nline 30")


# ---------- The REAL runner, against a fake model server ----------

class TestRealRunner(LauncherTestCase):
    def serve(self, mode):
        server = FakeModelServer(mode)
        self.addCleanup(server.close)
        return server, ModelConfig(base_url=server.url, model_id="fake-model", api_key=FAKE_KEY)

    def test_full_run_with_the_real_runner(self):
        server, config = self.serve("ok")
        process, folder = self.start(config)                      # real tasks/ai_runner.py
        self.assertEqual(process.wait(timeout=120), 0)

        described = run_launcher.describe_run(process, folder)
        self.assertEqual((described["state"], described["completed"], described["correct"]),
                         ("finished", len(EXPECTED), len(EXPECTED)))
        self.assertTrue(server.auth_headers)
        self.assertTrue(all(h == f"Bearer {FAKE_KEY}" for h in server.auth_headers))   # key reached the server
        self.assert_key_nowhere(process, folder)                                     # but was written nowhere
        self.assertEqual(sorted(p.name for p in folder.iterdir()), ["launcher.json", "results.json", "runner.log"])

    def test_rejected_key_is_not_written_to_the_log(self):
        server, config = self.serve("reject")                     # repeats the key in its error
        process, folder = self.start(config)
        self.assertEqual(process.wait(timeout=60), 1)
        self.assertEqual(run_launcher.describe_run(process, folder)["state"], "failed")
        log = (folder / "runner.log").read_text()
        self.assertIn("HTTP 401", log)
        self.assertIn("***", log)
        self.assert_key_nowhere(process, folder)

    def test_graceful_stop_of_the_real_runner(self):
        server, config = self.serve("slow")                       # 0.5 s per answer
        process, folder = self.start(config)
        self.wait_until(lambda: (run_launcher.read_progress(folder)["completed"] or 0) >= 2, timeout=60)
        self.assertEqual(run_launcher.stop_run(process), "interrupted")
        described = run_launcher.describe_run(process, folder)
        self.assertEqual(described["state"], "interrupted")
        self.assertLess(described["completed"], len(EXPECTED))
        self.assert_key_nowhere(process, folder)


# ---------- Process check for runs without a handle (after a dashboard restart) ----------

class TestProcessCheck(LauncherTestCase):
    def write_running_results(self, folder):
        (folder / "results.json").write_text(json.dumps({
            "run_status": "running",
            "summary": {"completed_tasks": 1, "total_tasks": 3, "correct": 1, "incorrect": 0, "errors": 0},
        }))

    def test_launcher_records_the_process_number_without_the_key(self):
        process, folder = self.start(ModelConfig(api_key=FAKE_KEY), runner=self.fake_runner(FAKE_OK))
        process.wait(timeout=20)
        info = json.loads((folder / "launcher.json").read_text())
        self.assertEqual(info["pid"], process.pid)
        self.assertEqual(set(info), {"pid", "started_at"})
        self.assert_key_nowhere(process, folder)

    def test_running_process_is_active(self):
        process, folder = self.start(ModelConfig(), runner=self.fake_runner(FAKE_SLOW))
        self.wait_until(lambda: (folder / "results.json").exists())
        self.assertEqual(run_launcher.check_process_alive(folder), "active")

    def test_ended_process_is_ended(self):
        process, folder = self.start(ModelConfig(), runner=self.fake_runner(FAKE_SLOW))
        self.wait_until(lambda: (folder / "results.json").exists())
        process.kill()
        process.wait()                                         # ended and cleaned up
        self.assertEqual(run_launcher.check_process_alive(folder), "ended")

    def test_ended_but_not_yet_cleaned_up_process_is_ended(self):
        # A "zombie": the program has ended but its parent hasn't collected it yet
        # (ps shows it as "<defunct>"). It must count as ended.
        process, folder = self.start(ModelConfig(), runner=self.fake_runner(FAKE_SLOW))
        self.wait_until(lambda: (folder / "results.json").exists())
        os.kill(process.pid, 9)
        time.sleep(0.5)                                        # deliberately not calling wait()
        self.assertEqual(run_launcher.check_process_alive(folder), "ended")

    def test_reused_process_number_is_not_mistaken_for_the_run(self):
        # The recorded number now belongs to an unrelated program (here: this test).
        folder = run_launcher.create_run_folder()
        (folder / "launcher.json").write_text(json.dumps({"pid": os.getpid()}))
        self.assertEqual(run_launcher.check_process_alive(folder), "ended")

    def test_missing_or_broken_launcher_info_is_unknown(self):
        folder = run_launcher.create_run_folder()
        self.assertEqual(run_launcher.check_process_alive(folder), "unknown")
        for text in ["{not json", "{}", '{"pid": "abc"}']:
            with self.subTest(text=text):
                (folder / "launcher.json").write_text(text)
                self.assertEqual(run_launcher.check_process_alive(folder), "unknown")

    def test_failed_process_check_is_unknown(self):
        folder = run_launcher.create_run_folder()
        (folder / "launcher.json").write_text(json.dumps({"pid": os.getpid()}))
        with mock.patch.object(run_launcher.subprocess, "run", side_effect=OSError("ps not available")):
            self.assertEqual(run_launcher.check_process_alive(folder), "unknown")

    def test_check_is_read_only(self):
        process, folder = self.start(ModelConfig(), runner=self.fake_runner(FAKE_SLOW))
        self.wait_until(lambda: (folder / "results.json").exists())
        with mock.patch.object(run_launcher.os, "kill") as kill:
            run_launcher.check_process_alive(folder)
        kill.assert_not_called()
        self.assertIsNone(process.poll())                      # still running, untouched

    def test_blocking_runs(self):
        active_process, active = self.start(ModelConfig(), runner=self.fake_runner(FAKE_SLOW))
        self.wait_until(lambda: (active / "results.json").exists())

        ended_process, ended = self.start(ModelConfig(), runner=self.fake_runner(FAKE_SLOW))
        self.wait_until(lambda: (ended / "results.json").exists())
        ended_process.kill()
        ended_process.wait()

        unknown = run_launcher.create_run_folder()             # "running" but no launcher.json
        self.write_running_results(unknown)

        before = fingerprint(self.runs_dir)
        blocking = dict(run_launcher.runs_blocking_a_new_start())
        self.assertEqual(blocking, {active: "active", unknown: "unknown"})   # "ended" does not block
        # The user confirmed the unknown one isn't running: only the active one blocks.
        blocking = dict(run_launcher.runs_blocking_a_new_start(confirmed_not_running={unknown.name}))
        self.assertEqual(blocking, {active: "active"})
        # A confirmation never unblocks a run that is actually active.
        blocking = dict(run_launcher.runs_blocking_a_new_start(confirmed_not_running={active.name, unknown.name}))
        self.assertEqual(blocking, {active: "active"})
        # Excluding the run this dashboard does have a handle for.
        self.assertEqual(dict(run_launcher.runs_blocking_a_new_start(exclude=[active],
                                                                     confirmed_not_running={unknown.name})), {})
        # Read-only (ignoring the still-running fake run's own progress file).
        after = fingerprint(self.runs_dir)
        self.assertEqual({k: v for k, v in after.items() if not k.startswith(active.name)},
                         {k: v for k, v in before.items() if not k.startswith(active.name)})


# ---------- Runs with no readable results yet (restart during start-up) ----------

class TestRunsWithoutFinalStatus(LauncherTestCase):
    """
    After a dashboard restart, a run may still be active but have no results.json yet
    (start-up or first task) or an unreadable one. Such runs must still be checked.
    """

    def start_live(self, source, ready_word):
        process, folder = self.start(ModelConfig(), runner=self.fake_runner(source))
        self.wait_until(lambda: ready_word in (folder / "runner.log").read_text())
        return process, folder

    def blocking(self, **kwargs):
        return dict(run_launcher.runs_blocking_a_new_start(**kwargs))

    def test_no_results_and_live_process_blocks(self):
        process, folder = self.start_live(FAKE_STARTUP, "starting")
        self.assertFalse((folder / "results.json").exists())
        self.assertEqual(run_launcher.runs_without_final_status(), [(folder, "not_started")])
        self.assertEqual(self.blocking(), {folder: "active"})

    def test_unreadable_results_and_live_process_blocks(self):
        process, folder = self.start_live(FAKE_UNREADABLE, "ready")
        self.assertEqual(run_launcher.read_progress(folder)["status"], "unreadable")
        self.assertEqual(run_launcher.runs_without_final_status(), [(folder, "unreadable")])
        self.assertEqual(self.blocking(), {folder: "active"})

    def test_no_results_and_ended_process_does_not_block(self):
        process, folder = self.start_live(FAKE_STARTUP, "starting")
        process.kill()
        process.wait()
        self.assertEqual(run_launcher.runs_without_final_status(), [(folder, "not_started")])
        self.assertEqual(run_launcher.check_process_alive(folder), "ended")
        self.assertEqual(self.blocking(), {})

    def test_unreadable_results_and_ended_process_does_not_block(self):
        process, folder = self.start_live(FAKE_UNREADABLE, "ready")
        process.kill()
        process.wait()
        self.assertEqual(self.blocking(), {})

    def test_no_results_and_unknown_process_blocks_until_confirmed(self):
        folder = run_launcher.create_run_folder()
        (folder / "launcher.json").write_text("{not json")      # can't tell which process it was
        self.assertEqual(run_launcher.check_process_alive(folder), "unknown")
        self.assertEqual(self.blocking(), {folder: "unknown"})   # never assumed to have stopped
        self.assertEqual(self.blocking(confirmed_not_running={folder.name}), {})

    def test_failed_process_check_on_a_live_run_blocks(self):
        process, folder = self.start_live(FAKE_STARTUP, "starting")
        with mock.patch.object(run_launcher.subprocess, "run", side_effect=OSError("ps not available")):
            self.assertEqual(self.blocking(), {folder: "unknown"})

    def test_folder_without_launcher_info_and_without_results_is_not_considered(self):
        # A failed start: the folder was created but no runner process was started.
        folder = run_launcher.create_run_folder()
        (folder / "runner.log").write_text("could not start")
        self.assertEqual(run_launcher.runs_without_final_status(), [])
        self.assertEqual(self.blocking(), {})

    def test_finished_or_interrupted_runs_are_never_considered(self):
        for status in ("finished", "interrupted"):
            with self.subTest(status=status):
                process, folder = self.start_live(FAKE_STARTUP, "starting")   # process still alive
                (folder / "results.json").write_text(json.dumps({
                    "run_status": status,
                    "summary": {"completed_tasks": 3, "total_tasks": 3, "correct": 3, "incorrect": 0, "errors": 0},
                }))
                self.assertNotIn(folder, dict(run_launcher.runs_without_final_status()))
                self.assertNotIn(folder, self.blocking())

    def test_running_status_is_still_included_with_or_without_launcher_info(self):
        without_info = run_launcher.create_run_folder()
        (without_info / "results.json").write_text(json.dumps({
            "run_status": "running",
            "summary": {"completed_tasks": 1, "total_tasks": 3, "correct": 1, "incorrect": 0, "errors": 0},
        }))
        self.assertEqual(run_launcher.runs_without_final_status(), [(without_info, "running")])
        self.assertEqual(self.blocking(), {without_info: "unknown"})   # same as before this change

    def test_unfinished_runs_is_unchanged(self):
        # unfinished_runs() still lists ONLY runs saying "running".
        _, live_no_results = self.start_live(FAKE_STARTUP, "starting")
        self.assertEqual(run_launcher.unfinished_runs(), [])
        self.assertEqual(run_launcher.runs_without_final_status(), [(live_no_results, "not_started")])

    def test_excluded_run_is_left_out(self):
        _, folder = self.start_live(FAKE_STARTUP, "starting")
        self.assertEqual(run_launcher.runs_without_final_status(exclude=[folder]), [])
        self.assertEqual(self.blocking(exclude=[folder]), {})

    def test_checks_are_read_only(self):
        _, live = self.start_live(FAKE_UNREADABLE, "ready")
        unknown = run_launcher.create_run_folder()
        (unknown / "launcher.json").write_text("{not json")
        before = fingerprint(self.runs_dir)
        run_launcher.runs_without_final_status()
        self.blocking()
        self.assertEqual(fingerprint(self.runs_dir), before)


# ---------- launcher.json is written BEFORE a process starts ----------

class TestLaunchInfoSafety(LauncherTestCase):
    """
    A runner process must never run without recoverable launcher.json, and a
    failure to write it must never leave a process running untracked.
    """

    def spy_popen(self):
        """Record every process start_run() starts, even if start_run() then raises."""
        started = []
        real_popen = subprocess.Popen

        def spy(*args, **kwargs):
            started.append(real_popen(*args, **kwargs))
            return started[-1]

        return started, mock.patch.object(run_launcher.subprocess, "Popen", side_effect=spy)

    def assert_reaped(self, process):
        self.assertIsNotNone(process.returncode)          # it ended and was waited for
        with self.assertRaises(ProcessLookupError):
            os.kill(process.pid, 0)                         # signal 0 = "does it exist?" (sends nothing)

    def fail_on_call(self, number):
        """A _write_launch_info() that fails on its Nth call and works otherwise."""
        real_write = run_launcher._write_launch_info
        calls = []

        def write(folder, info):
            calls.append(info)
            if len(calls) == number:
                raise OSError("No space left on device")
            return real_write(folder, info)

        return mock.patch.object(run_launcher, "_write_launch_info", side_effect=write)

    def test_launch_info_exists_before_the_process_starts(self):
        folder = run_launcher.create_run_folder()
        runner = self.fake_runner(FAKE_OK)
        seen = {}
        real_popen = subprocess.Popen

        def check_then_start(*args, **kwargs):
            seen["info"] = json.loads((folder / "launcher.json").read_text())
            return real_popen(*args, **kwargs)

        with mock.patch.object(run_launcher.subprocess, "Popen", side_effect=check_then_start):
            process = run_launcher.start_run(ModelConfig(), folder, runner=runner)
        self.processes.append(process)
        self.assertIsNone(seen["info"]["pid"])                           # placeholder already there
        process.wait(timeout=20)
        info = json.loads((folder / "launcher.json").read_text())
        self.assertEqual(info, {"pid": process.pid, "started_at": seen["info"]["started_at"]})
        self.assertFalse((folder / "launcher.json.tmp").exists())

    def test_placeholder_counts_as_unknown_and_blocks(self):
        # A restart between writing the placeholder and recording the number:
        # the run must block new runs, not be missed.
        folder = run_launcher.create_run_folder()
        run_launcher._write_launch_info(folder, {"pid": None, "started_at": "2026-10-13T00:00:00"})
        self.assertEqual(run_launcher.check_process_alive(folder), "unknown")
        self.assertEqual(dict(run_launcher.runs_blocking_a_new_start()), {folder: "unknown"})

    def test_initial_write_failure_starts_no_process(self):
        folder = run_launcher.create_run_folder()
        runner = self.fake_runner(FAKE_STARTUP)
        started, popen_patch = self.spy_popen()
        with popen_patch, self.fail_on_call(1):
            with self.assertRaises(run_launcher.LaunchError) as caught:
                run_launcher.start_run(ModelConfig(), folder, runner=runner)
        self.assertEqual(started, [])                                     # nothing was started
        self.assertIn("so the run was not started", str(caught.exception))
        self.assertEqual(list(folder.iterdir()), [])                      # no launcher.json, log or .tmp
        self.assertEqual(run_launcher.runs_without_final_status(), [])

    def test_real_disk_write_failure_starts_no_process(self):
        # The same, but failing inside the real write (not a replaced function).
        folder = run_launcher.create_run_folder()
        runner = self.fake_runner(FAKE_STARTUP)
        started, popen_patch = self.spy_popen()
        with popen_patch, mock.patch.object(run_launcher.Path, "write_text", side_effect=OSError("Read-only file system")):
            with self.assertRaises(run_launcher.LaunchError):
                run_launcher.start_run(ModelConfig(), folder, runner=runner)
        self.assertEqual(started, [])
        self.assertEqual(list(folder.iterdir()), [])

    def test_write_failure_after_start_stops_and_reaps_the_process(self):
        folder = run_launcher.create_run_folder()
        runner = self.fake_runner(FAKE_STARTUP)                           # no results until stopped
        started, popen_patch = self.spy_popen()
        with popen_patch, self.fail_on_call(2):
            with self.assertRaises(run_launcher.LaunchError) as caught:
                run_launcher.start_run(ModelConfig(api_key=FAKE_KEY), folder, runner=runner)
        self.assertEqual(len(started), 1)
        self.assert_reaped(started[0])                                    # no orphan process
        message = str(caught.exception)
        self.assertIn("so the run was stopped", message)
        self.assertNotIn("could not be confirmed", message)
        self.assertFalse(FAKE_KEY in message, "API key found in the error message")
        self.assertFalse((folder / "launcher.json").exists())             # placeholder removed
        self.assertEqual(run_launcher.runs_without_final_status(), [])    # looks like a failed start
        self.assertEqual(dict(run_launcher.runs_blocking_a_new_start()), {})

    def test_post_start_failure_when_launch_info_cannot_be_removed(self):
        folder = run_launcher.create_run_folder()
        runner = self.fake_runner(FAKE_STARTUP)
        started, popen_patch = self.spy_popen()
        with popen_patch, self.fail_on_call(2), \
             mock.patch.object(run_launcher.Path, "unlink", side_effect=OSError("Read-only file system")):
            with self.assertRaises(run_launcher.LaunchError) as caught:
                run_launcher.start_run(ModelConfig(), folder, runner=runner)
        self.assert_reaped(started[0])                                    # still no orphan
        self.assertIn("could not be removed", str(caught.exception))      # reported, not ignored
        # The leftover placeholder errs on the safe side: unknown, blocks until confirmed.
        self.assertIsNone(json.loads((folder / "launcher.json").read_text())["pid"])
        self.assertEqual(dict(run_launcher.runs_blocking_a_new_start()), {folder: "unknown"})

    def test_process_start_failure_removes_the_placeholder(self):
        folder = run_launcher.create_run_folder()
        runner = self.fake_runner(FAKE_OK)
        with mock.patch.object(run_launcher.subprocess, "Popen", side_effect=OSError("interpreter not found")):
            with self.assertRaises(OSError):                              # same error type as before
                run_launcher.start_run(ModelConfig(), folder, runner=runner)
        self.assertFalse((folder / "launcher.json").exists())
        self.assertEqual(run_launcher.runs_without_final_status(), [])

    def test_process_start_failure_when_placeholder_cannot_be_removed(self):
        folder = run_launcher.create_run_folder()
        runner = self.fake_runner(FAKE_OK)
        with mock.patch.object(run_launcher.subprocess, "Popen", side_effect=OSError("interpreter not found")), \
             mock.patch.object(run_launcher.Path, "unlink", side_effect=OSError("Read-only file system")):
            with self.assertRaises(run_launcher.LaunchError) as caught:
                run_launcher.start_run(ModelConfig(), folder, runner=runner)
        self.assertIn("could not be started", str(caught.exception))
        self.assertIn("could not be removed", str(caught.exception))

    def child_pids(self):
        """Process numbers of programs this test process has started and not yet reaped."""
        out = subprocess.run(["pgrep", "-P", str(os.getpid())], capture_output=True, text=True).stdout
        return set(out.split())

    def test_non_oserror_during_start_removes_the_placeholder(self):
        # Regression: e.g. a setting with an invalid null character makes the
        # process start raise ValueError, not OSError. It must be cleaned up too.
        folder = run_launcher.create_run_folder()
        runner = self.fake_runner(FAKE_OK)
        children_before = self.child_pids()
        with mock.patch.object(run_launcher.subprocess, "Popen", side_effect=ValueError("embedded null byte")):
            with self.assertRaises(ValueError) as caught:                 # original error, unchanged
                run_launcher.start_run(ModelConfig(api_key=FAKE_KEY), folder, runner=runner)
        self.assertEqual(str(caught.exception), "embedded null byte")
        self.assertFalse((folder / "launcher.json").exists())             # placeholder removed
        self.assertEqual(run_launcher.runs_without_final_status(), [])    # doesn't block future runs
        self.assertEqual(dict(run_launcher.runs_blocking_a_new_start()), {})
        self.assertEqual(self.child_pids(), children_before)              # no process left running

    def test_real_invalid_setting_is_cleaned_up_without_starting_a_process(self):
        # The same, without a mock: Python itself rejects a null character.
        folder = run_launcher.create_run_folder()
        runner = self.fake_runner(FAKE_OK)
        children_before = self.child_pids()
        with self.assertRaises(ValueError):
            run_launcher.start_run(ModelConfig(api_key="fake\x00key"), folder, runner=runner)
        self.assertFalse((folder / "launcher.json").exists())
        self.assertEqual(dict(run_launcher.runs_blocking_a_new_start()), {})
        self.assertEqual(self.child_pids(), children_before)

    def test_non_oserror_when_cleanup_also_fails_is_reported_without_the_key(self):
        folder = run_launcher.create_run_folder()
        runner = self.fake_runner(FAKE_OK)
        children_before = self.child_pids()
        # An error message that happens to contain the key, to prove it is hidden.
        with mock.patch.object(run_launcher.subprocess, "Popen", side_effect=ValueError(f"bad value {FAKE_KEY}")), \
             mock.patch.object(run_launcher.Path, "unlink", side_effect=OSError("Read-only file system")):
            with self.assertRaises(run_launcher.LaunchError) as caught:
                run_launcher.start_run(ModelConfig(api_key=FAKE_KEY), folder, runner=runner)
        message = str(caught.exception)
        self.assertIn("could not be started (ValueError", message)        # first failure reported
        self.assertIn("could not be removed", message)                    # second failure reported
        self.assertFalse(FAKE_KEY in message, "API key found in the error message")
        self.assertIn("***", message)
        self.assertIsInstance(caught.exception.__cause__, ValueError)     # original kept as the cause
        self.assertEqual(self.child_pids(), children_before)
        # The leftover placeholder still errs on the safe side: unknown, blocks until confirmed.
        self.assertEqual(dict(run_launcher.runs_blocking_a_new_start()), {folder: "unknown"})


if __name__ == "__main__":
    unittest.main()
