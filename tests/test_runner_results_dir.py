"""
Tests for the runner's optional --results-dir option.
Run from the project root with:
    python3 -m unittest discover -s tests -v

Every test uses FAKE model answers and TEMPORARY folders. No model is
contacted, and each test checks that the real results/ folder is unchanged.
"""

import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "tasks"))

import ai_runner  # noqa: E402

REAL_RESULTS = PROJECT / "results"
RUNNER = PROJECT / "tasks" / "ai_runner.py"

# The runner's real default paths, recorded before any test changes them.
ORIGINAL_PATHS = (ai_runner.RESULTS_DIR, ai_runner.RESULT_FILE, ai_runner.HISTORY_DIR)
ORIGINAL_DEFAULT_DIR = ai_runner.DEFAULT_RESULTS_DIR

# Correct answers for the fake model, looked up from the real dataset (read-only).
EXPECTED = {
    task["question"]: task["expected_answer"]
    for task in json.loads((PROJECT / "tasks" / "benchmark.json").read_text())
}


def fake_ask_model(question):
    """Stands in for the real model: instantly 'answers' with the expected number."""
    return f"Final answer: {EXPECTED[question]}", "fake reasoning", "stop"


def fingerprint(folder):
    """A hash of every file under a folder, so any change, addition or removal is noticed."""
    return {
        str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


class ResultsDirTestCase(unittest.TestCase):
    def setUp(self):
        self.real_results_before = fingerprint(REAL_RESULTS)

        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.temp = Path(temp.name).resolve()

        # Safety net: even if the runner had a bug, it must not be able to reach
        # the real results/ or the project folder during a test. So the project
        # folder and the "default" results folder are stand-ins inside the temp
        # folder, and every test runs from inside the temp folder.
        stand_in_project = self.temp / "stand_in_project"
        (stand_in_project / "tasks").mkdir(parents=True)
        task_file = stand_in_project / "tasks" / "benchmark.json"
        task_file.write_bytes((PROJECT / "tasks" / "benchmark.json").read_bytes())
        self.default_dir = self.temp / "stand_in_default_results"
        mock.patch.multiple(
            ai_runner,
            PROJECT_DIR=stand_in_project,
            TASK_FILE=task_file,
            MODEL_CONFIG=ai_runner.MODEL_CONFIG,
            DEFAULT_RESULTS_DIR=self.default_dir,
            RESULTS_DIR=self.default_dir,
            RESULT_FILE=self.default_dir / "results.json",
            HISTORY_DIR=self.default_dir / "history",
        ).start()
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(self.temp)
        self.temp_before = fingerprint(self.temp)
        # Fake model: the availability check passes and answers are instant.
        self.check = mock.patch.object(
            ai_runner.model_adapter, "check_model_available", return_value=None
        ).start()
        self.ask = mock.patch.object(ai_runner, "ask_model", side_effect=fake_ask_model).start()
        # Ignore any BENCHMARK_* settings in the real terminal.
        mock.patch.dict(os.environ).start()
        for name in list(os.environ):
            if name.startswith("BENCHMARK_"):
                del os.environ[name]
        self.addCleanup(mock.patch.stopall)

    def tearDown(self):
        self.assertEqual(
            fingerprint(REAL_RESULTS), self.real_results_before,
            "the real results/ folder changed",
        )

    def run_main(self, *args):
        """Run ai_runner.main() quietly. Returns the printed output."""
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            ai_runner.main(list(args))
        return output.getvalue()

    def assert_stops_with_error(self, args, message):
        with self.assertRaises(SystemExit) as caught:
            self.run_main(*args)
        self.assertIn(message, str(caught.exception.code))


class TestDefaultUnchanged(ResultsDirTestCase):
    def test_default_paths_are_exactly_the_old_ones(self):
        # The runner's real defaults, as recorded when the module was imported.
        self.assertEqual(
            ORIGINAL_PATHS,
            (REAL_RESULTS, REAL_RESULTS / "results.json", REAL_RESULTS / "history"),
        )
        self.assertEqual(ORIGINAL_DEFAULT_DIR, REAL_RESULTS)
        # results_paths(None) gives exactly the default folder's three paths.
        self.assertEqual(
            ai_runner.results_paths(None),
            (self.default_dir, self.default_dir / "results.json", self.default_dir / "history"),
        )

    def test_omitting_the_option_keeps_default_paths(self):
        # Stop at the model check (before anything is written) and look at the paths.
        self.check.return_value = "server unavailable"
        self.assert_stops_with_error([], "server unavailable")
        self.assertEqual(ai_runner.RESULTS_DIR, self.default_dir)
        self.assertEqual(ai_runner.RESULT_FILE, self.default_dir / "results.json")
        self.assertEqual(ai_runner.HISTORY_DIR, self.default_dir / "history")
        self.assertFalse(self.default_dir.exists())   # nothing was written

    def test_omitting_the_option_writes_to_the_default_folder(self):
        # A full run with no option goes to the (stand-in) default folder.
        self.run_main()
        self.assertTrue((self.default_dir / "results.json").exists())


class TestCustomFolder(ResultsDirTestCase):
    def test_full_run_writes_only_into_the_custom_folder(self):
        out = self.temp / "run1"
        self.run_main("--results-dir", str(out))

        saved = json.loads((out / "results.json").read_text())
        self.assertEqual(saved["run_status"], "finished")
        self.assertEqual(saved["summary"]["completed_tasks"], len(EXPECTED))
        self.assertEqual(saved["summary"]["correct"], len(EXPECTED))
        # Only results.json: no leftover temporary file, and no history on a first run.
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["results.json"])

    def test_all_three_paths_come_from_the_custom_folder(self):
        out = self.temp / "run1"
        replaced = []
        real_replace = os.replace

        def spy_replace(src, dst):
            replaced.append((Path(src), Path(dst)))
            return real_replace(src, dst)

        with mock.patch.object(ai_runner.os, "replace", side_effect=spy_replace):
            self.run_main("--results-dir", str(out))

        self.assertEqual(ai_runner.RESULTS_DIR, out)
        self.assertEqual(ai_runner.RESULT_FILE, out / "results.json")
        self.assertEqual(ai_runner.HISTORY_DIR, out / "history")
        # The temporary save file is created in the custom folder too.
        self.assertTrue(replaced)
        for src, dst in replaced:
            self.assertEqual(src, out / "results.json.tmp")
            self.assertEqual(dst, out / "results.json")

    def test_history_backup_goes_into_the_custom_folder(self):
        out = self.temp / "run1"
        self.run_main("--results-dir", str(out))
        first_run = (out / "results.json").read_text()
        self.run_main("--results-dir", str(out))

        backups = list((out / "history").glob("results_*.json"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_text(), first_run)   # a real copy of run 1

    def test_relative_path_is_taken_from_the_current_folder(self):
        # setUp already moved into the temp folder.
        self.run_main("--results-dir", "relative/out")
        self.assertTrue((self.temp / "relative" / "out" / "results.json").exists())

    def test_interrupted_run_is_saved_in_the_custom_folder(self):
        # The second question "presses Ctrl+C".
        self.ask.side_effect = [fake_ask_model(q) for q in list(EXPECTED)[:1]] + [KeyboardInterrupt()]
        out = self.temp / "run1"
        self.run_main("--results-dir", str(out))
        saved = json.loads((out / "results.json").read_text())
        self.assertEqual(saved["run_status"], "interrupted")
        self.assertEqual(saved["summary"]["completed_tasks"], 1)


class TestFailsSafely(ResultsDirTestCase):
    def test_path_to_an_existing_file_is_refused(self):
        existing = self.temp / "notes.txt"
        existing.write_text("keep me")
        before = fingerprint(self.temp)
        self.assert_stops_with_error(["--results-dir", str(existing)], "points to a file")
        self.assertEqual(fingerprint(self.temp), before)   # nothing written or changed
        self.check.assert_not_called()   # stopped before any network request

    def test_empty_path_is_refused(self):
        for value in ["", "   "]:
            with self.subTest(value=value):
                self.assert_stops_with_error(["--results-dir", value], "must not be empty")
        self.check.assert_not_called()

    def test_folder_that_cannot_be_created_read_only_parent(self):
        if os.geteuid() == 0:
            self.skipTest("folder permissions do not apply to the root user")
        locked = self.temp / "locked"
        locked.mkdir()
        locked.chmod(0o500)                     # read + enter only, no writing
        self.addCleanup(locked.chmod, 0o700)
        self.assert_stops_with_error(
            ["--results-dir", str(locked / "out")], "Could not prepare the results folder"
        )
        self.assertFalse((locked / "out").exists())

    def test_folder_that_cannot_be_created_mocked_permission_error(self):
        out = self.temp / "out"
        with mock.patch.object(ai_runner.Path, "mkdir", side_effect=PermissionError("Permission denied")):
            self.assert_stops_with_error(["--results-dir", str(out)], "Permission denied")
        self.assertFalse(out.exists())
        self.ask.assert_not_called()             # no task was run

    def test_unknown_or_misspelled_option_is_refused(self):
        for args in (["--result-dir", "x"], ["--results", "x"], ["--unknown"]):
            with self.subTest(args=args):
                with self.assertRaises(SystemExit) as caught:
                    self.run_main(*args)
                self.assertEqual(caught.exception.code, 2)   # argparse's "bad usage" code
        self.check.assert_not_called()
        self.assertEqual(fingerprint(self.temp), self.temp_before)   # nothing written

    def test_model_unavailable_stops_before_the_folder_is_created(self):
        self.check.return_value = "Could not connect to the model endpoint"
        out = self.temp / "out"
        self.assert_stops_with_error(["--results-dir", str(out)], "Could not connect")
        self.assertFalse(out.exists())


class TestDefaultFolderRefused(ResultsDirTestCase):
    """--results-dir must not be a second way to write into the default results folder."""

    MESSAGE = "points to the default results folder"

    def make_default_folder_with_a_result(self):
        self.default_dir.mkdir()
        (self.default_dir / "results.json").write_text('{"keep": "me"}')
        return fingerprint(self.temp)

    def assert_refused(self, value):
        before = fingerprint(self.temp)
        self.assert_stops_with_error(["--results-dir", value], self.MESSAGE)
        self.assertEqual(fingerprint(self.temp), before)   # nothing written or changed
        self.check.assert_not_called()                     # no network request
        self.ask.assert_not_called()                       # no task run

    def test_relative_paths_to_the_default_folder_are_refused(self):
        self.make_default_folder_with_a_result()
        for value in [
            "stand_in_default_results",
            "./stand_in_default_results",
            "stand_in_default_results/",
            "stand_in_project/../stand_in_default_results",
        ]:
            with self.subTest(value=value):
                self.assert_refused(value)

    def test_absolute_paths_to_the_default_folder_are_refused(self):
        self.make_default_folder_with_a_result()
        for value in [str(self.default_dir), str(self.default_dir) + "/"]:
            with self.subTest(value=value):
                self.assert_refused(value)

    def test_symlink_to_the_default_folder_is_refused(self):
        self.make_default_folder_with_a_result()
        (self.temp / "shortcut").symlink_to(self.default_dir, target_is_directory=True)
        self.assert_refused("shortcut")

    def test_different_letter_case_is_refused_where_the_disk_ignores_case(self):
        self.make_default_folder_with_a_result()
        upper = self.temp / "STAND_IN_DEFAULT_RESULTS"
        if not upper.exists():
            self.skipTest("this disk is case-sensitive, so the names are different folders")
        self.assert_refused(str(upper))

    def test_default_folder_is_refused_even_before_it_exists(self):
        self.assertFalse(self.default_dir.exists())
        for value in ["stand_in_default_results", str(self.default_dir)]:
            with self.subTest(value=value):
                self.assert_refused(value)
        self.assertFalse(self.default_dir.exists())

    def test_real_default_folder_is_recognised(self):
        # Read-only: only the checking function is called; nothing is run or written.
        with mock.patch.object(ai_runner, "DEFAULT_RESULTS_DIR", ORIGINAL_DEFAULT_DIR):
            for value in [str(REAL_RESULTS), os.path.relpath(REAL_RESULTS, self.temp)]:
                with self.subTest(value=value):
                    self.assertIn(self.MESSAGE, ai_runner.check_results_dir(value))

    def test_folder_inside_the_default_folder_is_still_allowed(self):
        self.make_default_folder_with_a_result()
        out = self.default_dir / "runs" / "run1"
        self.run_main("--results-dir", str(out))
        self.assertTrue((out / "results.json").exists())
        self.assertEqual((self.default_dir / "results.json").read_text(), '{"keep": "me"}')

    def test_separate_custom_folder_still_works(self):
        out = self.temp / "my_runs" / "run1"
        self.run_main("--results-dir", str(out))
        saved = json.loads((out / "results.json").read_text())
        self.assertEqual(saved["run_status"], "finished")
        self.assertFalse(self.default_dir.exists())   # the default folder was not used

    def test_omitting_the_option_still_uses_the_default_folder(self):
        self.make_default_folder_with_a_result()
        self.run_main()
        saved = json.loads((self.default_dir / "results.json").read_text())
        self.assertEqual(saved["run_status"], "finished")
        self.assertEqual(len(list((self.default_dir / "history").glob("results_*.json"))), 1)


class TestRealCommandLine(ResultsDirTestCase):
    """Runs the runner as a real separate program, the way the terminal does."""

    def run_program(self, *args, base_url=None):
        env = {k: v for k, v in os.environ.items() if not k.startswith("BENCHMARK_")}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        if base_url:
            env["BENCHMARK_BASE_URL"] = base_url
        return subprocess.run(
            [sys.executable, str(RUNNER), *args],
            cwd=self.temp, env=env, capture_output=True, text=True, timeout=60,
        )

    def test_help_mentions_the_option(self):
        completed = self.run_program("--help")
        self.assertEqual(completed.returncode, 0)
        self.assertIn("--results-dir", completed.stdout)

    def test_real_program_reads_the_option_and_stops_safely(self):
        # Nothing listens on port 9, so no model can be reached: the run must
        # stop at the model check, before creating or writing anything.
        out = self.temp / "out"
        completed = self.run_program("--results-dir", str(out), base_url="http://127.0.0.1:9/v1")
        self.assertEqual(completed.returncode, 1)
        self.assertIn("Could not connect", completed.stderr)
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
