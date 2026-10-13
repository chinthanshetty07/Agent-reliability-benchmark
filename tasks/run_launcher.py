"""
Run launcher: start the benchmark runner as a SEPARATE program, check on its
progress, and stop it safely. The dashboard will use this in the next step.

How a run works:
1. create_run_folder() makes a brand-new, empty folder under results/runs/.
   It never reuses or overwrites an existing folder.
2. start_run() starts `python tasks/ai_runner.py --results-dir <that folder>`
   and returns immediately, so the caller (the dashboard) doesn't freeze.
   The runner's printed output goes to runner.log in the run folder.
3. read_progress() / describe_run() read the run's results.json, which the
   runner saves after every task.
4. stop_run() asks the runner to stop (like pressing Ctrl+C), so it saves
   "interrupted". Only if it doesn't stop in time is it forcibly ended.

Security notes (local, single-user use):
- The model settings, including the API key, are passed to the runner ONLY
  through its environment variables. The key is never put on the command
  line, in runner.log, or in result files.
- Only the settings given to start_run() are passed. Any BENCHMARK_* values
  already in this program's environment are removed first, so an empty key
  field never silently uses a key from elsewhere.
- Environment variables are not a perfect security boundary: on macOS, other
  programs running as the same user (and administrators) can read a
  program's environment, for example with `ps -E`. That is acceptable for a
  local tool on your own computer, not for a shared or hosted server.
"""

import json
import os
import secrets
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import model_adapter

PROJECT_DIR = Path(__file__).resolve().parent.parent
RUNNER = PROJECT_DIR / "tasks" / "ai_runner.py"
RUNS_DIR = PROJECT_DIR / "results" / "runs"
LOG_NAME = "runner.log"
RESULTS_NAME = "results.json"
LAUNCH_INFO_NAME = "launcher.json"   # the run's process number (no key, no settings)

STOP_GRACE_SECONDS = 10   # how long stop_run() waits before forcing the run to end
FOLDER_ATTEMPTS = 5       # tries to find an unused folder name (a clash is very unlikely)


class LaunchError(RuntimeError):
    """A run could not be started safely."""


def create_run_folder(runs_dir=None):
    """
    Create and return a new, empty folder such as results/runs/20261013-142530-a1b2c3.
    An existing folder is never reused, so nothing can be overwritten.
    """
    runs_dir = Path(runs_dir) if runs_dir is not None else RUNS_DIR
    runs_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(FOLDER_ATTEMPTS):
        name = f"{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"
        folder = runs_dir / name
        try:
            folder.mkdir()          # fails if the folder already exists
        except FileExistsError:
            continue
        return folder
    raise LaunchError(f"Could not create a new run folder in {runs_dir}")


def build_runner_env(config, base_env=None):
    """
    The environment for the runner: a copy of this program's environment with
    every BENCHMARK_* value removed, then ONLY the given settings added.
    """
    base_env = os.environ if base_env is None else base_env
    env = {name: value for name, value in base_env.items() if not name.startswith("BENCHMARK_")}
    env[model_adapter.ENV_BASE_URL] = config.base_url
    env[model_adapter.ENV_MODEL_ID] = config.model_id
    if config.api_key:
        env[model_adapter.ENV_API_KEY] = config.api_key
    env["PYTHONUNBUFFERED"] = "1"   # write progress lines to runner.log straight away
    return env


def start_run(config, folder, runner=None):
    """
    Start the runner in the background, writing into `folder`, and return the
    process handle (subprocess.Popen). Returns immediately; does not wait.
    `folder` must be an existing, empty folder from create_run_folder().
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise LaunchError(f"Run folder does not exist: {folder}")
    if any(folder.iterdir()):
        raise LaunchError(f"Run folder is not empty, refusing to reuse it: {folder}")

    # 1. Write launcher.json BEFORE starting anything, so a run can never exist
    #    without it. Its process number isn't known yet (null), which
    #    check_process_alive() reports as "unknown": a dashboard restart in
    #    this short window blocks new runs instead of missing this one.
    #    If it can't be written, no process is started.
    started_at = datetime.now().isoformat(timespec="seconds")
    try:
        _write_launch_info(folder, {"pid": None, "started_at": started_at})
    except OSError as error:
        raise LaunchError(
            f"Could not write {LAUNCH_INFO_NAME} in {folder}, so the run was not started: {error}"
        ) from error

    # The key is NOT in this list; it only travels in the environment.
    command = [sys.executable, str(runner or RUNNER), "--results-dir", str(folder)]
    try:
        with open(folder / LOG_NAME, "w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_DIR,
                env=build_runner_env(config),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,   # errors go into the same log
            )
    except Exception as error:
        # No process was started (any error, e.g. OSError or a ValueError for an
        # invalid setting): remove the placeholder, so this folder looks like the
        # failed start it is (otherwise it would block as "unknown"). The original
        # error is re-raised unchanged. (Ctrl+C is not caught here: it leaves the
        # placeholder, which errs on the safe side by blocking as "unknown".)
        problem = _discard_launch_info(folder)
        if problem:
            raise LaunchError(model_adapter.redact(
                f"The run could not be started ({type(error).__name__}: {error}), and {problem}", config
            )) from error
        raise

    # 2. Record the real process number. Without it, a restarted dashboard
    #    couldn't recognise this run, so never leave it running untracked.
    try:
        _write_launch_info(folder, {"pid": process.pid, "started_at": started_at})
    except OSError as error:
        stopped = _stop_and_reap(process)
        problem = _discard_launch_info(folder)
        message = (f"Could not record the run's process number in {LAUNCH_INFO_NAME} ({error}), "
                   f"so the run was stopped{'' if stopped else ' (but it could not be confirmed to have ended)'}.")
        if problem:
            message += f" Also, {problem}"
        raise LaunchError(message) from error
    return process


def _write_launch_info(folder, info):
    """Write launcher.json in one step: a half-written file is never left behind."""
    target = Path(folder) / LAUNCH_INFO_NAME
    temp = target.with_name(target.name + ".tmp")
    try:
        temp.write_text(json.dumps(info), encoding="utf-8")
        os.replace(temp, target)
    except OSError:
        temp.unlink(missing_ok=True)
        raise


def _discard_launch_info(folder):
    """
    Remove launcher.json after a start that did not leave a running process.
    Returns None on success, or a description of the problem (never raises).
    """
    try:
        (Path(folder) / LAUNCH_INFO_NAME).unlink(missing_ok=True)
        return None
    except OSError as error:
        return (f"its {LAUNCH_INFO_NAME} could not be removed ({error}); this run will show "
                "as status unknown until you confirm it isn't running.")


def _stop_and_reap(process, timeout=10):
    """End a process we just started and wait for it, so no orphan is left. True if it ended."""
    try:
        process.kill()
        process.wait(timeout=timeout)
        return True
    except (OSError, subprocess.TimeoutExpired):
        return process.poll() is not None


def check_process_alive(folder):
    """
    For a run we have NO process handle for (e.g. after a dashboard restart):
    is its program still running? Read-only; never sends a signal.
    Returns:
      "active"   a process with the recorded number is running AND its command
                 line points at this run's folder
      "ended"    no such process, or the number now belongs to something else
                 (process numbers get reused), or it has ended but not been
                 cleaned up yet (shown by ps as "<defunct>")
      "unknown"  no usable launcher.json, or the check itself failed
    """
    folder = Path(folder)
    try:
        pid = int(json.loads((folder / LAUNCH_INFO_NAME).read_text(encoding="utf-8"))["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return "unknown"
    try:
        completed = subprocess.run(
            ["ps", "-ww", "-o", "command=", "-p", str(pid)],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if completed.returncode != 0:
        return "ended"
    command_line = completed.stdout.strip()
    return "active" if f"--results-dir {folder}" in command_line else "ended"


def read_progress(folder):
    """
    Read the run's results.json. Returns a dict with:
      status     "not_started" (no results yet), "running", "finished",
                 "interrupted", or "unreadable"
      completed, total, correct, incorrect, errors   (None until known)
    """
    progress = {"status": "not_started", "completed": None, "total": None,
                "correct": None, "incorrect": None, "errors": None}
    results_file = Path(folder) / RESULTS_NAME
    if not results_file.exists():
        return progress
    try:
        data = json.loads(results_file.read_text(encoding="utf-8"))
        summary = data["summary"]
        progress.update(
            status=data["run_status"],
            completed=summary["completed_tasks"],
            total=summary["total_tasks"],
            correct=summary["correct"],
            incorrect=summary["incorrect"],
            errors=summary["errors"],
        )
    except (OSError, ValueError, KeyError, TypeError):
        progress["status"] = "unreadable"
    return progress


def describe_run(process, folder):
    """
    Combine the process state and the saved progress into one overall state:
      "running"              still working
      "finished"             ended normally (check errors: a run can finish with failed tasks)
      "interrupted"          stopped on request; completed tasks were saved
      "stopped_before_start" stopped on request (Ctrl+C) before any results were saved
      "failed"               ended with an error before saving results (see runner.log)
      "stopped_unexpectedly" ended without saving a final status (e.g. forcibly ended)
    """
    # Check the process FIRST: if it ended, its final save already happened,
    # so the file read next is up to date.
    exit_code = process.poll()
    progress = read_progress(folder)
    if exit_code is None:
        state = "running"
    elif progress["status"] in ("finished", "interrupted"):
        state = progress["status"]
    elif progress["status"] == "not_started":
        # Exit code -2 means "ended by Ctrl+C (SIGINT)". The runner only saves
        # "interrupted" once its task loop has started, so a stop during start-up
        # leaves no results; that is a stop, not a failure.
        state = "stopped_before_start" if exit_code == -signal.SIGINT else "failed"
    else:
        state = "stopped_unexpectedly"
    return {"state": state, "exit_code": exit_code, **progress}


def outcome(described):
    """
    Turn describe_run()'s result into (level, message) for display.
    level is "info", "success", "warning" or "error".
    Uses the saved COUNTS, not the exit code: the runner exits with 0 even
    when every task failed.
    """
    state = described["state"]
    done, total = described.get("completed"), described.get("total")
    correct, errors = described.get("correct"), described.get("errors")

    if state == "running":
        if done is None:
            return "info", "Starting: checking the model and loading tasks."
        return "info", f"Running: {done} of {total} tasks done."
    if state == "finished":
        if not done:
            return "warning", "Finished, but no tasks ran."
        if errors == done:
            return "error", f"Finished, but every task failed ({errors} errors). See runner.log."
        if errors:
            return "warning", f"Finished with {errors} error(s): {correct}/{done} correct."
        return "success", f"Finished: {correct}/{done} correct."
    if state == "interrupted":
        return "warning", f"Stopped: {done} of {total} tasks saved."
    if state == "stopped_before_start":
        return "warning", "Stopped before the first task. No results were saved."
    if state == "failed":
        return "error", "Could not start the benchmark. See runner.log for the reason."
    if state == "stopped_unexpectedly":
        return "error", ("Ended without saving a final status. "
                         "Results up to the last saved task are kept.")
    return "error", f"Unknown run state: {state}"


def unfinished_runs(exclude=(), runs_dir=None):
    """
    Run folders whose saved status is still "running", newest first, leaving out
    `exclude`. Without a live process handle, such a run's real state is
    UNKNOWN: it may still be running, or it ended without saving. Read-only.
    """
    runs_dir = Path(runs_dir) if runs_dir is not None else RUNS_DIR
    if not runs_dir.is_dir():
        return []
    excluded = {Path(folder).resolve() for folder in exclude}
    return [
        folder
        for folder in sorted(runs_dir.iterdir(), reverse=True)
        if folder.is_dir()
        and folder.resolve() not in excluded
        and read_progress(folder)["status"] == "running"
    ]


def runs_without_final_status(exclude=(), runs_dir=None):
    """
    Run folders that have no final status yet, newest first, leaving out `exclude`.
    Each item is (folder, saved_status). Included:
      - saved status "running" (with or without launcher.json), and
      - no results yet ("not_started") or unreadable results ("unreadable"),
        but ONLY if launcher.json exists, i.e. a runner process was started.
        (Folders from failed starts have no launcher.json and are left out.)
    Runs saying "finished" or "interrupted" are complete and never included.
    Read-only.
    """
    runs_dir = Path(runs_dir) if runs_dir is not None else RUNS_DIR
    if not runs_dir.is_dir():
        return []
    excluded = {Path(folder).resolve() for folder in exclude}
    found = []
    for folder in sorted(runs_dir.iterdir(), reverse=True):
        if not folder.is_dir() or folder.resolve() in excluded:
            continue
        status = read_progress(folder)["status"]
        launched = (folder / LAUNCH_INFO_NAME).exists()
        if status == "running" or (status in ("not_started", "unreadable") and launched):
            found.append((folder, status))
    return found


def runs_blocking_a_new_start(exclude=(), confirmed_not_running=()):
    """
    Runs without a process handle that might still be active, so a new run
    must not start yet. Each item is (folder, liveness):
      "active"  always blocks
      "unknown" blocks unless its folder name is in `confirmed_not_running`
                (the user checked it; kept in memory only, never written to disk).
                An unknown process is never assumed to have stopped.
    Runs whose process has "ended" never block. Read-only.
    """
    blocking = []
    for folder, _status in runs_without_final_status(exclude=exclude):
        liveness = check_process_alive(folder)
        if liveness == "active" or (liveness == "unknown" and folder.name not in confirmed_not_running):
            blocking.append((folder, liveness))
    return blocking


def read_log_tail(folder, max_lines=15):
    """The last lines of the run's runner.log, or "" if there is none. Read-only."""
    log = Path(folder) / LOG_NAME
    if not log.exists():
        return ""
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def stop_run(process, grace_seconds=None):
    """
    Ask the run to stop, like pressing Ctrl+C (SIGINT), so it saves "interrupted".
    If it is still running after `grace_seconds` (default STOP_GRACE_SECONDS),
    end it forcibly. Returns "already_stopped", "interrupted", or "killed".
    The return value only says how the PROCESS ended; use describe_run() to
    see what was actually saved.
    """
    if grace_seconds is None:
        grace_seconds = STOP_GRACE_SECONDS
    if process.poll() is not None:
        return "already_stopped"
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=grace_seconds)
        return "interrupted"
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        return "killed"
