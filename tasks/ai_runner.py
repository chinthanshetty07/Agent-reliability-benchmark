import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

import model_adapter
from grading import grade_numeric_answer

# Which model to use and how to reach it. These are the local Ollama defaults;
# main() replaces them with any BENCHMARK_* environment variable settings.
# (Model communication itself lives in model_adapter.py.)
MODEL_CONFIG = model_adapter.ModelConfig()

PROJECT_DIR = Path(__file__).parent.parent
TASK_FILE = PROJECT_DIR / "tasks" / "benchmark.json"
RESULTS_DIR = PROJECT_DIR / "results"
RESULT_FILE = RESULTS_DIR / "results.json"
HISTORY_DIR = RESULTS_DIR / "history"
# Where results go when --results-dir is not given.
DEFAULT_RESULTS_DIR = RESULTS_DIR

# Asks for a predictable last line so the grader can find the final answer.
# It does not reveal any answers.
SYSTEM_PROMPT = (
    "Solve the user's question. End your reply with exactly one line in the form "
    "'Final answer: <number>'. Write only the number on that line (no $ sign or words). "
    "Use a negative number for a loss."
)

# Every task must have these. The category may be called "category" or "domain".
REQUIRED_FIELDS = ["task_id", "question", "expected_answer"]


def load_tasks():
    """Read benchmark.json and give a clear message if it is missing or broken."""
    if not TASK_FILE.exists():
        sys.exit(f"ERROR: Task file not found: {TASK_FILE}")

    try:
        with open(TASK_FILE, "r", encoding="utf-8") as file:
            tasks = json.load(file)
    except json.JSONDecodeError as error:
        sys.exit(
            f"ERROR: {TASK_FILE} is not valid JSON "
            f"(line {error.lineno}, column {error.colno}): {error.msg}"
        )

    if not isinstance(tasks, list):
        sys.exit(f"ERROR: {TASK_FILE} must contain a JSON list of tasks.")

    return tasks


def check_task(task):
    """Return a list of problems with a task (an empty list means the task is OK)."""
    if not isinstance(task, dict):
        return ["task is not a JSON object"]

    problems = []
    for field in REQUIRED_FIELDS:
        value = task.get(field)
        if value is None or str(value).strip() == "":
            problems.append(f"missing or empty field '{field}'")

    if not task.get("category") and not task.get("domain"):
        problems.append("missing field 'category' (or 'domain')")

    return problems


def get_category(task):
    """Support both the new 'category' key and the older 'domain' key."""
    return task.get("category") or task.get("domain") or "unknown"


def ask_model(question):
    """
    Send one question to the configured model (see model_adapter.py).
    Returns (answer_text, reasoning_text, finish_reason).
    Raises an exception with a readable message if anything goes wrong.
    """
    return model_adapter.ask_model(MODEL_CONFIG, SYSTEM_PROMPT, question)


def run_task(task):
    """Run one task and return a result record. Never raises."""
    task_id = task.get("task_id", "(no task_id)") if isinstance(task, dict) else "(invalid task)"

    result = {
        "task_id": task_id,
        "category": get_category(task) if isinstance(task, dict) else "unknown",
        "difficulty": task.get("difficulty", "unknown") if isinstance(task, dict) else "unknown",
        "question": task.get("question") if isinstance(task, dict) else None,
        "expected_answer": task.get("expected_answer") if isinstance(task, dict) else None,
        "model": MODEL_CONFIG.model_id,
        "model_answer": "",
        "model_reasoning": "",
        "extracted_answer": None,
        "extraction_method": None,
        "grading_note": None,
        "correct": False,
        "status": "error",   # becomes "correct" or "incorrect" once graded
        "response_time_seconds": None,
        "finish_reason": None,
        "error": None,
        "warning": None,
    }

    problems = check_task(task)
    if problems:
        result["error"] = "Invalid task: " + "; ".join(problems)
        return result

    start_time = time.perf_counter()
    try:
        answer, reasoning, finish_reason = ask_model(task["question"])
    except requests.ConnectTimeout:
        result["error"] = (
            f"Timed out connecting to the model endpoint after "
            f"{MODEL_CONFIG.connect_timeout} seconds"
        )
    except requests.ReadTimeout:
        result["error"] = (
            f"The model endpoint sent nothing for {MODEL_CONFIG.read_timeout} seconds "
            "(read timeout)"
        )
    except requests.ConnectionError as error:
        result["error"] = f"Lost connection to the model endpoint: {error}"
    except (requests.RequestException, RuntimeError, ValueError, TimeoutError) as error:
        result["error"] = str(error)
    else:
        result["model_answer"] = answer
        result["model_reasoning"] = reasoning
        result["finish_reason"] = finish_reason
        if finish_reason == "length":
            result["warning"] = f"Response was cut off at max_tokens={MODEL_CONFIG.max_tokens}"
        if not answer.strip():
            result["error"] = "Model returned an empty answer"

    # Error text can come from the server, so make sure no secret is saved.
    result["error"] = model_adapter.redact(result["error"], MODEL_CONFIG)

    result["response_time_seconds"] = round(time.perf_counter() - start_time, 2)

    # Grade only if the model actually answered.
    if result["error"] is None:
        grade = grade_numeric_answer(result["model_answer"], task["expected_answer"])
        result.update(grade)
        result["status"] = "correct" if grade["correct"] else "incorrect"

    return result


def summarize(results, total_tasks):
    """Correctness rate and response-time stats for the results so far."""
    correct = sum(1 for r in results if r["status"] == "correct")
    incorrect = sum(1 for r in results if r["status"] == "incorrect")
    errors = sum(1 for r in results if r["status"] == "error")
    times = [
        r["response_time_seconds"]
        for r in results
        if r["status"] != "error" and r["response_time_seconds"] is not None
    ]

    return {
        "total_tasks": total_tasks,
        "completed_tasks": len(results),
        "correct": correct,
        "incorrect": incorrect,
        "errors": errors,
        # Errors count as not correct, so failures cannot inflate the score.
        "correctness_rate": round(correct / len(results), 4) if results else None,
        "avg_response_time_seconds": round(sum(times) / len(times), 2) if times else None,
        "min_response_time_seconds": min(times) if times else None,
        "max_response_time_seconds": max(times) if times else None,
    }


def save_results(run_info, results, total_tasks):
    """Write results safely: write a temp file first, then swap it in."""
    data = dict(run_info)
    data["summary"] = summarize(results, total_tasks)
    data["results"] = results

    temp_file = RESULT_FILE.with_suffix(".json.tmp")
    temp_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp_file, RESULT_FILE)


def backup_previous_results():
    """Keep the last run's results instead of overwriting them."""
    if RESULT_FILE.exists():
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = HISTORY_DIR / f"results_{stamp}.json"
        shutil.copy2(RESULT_FILE, backup)
        print(f"Previous results backed up to: {backup}")


def parse_args(argv=None):
    """Read the command-line options. Unknown or misspelled options stop the program."""
    parser = argparse.ArgumentParser(
        description="Run the benchmark against the configured model.",
        allow_abbrev=False,   # "--results" must not silently mean "--results-dir"
    )
    parser.add_argument(
        "--results-dir",
        metavar="PATH",
        help="Folder for results.json and its history/ backups "
             "(default: results/ in the project). A relative path is taken "
             "from the folder you run the command in.",
    )
    return parser.parse_args(argv)


def results_paths(results_dir=None):
    """
    Return (results folder, results file, history folder).
    All three come from ONE folder, so they can never point to different places.
    No folder given -> exactly the default paths under results/.
    """
    if results_dir is None:
        folder = DEFAULT_RESULTS_DIR
    else:
        # A relative path is joined onto the current working directory;
        # an absolute path stays as it is.
        folder = Path.cwd() / Path(results_dir).expanduser()
    return folder, folder / "results.json", folder / "history"


def check_results_dir(results_dir):
    """Return an error message if the --results-dir value is unusable, else None."""
    if results_dir is None:
        return None
    if not results_dir.strip():
        return "--results-dir must not be empty."
    folder = results_paths(results_dir)[0]
    if folder.exists() and not folder.is_dir():
        return f"--results-dir points to a file, not a folder: {folder}"
    if is_same_folder(folder, DEFAULT_RESULTS_DIR):
        return (
            f"--results-dir points to the default results folder ({DEFAULT_RESULTS_DIR}). "
            "Choose a different folder, or leave out --results-dir to write there."
        )
    return None


def is_same_folder(a, b):
    """
    True if two paths are the same folder on disk, however they are written
    (relative, absolute, with "..", through a symlink, or different letter case on macOS).
    """
    if a.exists() and b.exists():
        return os.path.samefile(a, b)      # ask the operating system
    return a.resolve() == b.resolve()      # folder not created yet: compare cleaned-up paths


def main(argv=None):
    # Replace the defaults with this run's settings.
    global MODEL_CONFIG, RESULTS_DIR, RESULT_FILE, HISTORY_DIR

    # Options are checked first, before any setting, file, or network request.
    args = parse_args(argv)
    problem = check_results_dir(args.results_dir)
    if problem:
        sys.exit(f"ERROR: {problem}")
    if args.results_dir is not None:
        RESULTS_DIR, RESULT_FILE, HISTORY_DIR = results_paths(args.results_dir)

    try:
        MODEL_CONFIG = model_adapter.load_model_config()
    except model_adapter.ModelConfigError as error:
        sys.exit(f"ERROR: {error}")

    tasks = load_tasks()
    problem = model_adapter.check_model_available(MODEL_CONFIG)
    if problem:
        sys.exit(f"ERROR: {problem}")

    # Nothing has been written yet. If the folder can't be prepared, stop here.
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        backup_previous_results()
    except OSError as error:
        sys.exit(f"ERROR: Could not prepare the results folder {RESULTS_DIR}: {error}")

    run_info = {
        "model": MODEL_CONFIG.model_id,
        # Saved without any login details or ?query part; the API key is never saved.
        "base_url": model_adapter.safe_base_url(MODEL_CONFIG.base_url),
        "task_file": str(TASK_FILE.relative_to(PROJECT_DIR)),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "finished_at": None,
        "run_status": "running",
    }
    results = []

    print(f"Loaded {len(tasks)} benchmark tasks.")
    print(f"Using model: {MODEL_CONFIG.model_id}")
    print(f"Endpoint: {model_adapter.safe_base_url(MODEL_CONFIG.base_url)}")

    try:
        for index, task in enumerate(tasks, start=1):
            task_id = task.get("task_id", "?") if isinstance(task, dict) else "?"
            print(f"\n[{index}/{len(tasks)}] Running {task_id}...")
            result = run_task(task)
            results.append(result)

            if result["status"] == "error":
                print(f"  ERROR: {result['error']}")
            else:
                print(f"  Expected: {result['expected_answer']}")
                print(f"  Model's final answer: {result['extracted_answer']}")
                print(f"  Result: {result['status'].upper()}")
            if result["warning"]:
                print(f"  Warning: {result['warning']}")
            if result["response_time_seconds"] is not None:
                print(f"  Time: {result['response_time_seconds']} seconds")

            # Save after every task, so progress is not lost.
            save_results(run_info, results, len(tasks))

        run_info["run_status"] = "finished"
    except KeyboardInterrupt:
        run_info["run_status"] = "interrupted"
        print("\nStopped by user. Saving the tasks completed so far.")

    run_info["finished_at"] = datetime.now().isoformat(timespec="seconds")
    save_results(run_info, results, len(tasks))

    summary = summarize(results, len(tasks))
    print("\n===== Summary =====")
    print(f"Tasks run: {summary['completed_tasks']} of {summary['total_tasks']}")
    print(
        f"Correct: {summary['correct']}   Incorrect: {summary['incorrect']}   "
        f"Errors: {summary['errors']}"
    )
    if summary["correctness_rate"] is not None:
        print(f"Correctness rate: {summary['correctness_rate'] * 100:.1f}%")
    if summary["avg_response_time_seconds"] is not None:
        print(f"Average response time: {summary['avg_response_time_seconds']} seconds")
    print(f"\nResults saved to: {RESULT_FILE}")


if __name__ == "__main__":
    main()
