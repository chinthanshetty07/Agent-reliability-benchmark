import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

from grading import grade_numeric_answer

BASE_URL = "http://127.0.0.1:11434/v1"
MODEL_ID = "qwen3:4b"

PROJECT_DIR = Path(__file__).parent.parent
TASK_FILE = PROJECT_DIR / "tasks" / "benchmark.json"
RESULTS_DIR = PROJECT_DIR / "results"
RESULT_FILE = RESULTS_DIR / "results.json"
HISTORY_DIR = RESULTS_DIR / "history"

# qwen3 "thinks" before answering, and the thinking counts toward max_tokens.
# With only 300 tokens the model ran out while still thinking and returned
# an empty answer, so we allow much more room.
MAX_TOKENS = 4096

CONNECT_TIMEOUT = 10    # seconds to wait for Ollama to accept the connection
READ_TIMEOUT = 120      # seconds to wait for the NEXT streamed chunk (covers model loading)
TASK_TIME_LIMIT = 300   # seconds allowed for one whole task, so nothing hangs forever

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


def check_ollama():
    """Make sure Ollama is running and the model is installed before starting."""
    try:
        response = requests.get(f"{BASE_URL}/models", timeout=CONNECT_TIMEOUT)
        response.raise_for_status()
        model_ids = [model.get("id") for model in response.json().get("data", [])]
    except requests.ConnectionError:
        sys.exit(
            f"ERROR: Could not connect to Ollama at {BASE_URL}.\n"
            "Start it with the Ollama app or run: ollama serve"
        )
    except (requests.RequestException, ValueError) as error:
        sys.exit(f"ERROR: Ollama responded unexpectedly at {BASE_URL}/models: {error}")

    if MODEL_ID not in model_ids:
        sys.exit(
            f"ERROR: Model '{MODEL_ID}' is not installed in Ollama.\n"
            f"Installed models: {', '.join(model_ids) or '(none)'}"
        )


def ask_model(question):
    """
    Send one question to Ollama using streaming.
    Returns (answer_text, reasoning_text, finish_reason).
    Raises an exception with a readable message if anything goes wrong.
    """
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "stream": True,
    }

    answer_parts = []
    reasoning_parts = []
    finish_reason = None
    deadline = time.perf_counter() + TASK_TIME_LIMIT

    with requests.post(
        f"{BASE_URL}/chat/completions",
        json=payload,
        stream=True,
        timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
    ) as response:
        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama returned HTTP {response.status_code}: {response.text[:300]}"
            )

        for line in response.iter_lines(decode_unicode=True):
            if time.perf_counter() > deadline:
                raise TimeoutError(f"Task took longer than {TASK_TIME_LIMIT} seconds")

            if not line or not line.startswith("data: "):
                continue

            data = line[len("data: "):]
            if data == "[DONE]":
                break

            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                raise ValueError(f"Malformed data from Ollama: {data[:200]}")

            if "error" in chunk:
                raise RuntimeError(f"Ollama error: {chunk['error']}")

            choices = chunk.get("choices") or []
            if not choices:
                continue

            delta = choices[0].get("delta") or {}
            if delta.get("content"):
                answer_parts.append(delta["content"])
            # qwen3's thinking arrives separately in "reasoning".
            if delta.get("reasoning"):
                reasoning_parts.append(delta["reasoning"])
            if choices[0].get("finish_reason"):
                finish_reason = choices[0]["finish_reason"]

    return "".join(answer_parts), "".join(reasoning_parts), finish_reason


def run_task(task):
    """Run one task and return a result record. Never raises."""
    task_id = task.get("task_id", "(no task_id)") if isinstance(task, dict) else "(invalid task)"

    result = {
        "task_id": task_id,
        "category": get_category(task) if isinstance(task, dict) else "unknown",
        "difficulty": task.get("difficulty", "unknown") if isinstance(task, dict) else "unknown",
        "question": task.get("question") if isinstance(task, dict) else None,
        "expected_answer": task.get("expected_answer") if isinstance(task, dict) else None,
        "model": MODEL_ID,
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
        result["error"] = f"Timed out connecting to Ollama after {CONNECT_TIMEOUT} seconds"
    except requests.ReadTimeout:
        result["error"] = f"Ollama sent nothing for {READ_TIMEOUT} seconds (read timeout)"
    except requests.ConnectionError as error:
        result["error"] = f"Lost connection to Ollama: {error}"
    except (requests.RequestException, RuntimeError, ValueError, TimeoutError) as error:
        result["error"] = str(error)
    else:
        result["model_answer"] = answer
        result["model_reasoning"] = reasoning
        result["finish_reason"] = finish_reason
        if finish_reason == "length":
            result["warning"] = f"Response was cut off at max_tokens={MAX_TOKENS}"
        if not answer.strip():
            result["error"] = "Model returned an empty answer"

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


def main():
    tasks = load_tasks()
    check_ollama()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    backup_previous_results()

    run_info = {
        "model": MODEL_ID,
        "base_url": BASE_URL,
        "task_file": str(TASK_FILE.relative_to(PROJECT_DIR)),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "finished_at": None,
        "run_status": "running",
    }
    results = []

    print(f"Loaded {len(tasks)} benchmark tasks.")
    print(f"Using model: {MODEL_ID}")

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
