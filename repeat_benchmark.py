
import json
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
RESULTS_FILE = ROOT / "results" / "results.json"
OUTPUT_DIR = ROOT / "results" / "repeated_trials"
RUNNER = ROOT / "tasks" / "ai_runner.py"

TRIALS = 2  # Two additional runs; your existing run is trial 1


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not RESULTS_FILE.exists():
        print("Error: Existing results.json was not found.")
        sys.exit(1)

    with RESULTS_FILE.open() as f:
        original = json.load(f)

    if original.get("run_status") != "finished":
        print("Error: Existing benchmark did not finish successfully.")
        sys.exit(1)

    original_tasks = original.get("results", [])
    if not original_tasks:
        print("Error: No task results found.")
        sys.exit(1)

    task_ids = [r.get("task_id") for r in original_tasks]
    if len(task_ids) != len(set(task_ids)):
        print("Error: Duplicate task IDs in existing results.")
        sys.exit(1)

    print(f"Existing run: {len(original_tasks)} task results")
    print(f"Additional trials to run: {TRIALS}")
    print("Each trial runs the full benchmark and saves a separate copy.")

    # Save trial 1 to its own file. results.json is overwritten by the
    # trials below, so without this copy trial 1's answers would be lost.
    trial_1_file = OUTPUT_DIR / "trial_1.json"
    shutil.copy2(RESULTS_FILE, trial_1_file)
    print(f"Trial 1 saved to {trial_1_file}")

    all_runs = [{
        "trial": 1,
        "source": str(trial_1_file.relative_to(ROOT)),
        "summary": original.get("summary", {}),
        "results": original_tasks,
    }]

    for trial in range(2, TRIALS + 2):
        print(f"\nStarting trial {trial} of {TRIALS + 1}...")

        # Preserve the current results before the runner replaces them.
        with RESULTS_FILE.open() as f:
            previous_content = f.read()

        try:
            completed = subprocess.run(
                [sys.executable, str(RUNNER)],
                cwd=ROOT,
                check=False,
            )

            if completed.returncode != 0:
                print(f"Trial {trial} failed with exit code "
                      f"{completed.returncode}.")
                print("Restoring the previous results file.")
                RESULTS_FILE.write_text(previous_content)
                break

            with RESULTS_FILE.open() as f:
                run_data = json.load(f)

            if run_data.get("run_status") != "finished":
                print(f"Trial {trial} did not finish.")
                print("Restoring the previous results file.")
                RESULTS_FILE.write_text(previous_content)
                break

            current_ids = [r.get("task_id") for r in run_data.get("results", [])]
            if current_ids != task_ids:
                print("Task IDs/order changed. Restoring previous results.")
                RESULTS_FILE.write_text(previous_content)
                break

            saved_copy = OUTPUT_DIR / f"trial_{trial}.json"
            saved_copy.write_text(json.dumps(run_data, indent=2))

            all_runs.append({
                "trial": trial,
                "source": str(saved_copy.relative_to(ROOT)),
                "summary": run_data.get("summary", {}),
                "results": run_data.get("results", []),
            })

            print(f"Trial {trial} saved to {saved_copy}")

        except (OSError, json.JSONDecodeError) as error:
            print(f"Could not save/read trial {trial}: {error}")
            RESULTS_FILE.write_text(previous_content)
            break

    # Compare task correctness across successfully completed trials.
    task_report = []
    for index, task_id in enumerate(task_ids):
        outcomes = []
        for run in all_runs:
            rows = run["results"]
            row = next(
                (r for r in rows if r.get("task_id") == task_id),
                None,
            )
            outcomes.append(bool(row and row.get("correct") is True))

        task_report.append({
            "task_id": task_id,
            "trials": len(outcomes),
            "correct_trials": sum(outcomes),
            "consistency_rate": round(
                sum(outcomes) / len(outcomes) * 100, 2
            ) if outcomes else 0,
            "outcomes": outcomes,
        })

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": original.get("model"),
        "total_trials": len(all_runs),
        "task_count": len(task_ids),
        "trials": [
            {
                "trial": run["trial"],
                "source": run["source"],
                "summary": run["summary"],
            }
            for run in all_runs
        ],
        "task_consistency": task_report,
    }

    report_file = OUTPUT_DIR / "reliability_report.json"
    report_file.write_text(json.dumps(report, indent=2))

    print("\n===== Reliability Report =====")
    print(f"Completed trials: {len(all_runs)}")
    print(f"Tasks compared: {len(task_ids)}")
    print(f"Report saved to: {report_file}")
    print("\nPer-task consistency:")
    for task in task_report:
        print(
            f"{task['task_id']}: "
            f"{task['correct_trials']}/{task['trials']} correct"
        )


if __name__ == "__main__":
    main()