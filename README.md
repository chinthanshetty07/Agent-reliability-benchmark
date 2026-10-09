# Agent Reliability Benchmark (early prototype)

Asks a local AI model a set of finance questions, checks whether each final
answer is actually correct, and shows the results in a dashboard.

- Model: `qwen3:4b`, served by [Ollama](https://ollama.com) at `http://127.0.0.1:11434/v1`
- Dataset: 16 finance tasks in `tasks/benchmark.json`

## Requirements

- macOS, Linux, or Windows (tested on macOS, Apple Silicon M1, 8 GB RAM)
- Python 3.11 or newer (tested with Python 3.14)
- Ollama installed and running

## Install

Run these from the project folder.

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install the Python packages listed in `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the model (one time, about 2.5 GB):

   ```bash
   ollama pull qwen3:4b
   ```

4. Check that Ollama is running and the model is listed:

   ```bash
   curl http://127.0.0.1:11434/v1/models
   ```

## Run

All commands are run from the project folder with the virtual environment active.

### 1. Run the benchmark

```bash
python3 tasks/ai_runner.py
```

Each task takes roughly 20–60 seconds on an M1 Mac. Results are saved after every
task to `results/results.json`. Before a new run, the previous file is copied to
`results/history/`.

### 2. Run repeated trials (optional)

```bash
python3 repeat_benchmark.py
```

Treats the current `results/results.json` as trial 1, then runs the full
benchmark 2 more times. Writes `results/repeated_trials/trial_N.json` and
`results/repeated_trials/reliability_report.json`, which shows how often each
task was answered correctly.

### 3. Open the dashboard

```bash
streamlit run dashboard.py
```

Then open http://localhost:8501. It reads `results/results.json` and, if present,
`results/repeated_trials/reliability_report.json`.

### Tests (no Ollama needed)

```bash
python3 -m unittest discover -s tests -v
```

## How correctness is checked

The model is asked to end its reply with `Final answer: <number>`.
`tasks/grading.py` reads only that final answer, not numbers in the question or
the working. It understands `$3,000`, `-1,200`, `(1,200)` and phrases like
"a loss of $1,200". It then compares the number with `expected_answer`. The sign
matters: `1200` does not match `-1200`. Failed or empty responses count as not
correct.

## Files

| Path | Purpose |
|---|---|
| `tasks/benchmark.json` | The questions. Each needs `task_id`, `question`, `expected_answer`, and `category` (the older name `domain` also works). |
| `tasks/ai_runner.py` | Sends each question to the model and records the results. |
| `tasks/grading.py` | Finds the model's final number and compares it with `expected_answer`. |
| `repeat_benchmark.py` | Runs the benchmark several times and measures per-task consistency. |
| `dashboard.py` | Streamlit dashboard for the results. |
| `tests/test_benchmark.py` | Unit tests for grading, task validation, and error handling. |
| `results/` | Benchmark output. |

## Scope and limitations

- 16 mostly short arithmetic finance questions, one model, a few trials. A high
  score here is a smoke test, not a scientific measure of agent reliability.
- Only numeric answers can be graded.
- Difficulty labels are the author's judgement, not a calibrated scale.
