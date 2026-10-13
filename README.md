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
streamlit run dashboard.py --server.address localhost
```

Then open http://localhost:8501. It reads `results/results.json` and, if present,
`results/repeated_trials/reliability_report.json`.

`--server.address localhost` makes the dashboard reachable only from this computer.
Without it, Streamlit also listens on your network address, so other devices on
the same network (for example, the same Wi-Fi) could open it.

### Tests (no Ollama needed)

```bash
python3 -m unittest discover -s tests -v
```

The tests use fake (mocked) server replies, so they need no network, model, or API key.

## Use a different model

By default the benchmark uses local Ollama with `qwen3:4b`; you don't need to set
anything. To use another model, set these environment variables in your terminal
before running:

| Variable | Default | Meaning |
|---|---|---|
| `BENCHMARK_BASE_URL` | `http://127.0.0.1:11434/v1` | Address of the model server |
| `BENCHMARK_MODEL_ID` | `qwen3:4b` | Exact model name on that server |
| `BENCHMARK_API_KEY` | *(none)* | Private key, only for services that need one |

Another model on your local Ollama (download it first with `ollama pull`):

```bash
BENCHMARK_MODEL_ID=llama3.2:3b python3 tasks/ai_runner.py
```

A hosted OpenAI-compatible service needs its address, model name and key. To
avoid saving the key in your shell history, type it into a hidden prompt:

```bash
read -s BENCHMARK_API_KEY && export BENCHMARK_API_KEY
```

```bash
export BENCHMARK_BASE_URL=https://your-provider.example/v1
export BENCHMARK_MODEL_ID=provider-model-name
python3 tasks/ai_runner.py
```

`repeat_benchmark.py` uses the same variables, because it starts the runner from
the same terminal.

**Keeping the key safe**
- The key is read only from `BENCHMARK_API_KEY`. Never put it in a file in this project.
- It is sent to the model server only as an `Authorization: Bearer` header.
- It is never saved in `results/`, and it is replaced with `***` in error messages.
- `base_url` is saved in results without any `user:password@` or `?query` part.

**Compatibility.** The adapter (`tasks/model_adapter.py`) uses the common
OpenAI-style API: `GET /models` to check the model exists, and streaming
`POST /chat/completions` to ask questions. **Only local Ollama has been tested.**
Other providers may differ: some don't list models, use other parameter names,
or reject `max_tokens=4096`. If something doesn't match, the runner stops or
records a clear error; it doesn't record a wrong score.

**Comparing models.** `repeat_benchmark.py` does not yet check that every trial
used the same model. Don't switch models between trial 1 and later trials.

### Test a model connection from the dashboard

The dashboard sidebar has a **Configure model** form:

1. Run `streamlit run dashboard.py --server.address localhost` and open
   http://localhost:8501.
2. Check or change the **Base URL** and **Model ID**. They are pre-filled from
   `BENCHMARK_BASE_URL` and `BENCHMARK_MODEL_ID` when set, otherwise the Ollama
   defaults. Login details in a pre-filled address are not shown.
3. Enter an **API key** only if the service needs one. It is never pre-filled.
4. Click **Test connection**.

What the test does:
- It first checks that the Base URL starts with `http://` or `https://` and that
  the Model ID isn't empty. If not, nothing is sent.
- It then asks the server for its model list (`GET /models`) and reports whether
  your model is on it. It does **not** ask the model a question, so no text is
  generated and no tokens are used. Some providers may still count it toward
  request limits.

What it does not do:
- Settings are kept **only in the dashboard's memory** for your current browser
  session. Nothing is saved to disk.
- Refreshing the page clears the visible form and starts a new session. The
  previous, disconnected session (including its settings and key) may stay in
  the dashboard's memory for a while before Streamlit removes it; by default,
  at least about 2 minutes (`server.disconnectedSessionTTL`).
- Stopping the dashboard program clears all of its in-memory settings.
- It does **not** change which model the benchmark uses. `python3 tasks/ai_runner.py`
  still reads the `BENCHMARK_*` environment variables.
- The key is never shown on the page or in error messages, and never saved.

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
| `tasks/ai_runner.py` | Runs the benchmark: asks each question through the adapter, grades the answer and records the results. |
| `tasks/model_adapter.py` | Talks to the model server: settings, API key, streaming request, and model check. |
| `tasks/grading.py` | Finds the model's final number and compares it with `expected_answer`. |
| `repeat_benchmark.py` | Runs the benchmark several times and measures per-task consistency. |
| `dashboard.py` | Streamlit dashboard for the results, plus a sidebar form to test a model connection. |
| `tests/test_benchmark.py` | Unit tests for grading, task validation, and error handling. |
| `tests/test_model_adapter.py` | Unit tests for model settings, requests, and keeping the API key out of results. |
| `tests/test_dashboard_model_config.py` | Tests for the dashboard's connection form, using fake server replies. |
| `results/` | Benchmark output. |

## Scope and limitations

- 16 mostly short arithmetic finance questions, one model, a few trials. A high
  score here is a smoke test, not a scientific measure of agent reliability.
- Only numeric answers can be graded.
- Difficulty labels are the author's judgement, not a calibrated scale.
