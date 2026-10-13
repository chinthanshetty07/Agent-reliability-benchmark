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

To save a run somewhere else, without touching `results/results.json`, add
`--results-dir`:

```bash
python3 tasks/ai_runner.py --results-dir results/runs/my-test
```

The run then writes `results.json` (and its `history/` backups) only inside that
folder, and creates the folder if needed. A relative path is taken from the folder
you run the command in. Without `--results-dir`, everything goes to `results/` as
before. The run stops with an error, before contacting the model or writing
anything, if the path is empty, points to an existing file, points to the default
`results/` folder itself (however it is written, e.g. `results`, `./results/` or
the full path), or the option is misspelled. Folders inside `results/`, such as
`results/runs/my-test`, are allowed.

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
- It does **not** change which model terminal runs use: `python3 tasks/ai_runner.py`
  still reads the `BENCHMARK_*` environment variables. (**Run benchmark**, below,
  does use the form's settings.)
- The key is never shown on the page or in error messages, and never saved.

### Run a benchmark from the dashboard

In the same sidebar form, click **Run benchmark**:

1. The Base URL and Model ID are checked first. If they're invalid, nothing starts.
   (There is no automatic connection test; if the model can't be reached, the run
   stops and its log explains why.)
2. A **new folder** is created for the run, e.g. `results/runs/20261013-142530-a1b2c3/`,
   and the benchmark runs there as a separate program. Existing results, including
   `results/results.json`, are never overwritten. `results/runs/` is ignored by git.
3. The **Benchmark run** section in the sidebar refreshes itself every 2 seconds with
   the progress, the correct/incorrect/error counts, and a plain-English outcome. The
   rest of the page is not redrawn.
4. **Stop benchmark** asks the run to save its progress and exit (like Ctrl+C). If it
   doesn't stop within 10 seconds, it is ended forcibly. The dashboard then shows the
   run's real saved state, for example "Stopped: 7 of 16 tasks saved", "Stopped before
   the first task", or "Ended without saving a final status".

Things to know:
- **One run at a time**, across all tabs.
- The outcome is based on the saved counts, not just on the run finishing. A run
  where every task failed is shown as an error, even though the program ended normally.
- Each run's files are `results.json`, `runner.log` (the program's printed output), and
  `launcher.json` (its process number, used only for the restart check above).
  The main results view above still shows `results/results.json` from terminal runs.
- **Refreshing the page or opening a second tab keeps control of the run**, because the
  dashboard keeps it in a registry shared by all tabs. **Restarting the dashboard loses
  that control**, so a run still in progress then can't be stopped from the dashboard.
  For any run whose saved results still say "running" but that has no live process
  handle, the dashboard checks (read-only, with `ps`) whether the process number in its
  `launcher.json` is still running **this** run:
  - **Still running:** shown as such, and **new runs are blocked** until it finishes.
  - **Ended:** shown as "ended without saving a final status"; new runs are allowed.
  - **Can't tell** (for example, no `launcher.json`): shown as **status unknown**, and new
    runs are blocked until you click *"I've checked: it isn't running. Allow new runs"*.
    That confirmation is kept in memory only and is asked again after a restart.

  These old runs' files are only read, never changed.
- **API key safety:** the key is passed to the run only through its environment
  variables. It is never put on the command line, in `runner.log`, in result files, or in
  the dashboard's shared registry, and never shown on the page. Environment variables
  are not a perfect security boundary: on macOS, other programs running as your user
  (and administrators) can read them. That is fine for a local tool on your own
  computer, not for a shared or hosted server.
- **The password box is not a vault.** While the key is typed into the API key box, it
  sits in that box in your browser (in the page's HTML), as with any web password field.
  It isn't displayed, but browser extensions or developer tools can read it. Keep the
  dashboard bound to this computer (`--server.address localhost`) and don't treat the
  password box as making the key inaccessible.

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
| `tests/test_runner_results_dir.py` | Tests for the runner's `--results-dir` option, using fake answers and temporary folders. |
| `tasks/run_launcher.py` | Starts a benchmark as a separate program in a new folder, reads its progress, and stops it safely. |
| `tests/test_run_launcher.py` | Tests for the launcher, using fake runners and a fake local model server. |
| `tests/test_dashboard_run.py` | Tests for the dashboard's Run and Stop buttons and progress section, using fake runners. |
| `results/` | Benchmark output. |

## Scope and limitations

- 16 mostly short arithmetic finance questions, one model, a few trials. A high
  score here is a smoke test, not a scientific measure of agent reliability.
- Only numeric answers can be graded.
- Difficulty labels are the author's judgement, not a calibrated scale.
