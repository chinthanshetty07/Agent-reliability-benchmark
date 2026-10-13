
import json
import os
import sys
import threading
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS_FILE = ROOT / "results" / "results.json"
REPORT_FILE = (
    ROOT / "results" / "repeated_trials" / "reliability_report.json"
)

# The model adapter and run launcher live in tasks/. Add it to the import path once.
TASKS_DIR = str(ROOT / "tasks")
if TASKS_DIR not in sys.path:
    sys.path.insert(0, TASKS_DIR)
import model_adapter  # noqa: E402
import run_launcher  # noqa: E402

st.set_page_config(
    page_title="AI Agent Reliability Benchmark",
    page_icon="📊",
    layout="wide",
)


def run_connection_test(base_url, model_id, api_key):
    """
    Check the typed settings, then ask the server which models it has
    (GET /models). This never asks the model a question, so it generates
    no output and uses no tokens. Every message shown has the key removed.
    Deliberately NOT cached: Streamlit caches are shared by all users.
    """
    try:
        # Validates the address first; nothing is sent if it is invalid.
        config = model_adapter.build_model_config(base_url, model_id, api_key)
    except model_adapter.ModelConfigError as error:
        st.error(str(error))
        return

    try:
        problem = model_adapter.check_model_available(config)
    except Exception as error:   # show a short, redacted message instead of a traceback
        problem = f"Unexpected error while testing the connection: {type(error).__name__}: {error}"

    if problem:
        st.error(model_adapter.redact(problem, config))
    else:
        where = model_adapter.safe_base_url(config.base_url)
        st.success(f"Connected to {where}. Model '{config.model_id}' is available.")


def model_connection_panel():
    """Sidebar form for trying a model connection. Nothing here is saved to disk."""
    # Pre-fill from the same environment variables the runner uses.
    # The address is shown without any login details; the key is never pre-filled.
    default_url = model_adapter.safe_base_url(
        os.environ.get(model_adapter.ENV_BASE_URL, "").strip() or model_adapter.DEFAULT_BASE_URL
    )
    default_model = (
        os.environ.get(model_adapter.ENV_MODEL_ID, "").strip() or model_adapter.DEFAULT_MODEL_ID
    )

    with st.sidebar:
        st.header("Configure model")
        st.caption(
            "Settings stay in this browser session only and are never saved. "
            "**Test connection** checks them; **Run benchmark** starts a run with them. "
            "Runs started from the terminal (`python3 tasks/ai_runner.py`) still read "
            "the `BENCHMARK_*` environment variables."
        )
        with st.form("model_connection"):
            base_url = st.text_input("Base URL", value=default_url)
            model_id = st.text_input("Model ID", value=default_model)
            api_key = st.text_input(
                "API key (optional)",
                type="password",
                help="Only for services that need one. Never saved or displayed.",
            )
            submitted = st.form_submit_button("Test connection")
            run_clicked = st.form_submit_button("Run benchmark")

        if submitted:
            run_connection_test(base_url, model_id, api_key)
        if run_clicked:
            start_benchmark(base_url, model_id, api_key)


# ---------- Running a benchmark from the dashboard ----------

@st.cache_resource
def run_registry():
    """
    The active benchmark run, shared by every tab and kept across browser
    refreshes while this dashboard program runs. A dashboard restart clears it.
    Takes no arguments and never holds the API key or the model settings:
    only the process handle, run folder, model ID, cleaned address and times,
    plus the folder NAMES of old runs the user confirmed are not running.
    """
    return {"lock": threading.Lock(), "run": None, "confirmed_not_running": set()}


def start_benchmark(base_url, model_id, api_key):
    """Check the form settings, then start ONE run in a brand-new folder."""
    try:
        # Validates the address and model ID; nothing is started if they are invalid.
        config = model_adapter.build_model_config(base_url, model_id, api_key)
    except model_adapter.ModelConfigError as error:
        st.error(str(error))
        return

    registry = run_registry()
    with registry["lock"]:   # two tabs clicking at once still start only one run
        current = registry["run"]
        if current is not None and current["process"].poll() is None:
            st.error("A benchmark is already running. Stop it or wait for it to finish.")
            return
        # Runs this dashboard has no handle for (e.g. started before a restart)
        # may still be active; never start a second run alongside them.
        blocking = run_launcher.runs_blocking_a_new_start(
            exclude=[current["folder"]] if current is not None else [],
            confirmed_not_running=registry["confirmed_not_running"],
        )
        if blocking:
            names = ", ".join(f"`{folder.name}`" for folder, _ in blocking)
            st.error(
                f"New runs are blocked: {names} may still be running (started before the "
                "dashboard restarted). See **Benchmark run** below."
            )
            return
        try:
            folder = run_launcher.create_run_folder()
            # The key goes only into the run's environment, never into the registry.
            process = run_launcher.start_run(config, folder)
        except (run_launcher.LaunchError, OSError) as error:
            st.error(model_adapter.redact(f"Could not start the benchmark: {error}", config))
            return
        except Exception as error:
            # Anything else (e.g. a ValueError for an invalid setting): show a short,
            # redacted message instead of a traceback. No run is registered.
            st.error(model_adapter.redact(
                f"Could not start the benchmark ({type(error).__name__}): {error}", config
            ))
            return
        registry["run"] = {
            "process": process,
            "folder": folder,
            "model_id": config.model_id,
            "endpoint": model_adapter.safe_base_url(config.base_url),
            "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "stop_result": None,
        }


def stop_benchmark(run):
    """Ask the run to stop; it is ended forcibly only if it ignores the request."""
    with st.spinner("Stopping: asking the run to save its progress and exit..."):
        run["stop_result"] = run_launcher.stop_run(run["process"])


SHOW_MESSAGE = {"info": st.info, "success": st.success, "warning": st.warning, "error": st.error}


def show_run(run):
    """Draw the run's REAL state: process status plus what it actually saved."""
    described = run_launcher.describe_run(run["process"], run["folder"])
    level, message = run_launcher.outcome(described)
    st.caption(
        f"Model `{run['model_id']}` at {run['endpoint']} · started {run['started_at']} · "
        f"folder `results/runs/{run['folder'].name}`"
    )
    if described["total"]:
        done = described["completed"] or 0
        st.progress(min(done / described["total"], 1.0), text=f"{done} of {described['total']} tasks")
        st.caption(
            f"Correct {described['correct']} · Incorrect {described['incorrect']} · "
            f"Errors {described['errors']}"
        )
    SHOW_MESSAGE[level](message)
    if run["stop_result"] == "killed":
        st.caption("The run did not stop when asked, so it was ended forcibly.")
    if described["state"] in ("failed", "stopped_before_start", "stopped_unexpectedly"):
        tail = run_launcher.read_log_tail(run["folder"])
        if tail:
            st.code(tail, language=None)
    return described


@st.fragment(run_every="2s")
def live_run_section():
    """Redrawn on its own every 2 seconds while a run is active; the rest of the page is not."""
    run = run_registry()["run"]
    if run is None:
        return
    described = show_run(run)
    if described["state"] == "running":
        if st.button("Stop benchmark"):
            stop_benchmark(run)
            st.rerun()   # redraw everything with the real final state
    else:
        st.rerun()       # the run just ended: one full redraw turns this timer off


def benchmark_run_section():
    with st.sidebar:
        st.header("Benchmark run")
        run = run_registry()["run"]
        if run is not None and run["process"].poll() is None:
            live_run_section()
        elif run is not None:
            show_run(run)
        else:
            st.caption("No benchmark has been started from this dashboard yet.")

        show_runs_without_handle(run)


def show_runs_without_handle(run):
    """
    Runs with no final status that this dashboard has no process handle for
    (e.g. started before a restart): saved status "running", or no readable
    results yet. Their files are only read, never written. A process check
    (read-only) tells which may be active.
    """
    registry = run_registry()
    exclude = [run["folder"]] if run is not None else []
    for folder, saved_status in run_launcher.runs_without_final_status(exclude=exclude):
        name = folder.name
        liveness = run_launcher.check_process_alive(folder)
        has_results = saved_status == "running"
        if saved_status == "not_started":
            what_was_saved = "it hasn't saved any results yet (it may still be starting up or on its first task)"
        else:
            what_was_saved = "its results file can't be read"

        if liveness == "active":
            if has_results:
                st.warning(
                    f"`{name}` is still running: it was started before the dashboard restarted, "
                    "so it can't be stopped from here. New runs are blocked until it finishes."
                )
            else:
                st.warning(
                    f"`{name}` is still running, but {what_was_saved}. It was started before the "
                    "dashboard restarted, so it can't be stopped from here. New runs are blocked "
                    "until it finishes."
                )
        elif liveness == "ended":
            # Runs that ended without saving readable results are not listed here
            # (e.g. every failed start); run history covers them separately.
            if has_results:
                st.info(
                    f"`{name}` ended without saving a final status (its program is no longer "
                    "running). Results up to its last saved task are kept; its files are left untouched."
                )
        elif name in registry["confirmed_not_running"]:
            st.info(
                f"Status unknown for `{name}`. You confirmed it isn't running, so new runs are "
                "allowed (this confirmation is forgotten when the dashboard restarts)."
            )
        else:
            if has_results:
                st.warning(
                    f"Status unknown for `{name}`: its saved results still say \"running\", but this "
                    "dashboard has no live process for it and can't check whether it is still "
                    "running. It can't be stopped from here, and its files are left untouched. "
                    "New runs are blocked until you confirm it isn't running."
                )
            else:
                st.warning(
                    f"Status unknown for `{name}`: {what_was_saved}, and this dashboard has no live "
                    "process for it and can't check whether it is still running. It can't be "
                    "stopped from here, and its files are left untouched. New runs are blocked "
                    "until you confirm it isn't running."
                )
            if st.button("I've checked: it isn't running. Allow new runs", key=f"confirm-{name}"):
                registry["confirmed_not_running"].add(name)
                st.rerun()


# Drawn before the results below, so it still works when no results exist yet.
model_connection_panel()
benchmark_run_section()

st.title("AI Agent Reliability Benchmark")
st.caption("Finance domain · Local model evaluation")

if not RESULTS_FILE.exists():
    st.error("results/results.json was not found. Run the benchmark first.")
    st.stop()

with RESULTS_FILE.open() as file:
    latest = json.load(file)

summary = latest.get("summary", {})
results = latest.get("results", [])

if not results:
    st.error("No task results were found.")
    st.stop()

df = pd.DataFrame(results)

total = len(df)
correct = int(df["correct"].fillna(False).sum())
errors = int((df["status"] == "error").sum()) if "status" in df else 0
incorrect = total - correct - errors
accuracy = correct / total * 100 if total else 0
avg_time = (
    pd.to_numeric(df["response_time_seconds"], errors="coerce").mean()
    if "response_time_seconds" in df
    else 0
)

st.subheader("Latest benchmark run")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{accuracy:.1f}%")
c2.metric("Correct", f"{correct}/{total}")
c3.metric("Incorrect", incorrect)
c4.metric("Errors", errors)

c5, c6, c7 = st.columns(3)
c5.metric("Average response time", f"{avg_time:.2f} sec")
c6.metric("Model", latest.get("model", "Unknown"))
c7.metric("Run status", latest.get("run_status", "Unknown").title())

st.divider()
st.subheader("Task-by-task results")

display_columns = [
    column for column in [
        "task_id",
        "category",
        "difficulty",
        "expected_answer",
        "extracted_answer",
        "correct",
        "status",
        "response_time_seconds",
    ]
    if column in df.columns
]

st.dataframe(
    df[display_columns],
    width="stretch",
    hide_index=True,
)

st.subheader("Response time by task")

if "response_time_seconds" in df.columns:
    chart_df = df[["task_id", "response_time_seconds"]].copy()
    chart_df["response_time_seconds"] = pd.to_numeric(
        chart_df["response_time_seconds"], errors="coerce"
    )
    st.bar_chart(chart_df.set_index("task_id"))

st.divider()
st.subheader("Repeated-run reliability")

if REPORT_FILE.exists():
    with REPORT_FILE.open() as file:
        report = json.load(file)

    trials = report.get("trials", [])
    consistency = report.get("task_consistency", [])

    trial_rows = []
    for trial in trials:
        trial_summary = trial.get("summary", {})
        trial_rows.append({
            "Trial": f"Trial {trial.get('trial', '?')}",
            "Correct": trial_summary.get("correct", 0),
            "Incorrect": trial_summary.get("incorrect", 0),
            "Errors": trial_summary.get("errors", 0),
            "Accuracy (%)": round(
                trial_summary.get("correctness_rate", 0) * 100, 2
            ),
            "Average time (sec)": trial_summary.get(
                "avg_response_time_seconds", 0
            ),
        })

    if trial_rows:
        trial_df = pd.DataFrame(trial_rows)
        st.dataframe(trial_df, width="stretch", hide_index=True)
        st.bar_chart(trial_df.set_index("Trial")["Accuracy (%)"])

    st.caption(
        "Consistency means the percentage of recorded trials in which "
        "each task was answered correctly."
    )

    if consistency:
        consistency_df = pd.DataFrame(consistency)
        st.dataframe(
            consistency_df,
            width="stretch",
            hide_index=True,
        )

    st.metric("Completed trials", report.get("total_trials", len(trials)))
else:
    st.info(
        "Repeated-run report not found. Run repeat_benchmark.py "
        "to generate it."
    )

st.divider()
st.caption(
    "Research prototype. Results reflect this model, dataset, "
    "configuration, and test environment; they do not prove general AI reliability."
)