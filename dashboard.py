
import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS_FILE = ROOT / "results" / "results.json"
REPORT_FILE = (
    ROOT / "results" / "repeated_trials" / "reliability_report.json"
)

# The model adapter lives in tasks/. Add it to the import path once.
TASKS_DIR = str(ROOT / "tasks")
if TASKS_DIR not in sys.path:
    sys.path.insert(0, TASKS_DIR)
import model_adapter  # noqa: E402

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
            "They are used for this connection test only: the benchmark runner "
            "(`python3 tasks/ai_runner.py`) still reads the `BENCHMARK_*` environment variables."
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

        if submitted:
            run_connection_test(base_url, model_id, api_key)


# Drawn before the results below, so it still works when no results exist yet.
model_connection_panel()

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