
import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS_FILE = ROOT / "results" / "results.json"
REPORT_FILE = (
    ROOT / "results" / "repeated_trials" / "reliability_report.json"
)

st.set_page_config(
    page_title="AI Agent Reliability Benchmark",
    page_icon="📊",
    layout="wide",
)

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