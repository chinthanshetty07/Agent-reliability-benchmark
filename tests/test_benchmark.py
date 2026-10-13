"""
Run from the project root with:
    python3 -m unittest discover -s tests -v
These tests do not call Ollama.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent / "tasks"))

import ai_runner  # noqa: E402
from grading import grade_numeric_answer, parse_number  # noqa: E402


class TestNumberParsing(unittest.TestCase):
    def test_currency_and_commas(self):
        self.assertEqual(parse_number("$3,000"), 3000)
        self.assertEqual(parse_number("3,000.00"), 3000)
        self.assertEqual(parse_number("7500"), 7500)

    def test_negative_forms(self):
        self.assertEqual(parse_number("-1200"), -1200)
        self.assertEqual(parse_number("-$1,200"), -1200)
        self.assertEqual(parse_number("$-1,200"), -1200)
        self.assertEqual(parse_number("(1,200)"), -1200)

    def test_loss_word_makes_negative(self):
        self.assertEqual(parse_number("The business has a loss of $1,200."), -1200)
        self.assertEqual(parse_number("Profit of $3,000 (no loss)."), 3000)

    def test_subtraction_is_not_a_negative_sign(self):
        self.assertEqual(parse_number("20,000 - 12,500 = 7,500"), 7500)

    def test_digits_inside_words_are_ignored(self):
        self.assertIsNone(parse_number("model qwen3"))


class TestGrading(unittest.TestCase):
    def grade(self, answer, expected):
        return grade_numeric_answer(answer, expected)

    def test_final_answer_line(self):
        reply = "Profit = $10,000 - $7,000 = $3,000\n\nFinal answer: 3000"
        result = self.grade(reply, "3000")
        self.assertTrue(result["correct"])
        self.assertEqual(result["extraction_method"], "final_answer_line")

    def test_dollar_formatted_answer_matches(self):
        self.assertTrue(self.grade("Final answer: $3,000", "3000")["correct"])

    def test_sign_matters(self):
        self.assertFalse(self.grade("Final answer: 1200", "-1200")["correct"])
        self.assertFalse(self.grade("Final answer: -1200", "1200")["correct"])
        self.assertTrue(self.grade("Final answer: -1200", "-1200")["correct"])

    def test_loss_sentence(self):
        reply = "8,000 - 9,200 = -1,200\n\n**Final Answer:** The business has a **loss of $1,200**."
        self.assertTrue(self.grade(reply, "-1200")["correct"])

    def test_profit_instead_of_loss_is_wrong(self):
        self.assertFalse(self.grade("Final answer: a profit of $1,200", "-1200")["correct"])

    def test_question_numbers_do_not_count(self):
        # Mentions 10,000 and 7,000 (from the question) but final answer is wrong.
        reply = "Revenue is $10,000 and expenses are $7,000.\nFinal answer: 17000"
        self.assertFalse(self.grade(reply, "3000")["correct"])
        self.assertFalse(self.grade(reply, "10000")["correct"])

    def test_expected_number_only_in_working_does_not_count(self):
        # 3000 appears in the working, but the final answer is 4000.
        reply = "First I thought it was 3000, but rechecking...\nFinal answer: 4000"
        self.assertFalse(self.grade(reply, "3000")["correct"])

    def test_last_final_answer_wins(self):
        reply = "Final answer: 4000\nWait, that's wrong.\nFinal answer: 3000"
        self.assertTrue(self.grade(reply, "3000")["correct"])

    def test_boxed(self):
        self.assertTrue(self.grade("The profit is \\boxed{7500}.", "7500")["correct"])

    def test_thinking_block_is_ignored(self):
        reply = "Maybe 3000? <think>Final answer: 3000</think>\nFinal answer: 2500"
        self.assertFalse(self.grade(reply, "3000")["correct"])

    def test_fallback_without_marker(self):
        result = self.grade("So the profit is $7,500.", "7500")
        self.assertTrue(result["correct"])
        self.assertEqual(result["extraction_method"], "last_line_with_number")

    def test_empty_and_no_number(self):
        self.assertFalse(self.grade("", "3000")["correct"])
        self.assertFalse(self.grade("I don't know.", "3000")["correct"])

    def test_non_numeric_expected_answer_is_not_marked_correct(self):
        result = self.grade("Final answer: 3000", "three thousand")
        self.assertFalse(result["correct"])
        self.assertEqual(result["extraction_method"], "not_graded")


class TestTaskValidation(unittest.TestCase):
    base = {
        "task_id": "t1",
        "difficulty": "easy",
        "question": "Q?",
        "expected_answer": "1",
    }

    def test_category_and_domain_both_work(self):
        self.assertEqual(ai_runner.check_task({**self.base, "category": "c"}), [])
        self.assertEqual(ai_runner.check_task({**self.base, "domain": "d"}), [])
        self.assertEqual(ai_runner.get_category({**self.base, "domain": "d"}), "d")

    def test_missing_fields_reported(self):
        problems = ai_runner.check_task({"task_id": "t1"})
        text = " ".join(problems)
        self.assertIn("question", text)
        self.assertIn("expected_answer", text)
        self.assertIn("category", text)

    def test_real_dataset_is_valid(self):
        tasks = ai_runner.load_tasks()
        self.assertGreaterEqual(len(tasks), 3)
        for task in tasks:
            self.assertEqual(ai_runner.check_task(task), [], task.get("task_id"))

    def test_real_dataset_task_ids_are_unique(self):
        ids = [task["task_id"] for task in ai_runner.load_tasks()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_real_dataset_answers_are_gradable(self):
        # A perfect reply must grade as correct, otherwise the task can never pass.
        for task in ai_runner.load_tasks():
            reply = f"Final answer: {task['expected_answer']}"
            result = grade_numeric_answer(reply, task["expected_answer"])
            self.assertTrue(result["correct"], task["task_id"])


class TestRunTask(unittest.TestCase):
    task = {
        "task_id": "finance_003",
        "category": "financial_calculation",
        "difficulty": "easy",
        "question": "Revenue 8000, expenses 9200. Profit or loss?",
        "expected_answer": "-1200",
    }

    def test_invalid_task_is_error_not_crash(self):
        result = ai_runner.run_task({"task_id": "broken"})
        self.assertEqual(result["status"], "error")
        self.assertFalse(result["correct"])
        self.assertIn("Invalid task", result["error"])

    def test_non_dict_task(self):
        result = ai_runner.run_task("not a task")
        self.assertEqual(result["status"], "error")

    def test_connection_error_is_recorded(self):
        # Port 9 has nothing listening, so the connection is refused.
        with mock.patch.object(ai_runner, "MODEL_CONFIG", ai_runner.model_adapter.ModelConfig(base_url="http://127.0.0.1:9/v1")):
            result = ai_runner.run_task(self.task)
        self.assertEqual(result["status"], "error")
        self.assertFalse(result["correct"])
        self.assertTrue(result["error"])

    def test_empty_answer_is_error(self):
        with mock.patch.object(ai_runner, "ask_model", return_value=("", "thinking...", "length")):
            result = ai_runner.run_task(self.task)
        self.assertEqual(result["status"], "error")
        self.assertIn("empty", result["error"])
        self.assertIn("cut off", result["warning"])

    def test_correct_and_incorrect_answers(self):
        with mock.patch.object(ai_runner, "ask_model", return_value=("Final answer: -1200", "", "stop")):
            self.assertEqual(ai_runner.run_task(self.task)["status"], "correct")
        with mock.patch.object(ai_runner, "ask_model", return_value=("Final answer: 1200", "", "stop")):
            self.assertEqual(ai_runner.run_task(self.task)["status"], "incorrect")

    def test_timeout_is_recorded(self):
        with mock.patch.object(ai_runner, "ask_model", side_effect=TimeoutError("too slow")):
            result = ai_runner.run_task(self.task)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"], "too slow")


class TestSaving(unittest.TestCase):
    def test_saved_json_is_valid_and_summary_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            result_file = Path(folder) / "results.json"
            with mock.patch.object(ai_runner, "RESULT_FILE", result_file):
                results = [
                    {"status": "correct", "response_time_seconds": 10.0},
                    {"status": "error", "response_time_seconds": 0.1},
                ]
                ai_runner.save_results({"model": "m"}, results, total_tasks=3)
            data = json.loads(result_file.read_text())
        self.assertEqual(data["summary"]["completed_tasks"], 2)
        self.assertEqual(data["summary"]["correct"], 1)
        self.assertEqual(data["summary"]["errors"], 1)
        self.assertEqual(data["summary"]["correctness_rate"], 0.5)
        # Errors are excluded from the response-time average.
        self.assertEqual(data["summary"]["avg_response_time_seconds"], 10.0)


sys.path.insert(0, str(Path(__file__).parent.parent))
import repeat_benchmark  # noqa: E402

# Stands in for tasks/ai_runner.py: writes a finished run where the answer is wrong,
# so we can tell its results apart from trial 1's.
FAKE_RUNNER = """
import json, pathlib
results = [{"task_id": "t1", "correct": False, "status": "incorrect"}]
data = {"run_status": "finished", "summary": {"correct": 0}, "results": results}
pathlib.Path("results/results.json").write_text(json.dumps(data))
"""


class TestRepeatBenchmark(unittest.TestCase):
    def test_trial_1_is_saved_before_results_are_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "results").mkdir()
            results_file = root / "results" / "results.json"
            trial_1 = {
                "run_status": "finished",
                "summary": {"correct": 1},
                "results": [{"task_id": "t1", "correct": True, "status": "correct"}],
            }
            results_file.write_text(json.dumps(trial_1))
            runner = root / "fake_runner.py"
            runner.write_text(FAKE_RUNNER)

            with mock.patch.multiple(
                repeat_benchmark,
                ROOT=root,
                RESULTS_FILE=results_file,
                OUTPUT_DIR=root / "results" / "repeated_trials",
                RUNNER=runner,
            ), mock.patch("builtins.print"):
                repeat_benchmark.main()

            out = root / "results" / "repeated_trials"
            # Trial 1 keeps its own (correct) answers even though results.json changed.
            self.assertEqual(json.loads((out / "trial_1.json").read_text()), trial_1)
            self.assertFalse(json.loads(results_file.read_text())["results"][0]["correct"])

            report = json.loads((out / "reliability_report.json").read_text())
            self.assertEqual(report["trials"][0]["source"], "results/repeated_trials/trial_1.json")
            self.assertEqual(report["task_consistency"][0]["outcomes"], [True, False, False])


if __name__ == "__main__":
    unittest.main()
