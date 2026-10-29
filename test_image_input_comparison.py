import unittest

from compare_image_inputs import summarize_runs


class ComparisonSummaryTests(unittest.TestCase):
    def test_failures_and_late_successes_stay_in_denominator(self):
        records = [
            {"strategy": "full_image_only", "wall_seconds": seconds, "result": {"status": status}}
            for seconds, status in [(4.0, "success"), (8.1, "success"), (0.2, "error")]
        ]
        result = summarize_runs(records, 8)["full_image_only"]
        self.assertEqual(result["attempts"], 3)
        self.assertEqual(result["successful_reports"], 2)
        self.assertEqual(result["successful_within_budget"], 1)
        self.assertAlmostEqual(result["within_budget_rate"], 1 / 3)
        self.assertEqual(result["max_success_wall_seconds"], 8.1)

    def test_dry_run_is_not_counted_as_successful_detection(self):
        result = summarize_runs([{
            "strategy": "full_plus_detail_crop", "wall_seconds": 0.2,
            "result": {"status": "dry_run_ok"},
        }], 8)["full_plus_detail_crop"]
        self.assertEqual(result["successful_reports"], 0)
        self.assertEqual(result["within_budget_rate"], 0)
        self.assertIsNone(result["median_success_wall_seconds"])


if __name__ == "__main__":
    unittest.main()
