from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import prepare_backend


class BackendPackagingTests(unittest.TestCase):
    def test_sample_shapes_and_columns_match_recovered_tables(self):
        frames = prepare_backend.sample_frames()
        expected = {
            "DAILY_SIGNUPS": (365, ["date", "signup_rate"]),
            "REGION_REVENUE": (8, ["region", "revenue"]),
            "PRODUCT_REVENUE": (4380, ["product_line", "date", "revenue"]),
            "CORRELATED_METRICS": (200, ["marketing_spend", "signups", "revenue", "churn_rate"]),
            "BUDGET_VARIANCE": (6, ["department", "actual_minus_budget"]),
            "RANK_CHANGE": (6, ["product", "rank_last_year", "rank_this_year"]),
        }
        for name, position, types in prepare_backend.TABLES:
            with self.subTest(table=name):
                frame = frames[position]
                self.assertEqual((len(frame), list(frame.columns)), expected[name])
                self.assertEqual(len(types), len(frame.columns))
                pd.testing.assert_frame_equal(frame, prepare_backend.sample_frames()[position])

    def test_identifiers_reject_punctuation(self):
        self.assertEqual(prepare_backend.identifier("chartify_demo"), "CHARTIFY_DEMO")
        for value in ("", "has space", "has.dot", "has-dash", "1starts_with_digit"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                prepare_backend.identifier(value)

    def test_literal_serialization(self):
        self.assertEqual(prepare_backend.sql_literal("O'Brien"), "'O''Brien'")
        self.assertEqual(prepare_backend.sql_literal(pd.Timestamp("2025-01-01")), "'2025-01-01'")
        with self.assertRaises(ValueError):
            prepare_backend.sql_literal(float("nan"))

    def test_sql_is_explicit_and_non_replacing(self):
        sql = prepare_backend.sample_sql("demo", "app")
        self.assertEqual(sql.count("CREATE TABLE DEMO.APP."), 6)
        self.assertNotIn("CREATE OR REPLACE", sql)
        self.assertNotIn("DEVREL", sql)
        self.assertEqual(sql.count("INSERT INTO DEMO.APP.PRODUCT_REVENUE"), 9)
        self.assertIn("BEGIN TRANSACTION;", sql)
        self.assertTrue(sql.endswith("COMMIT;\n"))

    def test_submission_preserves_payload_and_rule_parsing(self):
        sql = prepare_backend.submission_sql("demo", "app")
        body = sql.split("$$")[1]
        namespace = {}
        exec(compile(body, "submission", "exec"), namespace)
        result = namespace["submit"](None, "bar_chart", "descending", "sequential", "", True, "Observed values", "2, 12, invalid, 15")
        self.assertEqual(result["rule_ids"], [2, 12, 15])
        self.assertEqual(result["chart_type"], "bar_chart")
        self.assertEqual(len(result), 7)

    def test_preparation_never_overwrites_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            args = ["prepare_backend.py", "--database", "demo", "--schema", "app", "--output", directory]
            destination = Path(directory) / "01_sample_tables.sql"
            destination.write_text("existing")
            with patch("sys.argv", args), self.assertRaises(FileExistsError):
                prepare_backend.main()
            self.assertEqual(destination.read_text(), "existing")


if __name__ == "__main__":
    unittest.main()
