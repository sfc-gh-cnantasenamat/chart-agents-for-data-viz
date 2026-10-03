import json
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import pyarrow as pa
from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).parent / "app" / "app.py"


class ChartOrderingTests(unittest.TestCase):
    def render(self, chart_type, sort_order=None):
        agent = ModuleType("chart_agent")
        agent.RULE_SOURCES = {}
        agent.recommend_chart = Mock()
        agent.suggest_followups = Mock()
        recommendation = {"chart_type": chart_type}
        if sort_order is not None:
            recommendation["sort_order"] = sort_order
        app = AppTest.from_file(str(APP_PATH), default_timeout=10)
        app.session_state["messages"] = [
            {"question": "Show me revenue by region", "rec": recommendation, "followups": []}
        ]
        with patch.object(sys, "path", [str(APP_PATH.parent), *sys.path]), patch.dict(sys.modules, {"chart_agent": agent}):
            app.run()
        self.assertFalse(app.exception)
        agent.recommend_chart.assert_not_called()
        agent.suggest_followups.assert_not_called()
        chart = app.get("vega_lite_chart")[0]
        spec = json.loads(chart.proto.spec)
        self.assertEqual(spec["mark"], "bar")
        if chart.proto.datasets:
            dataset = chart.proto.datasets[0].data.data
        else:
            dataset = chart.proto.data.data
        data = pa.ipc.open_stream(dataset).read_all().to_pandas()
        return spec, data

    def test_stacked_example_matches_top_five_product_denominator(self):
        from prepare_backend import sample_frames

        spec, data = self.render("stacked_bar_chart", "descending")
        expected = sample_frames()[2].groupby("product_line", as_index=False)["revenue"].sum().nlargest(5, "revenue")
        self.assertEqual(set(data["product_line"]), set(expected["product_line"]))
        self.assertAlmostEqual(data["revenue"].sum(), expected["revenue"].sum())
        self.assertEqual(spec["encoding"]["x"]["stack"], "normalize")
        source = APP_PATH.read_text()
        self.assertIn("Show each of the top five product lines as a separate share of those five product lines' combined revenue", source)
        self.assertNotIn("across North America, EMEA, APAC, LATAM, and ANZ as a share", source)

    def test_ranked_bars_default_to_descending(self):
        spec, data = self.render("horizontal_bar_chart")
        self.assertIn("sort", spec["encoding"]["y"])
        self.assertEqual(spec["encoding"]["y"]["sort"], data["region"].tolist())
        self.assertTrue(data["revenue"].is_monotonic_decreasing)
        self.assertNotEqual(data["region"].tolist(), sorted(data["region"]))

    def test_ranked_bars_descending(self):
        spec, data = self.render("horizontal_bar_chart", "descending")
        self.assertEqual(spec["encoding"]["y"]["sort"], data["region"].tolist())
        self.assertTrue(data["revenue"].is_monotonic_decreasing)

    def test_ranked_bars_ascending(self):
        spec, data = self.render("horizontal_bar_chart", "ascending")
        self.assertEqual(spec["encoding"]["y"]["sort"], data["region"].tolist())
        self.assertTrue(data["revenue"].is_monotonic_increasing)

    def test_ranked_bars_natural(self):
        spec, data = self.render("horizontal_bar_chart", "natural")
        self.assertEqual(spec["encoding"]["y"]["sort"], data["region"].tolist())
        self.assertEqual(
            data["region"].tolist(),
            ["North America", "EMEA", "APAC", "LATAM", "ANZ", "MEA", "Nordics", "Other"],
        )

    def test_diverging_bars_respect_requested_order(self):
        for sort_order in ("descending", "ascending", "natural"):
            with self.subTest(sort_order=sort_order):
                spec, data = self.render("diverging_bar_chart", sort_order)
                self.assertEqual(spec["encoding"]["y"]["sort"], data["department"].tolist())
                if sort_order == "descending":
                    self.assertTrue(data["actual_minus_budget"].is_monotonic_decreasing)
                elif sort_order == "ascending":
                    self.assertTrue(data["actual_minus_budget"].is_monotonic_increasing)
                else:
                    self.assertEqual(
                        data["department"].tolist(),
                        ["Sales", "Marketing", "Engineering", "Support", "Finance", "Ops"],
                    )

    def test_weekdays_keep_calendar_order(self):
        spec, data = self.render("bar_chart", "natural")
        self.assertEqual(spec["encoding"]["x"]["sort"], data["day_of_week"].tolist())
        self.assertEqual(
            data["day_of_week"].tolist(),
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        )

    def test_sorting_does_not_change_region_values(self):
        _, natural = self.render("horizontal_bar_chart", "natural")
        _, ranked = self.render("horizontal_bar_chart", "descending")
        pd.testing.assert_frame_equal(
            natural.sort_values("region").reset_index(drop=True),
            ranked.sort_values("region").reset_index(drop=True),
        )


if __name__ == "__main__":
    unittest.main()
