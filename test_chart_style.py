import copy
import importlib.util
from pathlib import Path
import unittest

import pandas as pd
from test_chart_eval import evaluator, fixture

SPEC = importlib.util.spec_from_file_location("chart_style", Path(__file__).parent / "app" / "chart_style.py")
style_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(style_module)
style_chart = style_module.style_chart


class ChartStyleTests(unittest.TestCase):
    def test_highlight_uses_literal_predicate_and_preserves_inputs(self):
        rec, evidence = fixture()
        label = "B'); alert('not executable')"
        evidence["data"].loc[0, "category"] = label
        rec["highlight"] = label
        original_spec = copy.deepcopy(evidence["spec"])
        original_data = evidence["data"].copy(deep=True)
        styled = style_chart(evidence["data"], evidence["spec"], rec)
        self.assertEqual(styled["encoding"]["color"]["condition"]["test"], {"field": "category", "equal": label})
        self.assertEqual(evidence["spec"], original_spec)
        pd.testing.assert_frame_equal(original_data, evidence["data"])
        evidence["spec"] = styled
        checks = evaluator.evaluate_chart(rec, evidence)
        self.assertEqual(next(item["Status"] for item in checks if item["Check"] == "Requested highlight"), "Pass")

    def test_absent_highlight_is_not_invented(self):
        rec, evidence = fixture()
        rec["highlight"] = "Missing"
        styled = style_chart(evidence["data"], evidence["spec"], rec)
        self.assertNotIn("condition", styled["encoding"]["color"])
        evidence["spec"] = styled
        self.assertEqual(evaluator._highlight(rec, evidence, evidence["data"])[0], "Fail")

    def test_palette_conflicts_stay_visible(self):
        for requested, expected in (("sequential", "Pass"), ("qualitative", "Pass"), ("diverging", "Fail")):
            rec, evidence = fixture()
            rec["color_scheme"] = requested
            evidence["spec"] = style_chart(evidence["data"], evidence["spec"], rec)
            self.assertEqual(evaluator._palette(rec, evidence)[0], expected)

    def test_diverging_highlight_preserves_signed_scale(self):
        rec, evidence = fixture()
        rec.update(chart_type="diverging_bar_chart", highlight="B", color_scheme="diverging")
        evidence["data"]["value"] = [30, 10, -20]
        evidence["spec"]["encoding"]["color"] = {"field": "value", "type": "quantitative", "scale": {"scheme": "redblue", "domainMid": 0}}
        evidence["spec"] = style_chart(evidence["data"], evidence["spec"], rec)
        self.assertEqual(evidence["spec"]["encoding"]["color"]["scale"]["domainMid"], 0)
        self.assertNotIn("domainMin", evidence["spec"]["encoding"]["x"]["scale"])
        self.assertEqual(evaluator._highlight(rec, evidence, evidence["data"])[0], "Pass")
        self.assertEqual(evaluator._baseline(rec["chart_type"], evidence, evidence["data"])[0], "Pass")

    def test_heatmap_does_not_destroy_continuous_colors(self):
        data = pd.DataFrame({"metric_a": ["A"], "metric_b": ["A"], "correlation": [1.0]})
        spec = {"mark": "rect", "encoding": {"x": {"field": "metric_a", "type": "nominal"}, "y": {"field": "metric_b", "type": "nominal"}, "color": {"field": "correlation", "type": "quantitative", "scale": {"domain": [-1, 1], "scheme": "redblue"}}}}
        styled = style_chart(data, spec, {"chart_type": "heatmap", "highlight": "A", "color_scheme": "sequential"})
        self.assertEqual(styled["encoding"]["color"], spec["encoding"]["color"])
        self.assertNotIn("strokeWidth", styled["encoding"])

    def test_stack_order_and_highlight_preserve_grouping(self):
        rec, evidence = fixture()
        rec.update(chart_type="stacked_bar_chart", highlight="B", color_scheme="qualitative")
        evidence["data"]["group"] = "Total"
        evidence["spec"]["encoding"].update(y={"field": "group", "type": "nominal"}, color={"field": "category", "type": "nominal"})
        evidence["spec"]["encoding"]["x"]["stack"] = "normalize"
        for requested in ("ascending", "descending"):
            rec["sort_order"] = requested
            styled = style_chart(evidence["data"], evidence["spec"], rec)
            self.assertEqual(styled["encoding"]["color"]["field"], "category")
            self.assertEqual(evaluator._ordering(rec, {**evidence, "spec": styled}, evidence["data"])[0], "Pass")

    def test_weekday_request_conflict_preserves_calendar_order(self):
        data = pd.DataFrame({"day_of_week": evaluator.WEEKDAYS, "value": range(7)})
        spec = {"mark": "bar", "encoding": {"x": {"field": "day_of_week", "type": "nominal", "sort": evaluator.WEEKDAYS}, "y": {"field": "value", "type": "quantitative"}}}
        rec = {"chart_type": "bar_chart", "sort_order": "descending"}
        styled = style_chart(data, spec, rec)
        self.assertEqual(styled["encoding"]["x"]["sort"], evaluator.WEEKDAYS)
        self.assertEqual(evaluator._ordering(rec, {"spec": styled}, data)[0], "Fail")

    def test_slope_emphasis_retains_series_color_field(self):
        data = pd.DataFrame({"product": ["A", "B"], "period": ["Before", "After"], "rank": [1, 2]})
        spec = {"mark": "line", "encoding": {"x": {"field": "period", "type": "nominal"}, "y": {"field": "rank", "type": "quantitative"}, "color": {"field": "product", "type": "nominal"}}}
        rec = {"chart_type": "slope_chart", "highlight": "A", "color_scheme": "qualitative"}
        styled = style_chart(data, spec, rec)
        self.assertEqual(styled["encoding"]["color"]["field"], "product")
        self.assertEqual(evaluator._highlight(rec, {"renderer": "vega_lite", "spec": styled}, data)[0], "Pass")


if __name__ == "__main__":
    unittest.main()
