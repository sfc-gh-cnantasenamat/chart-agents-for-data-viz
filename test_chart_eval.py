import copy
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import pandas.core.arrays.arrow.extension_types
import pyarrow as pa
import streamlit as st
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).parent / "app" / "app.py"
SPEC = importlib.util.spec_from_file_location("chart_eval", APP_PATH.with_name("chart_eval.py"))
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


def fixture():
    rec = {"chart_type": "horizontal_bar_chart", "sort_order": "descending",
           "color_scheme": "sequential", "highlight": "", "reason": "B leads A.", "rule_ids": []}
    evidence = {"renderer": "vega_lite", "source": "fixture", "natural_order": ["B", "A", "C"],
                "data": pd.DataFrame({"category": ["B", "A", "C"], "value": [30, 20, 10]}),
                "spec": {"mark": "bar", "encoding": {
                    "x": {"field": "value", "type": "quantitative", "scale": {"zero": True}},
                    "y": {"field": "category", "type": "nominal", "sort": ["B", "A", "C"]},
                }}}
    return rec, evidence


class EvaluationTests(unittest.TestCase):
    def status(self, rec, evidence, check):
        results = evaluator.evaluate_chart(rec, evidence)
        return next(item["Status"] for item in results if item["Check"] == check)

    def test_order_uses_axis_not_row_order(self):
        rec, evidence = fixture()
        self.assertEqual(self.status(rec, evidence, "Rendered category order"), "Pass")
        evidence["spec"]["encoding"]["y"]["sort"] = ["A", "B", "C"]
        self.assertEqual(self.status(rec, evidence, "Rendered category order"), "Fail")

    def test_reversed_order_and_ascending(self):
        rec, evidence = fixture()
        evidence["spec"]["encoding"]["y"]["sort"] = ["C", "A", "B"]
        self.assertEqual(self.status(rec, evidence, "Rendered category order"), "Fail")
        rec["sort_order"] = "ascending"
        self.assertEqual(self.status(rec, evidence, "Rendered category order"), "Pass")

    def test_natural_and_unsupported_order(self):
        rec, evidence = fixture()
        rec["sort_order"] = "natural"
        self.assertEqual(self.status(rec, evidence, "Rendered category order"), "Pass")
        evidence["spec"]["encoding"]["y"]["sort"] = "ascending"
        self.assertEqual(self.status(rec, evidence, "Rendered category order"), "Not Checked")

    def test_weekday_order(self):
        rec, evidence = fixture()
        rec.update(chart_type="bar_chart", sort_order="natural")
        evidence["data"] = pd.DataFrame({"day_of_week": evaluator.WEEKDAYS, "value": range(7)})
        evidence["spec"]["encoding"] = {
            "x": {"field": "day_of_week", "type": "nominal", "sort": evaluator.WEEKDAYS},
            "y": {"field": "value", "type": "quantitative"},
        }
        self.assertEqual(self.status(rec, evidence, "Rendered category order"), "Pass")
        evidence["spec"]["encoding"]["x"]["sort"] = sorted(evaluator.WEEKDAYS)
        self.assertEqual(self.status(rec, evidence, "Rendered category order"), "Fail")

    def test_zero_and_clipped_domains(self):
        rec, evidence = fixture()
        for scale, expected in (({"zero": True}, "Pass"), ({"zero": False}, "Fail"),
                                ({"domain": [5, 40]}, "Fail"), ({"domain": [0, 25]}, "Fail"),
                                ({"domain": [0, 40]}, "Pass"), ({}, "Not Checked"),
                                ({"type": "log"}, "Fail")):
            with self.subTest(scale=scale):
                evidence["spec"]["encoding"]["x"]["scale"] = scale
                self.assertEqual(self.status(rec, evidence, "Magnitude axis zero baseline"), expected)

    def test_signed_domain_includes_zero(self):
        rec, evidence = fixture()
        rec["chart_type"] = "diverging_bar_chart"
        evidence["data"]["value"] = [30, 10, -20]
        evidence["spec"]["encoding"]["x"]["scale"] = {"domain": [-30, 40]}
        self.assertEqual(self.status(rec, evidence, "Magnitude axis zero baseline"), "Pass")

    def test_part_to_whole(self):
        rec, evidence = fixture()
        rec["chart_type"] = "pie_chart"
        evidence["spec"] = {"mark": "arc", "encoding": {
            "theta": {"field": "value", "type": "quantitative"},
            "color": {"field": "category", "type": "nominal"},
        }}
        self.assertEqual(self.status(rec, evidence, "Part-to-whole structure"), "Pass")
        evidence["data"].loc[3] = ["D", 5]
        self.assertEqual(self.status(rec, evidence, "Part-to-whole structure"), "Fail")
        rec["chart_type"] = "stacked_bar_chart"
        evidence["data"]["group"] = "Total"
        evidence["spec"] = {"mark": "bar", "encoding": {
            "x": {"field": "value", "type": "quantitative", "stack": "normalize"},
            "y": {"field": "group", "type": "nominal"},
            "color": {"field": "category", "type": "nominal"},
        }}
        self.assertEqual(self.status(rec, evidence, "Part-to-whole structure"), "Pass")
        evidence["spec"]["encoding"]["x"].pop("stack")
        self.assertEqual(self.status(rec, evidence, "Part-to-whole structure"), "Fail")

    def test_diverging_midpoint(self):
        rec, evidence = fixture()
        rec.update(chart_type="diverging_bar_chart", color_scheme="diverging")
        evidence["spec"]["encoding"]["color"] = {
            "field": "value", "type": "quantitative", "scale": {"scheme": "redblue", "domainMid": 0}}
        self.assertEqual(self.status(rec, evidence, "Variance color midpoint"), "Pass")
        evidence["spec"]["encoding"]["color"]["scale"]["domainMid"] = 5
        self.assertEqual(self.status(rec, evidence, "Variance color midpoint"), "Fail")

    def test_highlight_absent_ignored_and_explicit(self):
        rec, evidence = fixture()
        rec["highlight"] = "B"
        self.assertEqual(self.status(rec, evidence, "Requested highlight"), "Fail")
        evidence["spec"]["encoding"]["color"] = {
            "condition": {"test": {"field": "category", "equal": "B"}, "value": "blue"}, "value": "gray"}
        self.assertEqual(self.status(rec, evidence, "Requested highlight"), "Pass")
        rec["highlight"] = "Z"
        self.assertEqual(self.status(rec, evidence, "Requested highlight"), "Fail")

    def test_palette_families(self):
        rec, evidence = fixture()
        for scheme, expected in (("blues", "Pass"), ("category10", "Fail"), ("unknown", "Not Checked")):
            evidence["spec"]["encoding"]["color"] = {"field": "value", "type": "quantitative", "scale": {"scheme": scheme}}
            self.assertEqual(self.status(rec, evidence, "Palette family agreement"), expected)

    def test_caption_and_accessibility_are_not_certified(self):
        rec, evidence = fixture()
        rec["color_blind_safe"] = True
        self.assertEqual(self.status(rec, evidence, "Caption presence"), "Pass")
        self.assertEqual(self.status(rec, evidence, "Caption accuracy"), "Not Checked")
        self.assertEqual(self.status(rec, evidence, "Visual accessibility"), "Not Checked")
        rec["reason"] = " "
        self.assertEqual(self.status(rec, evidence, "Caption presence"), "Fail")

    def test_native_evidence_limits(self):
        rec, evidence = fixture()
        rec.update(chart_type="histogram", highlight="B")
        evidence = {"renderer": "bar_chart", "source": "bins", "args": {},
                    "data": pd.Series([2, 3], index=pd.IntervalIndex.from_breaks([0, 1, 2]))}
        self.assertEqual(self.status(rec, evidence, "Distribution bin structure"), "Pass")
        for check in ("Magnitude axis zero baseline", "Palette family agreement", "Requested highlight"):
            self.assertEqual(self.status(rec, evidence, check), "Not Checked")

    def test_malformed_evidence_is_not_a_pass(self):
        rec, evidence = fixture()
        for broken in ({}, {"data": []}, {**evidence, "spec": None},
                       {**evidence, "spec": {"layer": []}},
                       {**evidence, "spec": {"encoding": {"x": {"field": "missing"}}}}):
            results = evaluator.evaluate_chart(rec, broken)
            self.assertTrue(any(item["Check"] == "Evidence coverage" and item["Status"] == "Not Checked" for item in results))

    def test_checks_do_not_require_agent_citation(self):
        rec, evidence = fixture()
        results = evaluator.evaluate_chart(rec, evidence)
        ordering = next(item for item in results if item["Rule"] == 12)
        self.assertEqual(ordering["Status"], "Pass")
        self.assertFalse(ordering["Agent cited"])
        self.assertEqual({item["Rule"] for item in results if item["Rule"]}, set(range(1, 16)))


    def test_normalized_and_absolute_stack_domains(self):
        rec, evidence = fixture()
        rec["chart_type"] = "stacked_bar_chart"
        evidence["data"]["group"] = "Total"
        encoding = evidence["spec"]["encoding"]
        encoding["y"] = {"field": "group", "type": "nominal"}
        encoding["x"].update(stack="normalize", scale={"domain": [0, 1]})
        self.assertEqual(self.status(rec, evidence, "Magnitude axis zero baseline"), "Pass")
        encoding["x"].update(stack="zero", scale={"domain": [0, 35]})
        self.assertEqual(self.status(rec, evidence, "Magnitude axis zero baseline"), "Fail")
        encoding["x"]["scale"]["domain"] = [0, 60]
        self.assertEqual(self.status(rec, evidence, "Magnitude axis zero baseline"), "Pass")
        evidence["data"]["value"] = [30, -20, -10]
        encoding["x"]["scale"]["domain"] = [-30, 30]
        self.assertEqual(self.status(rec, evidence, "Magnitude axis zero baseline"), "Pass")

    def test_color_aliases_and_unsupported_colors(self):
        rec, evidence = fixture()
        rec["highlight"] = "B"
        for base, expected in (("#0000ff", "Fail"), ("rgb(0, 0, 255)", "Fail"), ("gray", "Pass"), ("var(--theme)", "Not Checked")):
            evidence["spec"]["encoding"]["color"] = {
                "condition": {"test": {"field": "category", "equal": "B"}, "value": "blue"}, "value": base}
            self.assertEqual(self.status(rec, evidence, "Requested highlight"), expected)

    def test_unsupported_color_does_not_suppress_order_or_baseline(self):
        rec, evidence = fixture()
        rec["highlight"] = "B"
        evidence["spec"]["encoding"]["color"] = ["invalid"]
        self.assertEqual(self.status(rec, evidence, "Requested highlight"), "Not Checked")
        self.assertEqual(self.status(rec, evidence, "Rendered category order"), "Pass")
        self.assertEqual(self.status(rec, evidence, "Magnitude axis zero baseline"), "Pass")

    def test_source_reconciliation_changes_drops_duplicates(self):
        rec, evidence = fixture()
        evidence["source_contract"] = {"operation": "identity", "keys": ["category"], "data": evidence["data"].copy()}
        self.assertEqual(self.status(rec, evidence, "Source reconciliation"), "Pass")
        for changed in (evidence["data"].iloc[:2], evidence["data"].assign(value=[30, 21, 10]),
                        pd.concat([evidence["data"], evidence["data"].iloc[:1]])):
            broken = {**evidence, "data": changed}
            self.assertEqual(self.status(rec, broken, "Source reconciliation"), "Fail")

    def test_aggregation_reconciliation(self):
        rec, evidence = fixture()
        source = pd.DataFrame({"category": ["B", "B", "A", "C"], "value": [10, 20, 20, 10]})
        evidence["source_contract"] = {"operation": "sum_top", "keys": ["category"], "data": source, "value": "value", "count": 3}
        self.assertEqual(self.status(rec, evidence, "Source reconciliation"), "Pass")
        evidence["data"]["value"] = [15, 20, 10]
        self.assertEqual(self.status(rec, evidence, "Source reconciliation"), "Fail")

    def test_time_order_duplicates_and_gaps(self):
        rec, _ = fixture()
        rec["chart_type"] = "line_chart"
        source = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=4), "rate": [0.1, 0.2, 0.3, 0.4]})
        evidence = {"renderer": "line_chart", "data": source.set_index("date"),
                    "source_contract": {"data": source, "operation": "indexed", "keys": ["date"], "value": "rate", "frequency": "D"}}
        self.assertEqual(self.status(rec, evidence, "Time-series coverage"), "Pass")
        self.assertEqual(self.status(rec, evidence, "Time-series order and uniqueness"), "Pass")
        for index in ([1, 0, 2, 3], [0, 0, 2, 3]):
            broken = {**evidence, "data": evidence["data"].iloc[index]}
            self.assertEqual(self.status(rec, broken, "Time-series order and uniqueness"), "Fail")
        evidence["data"] = evidence["data"].iloc[[0, 2, 3]]
        self.assertEqual(self.status(rec, evidence, "Time-series coverage"), "Fail")

    def test_fraction_units(self):
        rec, evidence = fixture()
        evidence["data"]["value"] = [0.3, 0.2, 0.1]
        evidence["units"] = {"value": "fraction"}
        for formatting, expected in ((".0%", "Pass"), (".2f", "Fail"), (None, "Not Checked")):
            evidence["spec"]["encoding"]["x"]["axis"] = {"format": formatting}
            self.assertEqual(self.status(rec, evidence, "Unit formatting"), expected)

    def test_missing_category_labels(self):
        rec, evidence = fixture()
        self.assertEqual(self.status(rec, evidence, "Categorical label completeness"), "Pass")
        for label in (None, " "):
            evidence["data"].loc[0, "category"] = label
            self.assertEqual(self.status(rec, evidence, "Categorical label completeness"), "Fail")

    def test_histogram_source_recount(self):
        rec, _ = fixture()
        rec["chart_type"] = "histogram"
        evidence = {"renderer": "bar_chart", "data": pd.Series([2, 1], index=pd.IntervalIndex.from_breaks([0, 1, 2])),
                    "source_contract": {"data": pd.DataFrame({"value": [0.2, 0.8, 1.5]}), "operation": "histogram", "value": "value"}}
        self.assertEqual(self.status(rec, evidence, "Source reconciliation"), "Pass")
        evidence["data"].iloc[0] = 1
        self.assertEqual(self.status(rec, evidence, "Source reconciliation"), "Fail")


class EvaluationIntegrationTests(unittest.TestCase):
    def test_all_renderers_evaluate_emitted_inputs(self):
        chart_types = ["line_chart", "horizontal_bar_chart", "bar_chart", "stacked_bar_chart",
                       "histogram", "scatter_chart", "pie_chart", "table", "heatmap",
                       "diverging_bar_chart", "slope_chart"]
        for chart_type in chart_types:
            with self.subTest(chart_type=chart_type):
                mock_agent = ModuleType("chart_agent")
                mock_agent.RULE_SOURCES = {}
                mock_agent.recommend_chart = Mock()
                mock_agent.suggest_followups = Mock()
                rec = {"chart_type": chart_type, "sort_order": "natural" if chart_type == "bar_chart" else "descending",
                       "reason": "A sample caption.", "highlight": "North America", "rule_ids": [], "color_scheme": "sequential"}
                app = AppTest.from_file(str(APP_PATH), default_timeout=10)
                app.session_state["messages"] = [{"question": "Show signup data", "rec": rec, "followups": []}]
                with ExitStack() as stack:
                    stack.enter_context(patch.object(sys, "path", [str(APP_PATH.parent), *sys.path]))
                    stack.enter_context(patch.dict(sys.modules, {"chart_agent": mock_agent, "chart_eval": evaluator}))
                    evaluate = stack.enter_context(patch.object(evaluator, "evaluate_chart", wraps=evaluator.evaluate_chart))
                    native_calls = {name: stack.enter_context(patch.object(st, name, wraps=getattr(st, name)))
                                    for name in ("line_chart", "scatter_chart", "bar_chart", "dataframe")}
                    app.run()
                self.assertFalse(app.exception)
                self.assertEqual(evaluate.call_count, 1)
                evidence = evaluate.call_args.args[1]
                self.assertEqual(len(app.session_state["messages"]), 1)
                self.assertIn("Evaluation", [tab.label for tab in app.tabs])
                checks = app.dataframe[-1].value
                self.assertTrue((checks["Status"] == "Not Checked").any())
                self.assertFalse((checks["Check"] == "Evidence coverage").any())
                source_status = checks.loc[checks["Check"] == "Source reconciliation", "Status"].iloc[0]
                self.assertEqual(source_status, "Pass", f"{chart_type}: source reconciliation")
                if evidence["renderer"] == "vega_lite":
                    chart = app.get("vega_lite_chart")[0]
                    spec = json.loads(chart.proto.spec)
                    self.assertEqual(spec["encoding"], evidence["spec"]["encoding"])
                    self.assertEqual(spec["mark"], evidence["spec"]["mark"])
                    encoded = chart.proto.datasets[0].data.data if chart.proto.datasets else chart.proto.data.data
                    plotted = pa.ipc.open_stream(encoded).read_all().to_pandas()
                    pd.testing.assert_frame_equal(plotted, evidence["data"])
                else:
                    call = native_calls[evidence["renderer"]].call_args_list[0]
                    self.assertIs(call.args[0], evidence["data"])
                    self.assertEqual(call.kwargs, evidence["args"])
                if chart_type == "horizontal_bar_chart":
                    self.assertEqual(checks.loc[checks["Check"] == "Requested highlight", "Status"].iloc[0], "Pass")
                mock_agent.recommend_chart.assert_not_called()
                with patch.object(sys, "path", [str(APP_PATH.parent), *sys.path]), patch.dict(sys.modules, {"chart_agent": mock_agent, "chart_eval": evaluator}), patch.object(evaluator, "evaluate_chart", wraps=evaluator.evaluate_chart) as rerun_evaluate:
                    app.run()
                self.assertFalse(app.exception)
                self.assertEqual(rerun_evaluate.call_count, 1)
                pd.testing.assert_frame_equal(app.dataframe[-1].value, checks)


if __name__ == "__main__":
    unittest.main()
