import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).parent / "app" / "app.py"
SPEC = importlib.util.spec_from_file_location("tested_chart_agent", APP_PATH.with_name("chart_agent.py"))
agent = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(agent)
RECOMMENDATION = {
    "chart_type": "horizontal_bar_chart", "sort_order": "descending",
    "color_scheme": "sequential", "highlight": "North America",
    "color_blind_safe": True, "reason": "North America leads revenue.",
    "rule_ids": [2, 12, 13, 14, 15],
}


def response(raw=None):
    return {"content": [
        {"type": "tool_use", "tool_use": {
            "name": "submit_chart_recommendation", "tool_use_id": "submit-1", "input": {},
        }},
        {"type": "tool_result", "tool_result": {
            "name": "submit_chart_recommendation", "tool_use_id": "submit-1",
            "status": "success", "content": [{"type": "json", "json": {
                "result": json.dumps(RECOMMENDATION) if raw is None else raw,
                "truncated": False,
            }}],
        }},
    ]}


class AgentConfigurationTests(unittest.TestCase):
    def test_default_preserves_existing_deployment(self):
        with patch.dict(agent.os.environ, {}, clear=True):
            self.assertEqual(agent.get_agent_fqn(), agent.DEFAULT_AGENT_FQN)

    def test_explicit_agent_override(self):
        with patch.dict(agent.os.environ, {"CHARTIFY_AGENT_FQN": " DEMO.APP.CHARTIFY_AGENT "}):
            self.assertEqual(agent.get_agent_fqn(), "DEMO.APP.CHARTIFY_AGENT")

    def test_blank_override_uses_default(self):
        with patch.dict(agent.os.environ, {"CHARTIFY_AGENT_FQN": "  "}):
            self.assertEqual(agent.get_agent_fqn(), agent.DEFAULT_AGENT_FQN)


class RecommendationParsingTests(unittest.TestCase):
    def parse(self, payload):
        with patch.object(agent, "_run_agent", return_value=payload) as run:
            try:
                return agent.recommend_chart("Show me revenue by region")
            finally:
                run.assert_called_once()

    def assert_failure(self, payload, code):
        with self.assertRaises(agent.RecommendationError) as failure:
            self.parse(payload)
        self.assertEqual(failure.exception.code, code)

    def test_nested_json_string(self):
        self.assertEqual(self.parse(response())["chart_type"], "horizontal_bar_chart")

    def test_dictionary_result(self):
        self.assertEqual(self.parse(response(copy.deepcopy(RECOMMENDATION)))["rule_ids"], [2, 12, 13, 14, 15])

    def test_scan_all_content_blocks_and_resolve_name(self):
        payload = response()
        result = payload["content"][1]["tool_result"]
        del result["name"]
        result["content"].insert(0, {"type": "text", "text": "Procedure completed"})
        self.assertEqual(self.parse(payload)["sort_order"], "descending")

    def test_missing_submission(self):
        self.assert_failure({"content": [{"type": "text", "text": "Use a bar chart"}]}, "missing_submission")

    def test_missing_result(self):
        self.assert_failure({"content": response()["content"][:1]}, "missing_result")

    def test_failed_result_is_not_accepted(self):
        payload = response()
        payload["content"][1]["tool_result"]["status"] = "error"
        self.assert_failure(payload, "submission_failed")

    def test_invalid_payloads(self):
        for raw in ("not json", "null", "[]", {}, {**RECOMMENDATION, "chart_type": "unknown"},
                    {**RECOMMENDATION, "color_blind_safe": "true"},
                    {**RECOMMENDATION, "rule_ids": [True]}, {**RECOMMENDATION, "rule_ids": [16]}):
            with self.subTest(raw=raw):
                self.assert_failure(response(raw), "invalid_payload")

    def test_truncated_payload(self):
        payload = response()
        payload["content"][1]["tool_result"]["content"][0]["json"]["truncated"] = True
        self.assert_failure(payload, "invalid_payload")

    def test_duplicate_submission(self):
        payload = response()
        payload["content"].append(copy.deepcopy(payload["content"][1]))
        self.assert_failure(payload, "duplicate_submission")

    def test_invalid_response(self):
        for payload in (None, [], {}, {"content": "invalid"}, {"content": [None]}):
            with self.subTest(payload=payload):
                self.assert_failure(payload, "invalid_response")


class RecommendationRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.mock_agent = ModuleType("chart_agent")
        self.mock_agent.RULE_SOURCES = {}
        self.mock_agent.recommend_chart = Mock(return_value=copy.deepcopy(RECOMMENDATION))
        self.mock_agent.suggest_followups = Mock(return_value=[])
        self.app = AppTest.from_file(str(APP_PATH), default_timeout=10)
        self.app.session_state["pending_question"] = "Show me revenue by region"

    def run_app(self):
        with patch.object(sys, "path", [str(APP_PATH.parent), *sys.path]), patch.dict(sys.modules, {"chart_agent": self.mock_agent}):
            self.app.run()
        self.assertFalse(self.app.exception)

    def test_failed_request_preserves_history_and_manual_retry_recovers(self):
        self.app.session_state["messages"] = [{
            "question": "Earlier question", "rec": copy.deepcopy(RECOMMENDATION), "followups": [],
        }]
        self.mock_agent.recommend_chart.side_effect = agent.RecommendationError("missing_submission")
        self.run_app()
        self.assertEqual(len(self.app.session_state["messages"]), 2)
        self.assertEqual(len(self.app.get("vega_lite_chart")), 1)
        self.mock_agent.suggest_followups.assert_not_called()
        self.mock_agent.recommend_chart.assert_called_once()
        self.mock_agent.recommend_chart.side_effect = None
        self.app.button(key="retry_1").click()
        self.run_app()
        self.assertEqual(len(self.app.session_state["messages"]), 2)
        self.assertEqual(len(self.app.get("vega_lite_chart")), 2)
        self.assertEqual(self.mock_agent.recommend_chart.call_count, 2)
        self.assertFalse(any("No usable chart recommendation" in warning.value for warning in self.app.warning))

    def test_followup_failure_does_not_discard_chart(self):
        self.mock_agent.suggest_followups.side_effect = RuntimeError("unavailable")
        self.run_app()
        self.assertEqual(len(self.app.get("vega_lite_chart")), 1)
        self.assertEqual(self.app.session_state["messages"][0]["followups"], [])
        self.assertTrue(self.app.session_state["messages"][0]["followups_unavailable"])

    def test_transport_error_has_recoverable_state(self):
        self.mock_agent.recommend_chart.side_effect = RuntimeError("private transport details")
        self.run_app()
        self.assertEqual(self.app.session_state["messages"][0]["error"], "request_failed")
        self.assertEqual(len(self.app.get("vega_lite_chart")), 0)
        self.assertEqual(self.app.button(key="retry_0").label, "Retry")


if __name__ == "__main__":
    unittest.main()
