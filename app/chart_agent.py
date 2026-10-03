"""Chart-recommendation agent grounded in data-visualization literature.

Given a plain-English question, calls the CHARTIFY_AGENT Cortex Agent (via
SNOWFLAKE.CORTEX.DATA_AGENT_RUN), which looks up real facts about the sample
data through its Cortex Analyst tool, self-checks its proposal with a
validate_recommendation tool, and returns its final answer through a
submit_chart_recommendation tool. Its input_schema describes tool arguments;
the app checks the returned payload and handles missing submissions explicitly. The agent's system
instructions encode chart-selection and color rules distilled from five
data-viz books plus a peer-reviewed figure-design paper (see the blog post
for citations); RULES_PROMPT below is kept as the source of truth for those
rules and for suggest_followups().
"""

import json
import os

import streamlit as st
from snowflake.connector.errors import DatabaseError

RULES_PROMPT = """You are a chart-recommendation engine. If the user explicitly
asks for a table, a raw data listing, or a list rather than a chart, respond
with chart_type table and leave rule_ids empty, regardless of the rules
below. Otherwise, apply these rules, drawn from
data visualization literature, in priority order:
1. Time-based trend (a date/time column vs a numeric column) -> line_chart.
2. Ranked comparison across more than 2 categories -> horizontal_bar_chart, sorted
   by value descending, unless the category has intrinsic order (e.g. days of
   week, size tiers), in which case use bar_chart in that natural order.
3. Part-to-whole with 3 or fewer categories -> pie_chart. Part-to-whole with
   4 or 5 categories -> stacked_bar_chart. Never recommend pie_chart for more
   than 3 categories, and never recommend either for more than 5 categories.
4. Distribution of a single numeric column across many rows -> histogram.
5. Relationship between two numeric variables -> scatter_chart.
6. Correlation or relationship across three or more numeric variables at once
   (not just two) -> heatmap, with a diverging color_scheme centered on zero.
7. A value's change relative to a baseline, target, or zero (variance,
   surplus/deficit, year-over-year change) -> diverging_bar_chart with a
   diverging color_scheme, not a plain bar_chart.
8. Ranking of categories that changes across exactly two time points (before
   vs after, this year vs last year) -> slope_chart. For more than two time
   points, prefer line_chart instead.
9. If the question implies comparing exactly one series against the rest, note
   in highlight which single category/series should be visually emphasized
   and the rest muted.
10. Never recommend a rainbow/qualitative palette for ordered or sequential data.
11. Flag color_blind_safe true only if the chosen encoding does not rely on
    red/green alone as the sole differentiator.
12. For any bar-shaped chart (bar_chart, horizontal_bar_chart, stacked_bar_chart,
    diverging_bar_chart), set sort_order to "descending" so the ranking or the
    extremes are visible at a glance, unless the category already has an
    intrinsic order (days of week, size tiers, time periods), in which case set
    sort_order to "natural" and do not reorder it. For chart types where sorting
    doesn't apply (line_chart, scatter_chart, heatmap, slope_chart, histogram,
    table), set sort_order to "natural".
13. Regardless of chart type, maximize the data-ink ratio: never suggest heavy
    gridlines, 3D effects, decorative borders, or a legend for information a
    direct label could show instead. This rule applies to every chart_type
    (except table), so include 13 in rule_ids whenever a chart is produced.
14. For any chart that encodes magnitude as bar length (bar_chart,
    horizontal_bar_chart, stacked_bar_chart, diverging_bar_chart, histogram),
    the value axis must start at zero. A truncated axis makes the visual size
    of a difference disproportionate to the real difference in the data
    (Tufte's Lie Factor); never suggest a non-zero baseline for these types.
15. Regardless of chart type (except table), the reason field is not optional
    decoration -- it must explain what the chart actually shows (the trend
    direction, the largest category, the outlier, the correlation) or point
    out what's noteworthy, not merely restate the chosen chart_type. A figure
    without a caption forces the viewer to guess. Include 15 in rule_ids
    whenever a chart is produced.
In rule_ids, list the numbers (1-15) of every rule above that was decisive in
reaching this recommendation, in the order they were applied.
Keep the reason field under 20 words.
"""

# Citation compilation: which authoritative source backs each numbered rule
# in RULES_PROMPT. Compiled in advance so the app can show the underlying
# best-practice citation for whichever rule(s) the agent says it applied,
# rather than asking the model to cite its own sources at inference time.
RULE_SOURCES = {
    1: {
        "principle": "A time-based trend belongs on a line chart, not bars per period.",
        "source": "Storytelling with Data — Cole Nussbaumer Knaflic (2015)",
    },
    2: {
        "principle": "Ranked comparisons across categories read fastest as a sorted bar chart.",
        "source": "Storytelling with Data — Cole Nussbaumer Knaflic (2015)",
    },
    3: {
        "principle": "Part-to-whole breakdowns fail past a handful of slices; pie charts "
                     "should be reserved for very few categories, per ColorWise's chart "
                     "selector guide.",
        "source": "ColorWise — Kate Strachnyi (2023)",
    },
    4: {
        "principle": "Distributions of a single numeric column are shown with a histogram, "
                      "not a bar per row.",
        "source": "Fundamentals of Data Visualization — Claus O. Wilke (2019)",
    },
    5: {
        "principle": "Relationships between two numeric variables belong on a scatter plot.",
        "source": "Fundamentals of Data Visualization — Claus O. Wilke (2019)",
    },
    6: {
        "principle": "Correlation across three or more variables at once calls for a "
                      "heatmap with a diverging scale centered on zero -- one of the three "
                      "fundamental colormap families (sequential, diverging, qualitative); "
                      "diverging is the right family here because deviation from zero, not "
                      "raw magnitude, is what matters.",
        "source": "Fundamentals of Data Visualization — Claus O. Wilke (2019); "
                  "ColorWise — Kate Strachnyi (2023); "
                  "Ten Simple Rules for Better Figures — Rougier, Droettboom & Bourne (2014)",
    },
    7: {
        "principle": "Values measured against a baseline or target should use a diverging "
                      "color encoding, not a single-hue bar -- the same sequential/diverging/"
                      "qualitative distinction used to pick any colormap: diverging is for "
                      "deviation from a reference point, sequential is for one-directional "
                      "magnitude.",
        "source": "ColorWise — Kate Strachnyi (2023); "
                  "Ten Simple Rules for Better Figures — Rougier, Droettboom & Bourne (2014)",
    },
    8: {
        "principle": "Comparing a ranking across two points in time is a distinct chart "
                      "form (slope chart), not a bar or line chart.",
        "source": "The Truthful Art — Alberto Cairo (2016)",
    },
    9: {
        "principle": "Give one series a distinct color and mute the rest so the "
                      "eye is drawn to it before conscious reading starts "
                      "(preattentive attributes).",
        "source": "Storytelling with Data — Cole Nussbaumer Knaflic (2015)",
    },
    10: {
        "principle": "A rainbow/qualitative palette on ordered or sequential data implies "
                      "categories that don't exist in the data. Rainbow (jet) colormaps are "
                      "singled out in the literature as actively harmful: they distort "
                      "perceived magnitude and hide real detail in parts of the range, as "
                      "documented in Borland & Taylor's 'Rainbow Color Map (Still) "
                      "Considered Harmful' (2007).",
        "source": "ColorWise — Kate Strachnyi (2023); "
                  "Ten Simple Rules for Better Figures — Rougier, Droettboom & Bourne (2014)",
    },
    11: {
        "principle": "Roughly 1 in 12 men have some form of color vision deficiency; "
                      "never rely on red/green alone as the only differentiator.",
        "source": "ColorWise — Kate Strachnyi (2023)",
    },
    12: {
        "principle": "Sort categorical data by value by default; alphabetical order is a "
                      "default, not a decision. Preserve order only when it's intrinsic "
                      "to the category (days of week, size tiers).",
        "source": "Storytelling with Data — Cole Nussbaumer Knaflic (2015)",
    },
    13: {
        "principle": "Every mark on a chart should represent data. Heavy gridlines, 3D "
                      "effects, and decorative borders are \"chartjunk\" — unnecessary or "
                      "confusing visual elements that add visual noise without adding "
                      "information, whether that's too many colors, too many labels, "
                      "gratuitously colored backgrounds, or gridlines that don't even align "
                      "with the data. They lower the data-ink ratio.",
        "source": "The Visual Display of Quantitative Information — Edward Tufte (1983); "
                  "Ten Simple Rules for Better Figures — Rougier, Droettboom & Bourne (2014)",
    },
    14: {
        "principle": "The visual size of an effect on a chart must be proportional to its "
                      "real size in the data (the \"Lie Factor\"); a truncated, non-zero "
                      "axis exaggerates comparisons even though every number is correct -- "
                      "and labeling the axis honestly doesn't fully cure it, because the "
                      "bars themselves remain the most visually salient thing on the chart.",
        "source": "The Visual Display of Quantitative Information — Edward Tufte (1983); "
                  "Ten Simple Rules for Better Figures — Rougier, Droettboom & Bourne (2014)",
    },
    15: {
        "principle": "A figure needs a caption explaining how to read it and pointing out "
                      "what's noteworthy -- don't expect the viewer to guess values or "
                      "relationships unaided.",
        "source": "Ten Simple Rules for Better Figures — Rougier, Droettboom & Bourne (2014)",
    },
}

RESPONSE_SCHEMA = {
    "type": "json",
    "schema": {
        "type": "object",
        "properties": {
            "chart_type": {
                "type": "string",
                "enum": [
                    "line_chart", "horizontal_bar_chart", "bar_chart",
                    "stacked_bar_chart", "histogram", "scatter_chart",
                    "pie_chart", "table", "heatmap", "diverging_bar_chart",
                    "slope_chart",
                ],
            },
            "sort_order": {"type": "string", "enum": ["ascending", "descending", "natural"]},
            "color_scheme": {"type": "string", "enum": ["sequential", "diverging", "qualitative"]},
            "highlight": {"type": "string"},
            "color_blind_safe": {"type": "boolean"},
            "reason": {"type": "string"},
            "rule_ids": {"type": "array", "items": {"type": "integer"}},
        },
        "required": [
            "chart_type", "sort_order", "color_scheme",
            "highlight", "color_blind_safe", "reason", "rule_ids",
        ],
    },
}


FOLLOWUP_SCHEMA = {
    "type": "json",
    "schema": {
        "type": "object",
        "properties": {
            "followups": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 3,
            },
        },
        "required": ["followups"],
    },
}

FOLLOWUP_PROMPT_TEMPLATE = """{rules}
A user asked: "{question}"
Columns available: {schema_summary}.
The chart-recommendation agent responded with:
- chart_type: {chart_type}
- sort_order: {sort_order}
- rule_ids applied: {rule_ids}
- reason: {reason}

Suggest exactly 3 short, natural follow-up questions (each under 12 words) that this user
might plausibly ask next about this same data, where each one would exercise a DIFFERENT
one of the numbered rules above than the ones already applied (e.g. sorting the other way,
a baseline/target comparison, a distribution, a correlation, or emphasizing one series).
Do not repeat the original question. Each suggestion must be answerable using only the
columns listed above.
"""


def suggest_followups(question: str, schema_summary: str, rec: dict) -> list:
    """Ask AI_COMPLETE for 3 follow-up questions grounded in the same rule set.

    Takes the just-returned recommendation (chart_type, sort_order, rule_ids,
    reason) as context so the suggestions probe rules that weren't already
    applied, rather than restating the same chart choice.
    """
    conn = st.connection("snowflake")
    prompt = FOLLOWUP_PROMPT_TEMPLATE.format(
        rules=RULES_PROMPT,
        question=question,
        schema_summary=schema_summary,
        chart_type=rec.get("chart_type"),
        sort_order=rec.get("sort_order"),
        rule_ids=rec.get("rule_ids"),
        reason=rec.get("reason"),
    )

    query = """
        SELECT AI_COMPLETE(
            model => 'claude-sonnet-5',
            prompt => ?,
            model_parameters => {'temperature': 0.3, 'max_tokens': 500},
            response_format => PARSE_JSON(?)
        )::string AS response
        """
    params = [prompt, json.dumps(FOLLOWUP_SCHEMA)]

    # ttl=0 disables st.connection's built-in query cache, which otherwise
    # keys on SQL text and can return a stale cached row for a different
    # question that happens to reuse this exact query string.
    try:
        rows = conn.query(query, params=params, ttl=0)
    except DatabaseError:
        conn.reset()
        conn = st.connection("snowflake")
        rows = conn.query(query, params=params, ttl=0)

    return json.loads(rows["RESPONSE"].iloc[0]).get("followups", [])


DEFAULT_AGENT_FQN = "DEVREL.CNANTASENAMAT_DEV.CHARTIFY_AGENT"


def get_agent_fqn() -> str:
    return os.environ.get("CHARTIFY_AGENT_FQN", "").strip() or DEFAULT_AGENT_FQN

# Tool names worth surfacing in the "Tool Calls" UI tab. chartify_analyst's own
# tool_use/tool_result is a large semantic-context dump; the actual generated
# SQL and query results come back under system_execute_sql, which is what's
# actually useful to show a user. submit_chart_recommendation is skipped here
# since its content is already the returned rec.
_DISPLAY_TOOL_NAMES = {"system_execute_sql": "chartify_analyst (SQL query)", "validate_recommendation": "validate_recommendation"}


def _run_agent(question: str) -> dict:
    """Call the Chartify Cortex Agent and return its parsed JSON response."""
    conn = st.connection("snowflake")
    request_body = json.dumps({"messages": [{"role": "user", "content": [{"type": "text", "text": question}]}]})

    query = """
        SELECT TRY_PARSE_JSON(
            SNOWFLAKE.CORTEX.DATA_AGENT_RUN(?, ?)
        )::string AS response
        """
    params = [get_agent_fqn(), request_body]

    try:
        rows = conn.query(query, params=params, ttl=0)
    except DatabaseError:
        # The underlying session can go stale after the app sits idle for a
        # while (expired OAuth token, closed connection). reset() forces
        # st.connection("snowflake") to reinitialize on next use.
        conn.reset()
        conn = st.connection("snowflake")
        rows = conn.query(query, params=params, ttl=0)

    try:
        return json.loads(rows["RESPONSE"].iloc[0])
    except (TypeError, json.JSONDecodeError, IndexError, KeyError) as error:
        raise RecommendationError("invalid_response") from error


class RecommendationError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(f"No usable chart recommendation returned ({code}).")


def _validate_recommendation(rec: object) -> dict:
    if not isinstance(rec, dict):
        raise RecommendationError("invalid_payload")
    schema = RESPONSE_SCHEMA["schema"]
    expected_types = {"string": str, "boolean": bool, "array": list}
    for field in schema["required"]:
        definition = schema["properties"][field]
        value = rec.get(field)
        if type(value) is not expected_types[definition["type"]]:
            raise RecommendationError("invalid_payload")
        if "enum" in definition and value not in definition["enum"]:
            raise RecommendationError("invalid_payload")
    if any(type(rule_id) is not int or rule_id not in RULE_SOURCES for rule_id in rec["rule_ids"]):
        raise RecommendationError("invalid_payload")
    return {field: rec[field] for field in schema["required"]}


def recommend_chart(question: str, schema_summary: str = "") -> dict:
    """Ask the Chartify Cortex Agent which chart to draw for a question.

    The agent grounds its answer in the real sample tables (via its
    chartify_analyst Cortex Analyst tool), self-checks the proposal with the
    validate_recommendation tool, and returns its final answer through the
    submit_chart_recommendation tool. The app validates its result against
    RESPONSE_SCHEMA; tool invocation and successful completion are not guaranteed. schema_summary is accepted for
    backwards compatibility with suggest_followups() but is no longer needed
    by the agent, which looks up real column facts itself.

    Returns a dict with the usual chart_type/sort_order/color_scheme/highlight/
    color_blind_safe/reason/rule_ids fields, plus a "tool_calls" list of the
    agent's real tool invocations (name, input, result) for display.
    """
    response = _run_agent(question)
    if not isinstance(response, dict) or not isinstance(response.get("content"), list):
        raise RecommendationError("invalid_response")
    content = response["content"]
    if any(not isinstance(item, dict) for item in content):
        raise RecommendationError("invalid_response")

    rec = None
    submission_results = 0
    tool_calls = []
    tool_use_by_id = {}

    for item in content:
        if item.get("type") != "tool_use":
            continue
        tool_use = item.get("tool_use", item)
        tool_use_by_id[tool_use.get("tool_use_id")] = tool_use

    for item in content:
        if item.get("type") != "tool_result":
            continue
        tool_result = item.get("tool_result", item)
        tool_use_id = tool_result.get("tool_use_id")
        name = tool_result.get("name") or tool_use_by_id.get(tool_use_id, {}).get("name")
        result_content = tool_result.get("content", [])
        if not isinstance(result_content, list):
            raise RecommendationError("invalid_response")
        json_blocks = [
            block["json"] for block in result_content
            if isinstance(block, dict) and isinstance(block.get("json"), dict)
        ]
        result_json = json_blocks[0] if json_blocks else None

        if name == "submit_chart_recommendation":
            submission_results += 1
            if submission_results > 1:
                raise RecommendationError("duplicate_submission")
            if tool_result.get("status") != "success":
                raise RecommendationError("submission_failed")
            payloads = [block for block in json_blocks if "result" in block]
            if len(payloads) != 1 or payloads[0].get("truncated"):
                raise RecommendationError("invalid_payload")
            raw = payloads[0]["result"]
            try:
                rec = _validate_recommendation(json.loads(raw) if isinstance(raw, str) else raw)
            except json.JSONDecodeError as error:
                raise RecommendationError("invalid_payload") from error

        display_name = _DISPLAY_TOOL_NAMES.get(name)
        if display_name:
            tool_calls.append(
                {
                    "name": display_name,
                    "input": tool_use_by_id.get(tool_use_id, {}).get("input"),
                    "result": result_json,
                }
            )

    if rec is None:
        submitted = any(
            tool.get("name") == "submit_chart_recommendation"
            for tool in tool_use_by_id.values()
        )
        raise RecommendationError("missing_result" if submitted else "missing_submission")

    rec["tool_calls"] = tool_calls
    return rec
