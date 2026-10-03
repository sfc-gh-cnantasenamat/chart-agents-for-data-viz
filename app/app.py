import numpy as np
import logging

import pandas as pd
import streamlit as st

from chart_agent import RULE_SOURCES, recommend_chart, suggest_followups
from chart_eval import evaluate_chart
from chart_style import style_chart

st.set_page_config(page_title="Chartify", page_icon=":material/bar_chart:", layout="wide")

st.title(":material/bar_chart: Chartify")
st.caption("Ask a question, get the chart data storytellers would actually recommend.")


@st.cache_data
def load_sample_data():
    dates = pd.date_range("2025-01-01", periods=365, freq="D")
    regions = ["North America", "EMEA", "APAC", "LATAM", "ANZ", "MEA", "Nordics", "Other"]
    product_lines = [f"Product {chr(65 + i)}" for i in range(12)]
    departments = ["Sales", "Marketing", "Engineering", "Support", "Finance", "Ops"]

    rng = np.random.default_rng(42)

    daily = pd.DataFrame(
        {
            "date": dates,
            "signup_rate": rng.normal(0.12, 0.015, len(dates)).clip(min=0),
        }
    )
    by_region = pd.DataFrame(
        {
            "region": regions,
            "revenue": rng.lognormal(9.5, 0.4, len(regions)),
        }
    )
    by_product = pd.DataFrame(
        {
            "product_line": np.repeat(product_lines, len(dates)),
            "date": np.tile(dates, len(product_lines)),
            "revenue": rng.lognormal(8.5, 0.5, len(dates) * len(product_lines)),
        }
    )

    # Correlated metrics for the heatmap scenario: marketing spend drives
    # signups and revenue up, churn down, with noise layered on top.
    marketing_spend = rng.normal(50_000, 12_000, 200)
    signups = marketing_spend * 0.02 + rng.normal(0, 200, 200)
    revenue = signups * 45 + rng.normal(0, 5_000, 200)
    churn_rate = 0.08 - marketing_spend * 0.0000008 + rng.normal(0, 0.01, 200)
    metrics = pd.DataFrame(
        {
            "marketing_spend": marketing_spend,
            "signups": signups,
            "revenue": revenue,
            "churn_rate": churn_rate,
        }
    )
    corr = metrics.corr()

    # Budget variance for the diverging-bar scenario.
    budget_variance = pd.DataFrame(
        {
            "department": departments,
            "actual_minus_budget": rng.normal(0, 40_000, len(departments)),
        }
    )

    # Rank changes for the slope-chart scenario.
    products = [f"Product {chr(65 + i)}" for i in range(6)]
    rank_last_year = rng.permutation(np.arange(1, len(products) + 1))
    rank_this_year = rng.permutation(np.arange(1, len(products) + 1))
    rank_change = pd.DataFrame(
        {
            "product": products,
            "rank_last_year": rank_last_year,
            "rank_this_year": rank_this_year,
        }
    )

    return daily, by_region, by_product, corr, budget_variance, rank_change, metrics


daily, by_region, by_product, corr, budget_variance, rank_change, metrics = load_sample_data()

SCHEMA_SUMMARY = (
    "REGION (string, 8 distinct values), REVENUE (number), "
    "MONTH/DATE (date), SIGNUP_RATE (number), "
    "PRODUCT_LINE (string, 12 distinct values), "
    "MARKETING_SPEND (number), SIGNUPS (number), CHURN_RATE (number), "
    "DEPARTMENT (string, 6 distinct values), ACTUAL_MINUS_BUDGET (number, can be negative), "
    "PRODUCT (string, 6 distinct values), RANK_LAST_YEAR (number), RANK_THIS_YEAR (number)"
)

QUICK_EXAMPLES = [
    "Show me revenue by region",
    "How has signup rate changed this year?",
    "How did product rankings change from last year to this year?",
]

# One question per remaining chart type, presented via a selectbox rather
# than buttons so all 11 chart types have coverage without crowding the page.
SELECT_EXAMPLES = [
    "How correlated are marketing spend, signups, revenue, and churn rate?",
    "How does actual revenue compare to budget by department?",
    "What is average signup rate by day of week?",
    "Show each of the top five product lines as a separate share of those five product lines' combined revenue",
    "What's the revenue split between the top 3 regions?",
    "What is the distribution of daily signup rates?",
    "What is the relationship between marketing spend and revenue?",
    "Give me a raw data table of budget variance by department",
]

SELECT_PLACEHOLDER = "More example questions..."


def _vega(data, spec, source, natural_order=None):
    return {"renderer": "vega_lite", "data": data, "spec": spec,
            "source": source, "natural_order": natural_order}


def _native(renderer, data, source, **kwargs):
    getattr(st, renderer)(data, **kwargs)
    return {"renderer": renderer, "data": data, "args": kwargs, "source": source}


def render_heatmap(corr_df: pd.DataFrame) -> dict:
    long_df = corr_df.reset_index().melt(id_vars="index", var_name="metric_b", value_name="correlation")
    long_df = long_df.rename(columns={"index": "metric_a"})
    spec = {
        "mark": "rect",
        "height": 480,
        "encoding": {
            "x": {"field": "metric_a", "type": "nominal"},
            "y": {"field": "metric_b", "type": "nominal"},
            "color": {
                "field": "correlation",
                "type": "quantitative",
                "scale": {"domain": [-1, 1], "scheme": "redblue"},
            },
            "tooltip": [{"field": "correlation", "type": "quantitative", "format": ".2f"}],
        },
    }
    return _vega(long_df, spec, "metrics.corr()")


def render_diverging_bar(df: pd.DataFrame, sort_order: str = "descending") -> dict:
    if sort_order == "natural":
        sorted_df = df
    else:
        sorted_df = df.sort_values(
            "actual_minus_budget", ascending=(sort_order == "ascending"), kind="stable"
        )
    spec = {
        "mark": "bar",
        "height": 480,
        "encoding": {
            "x": {"field": "actual_minus_budget", "type": "quantitative", "title": "Actual minus budget"},
            "y": {"field": "department", "type": "nominal", "sort": sorted_df["department"].tolist()},
            "color": {
                "field": "actual_minus_budget",
                "type": "quantitative",
                "scale": {"domainMid": 0, "scheme": "redblue"},
                "legend": None,
            },
        },
    }
    return _vega(sorted_df, spec, "budget_variance", df["department"].tolist())


def render_slope(df: pd.DataFrame) -> dict:
    long_df = df.melt(
        id_vars="product",
        value_vars=["rank_last_year", "rank_this_year"],
        var_name="period",
        value_name="rank",
    )
    long_df["period"] = long_df["period"].map(
        {"rank_last_year": "Last Year", "rank_this_year": "This Year"}
    )
    spec = {
        "mark": {"type": "line", "point": True},
        "height": 480,
        "encoding": {
            "x": {"field": "period", "type": "nominal", "sort": ["Last Year", "This Year"]},
            "y": {"field": "rank", "type": "quantitative", "scale": {"reverse": True}},
            "color": {"field": "product", "type": "nominal"},
        },
    }
    return _vega(long_df, spec, "rank_change")


def render_stacked_bar(df: pd.DataFrame, category_col: str, value_col: str) -> dict:
    plot_df = df.copy()
    plot_df["_group"] = "Total"
    spec = {
        "mark": "bar",
        "height": 200,
        "encoding": {
            "x": {
                "field": value_col,
                "type": "quantitative",
                "stack": "normalize",
                "title": "Share of revenue",
                "axis": {"format": "%"},
            },
            "y": {"field": "_group", "type": "nominal", "title": None},
            "color": {"field": category_col, "type": "nominal", "title": category_col.replace("_", " ").title()},
            "tooltip": [{"field": category_col}, {"field": value_col, "format": ",.0f"}],
        },
    }
    return _vega(plot_df, spec, "by_product: top 5 product lines")


def render_pie(df: pd.DataFrame, category_col: str, value_col: str) -> dict:
    spec = {
        "mark": {"type": "arc", "innerRadius": 0},
        "height": 420,
        "encoding": {
            "theta": {"field": value_col, "type": "quantitative"},
            "color": {"field": category_col, "type": "nominal", "title": category_col.replace("_", " ").title()},
            "tooltip": [{"field": category_col}, {"field": value_col, "format": ",.0f"}],
        },
    }
    return _vega(df, spec, "by_region: top 3 regions")


def render_chart(rec: dict, question: str) -> dict:
    evidence = _render_chart(rec, question)
    if evidence["renderer"] == "vega_lite":
        evidence["spec"] = style_chart(evidence["data"], evidence["spec"], rec)
        st.vega_lite_chart(evidence["data"], evidence["spec"], use_container_width=True)
    contracts = {
        "by_region": {"data": by_region, "operation": "identity", "keys": ["region"]},
        "budget_variance": {"data": budget_variance, "operation": "identity", "keys": ["department"]},
        "by_region: fallback": {"data": by_region, "operation": "identity", "keys": ["region"]},
        "by_region: top 3 regions": {"data": by_region, "operation": "top", "keys": ["region"], "value": "revenue", "count": 3},
        "by_product: top 5 product lines": {"data": by_product, "operation": "sum_top", "keys": ["product_line"], "value": "revenue", "count": 5},
        "daily": {"data": daily, "operation": "indexed", "keys": ["date"], "value": "signup_rate", "frequency": "D"},
        "by_product: daily revenue by product line": {"data": by_product, "operation": "pivot", "keys": ["date", "product_line"], "value": "revenue", "frequency": "D"},
        "daily: mean signup rate by weekday": {"data": daily, "operation": "weekday_mean", "keys": ["day_of_week"], "value": "signup_rate"},
        "metrics": {"data": metrics, "operation": "identity", "keys": []},
        "metrics.corr()": {"data": metrics, "operation": "correlation", "keys": ["metric_a", "metric_b"]},
        "rank_change": {"data": rank_change, "operation": "ranks", "keys": ["product", "period"]},
        "daily: signup rate bins": {"data": daily, "operation": "histogram", "bins": 20, "value": "signup_rate", "keys": []},
        "by_product: revenue bins": {"data": by_product, "operation": "histogram", "bins": 20, "value": "revenue", "keys": []},
    }
    evidence["source_contract"] = contracts[evidence["source"]]
    evidence["units"] = {"signup_rate": "fraction", "revenue": "currency_unspecified",
                         "actual_minus_budget": "currency_unspecified", "marketing_spend": "currency_unspecified"}
    return evidence


def _render_chart(rec: dict, question: str) -> dict:
    """Dispatch to the right chart renderer for one turn's recommendation."""
    chart_type = rec.get("chart_type")
    q_lower = question.lower()

    if chart_type == "heatmap":
        return render_heatmap(corr)
    elif chart_type == "diverging_bar_chart":
        return render_diverging_bar(budget_variance, sort_order=rec.get("sort_order", "descending"))
    elif chart_type == "slope_chart":
        return render_slope(rank_change)
    elif chart_type == "line_chart" and "signup" in q_lower:
        return _native("line_chart", daily.set_index("date")["signup_rate"], "daily", height=420)
    elif chart_type == "line_chart":
        pivoted = by_product.pivot(index="date", columns="product_line", values="revenue")
        return _native("line_chart", pivoted, "by_product: daily revenue by product line", height=420)
    elif chart_type == "horizontal_bar_chart":
        sort_order = rec.get("sort_order", "descending")
        if sort_order == "natural":
            sorted_region = by_region
        else:
            sorted_region = by_region.sort_values(
                "revenue", ascending=(sort_order == "ascending"), kind="stable"
            )
        spec = {
            "mark": "bar",
            "height": 420,
            "encoding": {
                "x": {"field": "revenue", "type": "quantitative", "title": None},
                "y": {
                    "field": "region",
                    "type": "nominal",
                    "sort": sorted_region["region"].tolist(),
                    "title": None,
                },
                "tooltip": [{"field": "region"}, {"field": "revenue", "format": ",.0f"}],
            },
        }
        return _vega(sorted_region, spec, "by_region", by_region["region"].tolist())
    elif chart_type == "bar_chart":
        # Days of week have intrinsic order, so this stays unsorted by value.
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        with_dow = daily.copy()
        with_dow["day_of_week"] = with_dow["date"].dt.day_name()
        by_dow = with_dow.groupby("day_of_week")["signup_rate"].mean().reindex(dow_order)
        spec = {
            "mark": "bar",
            "height": 420,
            "encoding": {
                "x": {"field": "day_of_week", "type": "nominal", "sort": dow_order, "title": None},
                "y": {"field": "signup_rate", "type": "quantitative", "title": None},
                "tooltip": [{"field": "day_of_week"}, {"field": "signup_rate"}],
            },
        }
        return _vega(by_dow.reset_index(), spec, "daily: mean signup rate by weekday", dow_order)
    elif chart_type == "stacked_bar_chart":
        top5 = by_product.groupby("product_line", as_index=False)["revenue"].sum().nlargest(5, "revenue")
        return render_stacked_bar(top5, "product_line", "revenue")
    elif chart_type == "pie_chart":
        top3 = by_region.nlargest(3, "revenue")
        return render_pie(top3, "region", "revenue")
    elif chart_type == "scatter_chart":
        return _native("scatter_chart", metrics, "metrics", x="marketing_spend", y="revenue", height=420)
    elif chart_type == "histogram" and "signup" in q_lower:
        return _native("bar_chart", daily["signup_rate"].value_counts(bins=20).sort_index(), "daily: signup rate bins", height=420)
    elif chart_type == "histogram":
        return _native("bar_chart", by_product["revenue"].value_counts(bins=20).sort_index(), "by_product: revenue bins", height=420)
    elif chart_type == "table":
        return _native("dataframe", budget_variance, "budget_variance", use_container_width=True)
    else:
        return _native("dataframe", by_region, "by_region: fallback", use_container_width=True)


def render_reasoning(rec: dict, evaluation: list) -> None:
    with st.expander("Agent's reasoning", expanded=False):
        text_tab, best_practice_tab, evaluation_tab, tool_calls_tab, json_tab = st.tabs(
            ["Text", "Best Practice", "Evaluation", "Tool Calls", "JSON"]
        )
        with evaluation_tab:
            counts = {status: sum(item["Status"] == status for item in evaluation)
                      for status in ("Pass", "Fail", "Not Checked", "Not Applicable")}
            st.write(" | ".join(f"{status}: {count}" for status, count in counts.items()))
            st.caption("Specification and data checks only. Unchecked items are not passes; this is not a visual or accessibility certification.")
            st.dataframe(pd.DataFrame(evaluation), use_container_width=True, hide_index=True)
        with text_tab:
            st.write(rec.get("reason", ""))
        with best_practice_tab:
            rule_ids = rec.get("rule_ids", [])
            if not rule_ids:
                st.write("No rules were reported for this recommendation.")
            for rule_id in rule_ids:
                citation = RULE_SOURCES.get(rule_id)
                if not citation:
                    continue
                st.markdown(f"**Rule {rule_id}.** {citation['principle']}")
                st.caption(citation["source"])
        with tool_calls_tab:
            tool_calls = rec.get("tool_calls", [])
            if not tool_calls:
                st.write("No tool calls were recorded for this recommendation.")
            for call in tool_calls:
                st.markdown(f"**{call['name']}**")
                if call.get("input"):
                    st.caption("Input")
                    st.json(call["input"])
                if call.get("result"):
                    st.caption("Result")
                    st.json(call["result"])
        with json_tab:
            st.json({k: v for k, v in rec.items() if k != "tool_calls"})


st.session_state.setdefault("messages", [])
st.session_state.setdefault("pending_question", None)


def _set_pending(value: str) -> None:
    st.session_state.pending_question = value


def _on_select_example() -> None:
    choice = st.session_state.get("example_select")
    if choice and choice != SELECT_PLACEHOLDER:
        st.session_state.pending_question = choice


def _retry_question(turn_index: int) -> None:
    turn = st.session_state.messages.pop(turn_index)
    st.session_state.pending_question = turn["question"]


def _clear_chat() -> None:
    st.session_state.messages = []
    st.session_state.pending_question = None


# st.chat_input always renders pinned to the bottom of the page regardless of
# where in the script it's called, so calling it here lets its return value
# feed into this run's logic the same way pending_question (from example and
# follow-up buttons) does.
chat_value = st.chat_input('Ask a question, e.g. "Show me revenue by region"')

new_question = st.session_state.pop("pending_question", None) or chat_value

if new_question:
    turn = {"question": new_question, "rec": None, "followups": []}
    try:
        with st.spinner("Thinking about the right chart..."):
            turn["rec"] = recommend_chart(question=new_question, schema_summary=SCHEMA_SUMMARY)
    except Exception as error:
        turn["error"] = getattr(error, "code", "request_failed")
        logging.getLogger(__name__).warning("Chart recommendation failed: %s", turn["error"])
    else:
        try:
            with st.spinner("Thinking of what to ask next..."):
                turn["followups"] = suggest_followups(
                    question=new_question, schema_summary=SCHEMA_SUMMARY, rec=turn["rec"]
                )
        except Exception:
            turn["followups_unavailable"] = True
            logging.getLogger(__name__).warning("Follow-up suggestions unavailable")
    st.session_state.messages.append(turn)

if not st.session_state.messages:
    st.subheader("Try asking")
    quick_cols = st.columns(3)
    for col, example in zip(quick_cols, QUICK_EXAMPLES):
        col.button(example, on_click=_set_pending, args=(example,), use_container_width=True)

    st.selectbox(
        "More example questions",
        [SELECT_PLACEHOLDER] + SELECT_EXAMPLES,
        key="example_select",
        on_change=_on_select_example,
        label_visibility="collapsed",
    )

for turn_index, turn in enumerate(st.session_state.messages):
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant", avatar=":material/bar_chart:"):
        if turn.get("error"):
            st.warning("No usable chart recommendation was returned. Your earlier charts are unchanged.")
            st.caption(f"Status: {turn['error']}")
            st.button(
                "Retry", key=f"retry_{turn_index}",
                on_click=_retry_question, args=(turn_index,),
            )
        else:
            evidence = render_chart(turn["rec"], turn["question"])
            evaluation = evaluate_chart(turn["rec"], evidence)
            failures = sum(item["Status"] == "Fail" for item in evaluation)
            if failures:
                st.warning(f"Chart evaluation: {failures} failed check(s). See Evaluation for evidence.")
            render_reasoning(turn["rec"], evaluation)
            if turn.get("followups_unavailable"):
                st.caption("Follow-up suggestions are unavailable. You can still ask another question.")

if st.session_state.messages:
    followups = st.session_state.messages[-1]["followups"]
    if followups:
        header_col, clear_col = st.columns([4, 1])
        header_col.subheader("Try next")
        clear_col.button("Clear chat", on_click=_clear_chat, use_container_width=True)
        followup_cols = st.columns(len(followups))
        for i, (col, followup) in enumerate(zip(followup_cols, followups)):
            col.button(
                followup,
                key=f"followup_{i}",
                on_click=_set_pending,
                args=(followup,),
                use_container_width=True,
            )
    else:
        st.button("Clear chat", on_click=_clear_chat)
