"""Independent checks of rendering inputs, not a visual or semantic audit."""

import math

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
from PIL import ImageColor

PASS, FAIL, NA, NC = "Pass", "Fail", "Not Applicable", "Not Checked"
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
PALETTES = {
    "redblue": "diverging", "blueorange": "diverging", "purplegreen": "diverging",
    "blues": "sequential", "viridis": "sequential", "cividis": "sequential",
    "tableau10": "qualitative", "category10": "qualitative", "set2": "qualitative",
}
RULE_NAMES = {
    1: "Temporal trend structure", 2: "Categorical bar structure",
    3: "Part-to-whole structure", 4: "Distribution bin structure",
    5: "Scatter numeric fields", 6: "Correlation matrix structure",
    7: "Variance encoding", 8: "Two-period ranking structure",
    9: "Requested highlight", 10: "Palette family agreement",
    11: "Visual accessibility", 12: "Rendered category order",
    13: "Clutter and readability", 14: "Magnitude axis zero baseline", 15: "Caption presence",
}
STRUCTURE_TYPES = {
    1: {"line_chart"}, 2: {"bar_chart", "horizontal_bar_chart"},
    3: {"pie_chart", "stacked_bar_chart"}, 4: {"histogram"},
    5: {"scatter_chart"}, 6: {"heatmap"}, 7: {"diverging_bar_chart"}, 8: {"slope_chart"},
}
BAR_TYPES = {"horizontal_bar_chart", "bar_chart", "diverging_bar_chart", "stacked_bar_chart"}


def _numeric(series):
    return (is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
            and len(series) > 0 and series.notna().all()
            and series.map(lambda value: math.isfinite(float(value))).all())


def _verdict(valid, detail):
    return (PASS if valid else FAIL), detail


def _frame(data):
    return data.to_frame() if isinstance(data, pd.Series) else data


def _encoding(evidence):
    spec = evidence.get("spec", {})
    if not isinstance(spec, dict) or any(key in spec for key in (
        "transform", "layer", "facet", "repeat", "concat", "hconcat", "vconcat"
    )):
        raise ValueError("Unsupported specification shape or browser transform")
    encoding = spec.get("encoding", {})
    if not isinstance(encoding, dict):
        raise ValueError("Invalid encoding")
    return encoding


def _channel(evidence, name):
    channel = _encoding(evidence).get(name, {})
    if not isinstance(channel, dict) or channel.get("aggregate") or channel.get("bin"):
        raise ValueError(f"Unsupported {name} encoding or browser transform")
    return channel


def _axes(chart_type, evidence):
    horizontal = chart_type in {"horizontal_bar_chart", "diverging_bar_chart", "stacked_bar_chart"}
    return _channel(evidence, "y" if horizontal else "x"), _channel(evidence, "x" if horizontal else "y")


def _zero_scale(encoding, values):
    if not _numeric(values):
        return FAIL, "Magnitude values are missing or nonfinite."
    scale = encoding.get("scale", {})
    if not isinstance(scale, dict):
        return NC, "Disabled or unsupported magnitude scale."
    if scale.get("type", "linear") != "linear":
        return FAIL, "Magnitude encoding does not use a linear scale."
    if "domain" in scale:
        domain = scale["domain"]
        if not isinstance(domain, list) or len(domain) != 2 or not all(
            type(value) in (int, float) and math.isfinite(value) for value in domain
        ):
            return NC, "Dynamic or unsupported magnitude domain."
        lower, upper = domain
        valid = lower <= 0 <= upper and lower <= values.min() and upper >= values.max()
        return _verdict(valid, f"Explicit domain {domain}; rendered extent [{values.min():g}, {values.max():g}].")
    if "domainMin" in scale or "domainMax" in scale:
        return NC, "Partial domain bounds need resolved scale inspection."
    if scale.get("zero") is True:
        return PASS, "Linear magnitude scale explicitly includes zero."
    if scale.get("zero") is False:
        return FAIL, "Magnitude scale disables zero inclusion."
    return NC, "Zero is not explicit; the browser-resolved default scale was not inspected."


def _baseline(chart_type, evidence, data):
    if chart_type == "histogram":
        return NC, "Native magnitude scale is not available in captured API arguments."
    if chart_type not in BAR_TYPES:
        return NA, "No bar-length magnitude encoding."
    category, value = _axes(chart_type, evidence)
    values = data[value["field"]]
    if chart_type == "stacked_bar_chart":
        if not _numeric(values) or data[category["field"]].isna().any():
            return FAIL, "Invalid stack values or group labels."
        stack = value.get("stack")
        if stack == "normalize":
            if (values < 0).any() or not data.groupby(category["field"])[value["field"]].sum().gt(0).all():
                return FAIL, "Normalized shares require nonnegative values and positive group totals."
            values = pd.Series([0.0, 1.0])
        elif stack == "zero":
            groups = data.groupby(category["field"])[value["field"]]
            positive = groups.apply(lambda series: series.clip(lower=0).sum())
            negative = groups.apply(lambda series: series.clip(upper=0).sum())
            values = pd.concat([positive, negative])
        else:
            return NC, "Only explicit normalize or zero stack offsets are supported."
    return _zero_scale(value, values)


def _centered(color):
    scale = color.get("scale", {})
    if not isinstance(scale, dict):
        return NC, "Unsupported color scale."
    scheme = scale.get("scheme")
    family = PALETTES.get(scheme) if isinstance(scheme, str) else None
    if family and family != "diverging":
        return FAIL, f"Palette {scheme} is {family}, not diverging."
    if not family:
        return NC, "No supported explicit diverging palette to inspect."
    domain = scale.get("domain")
    if "domainMid" in scale:
        valid = scale["domainMid"] == 0
        if domain is not None:
            valid = valid and isinstance(domain, list) and len(domain) == 2 and domain[0] < 0 < domain[1]
    else:
        valid = (isinstance(domain, list) and len(domain) == 2
                 and all(type(value) in (int, float) and math.isfinite(value) for value in domain)
                 and domain[0] < 0 < domain[1] and domain[0] == -domain[1])
    return _verdict(valid, f"Diverging scale: {scale}.")


def _structure(number, chart_type, evidence, data):
    if chart_type not in STRUCTURE_TYPES[number]:
        return NA, "This structural check does not apply to this chart type."
    if number == 1:
        return _verdict(is_datetime64_any_dtype(data.index) and all(_numeric(data[column]) for column in data),
                        "Checks datetime index and finite numeric series, not question intent.")
    if number == 4:
        valid = isinstance(data.index, pd.IntervalIndex) and data.index.is_monotonic_increasing and not data.index.is_overlapping
        valid = valid and all(_numeric(data[column]) and (data[column] >= 0).all() for column in data)
        return _verdict(valid, "Checks ordered nonoverlapping bins and nonnegative counts, not browser spacing.")
    if number == 5:
        args = evidence.get("args", {})
        valid = args.get("x") != args.get("y") and all(_numeric(data[args[channel]]) for channel in ("x", "y"))
        return _verdict(valid, f"Numeric pair: {args.get('x')} and {args.get('y')}.")
    category, value = _axes(chart_type, evidence)
    if number in (2, 7):
        valid = value.get("type") == "quantitative" and _numeric(data[value["field"]])
        valid = valid and category.get("type") in {"nominal", "ordinal"}
        return _verdict(valid, "Checks categorical axis and numeric magnitude; metric and baseline semantics require source review.")
    color = _channel(evidence, "color")
    if number == 3:
        magnitude = _channel(evidence, "theta") if chart_type == "pie_chart" else value
        categories, values = data[color["field"]], data[magnitude["field"]]
        count = categories.nunique()
        valid = (1 <= count <= 3 if chart_type == "pie_chart" else 4 <= count <= 5)
        valid = valid and not categories.isna().any() and _numeric(values) and (values >= 0).all() and values.sum() > 0
        if chart_type == "stacked_bar_chart":
            valid = valid and magnitude.get("stack") == "normalize" and data[category["field"]].nunique() == 1
        return _verdict(valid, f"{count} categories; requires nonnegative values, positive total and normalized stacking where applicable.")
    first, second = _channel(evidence, "x"), _channel(evidence, "y")
    if number == 6:
        first_values, second_values, values = data[first["field"]], data[second["field"]], data[color["field"]]
        valid = set(first_values) == set(second_values) and first_values.nunique() >= 3 and len(data) == first_values.nunique() ** 2
        valid = valid and not data.duplicated([first["field"], second["field"]]).any() and _numeric(values) and values.between(-1, 1).all()
        return _verdict(valid, "Checks complete 3+ metric grid and correlations in [-1, 1]; no source correlation recomputation.")
    period, ranks = data[first["field"]], data[second["field"]]
    valid = period.nunique() == 2 and _numeric(ranks) and (ranks >= 1).all() and (ranks % 1 == 0).all()
    valid = valid and not data.duplicated([color["field"], first["field"]]).any() and data.groupby(color["field"])[first["field"]].nunique().eq(2).all()
    valid = valid and second.get("scale", {}).get("reverse") is True
    order = first.get("sort")
    valid = valid and isinstance(order, list) and len(order) == 2 and set(order) == set(period)
    return _verdict(valid, "Requires two periods per series, explicit period order, integer ranks and reversed rank axis.")


def _ordering(rec, evidence, data):
    chart_type = rec.get("chart_type")
    if chart_type == "stacked_bar_chart":
        category, value = _axes(chart_type, evidence)
        color = _channel(evidence, "color")
        order = _channel(evidence, "order")
        if data[category["field"]].nunique() != 1 or data[color["field"]].duplicated().any():
            return NC, "Only single stacks with unique categories are supported."
        requested = rec.get("sort_order", "descending")
        if requested not in {"ascending", "descending"}:
            return NC, "Natural stack order requires an explicit category sequence."
        valid = order.get("field") == value["field"] and order.get("type") == "quantitative" and order.get("sort") == requested
        return _verdict(valid, f"Stack segment order encoding {order}; requested {requested}. This checks segment order, not legend order.")
    if chart_type not in BAR_TYPES:
        return NA, "Value-ranked categorical axis check does not apply."
    category, value = _axes(chart_type, evidence)
    order, requested = category.get("sort"), rec.get("sort_order", "descending")
    categories, values = data[category["field"]], data[value["field"]]
    if categories.isna().any() or categories.duplicated().any():
        return FAIL, "Missing or duplicate categories make the ranked axis ambiguous."
    if not isinstance(order, list) or not _numeric(values):
        return NC, "Requires an explicit category list and finite values."
    if len(order) != len(categories) or set(order) != set(categories):
        return FAIL, "Axis list does not match plotted categories exactly."
    if category.get("field") == "day_of_week" and order != WEEKDAYS:
        return FAIL, "Weekday order violates intrinsic Monday-to-Sunday order."
    if requested == "natural":
        natural = WEEKDAYS if category.get("field") == "day_of_week" else evidence.get("natural_order")
        return (NC if natural is None else PASS if order == natural else FAIL), f"Axis order {order}; expected {natural}."
    if requested not in {"ascending", "descending"}:
        return NC, f"Unsupported requested order: {requested}."
    ordered = data.set_index(category["field"]).loc[order, value["field"]]
    valid = ordered.is_monotonic_increasing if requested == "ascending" else ordered.is_monotonic_decreasing
    return _verdict(valid, f"Values in axis order checked against {requested}: {order}.")


def _highlight(rec, evidence, data):
    highlight = rec.get("highlight", "")
    if rec.get("chart_type") == "table" or not highlight:
        return NA, "No chart highlight requested."
    if evidence["renderer"] != "vega_lite":
        return NC, "Native emphasis encodings were not inspected."
    encoding = _encoding(evidence)
    fields = [definition.get("field") for definition in encoding.values()
              if isinstance(definition, dict) and definition.get("type") in {"nominal", "ordinal"} and definition.get("field") in data]
    if not any(highlight in data[column].astype(str).values for column in fields):
        return FAIL, f"Highlight {highlight!r} is absent from encoded categories."
    width = _channel(evidence, "strokeWidth")
    width_condition = width.get("condition", {})
    width_predicate = width_condition.get("test") if isinstance(width_condition, dict) else None
    if isinstance(width_predicate, dict) and set(width_predicate) == {"field", "equal"}:
        selected, base = width_condition.get("value"), width.get("value")
        valid = width_predicate["field"] in fields and width_predicate["equal"] == highlight
        valid = valid and type(selected) in (int, float) and type(base) in (int, float)
        valid = valid and math.isfinite(selected) and math.isfinite(base) and selected > base >= 0
        if rec.get("chart_type") != "slope_chart" and not _channel(evidence, "stroke").get("value"):
            return NC, "Outline width is defined but stroke color is not explicit."
        return _verdict(valid, "Selective stroke-width encoding checked; screen contrast and visibility still require visual review.")
    color = _channel(evidence, "color")
    condition = color.get("condition")
    predicate = condition.get("test") if isinstance(condition, dict) else None
    if isinstance(predicate, dict) and set(predicate) == {"field", "equal"}:
        if predicate["field"] not in fields or predicate["equal"] != highlight:
            return FAIL, "Highlight predicate targets the wrong field or category."
        try:
            foreground = ImageColor.getcolor(condition["value"].strip().lower(), "RGBA")
            background = ImageColor.getcolor(color["value"].strip().lower(), "RGBA")
        except (ValueError, TypeError, AttributeError, KeyError):
            return NC, "Color syntax cannot be resolved by the supported CSS color parser."
        if foreground[3] != 255 or background[3] != 255:
            return NC, "Transparent colors require browser compositing and background inspection."
        return _verdict(foreground != background,
                        f"Resolved highlight/base colors: {foreground[:3]} / {background[:3]}; perceptual contrast is not certified.")
    scale = color.get("scale", {})
    if any(isinstance(definition, dict) and "condition" in definition for definition in encoding.values()) or isinstance(scale, dict) and "range" in scale:
        return NC, "Custom emphasis needs additional inspection."
    return FAIL, f"{highlight!r} is present but no selective emphasis is supplied."


def _palette(rec, evidence):
    if rec.get("chart_type") == "table":
        return NA, "No chart palette."
    color = _channel(evidence, "color")
    if "condition" in color:
        return NC, "Highlight/base colors replace the palette scale; palette family is not certified."
    scale = color.get("scale", {})
    scheme = scale.get("scheme") if isinstance(scale, dict) else None
    family = PALETTES.get(scheme) if isinstance(scheme, str) else None
    if not family:
        return NC, "Default, conditional, or unrecognized palette."
    valid = family == rec.get("color_scheme") and not (color.get("type") in {"quantitative", "ordinal"} and family == "qualitative")
    return _verdict(valid, f"Palette {scheme}: {family}; recommendation: {rec.get('color_scheme')}. Family only.")


def _reconcile(evidence, plotted):
    contract = evidence.get("source_contract")
    if not contract:
        return NC, "No independent source and transformation contract supplied."
    source = contract["data"]
    operation = contract["operation"]
    keys = contract.get("keys", [])
    value = contract.get("value")
    if operation == "identity":
        expected = source.copy()
    elif operation in {"top", "sum_top"}:
        expected = source.groupby(keys, as_index=False, dropna=False)[value].sum() if operation == "sum_top" else source.copy()
        expected = expected.nlargest(contract["count"], value)
    elif operation == "indexed":
        expected = source.set_index(keys)[[value]]
    elif operation == "pivot":
        expected = source.pivot(index=keys[0], columns=keys[1], values=value)
    elif operation == "weekday_mean":
        expected = source.assign(day_of_week=source["date"].dt.day_name()).groupby("day_of_week")[value].mean().reindex(WEEKDAYS).reset_index()
    elif operation == "correlation":
        expected = source.corr().rename_axis("metric_a").reset_index().melt(id_vars="metric_a", var_name="metric_b", value_name="correlation")
    elif operation == "ranks":
        expected = source.melt(id_vars="product", value_vars=["rank_last_year", "rank_this_year"], var_name="period", value_name="rank")
        expected["period"] = expected["period"].map({"rank_last_year": "Last Year", "rank_this_year": "This Year"})
    elif operation == "histogram":
        if not isinstance(plotted.index, pd.IntervalIndex) or plotted.shape[1] != 1 or plotted.index.is_overlapping:
            return FAIL, "Histogram bins are ambiguous or overlapping."
        samples = source[value]
        if not _numeric(samples):
            return FAIL, "Histogram source contains missing or nonfinite observations."
        if "bins" in contract:
            assignments = pd.cut(samples, bins=contract["bins"], include_lowest=True)
            expected_bins = assignments.cat.categories
            counts = assignments.value_counts(sort=False).reindex(expected_bins, fill_value=0)
            valid = (not assignments.isna().any() and plotted.index.equals(expected_bins)
                     and counts.to_numpy().tolist() == plotted.iloc[:, 0].tolist())
            return _verdict(valid, f"Recomputed {contract['bins']} bins and counts from {len(samples)} source observations before display-boundary rounding.")
        counts = pd.Series([int(((samples > interval.left) & (samples <= interval.right)).sum())
                            if interval.closed == "right" else int(samples.map(lambda item: item in interval).sum())
                            for interval in plotted.index], index=plotted.index)
        valid = counts.sum() == len(samples) and counts.eq(plotted.iloc[:, 0]).all()
        return _verdict(valid, f"Recounted {len(samples)} source observations in emitted bins; assigned {counts.sum()}. Rounded boundaries can expose bin mismatches.")
    else:
        return NC, f"Unsupported transformation contract: {operation}."
    actual = plotted.copy()
    if operation == "sum_top":
        actual = actual.drop(columns=["_group"], errors="ignore")
    if keys and all(key in expected.columns and key in actual.columns for key in keys):
        if actual[keys].isna().any().any() or actual.duplicated(keys).any():
            return FAIL, "Missing or duplicate source keys in plotted data."
        expected = expected.sort_values(keys).reset_index(drop=True)
        actual = actual.sort_values(keys).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_names=False,
                                      check_exact=False, rtol=1e-9, atol=1e-12, check_like=True)
    except AssertionError:
        return FAIL, f"Plotted rows, categories, or values differ from the {operation} source contract."
    return PASS, f"Rows, categories and values match local source operation {operation}; this does not establish question intent or SQL parity."


def _time_check(rec, evidence, data, coverage=False):
    if rec.get("chart_type") != "line_chart":
        return NA, "Not a time-series chart."
    if not isinstance(data.index, pd.DatetimeIndex):
        return NC, "No inspectable datetime index."
    if not coverage:
        valid = not data.index.hasnans and data.index.is_unique and data.index.is_monotonic_increasing
        return _verdict(valid, "Checks nonmissing, unique, chronological timestamps per wide-format series.")
    contract = evidence.get("source_contract", {})
    frequency = contract.get("frequency")
    source = contract.get("data")
    if frequency is None or source is None:
        return NC, "No declared cadence and independent source time extent."
    dates = source[contract["keys"][0]]
    expected = pd.date_range(dates.min(), dates.max(), freq=frequency)
    valid = data.index.equals(expected) and not data.isna().any().any()
    return _verdict(valid, f"Expected {len(expected)} observations per series at cadence {frequency}; missing periods/values are not treated as zeros.")


def _units(rec, evidence, data):
    if rec.get("chart_type") == "table":
        return NC, "Table unit labels and formatting are not inspected."
    if evidence.get("renderer") != "vega_lite":
        return NC, "Native browser-resolved labels and formatting are unavailable."
    units = evidence.get("units", {})
    inspected, unknown, failures = [], [], []
    for name in ("x", "y", "theta"):
        channel = _channel(evidence, name)
        unit = "fraction" if channel.get("stack") == "normalize" else units.get(channel.get("field"))
        if not unit:
            continue
        if name == "theta":
            unknown.append("Arc labels and tooltips require separate inspection")
            continue
        axis = channel.get("axis", {})
        if not isinstance(axis, dict):
            unknown.append(f"{name}: hidden or unsupported axis")
            continue
        formatting = axis.get("format")
        if unit == "fraction":
            if channel.get("stack") != "normalize" and not data[channel["field"]].between(0, 1).all():
                failures.append(f"{name}: fractional rate falls outside [0, 1]")
            if not isinstance(formatting, str):
                unknown.append(f"{name}: percentage format is implicit")
            elif not formatting.endswith("%"):
                failures.append(f"{name}: fractional rate/share lacks percentage formatting")
            else:
                inspected.append(f"{name}: fraction displayed as percentage")
        elif unit == "currency_unspecified":
            unknown.append(f"{name}: source currency identity is unspecified")
    if failures:
        return FAIL, "; ".join(failures + unknown)
    if unknown:
        return NC, "; ".join(inspected + unknown)
    return (PASS, "; ".join(inspected)) if inspected else (NC, "No supported declared units on inspected axes.")


def _missing_labels(evidence, data):
    fields = [definition["field"] for definition in _encoding(evidence).values()
              if isinstance(definition, dict) and definition.get("type") in {"nominal", "ordinal"} and "field" in definition]
    if not fields:
        return NC, "No explicit categorical encodings to inspect."
    invalid = [field for field in fields if data[field].isna().any() or data[field].astype(str).str.strip().eq("").any()]
    return _verdict(not invalid, f"Missing or blank categorical labels: {invalid or 'none'}.")


def evaluate_chart(rec, evidence):
    """A failed check cannot suppress unrelated checks or imply compliance."""
    results = []
    cited = rec.get("rule_ids", [])
    chart_type = rec.get("chart_type")

    def add(check, status, detail, rule=None):
        results.append({"Check": check, "Rule": rule, "Status": status,
                        "Agent cited": rule in cited if rule else False, "Evidence": detail})

    def run(check, function, rule=None):
        try:
            status, detail = function()
        except (KeyError, TypeError, ValueError, AttributeError, IndexError, OverflowError) as error:
            status, detail = NC, f"Unsupported or incomplete evidence: {error}. Other checks continue."
        add(check, status, detail, rule)

    try:
        data = _frame(evidence["data"])
        if not isinstance(data, pd.DataFrame) or data.empty or not data.columns.is_unique:
            raise ValueError("Missing, empty, or ambiguous plotted data")
        _encoding(evidence)
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        add("Evidence coverage", NC, str(error))
        data = pd.DataFrame()

    def renderer_agreement():
        spec = evidence.get("spec", {})
        mark = spec.get("mark")
        mark = mark.get("type") if isinstance(mark, dict) else mark
        expected = {"horizontal_bar_chart": "bar", "bar_chart": "bar", "stacked_bar_chart": "bar",
                    "diverging_bar_chart": "bar", "pie_chart": "arc", "slope_chart": "line", "heatmap": "rect"}
        native = {"line_chart": "line_chart", "scatter_chart": "scatter_chart", "histogram": "bar_chart", "table": "dataframe"}
        matches = evidence["renderer"] == "vega_lite" and mark == expected[chart_type] if chart_type in expected else chart_type in native and evidence["renderer"] == native[chart_type]
        return _verdict(matches, f"Requested {chart_type}; renderer {evidence['renderer']}; mark {mark or 'native'}.")

    def plotted_data():
        fields = [definition["field"] for definition in _encoding(evidence).values() if isinstance(definition, dict) and "field" in definition]
        if data.empty or any(field not in data for field in fields):
            add("Evidence coverage", NC, "Empty data or encoded field absent.")
            return FAIL, "Empty data or an encoded field is absent."
        return PASS, f"Source: {evidence.get('source', 'unknown')}; {len(data)} rows; fields: {', '.join(map(str, data.columns))}. Presence only, not source correctness."

    run("Renderer agreement", renderer_agreement)
    run("Plotted data", plotted_data)
    for number in STRUCTURE_TYPES:
        run(RULE_NAMES[number], lambda number=number: _structure(number, chart_type, evidence, data), number)
    if chart_type in {"heatmap", "diverging_bar_chart"}:
        number = 6 if chart_type == "heatmap" else 7
        run("Correlation color midpoint" if number == 6 else "Variance color midpoint", lambda: _centered(_channel(evidence, "color")), number)
    run(RULE_NAMES[9], lambda: _highlight(rec, evidence, data), 9)
    run(RULE_NAMES[10], lambda: _palette(rec, evidence), 10)
    run(RULE_NAMES[12], lambda: _ordering(rec, evidence, data), 12)
    run(RULE_NAMES[14], lambda: _baseline(chart_type, evidence, data), 14)
    for number, detail in ((11, "No full accessibility audit; color_blind_safe is an agent claim."),
                           (13, "Clipping, clutter, overlap and readability need browser or human review.")):
        add(RULE_NAMES[number], NA if chart_type == "table" else NC, detail, number)
    present = isinstance(rec.get("reason"), str) and bool(rec["reason"].strip())
    add(RULE_NAMES[15], NA if chart_type == "table" else PASS if present else FAIL,
        "Caption presence only; meaning and numerical accuracy are not verified.", 15)
    if chart_type != "table":
        add("Caption accuracy", NC, "Caption claims have not been checked against source data.", 15)
    run("Source reconciliation", lambda: _reconcile(evidence, data))
    run("Time-series order and uniqueness", lambda: _time_check(rec, evidence, data))
    run("Time-series coverage", lambda: _time_check(rec, evidence, data, coverage=True))
    run("Unit formatting", lambda: _units(rec, evidence, data))
    run("Categorical label completeness", lambda: _missing_labels(evidence, data))
    add("Question-to-data alignment", NC, "No semantic question matching; local rendering data is separate from SQL grounding.")
    return results
