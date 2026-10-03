"""Apply supported display recommendations without changing plotting data."""

from copy import deepcopy


def style_chart(data, spec, rec):
    styled = deepcopy(spec)
    encoding = styled["encoding"]
    chart_type = rec.get("chart_type")
    palette = rec.get("color_scheme")
    if chart_type in {"horizontal_bar_chart", "bar_chart", "diverging_bar_chart", "stacked_bar_chart"}:
        magnitude = "x" if chart_type != "bar_chart" else "y"
        encoding[magnitude].setdefault("scale", {})["zero"] = True
    if chart_type in {"heatmap", "diverging_bar_chart"}:
        encoding["color"]["scale"]["scheme"] = "redblue"
    elif chart_type in {"pie_chart", "stacked_bar_chart", "slope_chart"}:
        encoding["color"]["scale"] = {"scheme": "tableau10"}
    elif chart_type in {"horizontal_bar_chart", "bar_chart"}:
        category_axis = "y" if chart_type == "horizontal_bar_chart" else "x"
        value_axis = "x" if category_axis == "y" else "y"
        if palette == "qualitative":
            encoding["color"] = {"field": encoding[category_axis]["field"], "type": "nominal",
                                 "scale": {"scheme": "tableau10"}, "legend": None}
        else:
            encoding["color"] = {"field": encoding[value_axis]["field"], "type": "quantitative",
                                 "scale": {"scheme": "blues"}, "legend": None}
    if chart_type == "stacked_bar_chart":
        requested = rec.get("sort_order", "descending")
        if requested in {"ascending", "descending"}:
            encoding["order"] = {"field": encoding["x"]["field"], "type": "quantitative", "sort": requested}
    highlight = rec.get("highlight")
    category_fields = {
        "horizontal_bar_chart": encoding.get("y", {}).get("field"),
        "bar_chart": encoding.get("x", {}).get("field"),
        "diverging_bar_chart": encoding.get("y", {}).get("field"),
        "pie_chart": encoding.get("color", {}).get("field"),
        "stacked_bar_chart": encoding.get("color", {}).get("field"),
        "slope_chart": encoding.get("color", {}).get("field"),
    }
    field = category_fields.get(chart_type)
    if not highlight or field not in data or not data[field].eq(highlight).any():
        return styled
    predicate = {"field": field, "equal": highlight}
    if chart_type in {"horizontal_bar_chart", "bar_chart"}:
        encoding["color"] = {"condition": {"test": predicate, "value": "#2878B5"}, "value": "#9AA0A6"}
    elif chart_type == "slope_chart":
        encoding["strokeWidth"] = {"condition": {"test": predicate, "value": 4}, "value": 1.5}
        encoding["opacity"] = {"condition": {"test": predicate, "value": 1}, "value": 0.35}
    else:
        encoding["stroke"] = {"value": "#111111"}
        encoding["strokeWidth"] = {"condition": {"test": predicate, "value": 3}, "value": 0}
    return styled
