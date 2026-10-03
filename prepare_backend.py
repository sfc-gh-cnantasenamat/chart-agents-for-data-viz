import argparse
import ast
from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
TABLES = (
    ("DAILY_SIGNUPS", 0, ("DATE", "FLOAT")),
    ("REGION_REVENUE", 1, ("VARCHAR", "FLOAT")),
    ("PRODUCT_REVENUE", 2, ("VARCHAR", "DATE", "FLOAT")),
    ("CORRELATED_METRICS", 6, ("FLOAT", "FLOAT", "FLOAT", "FLOAT")),
    ("BUDGET_VARIANCE", 4, ("VARCHAR", "FLOAT")),
    ("RANK_CHANGE", 5, ("VARCHAR", "NUMBER(38,0)", "NUMBER(38,0)")),
)


def identifier(value):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError("Use an unquoted Snowflake identifier containing letters, digits, or underscores.")
    return value.upper()


def sample_frames():
    tree = ast.parse((ROOT / "app" / "app.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "load_sample_data")
    function.decorator_list = []
    namespace = {"np": np, "pd": pd}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(ROOT / "app" / "app.py"), "exec"), namespace)
    return namespace["load_sample_data"]()


def sql_literal(value):
    if isinstance(value, pd.Timestamp):
        return "'" + value.strftime("%Y-%m-%d") + "'"
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if not np.isfinite(value):
        raise ValueError("Sample data must contain finite values.")
    return repr(float(value))


def sample_sql(database, schema):
    prefix = f"{identifier(database)}.{identifier(schema)}"
    frames = sample_frames()
    statements = ["BEGIN TRANSACTION;"]
    definitions = []
    for name, frame_index, types in TABLES:
        frame = frames[frame_index]
        columns = [identifier(column) for column in frame.columns]
        definitions.append(f"CREATE TABLE {prefix}.{name} (" + ", ".join(f"{column} {kind}" for column, kind in zip(columns, types, strict=True)) + ");")
        values = ["(" + ", ".join(sql_literal(value) for value in row) + ")" for row in frame.itertuples(index=False, name=None)]
        for offset in range(0, len(values), 500):
            statements.append(f"INSERT INTO {prefix}.{name} (" + ", ".join(columns) + ") VALUES\n" + ",\n".join(values[offset:offset + 500]) + ";")
    statements.append("COMMIT;")
    return "\n\n".join(definitions + statements) + "\n"


def submission_sql(database, schema):
    prefix = f"{identifier(database)}.{identifier(schema)}"
    return f'''CREATE PROCEDURE {prefix}.SUBMIT_CHART_RECOMMENDATION(
    CHART_TYPE VARCHAR, SORT_ORDER VARCHAR, COLOR_SCHEME VARCHAR,
    HIGHLIGHT VARCHAR, COLOR_BLIND_SAFE BOOLEAN, REASON VARCHAR, RULE_IDS VARCHAR
)
RETURNS VARIANT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('snowflake-snowpark-python')
HANDLER = 'submit'
EXECUTE AS OWNER
AS $$
def submit(session, chart_type, sort_order, color_scheme, highlight, color_blind_safe, reason, rule_ids):
    parsed_rule_ids = []
    for piece in (rule_ids or "").split(","):
        piece = piece.strip()
        if piece:
            try:
                parsed_rule_ids.append(int(piece))
            except ValueError:
                pass
    return {{
        "chart_type": chart_type,
        "sort_order": sort_order,
        "color_scheme": color_scheme,
        "highlight": highlight,
        "color_blind_safe": color_blind_safe,
        "reason": reason,
        "rule_ids": parsed_rule_ids,
    }}
$$;
'''


def main():
    parser = argparse.ArgumentParser(description="Prepare local Chartify SQL without connecting to Snowflake. Validator deployment remains blocked pending security review.")
    parser.add_argument("--database", required=True, type=identifier)
    parser.add_argument("--schema", required=True, type=identifier)
    parser.add_argument("--output", type=Path, default=ROOT / "_local" / "backend")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = {
        "01_sample_tables.sql": sample_sql(args.database, args.schema),
        "02_submission.sql": submission_sql(args.database, args.schema),
    }
    for name, content in outputs.items():
        destination = args.output / name
        with destination.open("x") as handle:
            handle.write(content)
        print(destination)
    print("Prepared only. No SQL executed. Validator and approved deployment targets are still required.")


if __name__ == "__main__":
    main()
