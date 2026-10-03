# Chartify: Literature-Grounded Chart Recommendations

Companion project for **I Curated the Best Data Visualization Tips, Then Taught an Agent to Use Them**. The Developer Blog draft remains `Hidden`; no publication or clean-account deployment is implied.

## Layout

- `app/app.py`: Streamlit chat interface, seeded sample data, eleven renderer mappings, and example questions.
- `app/chart_agent.py`: agent transport, submission validation, citations, and separate follow-up generation.
- `app/chart_eval.py` and `app/chart_style.py`: rendering-input evaluation and supported Vega-Lite styling.
- `cortex_project/`: exported agent and semantic-view specifications, with portable example resource references.
- `prepare_backend.py`: offline SQL preparation for the six sample tables and submission procedure.
- [Blog repository](https://github.com/sfc-gh-cnantasenamat/blog-streamlit-chart-agents-for-data-viz): Hidden article draft, publication PNGs, diagram generator, and editable diagram source.
- `test_chart_*.py`: local regression tests. They do not require a live agent.

## Architecture and Boundaries

The Cortex Agent is instructed to query `chartify_analyst`, call `validate_recommendation`, and finish with `submit_chart_recommendation`. Instructions and tool schemas do not guarantee execution or success. The app checks the returned submission before rendering and preserves earlier charts when a request fails.

The Analyst semantic view covers six Snowflake tables. Rendering uses separate, locally generated DataFrames, not the agent's SQL results. Chart mappings are fixed: regional horizontal bars, weekday signup bars, top-five product-line stacked shares, top-three region pies, correlations, budget variance, rank changes, and the remaining native chart cases. Free-form questions do not implement arbitrary filters or query-result rendering.

The reasoning expander has five tabs: Text, Best Practice, Evaluation, Tool Calls, and JSON. Evaluation checks explicit rendering inputs and local source transformations. It does not certify caption accuracy, question-to-data alignment, browser-resolved native encodings, label overlap, or full accessibility.

## Local Installation

The recorded local baseline is Python 3.11.6 with Streamlit 1.57.0, Pandas 2.3.3, NumPy 2.3.5, Snowflake Connector 4.7.1, Pillow 12.0.0, and PyArrow 18.1.0. Runtime dependencies and the test extra are declared in `app/pyproject.toml`.

From the project root:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install './app[test]'
.venv/bin/python -m unittest discover -s . -p 'test_chart_*.py'
```

Configure a local Snowflake connection using Streamlit's Snowflake connection settings. A non-secret template for `app/.streamlit/secrets.toml` is:

```toml
[connections.snowflake]
account = "YOUR_ORG-YOUR_ACCOUNT"
user = "YOUR_USER"
role = "YOUR_APPROVED_ROLE"
warehouse = "YOUR_APPROVED_WAREHOUSE"
authenticator = "externalbrowser"
```

Use your organization's approved authentication method. The real secrets file and environment files are excluded by `.gitignore`; do not include them in deployment artifacts. This template is for local execution only. Hosted Streamlit in Snowflake uses embedded identity.

Launch from the app directory so Streamlit finds its local configuration:

```bash
cd app
CHARTIFY_AGENT_FQN=CHARTIFY_DEMO.PUBLIC.CHARTIFY_AGENT ../.venv/bin/streamlit run app.py
```

Set `CHARTIFY_AGENT_FQN` to an agent you can access. When unset or blank, the code retains `DEVREL.CNANTASENAMAT_DEV.CHARTIFY_AGENT` solely for compatibility with the existing deployment; readers should not expect access to it. Both agent and follow-up queries use `ttl=0` to avoid reusing query results. This is not a claim that Streamlit ignores bind values in cache keys.

## Backend Preparation

**Clean-account installation is blocked pending validator security review.** The existing owner-executed validator interpolates requested column names into SQL. Its scope is also schema-wide rather than limited to the six demo tables. No adversarial testing was performed, and the deployed procedure was not changed. An executable validator deployment artifact is deliberately omitted rather than presenting the recovered implementation as safe public setup code. A reviewed, safely parameterized validator is required before exposing a new agent to users.

The remaining preparation is reproducible and does not connect to Snowflake:

```bash
.venv/bin/python prepare_backend.py --database CHARTIFY_DEMO --schema PUBLIC
```

This creates `_local/backend/01_sample_tables.sql` and `02_submission.sql`. Existing output files are not overwritten. The generator executes only the local app's `load_sample_data` function, without running the Streamlit UI, so it preserves the RNG seed and call order rather than maintaining a second sample generator. The six table row counts are 365 daily signups, 8 regional revenue rows, 4,380 product revenue rows, 200 correlated metrics, 6 budget variances, and 6 rank changes. These counts and column types were inspected in the existing deployment; full row-by-row equivalence with deployed data has not been verified.

The SQL uses `CREATE`, not `CREATE OR REPLACE`. Review every statement before execution and stop on any error. Table creation is not transactionally rolled back in Snowflake. Use a fresh, approved schema and do not rerun inserts into an existing populated schema.

The specifications were exported through `cortex agent-studio` from the deployed objects. Instructions, tools, budgets, and the submission schema are preserved. Only resource references were changed to `CHARTIFY_DEMO.PUBLIC` and `CHARTIFY_WH`, which are illustrative targets, not provisioned resources. The inherited submit-tool description still says rule IDs 1-12 while the instructions define 1-15; this inconsistency is recorded rather than silently changing the agent.

After validator review and separate approval of resources, the installation order is:

1. Have an administrator approve the database/schema, warehouse, role, authentication, and model access. No grants or compute resources are created by this package.
2. Review and execute the sample-table SQL in a fresh schema, then the submission procedure and reviewed validator. The provisioning role needs the corresponding object-creation privileges; the execution role needs access to the agent and its dependencies. Review owner-rights procedure access separately.
3. Use `cortex agent-studio sv-read --source workspace --file-path cortex_project/CHARTIFY_SEMANTIC_VIEW.sv.yaml`, adapt base-table references through the semantic-view editing workflow, validate, and deploy to the approved target.
4. Use `cortex agent-studio agent-read --source workspace --file-path cortex_project/CHARTIFY_AGENT.agent.yaml`, adapt the three tool resources and warehouse, and deploy through agent-studio. Confirm the intended version is live; a saved draft alone is not publication.
5. Configure the app's connection and agent FQN, run one bounded request, and inspect Analyst, validator, and submission evidence independently.
6. Test rendered charts in the actual target runtime before sharing the app.

No DDL, grants, new database, warehouse, compute pool, or agent deployment was executed during packaging.

## Streamlit in Snowflake

The existing app uses the container runtime. A new deployment needs an approved compute pool, query warehouse, and runtime-compatible dependencies. Including `pyproject.toml` in container artifacts triggers package resolution and requires suitable external access configuration. Do not reuse the author's resource names as defaults for another account.

Use an explicit artifact list containing `app.py`, `chart_agent.py`, `chart_eval.py`, `chart_style.py`, and the approved dependency file. Do not upload the entire project or `.streamlit/secrets.toml`. Configure the agent target for the hosted runtime before deployment. This package does not include a deploy-now manifest because targets and cost-bearing resources have not been approved.

The current sizing calls retain `use_container_width=True`, previously exercised in the deployed app. Streamlit 1.57 emits deprecation warnings; replacing them with `width` requires testing against the actual deployed version. Native histogram rendering also emits an interval-type inference warning. These warnings are not evidence of a failing test, but remain compatibility work before broader runtime support is claimed.
