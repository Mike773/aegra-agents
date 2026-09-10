# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

LangGraph agents for a self-hosted [aegra](https://github.com/ibbybuilds/aegra) server (LangGraph Platform alternative), LLM/embeddings via `langchain-gigachat`. Domain: a manager's chat assistant that analyzes employee metrics datasets (plan/fact/peer comparison/"stars") and grounds answers in a wiki knowledge base. **Code comments, docstrings, docs, prompts and commit messages are all in Russian** — keep it that way.

Several modules here are **stubs replaced by the production `langgraph_executor` repo** with identical public API: `agent/services/clients/gigachat.py`, `shared/{agent_dataset,orgstructure,assignments_service,settings}.py`, `aegra_agents/long_term_memory/`. Keep their signatures stable; don't build real logic into them.

## Commands

```bash
uv sync                                  # or: pip install -e .   (Python >=3.11; .venv is 3.12)
uv sync --extra causal                   # dowhy/pandas — only for the (removed) causal layer

# Tests — only these suites are live (~500 tests, ~3s, no network):
.venv/bin/python -m pytest tests/analyst_agent tests/metric_enricher tests/analytic_orchestrator_v4 tests/json_analyzer_v5 tests/long_term_memory tests/shared tests/test_agent_dataset_prod.py -q
.venv/bin/python -m pytest tests/json_analyzer_v5/test_star_metrics_v5.py -q            # one file
.venv/bin/python -m pytest tests/json_analyzer_v5/test_star_metrics_v5.py -k star -q    # one test by name
```

Bare `pytest` fails at collection: `tests/{json_analyzer,json_analyzer_v2,_v3,_v4,_causal,analytic_orchestrator,analytic_orchestrator_v2}` import packages that were deleted from the tree. Same for scripts `run_orchestrator.py`, `chat_orchestrator.py`, `diagnose_*.py`, `e2e_causal.py`, `smoke_causal.py`, `loadtest_inprocess.py`. Don't "fix" them unless asked; treat them as history.

Tests need no credentials: graph modules instantiate the GigaChat client at import time, so tests do `os.environ.setdefault("GIGACHAT_CREDENTIALS", "test-dummy")` and `sys.path.insert(0, repo_root)` before importing, then call `build_graph(object())` (nodes are lazy factories) or exercise pure helpers directly. Follow that pattern; there is no conftest.

Local runs against real GigaChat/Postgres (need `.env`: `GIGACHAT_CREDENTIALS`, `GIGACHAT_SCOPE`, `GIGACHAT_VERIFY_SSL`, `LLM_MODEL`, `POSTGRES_DSN`; `.env` is auto-loaded by `shared/clients.py` and `easyrag/db.py` by walking up from the module):

```bash
.venv/bin/python scripts/e2e_analyst.py               # analyst_agent on a samples_v2/*.json file, dataset service mocked (E2E_DASHBOARD=1, E2E_PEER=1, E2E_TURNS=2, E2E_WIKI=1)
.venv/bin/python scripts/sql_debug.py --dataset samples_v2/sample_declining.json --list   # run analyst_agent's .sql templates / free SQL on a dataset or a dumped .db
.venv/bin/python scripts/e2e_orchestrator.py          # first turn of orchestrator on a samples_v2/*.json file, dataset service mocked
.venv/bin/python scripts/chat_orchestrator_aegra.py    # interactive chat against a RUNNING aegra (http://localhost:8000)
.venv/bin/python scripts/run_kb_chat.py | run_doc_manager.py | run_wiki_ingest.py | smoke_gap_resolver.py | reprocess_direction.py
.venv/bin/python scripts/loadtest_orchestrator.py      # HTTP load test against running aegra
DIAG=1 uvicorn langgraph_executor.plugins.app:app      # pool/thread sampler from langgraph_executor/diag.py (see docs/diagnose_concurrency.md)
python -c "from langgraph_executor.aegra_agents.analytic_orchestrator_v4.graph import graph; print(graph.get_graph().draw_ascii())"
```

`langgraph_executor.plugins.app` (the aegra http app) is **not** in this repo; `aegra.json` references it from the production project. Postgres schema: apply `migrations/wiki_rag/0001_initial.sql`, `0002_ingest.sql` (idempotent; pgvector 1024-dim).

## Architecture

### Registration and graph contract

`aegra.json` → `./link.py:<name>_graph` → `langgraph_executor/aegra_agents/<graph>/graph.py`. `link.py` is the single entry point on purpose (aegra loads graphs by file path; pip-installed/PyInstaller bundles have no `.py` on disk) and also anchors PyInstaller import analysis — **every new graph must be imported there and listed in `__all__`**, then added to `aegra.json`.

Every `graph.py` follows: `def build_graph(llm, ...) -> compiled graph` plus module-level `llm = create_gigachat_client().get_llm(); graph = build_graph(llm)`. Nodes are closures from `make_*_node(llm, ...)` factories in `nodes.py`; runtime params come from `config["configurable"]` read inside nodes (no `config_schema`). Checkpointer is injected by aegra per `thread_id`; local scripts pass `MemorySaver`. Package versions are suffixed (`json_analyzer_v5`, `analytic_orchestrator_v4`); older versions have been deleted, only the newest lives in the tree.

### Graph inventory (registered names)

| Graph | Shape |
|---|---|
| `analyst_agent` | Turn-based manager chat, **the current product**. One graph, one business prompt, one tool loop per turn; text2sql over a normalized in-memory SQLite. See `docs/analyst_agent.md`. |
| `metric_enricher` | `load_catalog → enrich → finalize`. Background: fills `wiki_rag.metric_knowledge` with metric interpretations found in the wiki. |
| `analytic_orchestrator_v4` | Previous product: turn-based chat calling `json_analyzer_v5` and `easyrag` as subgraphs. Kept until `analyst_agent` replaces it in prod. |
| `json_analyzer_v5` | `gather → synthesize`. Metrics dataset → facts digest (not prose). |
| `easyrag` | `embed_query → retrieve → maybe_record_gap`. Alias lookup over `wiki_rag.wiki_alias` (exact match on a query word, then a per-alias vector) followed by pgvector search over `wiki_rag.wiki_section`; records unanswered queries as `query_gap`. |
| `wiki_ingest` | `load_pending → process → finalize`. Unprocessed `source_doc` → wiki pages/sections (chunk → LLM extract → resolve/merge → backlink → embed). |
| `doc_manager` | `classify → upload/list/delete`. Chat CRUD over `source_doc`. |
| `kb_chat` | `route → (retrieve →) respond`. Small-talk vs KB question via `easyrag`. |
| `gap_resolver` | `load_gaps → investigate → finalize`. Tries to answer `query_gap`s from source docs, creates stub pages. |
| `external_doc_loader` | `fetch_documents → load_documents → finalize`. Pulls docs from an external source (`shared/external_source.py`) into `source_doc`. |

Page aliases are a search surface of their own (`wiki_alias`, migration 0005): abbreviations dissolve inside a section vector, so each alias carries its own vector and an exact-match form. An alias hit expands to **all** sections of its page. The alias vector threshold is high on purpose (`DEFAULT_ALIAS_THRESH`): with the current embedding model any unrelated query scores ~0.78 against any alias, while a true match scores 0.96+. Chunks of source documents are **not** embedded (migration 0004) — `gap_resolver` judges them one by one.

All KB graphs share Postgres schema `wiki_rag` and isolate tenants by `direction_key` (state field or configurable). `easyrag/db.py` reuses aegra's `AsyncEngine` when running under aegra, else builds one from `POSTGRES_DSN` — never create a second engine.

### `analyst_agent` (current product)

Replaces `analytic_orchestrator_v4` + `json_analyzer_v5`; both stay registered until the switch. Full design: `docs/analyst_agent.md`.

- **One LLM loop per turn.** No router LLM, no separate fact-gathering subgraph, no synthesize stage: the model with the business prompt and tools gathers data and writes the final answer itself. `agent/loop.py` is a plain Python loop over `llm.bind_tools` living **inside one node** (aegra runs graphs with the default recursion limit; `langchain-gigachat` serialises only `tool_calls[0]`, so each tool call is one round trip).
- **Six tools** (`tools/`): `query_sql` (free read-only SELECT), `metric_card` (everything about one metric in one call), `peer_context`, `search_wiki`, `list_deviations`, `note_deviation`. Bound per flags.
- **Normalized SQLite** (`db/`): `person`, `metric`, `metric_edge`, `period`, `fact`, `fact_analytics`, `ranking`, `peer_aggregate`, `person_peer`, `deviation` + views the agent queries (`v_fact`, `v_fact_latest`, `v_tree`, `v_peer_latest`, `v_star`, …). `build_run_db()` returns a **ready** DB (data + derived fields + kind-based percent suppression) — callers must not call `compute_analytics` separately.
- **Dates never couple metrics**: "latest" is per series (person × metric × element, `fact.is_last_of_series`), the tree comes only from JSON structure. A child metric whose data lags the parent stays visible with its own date. Tests enforce this.
- **`analytics.py` is a line-by-line port of `json_analyzer_v5/analytics.py`**; `tests/analyst_agent/test_analytics_parity.py` compares all 24 columns against v5 on every `samples_v2` file — keep it green while v5 is in the tree (it builds the DB with `compute=False` to compare formulas only).
- **`.sql` templates** (`db/sql/*.sql`) with a machine-parsed header (`-- name/params/returns/render`) are shared by tools, prompt enrichment and `scripts/sql_debug.py`, so a prod query is reproducible verbatim.
- **Deviation map** (`deviations/`) is computed in code before the model runs: priority = impact × scale × controllability, lives in state for the whole dialog, materialized into the `deviation` table each turn. The main insight for `SendAssignmentsComponent` is picked from its top deterministically — no classify-insights LLM call.
- **Prompt** (`prompts/`): the business prompt is one variable, `BUSINESS_PROMPT` in `prompts/business.py` (role, principles, style, report structure); `compose_system_prompt` orders business prompt → tools guide (cross-cutting rules only; per-tool hints live in each tool's description) → data blocks → deviation map → memory/briefing → star wording (only when the dataset has stars) → dashboard task hint. The SQL schema doc (`db/schema_doc.py`) is **not** in the system prompt: it is the `query_sql` tool description, and its example queries go to GigaChat as the function's `few_shot_examples`. `system_prompt_override` replaces only the business prompt. The catalog block shows levels 1-2 and, for each level-2 metric, counters of deeper layers.
- **Metric knowledge** (`metric_knowledge/`, tables in migration `0003`): wiki interpretations cached per direction, lazily topped up on turn 1 (capped, timeboxed) and by the `metric_enricher` graph. Metric kind (уровень/вклад/индекс) is guessed from the name **always**, so ranks never get percentages even without Postgres.
- Contract with the client is v4's, verbatim (`contract/`): `orchestrator_step`/`orchestrator_final` tags, `answer_html`, `describe_answer` section, `reasoning_trace`. `format_first_answer` is accepted but does nothing.

### `analytic_orchestrator_v4` (nodes.py ~2000 lines, previous product)

- **Turn-based, no `interrupt()`**: one graph call = one incoming `{"messages": [...]}` → answer → END. State persists across turns via the checkpointer. `START → need_load` gate: first turn goes `load_data → load_memory → ground_wiki_initial → initial_analysis → auto_insight`; later turns go `route → {ground_wiki_analytics → call_json_analyzer | call_easyrag | save_insight | respond} → save_memory`.
- **State split** (`state.py`): `OrchestratorOutput` = public channels returned by run/stream; `OrchestratorState(OrchestratorOutput)` adds internal channels (`metrics` raw dataset, `aggregates`) that are checkpointed but never returned. Put new fields in the right class.
- **Message contract**: working nodes emit short step messages tagged `additional_kwargs.orchestrator_step`; the final answer is tagged `orchestrator_final` and is always last (`emit_progress_messages=false` disables steps). `reasoning_trace` is per-turn: the first node of a turn overwrites it, downstream nodes concatenate explicitly (no reducer).
- **Sticky context**: `briefing` (first message, kept for the whole dialog), `metrics_summary` (broad first-turn analysis, never overwritten), `analytics_answer` (latest narrow answer, overwritten per analytics turn).
- Data comes from `shared/agent_dataset.py` (`GetBatchAgentDatasetByFiltersComponent`) + `shared/orgstructure.py` filters; peer aggregates are per-person via `aggregates_ids`. Blocking data clients are called through `shared/offload.py` (dedicated thread pool + timeout) — use it for any sync external I/O from async nodes; never `asyncio.to_thread` them directly.
- Configurable keys: `boss_tabnum`, `employee_tabnum`, `position`, `dataset_name`, `direction_key`, `source_type`/`source_id` (both required for insights to be written), `run_mode` (`signal` → insight gets `author`/`confirmed`/`signal`/`signal_description`, shared logic in `shared/insight_signal.py`), `easyrag_enabled`, `easyrag_top_k`, `wiki_grounding_enabled`, `wiki_max_queries`, `use_peer_aggregates`, `gap_on_unanswered`, `emit_progress_messages`, `describe_answer`, `answer_html`, `system_prompt_override`.
- The orchestrator's `BUSINESS_SYSTEM_PROMPT` writes the prose; the analyzer only supplies facts+verdicts. Wording rules live in one place — the orchestrator.

### `json_analyzer_v5`

- Input shape is fixed (`loader.py` docstring is the spec): `{"me": person, "employees": [person]}`, metric tree with `fact/plan/benchmark/ex/rr/element/rankings/child_metrics`, optional `raw_aggregates` (peer-group batch stats by level), optional `wiki_context` text from the orchestrator. **Stars**: a node with non-empty `star_received` is a named award with no numbers; its `child_metrics` with `is_star_metric=true` are the influencing numeric metrics. Absent flag ≠ `False`.
- `gather` (async): load JSON → flatten into **in-memory SQLite** (`sqlite_store.py`) → deterministic derived analytics without LLM (`analytics.py`: plan/benchmark/pop deviations, trend, peer z-score/rank, anomalies) → LLM-derived per-direction caches in **LangGraph Store** (`store_cache.py` embeddings for semantic metric search, `relations_cache.py` metric-influence graph, `metric_kinds_cache.py` level/contribution/index classification that suppresses meaningless %) → tool-loop agent (`agent_classic.py`, `langchain.agents.create_agent`). Blocking work runs in `asyncio.to_thread`.
- **The agent never writes SQL** — it only calls typed tools in `tools.py`, which run parametric queries and render compact human-readable text. Loop guards: output cap, recursion limit, tool-call budget, repeated-call detection; all counters are per-invocation dicts (the graph is called concurrently).
- Store caches are keyed by `(direction_key, catalog_hash)` and the Store has no vector index — cosine search is done in memory. Bump `_VERSION`/namespace when changing embedding semantics or catalog composition.
- **Backward-compatibility rule enforced by tests**: when optional input fields (stars, aggregates, wiki_context, ex/rr) are absent, nothing changes — no new columns, prompt text, or tools (`test_nothing_changes_without_star_fields` etc.). New optional features must follow this.
- `docs/json_analyzer.md` is the long-form design doc for this graph and its orchestrator integration.

## Conventions

- Commits: `type(scope): описание` in Russian, scope = package name(s), e.g. `feat(json_analyzer_v5, analytic_orchestrator_v4): …`. Feature branches `feat/<topic>`, PRs into `main`.
- Design specs for larger changes go to `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`; `docs/tz_reasoning_trace_and_describe_answer.md` is the ТЗ behind `reasoning_trace`/`describe_answer`; `ideal_answers.md` is the business methodology the analysis targets.
- Sample datasets for manual/E2E runs: `samples_v2/*.json` (current format, incl. `sample_star.json`, `sample_declining_ex_rr.json`); `samples/` is the older format.
- `docs/analyst_agent.md` is the long-form design doc for the current agent; `docs/json_analyzer.md` describes the previous analyzer.
- `README.md` graph table and configurable list are stale (pre-v4/v5); this file and `aegra.json` are authoritative.
