# DailyPapers (`paperwatch`)

A local reading service that answers **"Which papers are worth my time today, and why?"**
It collects papers, has the local llama.cpp model (Qwen, `qwen-local` at `http://127.0.0.1:8080/v1`) review them,
and publishes one Markdown report per day. The full design is in [`Local_Paper_Monitor_Plan.md`](Local_Paper_Monitor_Plan.md).

**Status:** part 1 of 10 (application skeleton and contracts) is implemented.

## Project decisions (confirmed by the owner)

These answer open questions from the plan. Later build steps must follow them.

| Decision | Detail | Affects |
| --- | --- | --- |
| **Many topics, all active** | `topics.toml` will hold several `[[topics]]` entries. Each `question` is a full sentence. Every topic is accounted for in every run. Collection, screening, review slots, and report selection must cover all topics; none is a "primary" topic. | Parts 4, 6, 7, 9 |
| **Output stays local for now** | `output_dir = "output"` inside the paperwatch home. Once the pipeline is trusted, `output_dir` will point into an Obsidian vault. Keep note links relative and Obsidian-safe so the move is only a config change. `state/` never moves into the vault. | Parts 7, 9 |
| **Zotero via live API key** | Seeds come from the Zotero Web API with a read-only key, not a CSL JSON export. Per plan part 3, the key goes in a collector-only credentials file, never in `config.toml` or `topics.toml`. Support for CSL JSON import is optional. | Parts 3, 4 |

## Setup

```bash
uv sync                                   # Python 3.12 + httpx + jsonschema (+ pytest for dev)
uv run paperwatch --home ~/paperwatch init     # writes config.toml, topics.toml, state/, output/
$EDITOR ~/paperwatch/topics.toml               # add your research questions
uv run paperwatch --home ~/paperwatch doctor   # checks config, DB, and the llama.cpp server
```

`--home` defaults to `$PAPERWATCH_HOME`, then the current directory. Keep `state/` outside any synced notes folder;
`output_dir` in `config.toml` may point into a notes vault.

## Commands

| Command | What it does |
| --- | --- |
| `init` | Create the home layout and database; never overwrites existing config files. Safe to rerun (migrates the DB). |
| `doctor` | PASS/WARN/FAIL per component: config, paths, database, model health, alias, context budget, reasoning levels, a schema-constrained request with reasoning. |
| `collect [--fixture FILE]` | Import papers and queue review jobs. Only recorded fixtures for now (arXiv/OpenAlex/Zotero come in parts 4-5). |
| `work --once` / `work` | Run one job / loop until SIGTERM. Idle = zero inference calls. |
| `report [--date YYYY-MM-DD]` | Publish `output/daily/<date>.md` plus per-paper notes. One report per date; reruns produce identical bytes. |
| `status` | Topics, papers by status, jobs by kind/status, reviews by coverage, model pause state, last model call, last report. |

Failures always print `paperwatch: FAILED [<component>] ...` with a component-specific exit code
(2 config, 3 model-server unavailable, 4 database, 5 model call/output, 6 inference lock, 7 source, 8 output, 9 worker).

## How the model is used

- One fresh two-message context per job; the model gets plain data and **no tools**.
- Each role (`roles/`) has its own prompt, prompt version, JSON output schema, host-side checks, and hard
  time/token limits (`[roles.<name>]` in `config.toml`).
- Requests are schema-constrained (`response_format: json_schema`); the final answer is parsed separately from
  `reasoning_content`. Model identity, server build, and prompt version are stored with every call and review.
- One inference request at a time across all roles and processes (`state/inference.lock`).
- If the server is down, all model jobs pause together, health checks back off exponentially, the job goes back to the
  queue without using an attempt, and there is never a fallback to a hosted model.

## Layout

```text
src/paperwatch/
  cli.py, app_context.py        # argument parsing, shared config + DB + model client
  config.py, templates/         # typed config loading; default config.toml/topics.toml
  db.py                         # SQLite (WAL, foreign keys, busy timeout) + migrations
  catalog.py, review_inputs.py  # paper identity; what the reviewer reads and its input key
  jobs.py                       # queue with leases and lease-checked completion
  llm/                          # model client contract, llama.cpp client, fake client, lock, outage guard
  roles/                        # healthcheck + provisional abstract-only reviewer
  sources/                      # PaperSource protocol + recorded fixture source
  rendering/                    # Markdown escaping, notes, daily report, atomic writes
  commands/                     # one module per CLI command
tests/                          # CLI vertical slice (fake model + unavailable endpoint) and unit tests
```

## Tests

```bash
uv run pytest                               # offline, uses the fake model
PAPERWATCH_LIVE_MODEL=1 uv run pytest -m live_model   # also reviews the fixture with the real server
```
