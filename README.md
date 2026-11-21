# DailyPapers (`paperwatch`)
# This is a total WIP and completely vibe coded. This is traditionally for my own use. I do not take any responsibility for anything bad happening in a world where you use this.

**A private, local research assistant that answers one question every morning: _which papers are worth my time
today, and why?_**

`paperwatch` watches arXiv and OpenAlex for papers related to your research questions, screens them on your CPU,
has a **local LLM** (any model served by [llama.cpp](https://github.com/ggml-org/llama.cpp)) read the promising ones
closely, and publishes **one Markdown report per day**. Every claim in a review or report is backed by a quoted,
checkable passage from the paper.

Nothing is sent to a hosted AI service. Your research questions never leave your machine, and the model never gets
tools, network access, or the ability to influence which requests are made.

## Features

- **Topics defined your way**: by a private research question, by public search terms, by a set of seed papers from
  your Zotero library or a CSL JSON export, or any combination. Several topics can be active at once.
- **Current and older work**: daily arXiv feeds and recent OpenAlex results, plus a bounded historical search that
  leans toward recent work but still surfaces older influential papers. The arXiv feeds share a capped part of the
  daily candidate budget fairly across categories, so one broad category (e.g. cs.AI) cannot crowd out the others or
  the keyword searches.
- **Cheap screening first**: a pinned sentence-embedding model (BAAI/bge-base-en-v1.5 on ONNX Runtime) ranks every
  candidate on the CPU, and only the day's best (30 by default, shared fairly across topics) go to the LLM.
- **Evidence-backed reading**: the model reads the full text in bounded sections, and every finding must quote the
  passage it comes from. Invented numbers, unknown passages, and unsupported claims are rejected. When only a few
  items of an answer fail, the host fixes a quote cited from the wrong passage (only when exactly one passage holds
  it) and drops the items that still fail, instead of discarding the whole answer. Whatever is kept passes every
  check.
- **A transparent score**: each paper gets a 0-100 reading priority per topic, computed by code from the model's
  explained ratings (fit to the question, fit of method, strength of evidence, value relative to papers you already
  have).
- **Follows the trail**: important references of relevant papers are read too, and papers that address the same
  question are compared side by side.
- **Notes you keep**: each review becomes a Markdown note, and the daily report links to them with relative links,
  so the output folder works as (or inside) an Obsidian vault.
- **Built to run unattended**: crash-safe job queue, resumable collection, model outages handled by pausing (never by
  falling back to a hosted model), and a fallback report if the writer cannot finish in time. Stopping the worker
  (Ctrl+C, SIGTERM) aborts a running model call at once and puts its job back in the queue without using an attempt.
  Reports and notes are readable by the operator through the shared `paperwatch-state` group.

## How it works

```text
 arXiv / OpenAlex / Zotero          your CPU                   local llama.cpp server         output/
 ─────────────────────────   ─────────────────────────   ─────────────────────────────   ─────────────────
  collect ──► catalog ──────► screen & promote ─────────► read · score · compare ─────► papers/<id>/*.md
  (fixed, rate-limited        (embeddings, fair           (evidence checked by code)      comparisons/*.md
   requests only)              daily slots per topic)                                    daily/<date>.md
```

1. **`collect`** syncs your seed papers, polls arXiv feeds, searches OpenAlex, and catalogs everything it finds. Each
   work is stored once, however many routes report it, and its versions are kept apart. Full text is fetched for
   the papers chosen for reading and extracted inside a network-less container.
2. **Screening** (part of `collect`) scores every new candidate against every topic and promotes the day's best to
   the reading queue.
3. **`work`** runs the queue: it reads each paper, validates every citation, scores it for every topic, writes its
   note, picks references worth following, and compares related reviews. When the queue is empty it makes no model
   calls.
4. **`report`** freezes the reviews completed before the daily cutoff (07:00 by default), has the model write an
   overview whose every claim is checked against the cited passages, and publishes `output/daily/<date>.md`.
   Rerunning it produces the same file.

All state lives in one SQLite database (`state/`). Generated notes and reports go to `output/`.

## Requirements

- Linux and Python 3.12+, with [uv](https://docs.astral.sh/uv/)
- A [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` on `127.0.0.1` serving a model that supports
  JSON-schema-constrained output and a context of at least 32k tokens per slot (a reasoning model such as Qwen works
  well)
- About 450 MB of disk for the screening model
- Optional: **Podman** for PDF text extraction (without it, papers are reviewed from their abstracts)
- Optional: a **Zotero** API key (read-only) for seed collections, and a free **OpenAlex** API key for full search
  paging (without one, only the first page of each search is used)
- Optional, for a hardened deployment: systemd, nftables, and sudo access

## Quick start

Everything is set up from **one settings file**, `settings/paperwatch.toml`, by **one script**, `./setup.sh`:

```bash
git clone <this repository> DailyPapers && cd DailyPapers

./setup.sh                       # 1. first run: creates settings/paperwatch.toml from the example, then stops
$EDITOR settings/paperwatch.toml # 2. set [setup] home and mode, and describe your [[topics]]
./setup.sh                       # 3. installs everything the file describes, then checks it with `doctor`
```

`./setup.sh` is safe to re-run after any change. `./setup.sh --check` validates the file and changes nothing;
`./setup.sh --settings-only` re-applies only the settings (the quick path after editing topics).

The `[setup]` table at the top of the file decides how paperwatch is installed:

```toml
[setup]
home = "/srv/paperwatch"          # database, notes, reports, models
mode = "hardened"                 # "hardened": /opt install, service accounts, firewall, sandboxed services (sudo)
                                  # "local": everything runs as you from this repository, no sudo
credentials_file = "collector_credentials.toml"   # API keys stay in their own private file (mode 0600)
install_embedding_model = true    # the pinned screening model (about 450 MB, hash-checked)
build_extractor_image = true      # the sandboxed PDF extractor (Podman); its id is filled in automatically
daily_schedule = false            # true: collect, work, and report every day by themselves (systemd timers)
monitor_service = false           # true: keep the monitor page running (see "Monitor" below)
monitor_port = 8765
monitor_listen_address = "127.0.0.1"
```

Every other table is the configuration (daily limits, report time and timezone, model roles, budgets), each option
with its default and an explanation, followed by your `[[topics]]`. The model server is the one thing setup does
not start; run it yourself:

```bash
llama-server -m <your-model>.gguf --alias qwen-local --host 127.0.0.1 --port 8080 -c 32768 --offline
```

If your server uses another address or model name, change `[model] base_url` and `model_alias`. Run `llama-server`
without `--tools`, `--agent`, `--mcp-servers-*`, `--ui-mcp-proxy`, `-hf`, or `--model-url`; `doctor` fails if the
server exposes tools.

Without the daily schedule, run the cycle by hand (`<paperwatch>` is `/opt/paperwatch/venv/bin/paperwatch` in
hardened mode, `.venv/bin/paperwatch` in local mode; setup prints the exact commands):

```bash
<paperwatch> --home <home> collect
<paperwatch> --home <home> work            # runs until stopped (Ctrl+C); use --once for one job
<paperwatch> --home <home> report --early  # publish today's report now instead of at the cutoff
```

The report appears in `<home>/output/daily/`.

## Configuring topics

Each research interest is one `[[topics]]` block in `settings/paperwatch.toml`:

```toml
[[topics]]
id = "team-safety"
question = "Can a runtime monitor reject a joint robot plan before it blocks a human exit?"  # private, optional
public_queries = ["multi-robot runtime assurance"]   # may be sent to arXiv / OpenAlex
arxiv_categories = ["cs.RO", "cs.MA"]                # categories to watch
zotero_collections = ["ABCD2345"]                    # seed papers from Zotero (key in the credentials file)
csl_json_files = ["seeds/team-safety.json"]          # seed papers from a CSL JSON export, kept in settings/seeds/
```

- The **question** is only ever shown to the local model. Only `public_queries` and `arxiv_categories` are sent to
  paper providers.
- A topic needs keywords (a question or public queries), seeds, or both. A seeds-only topic watches its categories
  and ranks what it finds against your seed papers.
- Seed papers are never recommended back to you; they serve as reference points for "is this new to me?".
- To use a PDF you already have, drop it into `<home>/local_sources/` named after its arXiv id or DOI
  (`2401.01234v2.pdf`, `10.1109_xyz.pdf`). It is used instead of a download.

API keys go in `settings/collector_credentials.toml` (mode 0600; start from
[`settings/collector_credentials.example.toml`](settings/collector_credentials.example.toml)). Setup installs them
where only the collector can read them. See [`settings/README.md`](settings/README.md) for what goes where.

## Monitor

`paperwatch monitor` serves a live, read-only web page of what paperwatch is doing, has done, and will do:

- **Now**: the worker's current job, down to the reading stage and the model call in progress (e.g. "Section 3 of 7 ·
  Model call: section_reader (repair) · running 6m 12s"), the model server's health, the collector, and the
  countdown to the next report.
- **Timeline**: collect runs, jobs, every model call (by role; rejected answers hatched), and reports over the last
  24 hours or more (back into the previous day), with the rest of the schedule up to the next report to the right of
  the "now" line. Hover any bar for details.
- **Plan and queue**: the scheduled collections and report, how many queued readings fit before the report (from the
  measured time per reading), and the queue in the order the worker takes it.
- **Events, totals, and limits**: readings finished (with their scores), rejected model answers, failed jobs,
  collect runs, reports; per-day totals; today's budgets against their limits.

```bash
<paperwatch> --home <home> monitor               # http://127.0.0.1:8765
<paperwatch> --home <home> monitor --json        # one snapshot as JSON, no server
```

Run it as yourself. It opens the database read-only, connects nowhere, and listens only on loopback or this
machine's Tailscale address. To watch from another machine, tunnel the port over SSH:

```bash
ssh -L 8765:127.0.0.1:8765 <this host>           # then open http://localhost:8765 on that machine
```

Or, on a Tailscale network, serve it on this machine's Tailscale address (`[setup] monitor_listen_address`, or
`--listen-address $(tailscale ip -4)`) and open `http://<tailscale address>:8765` from any device on the tailnet.
With `[setup] monitor_service = true`, setup keeps it running as your user service.

## Commands

| Command | What it does |
| --- | --- |
| `init` | Create the home directory and database. Never overwrites existing files; safe to rerun after an upgrade. |
| `doctor` | Check the configuration, database, screening model, and model server (PASS / WARN / FAIL). |
| `collect` | Sync seeds, collect new and older papers, follow references, screen, and queue the day's readings. |
| `work [--once]` | Process queued readings, reference choices, and comparisons. Several workers can run at once. |
| `report [--date D] [--early]` | Publish the daily report for a date (default: the latest date whose cutoff has passed). |
| `reread PAPER_ID` | Request a fresh reading of a paper. |
| `retry-failed [--component C] [--job ID] [--dry-run]` | Put failed jobs back in the queue with fresh attempts, keeping their saved reading stages (default: failures of the model's answers). |
| `status` | Overview of topics, seed sources, papers, queue, searches, provider budgets, and the last report. |
| `capacity` | Measured model time per role and per paper, and an estimate of how many readings fit in a day. |
| `backup DIR` / `backup --check-restore DIR` | Write a consistent snapshot (no credentials, no raw PDFs) / verify that a copy restores. |
| `label-sample OUT.csv` | A shuffled sample of screened candidates without scores, for checking ranking quality by hand. |
| `monitor [--port P] [--json]` | Live, read-only web page: current step, timeline, plan, queue, events (see Monitor). |
| `access-check --component collector\|worker` | Verify the hardened deployment from a service account. |

Run `paperwatch <command> --help` for every option. Failures print `paperwatch: FAILED [<component>] ...` and exit
with a component-specific code (2 config, 3 model server unavailable, 4 database, 5 model output, 7 source, ...).

## Security and privacy

Papers are untrusted input: a PDF can contain text written to manipulate an AI. `paperwatch` is designed so that
even a fully compromised model cannot make it reach the network, run programs, or write outside its folders.

- **The model has no tools.** It receives plain data in a fresh context per call and must answer in a fixed JSON
  schema. Unknown fields, invented ids, and invented links are rejected.
- **Requests are built by code, not by the model.** The collector can only call a small set of fixed operations
  (search an approved query, poll a category, fetch a catalogued work). A model can only choose among ids it was
  offered, and each id is checked again before use.
- **Only the collector talks to the internet**, over HTTPS to an allow-list of hosts with per-provider rate limits and
  budgets. Every request is logged with credentials redacted. Other commands can reach only the local model port.
- **PDFs are extracted in a sandbox**: a pinned, rootless Podman container with no network, a read-only filesystem,
  and no capabilities.
- **Untrusted text is escaped** in every note and report, and links from papers are made non-clickable.

The [hardened deployment](#hardened-deployment) adds OS-level enforcement: separate service accounts and
nftables egress rules. A test in the suite plays a whole day with a hostile paper and a hostile model and checks that
none of the above can be bypassed.

## Hardened deployment

`mode = "hardened"` (the default) installs `paperwatch` read-only under `/opt/paperwatch` with the home in
`/srv/paperwatch`, separate collector and worker accounts, nftables egress rules (worker -> local model only,
collector -> DNS + public HTTPS only), and sandboxed systemd services. `./setup.sh` runs these steps, each of which
skips work that is already done, and asks for sudo where a step needs it:

| Step | Script |
| --- | --- |
| install the application read-only in `/opt/paperwatch` | `deploy/install_application.sh` |
| create the service accounts, the shared group, and permissions | `deploy/setup_service_accounts.sh` |
| install the pinned screening model | `deploy/install_embedding_model.sh` |
| build the PDF extractor image if it is missing | `deploy/build_extractor_image.sh` |
| install `config.toml`, `topics.toml`, keys, and seed files with the right owners | (setup itself) |
| create or migrate the database | `paperwatch init`, as the worker account |
| load the egress rules and keep them across reboots | `deploy/apply_egress_rules.sh` |
| verify the access limits from both accounts, then the model server | `deploy/check_access_as_service_accounts.sh`, `doctor` |
| daily schedule and monitor service on or off | `paperwatch.deployment.daily_schedule` |

The egress rules are loaded by their own `paperwatch-egress.service`; don't persist them through the stock
`nftables.service`, whose config begins with `flush ruleset` and would also remove other firewall rules (such as
UFW's).

With `daily_schedule = true`, `paperwatch-collect.timer` collects at `[schedule] collect_times`,
`paperwatch-worker.service` reads continuously, and `paperwatch-report.timer` publishes at `[report] publish_time`
(the report stops the worker while the writer runs and starts it again afterwards). Logs:
`journalctl -u 'paperwatch-*' -f` (keys and paper text redacted).

### Trial run

To try the full cycle unattended, `deploy/pilot/` installs a **one-day run** with one-shot systemd timers: it collects
now and at the configured collection times, runs the worker until the cutoff, publishes the next morning's report,
and then removes all of its units. It can never run a second time.

```bash
sudo env "PATH=$PATH" deploy/pilot/install_one_day_pilot.sh   # start
paperwatch --home /srv/paperwatch status                      # follow progress
journalctl -u 'paperwatch-pilot-*' -f                         # live logs (keys and paper text redacted)
sudo deploy/pilot/stop_one_day_pilot.sh                       # end early
```

## Project layout

```text
src/paperwatch/
  commands/       one module per CLI command
  collection/     arXiv feeds, OpenAlex searches, historical search, full-text acquisition
  seeds/          Zotero and CSL JSON seed sync
  screening/      embedding model, scoring, and daily promotion
  reading/        passages, reading plan, evidence checks, reading-priority rubric
  references/     reference fetching and selection
  comparisons/    comparisons of reviews on the same question
  reporting/      daily cutoff, report selection, writer synthesis and fallback
  rendering/      Markdown notes and reports (escaping, atomic writes)
  monitor/        the read-only monitor page: snapshot, schedule, web server, static page
  roles/          prompt, output schema, and checks for each model role
  llm/            llama.cpp client, inference lock, outage handling
  network/        provider rules, secure fetcher, request ledger, egress guard
  deployment/     settings-file split, daily schedule and pilot systemd units
  templates/      default config.toml and topics.toml
setup.sh          one-step install / update from settings/paperwatch.toml
deploy/           the steps setup.sh runs: installation, accounts, firewall, extractor image, trial run
settings/         your private settings file and keys (gitignored) and examples
tests/            offline test suite
```

## Development

```bash
uv run pytest      # offline: uses a fake model and a lightweight test embedder
```

Optional tests against real components:

```bash
PAPERWATCH_LIVE_MODEL=1 uv run pytest -m live_model                                       # real llama.cpp server
PAPERWATCH_TEST_EXTRACTOR_IMAGE=<image id> uv run pytest -k real_podman                   # real extractor sandbox
PAPERWATCH_TEST_EMBEDDING_MODEL_DIR=<home>/models/bge-base-en-v1.5 uv run pytest -k real_onnx   # real screening model
```

## License

[MIT](LICENSE)
