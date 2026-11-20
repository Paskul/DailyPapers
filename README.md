# DailyPapers (`paperwatch`)

A local reading service that answers **"Which papers are worth my time today, and why?"**
It collects papers, has the local llama.cpp model (Qwen, `qwen-local` at `http://127.0.0.1:8080/v1`) review them,
and publishes one Markdown report per day. The full design is in [`Local_Paper_Monitor_Plan.md`](Local_Paper_Monitor_Plan.md).

**Status:** parts 1-4 of 10 are implemented: the application skeleton and contracts; paper identity, editions,
reading identity, and crash-safe jobs; the access limits on what can reach the internet; and the research profile
built from topics, Zotero, and CSL JSON seeds.

## Project decisions (confirmed by the owner)

These answer open questions from the plan. Later build steps must follow them.

| Decision | Detail | Affects |
| --- | --- | --- |
| **Many topics, all active** | `topics.toml` will hold several `[[topics]]` entries. Each `question` is a full sentence. Every topic is accounted for in every run. Collection, screening, review slots, and report selection must cover all topics; none is a "primary" topic. | Parts 4, 6, 7, 9 |
| **Output stays local for now** | `output_dir = "output"` inside the paperwatch home. Once the pipeline is trusted, `output_dir` will point into an Obsidian vault. Keep note links relative and Obsidian-safe so the move is only a config change. `state/` never moves into the vault. | Parts 7, 9 |
| **Zotero API and CSL JSON are equal seed sources** | A topic's seeds come from Zotero collections (live Web API, read-only key) and/or CSL JSON exports (`csl_json_files`). Both are first-class: same record type, same tables, same role in relevance, same fallback to the last good seed set. Per plan part 3, the Zotero key goes in the collector-only credentials file, never in `config.toml` or `topics.toml`. | Parts 3, 4, 6, 7 |
| **OS controls applied by the operator** | Service accounts, nftables egress rules, and the Podman extractor image ship as scripts in `deploy/`. The owner runs them with sudo. `paperwatch access-check` verifies them from each service account. | Parts 3, 10 |
| **llama-server stays a desktop process for now** | It keeps running as the desktop user. No model-account egress rule is loaded, because that UID is the desktop session. `access-check` reports this as a WARN. `deploy/nftables/paperwatch-model-egress.nft` is ready for a later move to its own account. | Parts 3, 10 |
| **Extractor = poppler `pdftotext`** | PDF text comes from `pdftotext` (reading order, form feeds between pages) in a Debian-slim image pinned by image id. XML (JATS/TEI) comes from a small stdlib dumper in the same image. | Parts 3, 5, 7 |

## Setup

```bash
uv sync                                   # Python 3.12 + httpx + jsonschema (+ pytest for dev)
uv run paperwatch --home ~/paperwatch init     # writes config.toml, topics.toml, state/, output/
$EDITOR ~/paperwatch/topics.toml               # add your research questions
uv run paperwatch --home ~/paperwatch doctor   # checks config, DB, and the llama.cpp server
```

**Your settings live in [`settings/`](settings/README.md)** (topics, config, Zotero/OpenAlex keys, CSL JSON seeds).
Real files there are gitignored; only `*.example.toml` is committed. After editing, run `deploy/apply_settings.sh`
to validate them and install them into the deployed home (`/srv/paperwatch`) with the right owners and permissions.

`--home` defaults to `$PAPERWATCH_HOME`, then the current directory. Keep `state/` outside any synced notes folder;
`output_dir` in `config.toml` may point into a notes vault.

## Commands

| Command | What it does |
| --- | --- |
| `init` | Create the home layout and database; never overwrites existing config files. Safe to rerun (migrates the DB). |
| `doctor` | PASS/WARN/FAIL per component: config, paths, database, model health, alias, context budget, reasoning levels, a schema-constrained request with reasoning. |
| `collect [--fixture FILE]` | Sync seed papers (Zotero collections, CSL JSON files), import papers, queue review jobs, and print each topic's input mode. Paper sources are recorded fixtures for now (arXiv/OpenAlex come in part 5). |
| `work --once` / `work` | Run one job / loop until SIGTERM. Idle = zero inference calls. Several workers may run at once. |
| `reread PAPER_ID [--reason TEXT]` | Operator request for a fresh reading; gets its own generation id and a separate review. |
| `report [--date YYYY-MM-DD]` | Publish `output/daily/<date>.md` plus per-paper notes. One report per date; reruns produce identical bytes. |
| `status` | Topics with their input mode (keywords / seeds / both) and what is missing, seed sources and their last sync, papers by status, editions/aliases, unresolved possible duplicates, jobs by kind/status (incl. expired leases), reviews by coverage, model pause state, last model call, outbound requests today and paused providers, last report. |
| `access-check --component collector\|worker` | Run as a service account. Verifies the account and group; the read-only provider rules; the private credentials; the output, state, WAL, and cache permissions; the egress probes; llama-server flags and account; and the extractor sandbox (smoke test). |

Failures always print `paperwatch: FAILED [<component>] ...` with a component-specific exit code
(2 config, 3 model-server unavailable, 4 database, 5 model call/output, 6 inference lock, 7 source, 8 output, 9 worker,
10 source-access: refused by the access rules, 11 extractor).

## How the model is used

- One fresh two-message context per job; the model gets plain data and **no tools**.
- Each role (`roles/`) has its own prompt, prompt version, JSON output schema, host-side checks, and hard
  time/token limits (`[roles.<name>]` in `config.toml`).
- Requests are schema-constrained (`response_format: json_schema`); the final answer is parsed separately from
  `reasoning_content`. Model identity, server build, and prompt version are stored with every call and review.
- One inference request at a time across all roles and processes (`state/inference.lock`).
- If the server is down, all model jobs pause together, health checks back off exponentially, the job goes back to the
  queue without using an attempt, and there is never a fallback to a hosted model.

## Paper identity and readings (plan part 2)

- **One paper per work.** DOIs, arXiv ids and OpenAlex ids are normalized (`scholarly_identifiers.py`); the arXiv
  version is kept apart from the id. Each normalized identifier maps to exactly one paper. A record joins an existing
  paper only through an identifier: its own, or one its provider states explicitly (`linked_identifiers`, e.g. an
  OpenAlex work listing its arXiv DOI). Matching titles and authors are flagged in `paper_possible_matches` and are
  never merged. Neither is a record that links two existing papers.
- **Editions.** Each published version (arXiv v1, v2, a journal version) is its own row with its own text snapshot.
  First publication, latest revision and local discovery time are stored separately; the paper's title is never
  rewritten.
- **A reading = paper + normalized text hash + generation.** A changed title, another route, a topic or scoring
  change, a model/prompt upgrade, a restart, or an empty queue never cause another reading. Changed text (or an
  abstract appearing for a title-only record) does. `reread` adds an explicit generation. Topic assessments are
  stored apart from the reading (`review_topic_assessments`), keyed by the topic-profile hash.
- **Crash-safe jobs.** Claims are short transactions with short leases (`worker.lease_seconds`), which a background
  thread renews during long calls. Results and `job done` commit together only if the lease still matches, so a
  stale worker can never overwrite a newer result. A crashed worker's job is reclaimed when its lease expires.
- **Records for later parts:** `citation_edges` and `comparisons` (`research_graph.py`, part 8),
  `source_checkpoints` (`source_checkpoints.py`, part 5), and `paper_seed_memberships` (part 4).

## Research profile (plan part 4)

The question is the unit of relevance, and every topic is defined by keywords, seeds, or both:

```toml
[[topics]]
id = "team-safety"
question = "Can a runtime monitor reject a joint robot plan before it blocks a human exit?"  # private; optional
public_queries = ["multi-robot runtime assurance"]        # may be sent to arXiv / OpenAlex
arxiv_categories = ["cs.RO", "cs.MA"]                     # may be sent; enough for a seeds-only topic
zotero_collections = ["ABCD2345"]                         # live Zotero seeds (key in the credentials file)
csl_json_files = ["seeds/team-safety.json"]               # offline CSL JSON seeds, equal standing
```

- **Modes** (`research_profile.py`): *keywords* (a question and/or public queries), *seeds* (active seeds, no
  question: the topic searches its categories and ranks results against its seeds locally in part 6), or *both*. If
  a seed source is empty or offline, the topic keeps working from its keywords. `status` explains a topic with no
  input, a topic with nothing to search, and title-only seeds.
- **Seeds** (`seeds/`): Zotero items and CSL JSON items become one `SeedItem` type. Only bibliographic fields are
  read. Notes, attachments, annotations, and trashed items are dropped. Free-text fields such as Zotero's Extra and
  CSL's `note` are used only for structured `DOI:` / `arXiv:` lines. A seed without an abstract is stored as
  `title_only`, a recorded limitation. Each seed keeps its id (`zotero:<item key>` or `csl_json:<file key>#<id>`),
  so a review can name the seed that matched.
- **Sync** (`seeds/seed_sync.py`): each source is read completely before anything is committed. Then the seed set,
  memberships, and checkpoint (`source_checkpoints`) commit in one transaction.
  - **Zotero:** sends `If-Modified-Since-Version`, so an unchanged library costs one request (HTTP 304). Every page
    must report the same library version; if the library changes mid-sync, the sync is retried once, then left for
    the next run. Deleted, trashed, and moved items drop out of the full listing and are marked removed.
  - **CSL JSON:** a file is re-imported only when its SHA-256 changes.
  - **Failures:** a missing key, an outage, or a missing or invalid file marks the source blocked, and its last
    successful seed set stays in use.
- **Seeds in the catalog:** every seed is also a catalog paper (route `zotero-seed:` / `csl-seed:`), so a seed and
  the same work found on arXiv are one paper. Active seed papers are never queued for review; they serve as
  comparison anchors (parts 7 and 9).
- **What may leave the machine:** only the approved public queries and categories, and for Zotero only the collection
  key, library id, and paging values. Seed titles, notes, collection names, and the question never become a request.
  Looking up a work by an identifier known *only* from the seed library (e.g. a seed's DOI) discloses interest in it,
  so `get_work` refuses unless `[collector] allow_lookup_by_seed_identifier = true`.
- **Profile hash:** a topic assessment's profile hash now covers the active seeds' metadata too. A changed seed set
  makes old assessments visibly stale; the reading is reused. A topic without seeds keeps its previous hash.

## Access limits (plan part 3)

Only the collector can contact paper providers, and only through fixed operations. No text from a paper or a
model answer can turn into a request, a command, a filename, or a link.

- **Fixed operations** (`network/provider_gateway.py`): `search(topic_query_id, cursor)`, `get_work(provider_id)`,
  `get_text(known_edition_id)`, `list_seed_items(collection_key, start)`. Code builds every URL from validated ids
  (`network/provider_requests.py`):
  - A search takes only an id from the **approved public query catalog** (`public_query_catalog.py`). The catalog is
    built from `public_queries` and `arxiv_categories` in topics.toml, so the private question never leaves the
    machine.
  - Ids are **rechecked against source records at dispatch**. `get_work` needs a catalog alias or a citation that a
    provider supplied. `get_text` needs a catalogued edition whose identifier still maps to its paper. The worker
    likewise rechecks a queued edition before reading it.
- **Read-only provider rules** (`network/provider_rules.toml`, shipped in the package and root-owned once installed).
  Each provider operation gets one host, one path pattern, allowed parameter names, allowed redirect hosts
  (none by default), response bounds, a daily request budget, minimum spacing (3 s for arXiv), and where its
  credential goes. config.toml cannot widen any of this.
- **One secure HTTPS hop at a time** (`network/secure_fetcher.py`):
  - strict TLS (system CAs, TLS 1.2 or newer); no proxies from the environment; no automatic redirects;
  - the host must resolve only to public addresses, and the connected address is checked again;
  - bounded wire bytes, decompressed bytes (gzip bombs stop at the limit), and time; content types must match.
  - A redirect is followed only to a listed host, over HTTPS, and credentials are dropped on any host change.
- **Ledger** (`network/outbound_ledger.py`): every hop is logged to `outbound_requests` with its outcome, and
  credentials are redacted. Budgets and spacing are shared across processes. A 429, a 5xx, or a refused key pauses
  that provider (`service_state`, `provider:<name>`), and a Retry-After lengthens the pause.
- **Credentials** live in one collector-only file, `credentials/collector.toml` (mode 0600, owned by the collector).
  Only `collect` loads it. Secrets never appear in `repr`, logs, or the ledger.
- **Egress guard in every process** (`network/egress_guard.py`, a PEP 578 audit hook). It is defence in depth
  behind nftables:
  - `collect` may resolve only the rule hosts, connect only to public addresses on port 443, and launch only the
    container runtime.
  - `work`, `report`, `reread`, `status`, and `doctor` may connect only to the local model port, and may launch no
    programs.
  - `init` may use no network at all.
  - `access-check` arms nothing, because it must try forbidden connections.
- **The model has zero tools.** Requests never carry `tools` or `functions`. Each answer is schema-checked, and
  unknown fields such as an invented `actions` list are rejected. Topic ids outside the input and web addresses
  absent from the input (invented references) are rejected too. `doctor` fails if llama-server exposes built-in
  tools or the MCP proxy. Untrusted text in notes is escaped, and web addresses are defanged (`https[:]//…`), so
  they are never clickable.
- **Sandboxed extractor** (`sandboxed_extractor.py`, image in `deploy/extractor/`). Rootless Podman runs with
  `--network=none`, `--read-only`, `--cap-drop=all`, and no-new-privileges. It gets one read-only input, which must
  be a cached file named by internal ids, plus bounded tmpfs, memory, CPU, and processes. A timeout kills the
  container. The image is pinned by id and never pulled at run time.
- **MCP option:** not built. The first release uses direct Python functions, as the plan allows.

### Deploying the access limits (operator, with sudo)

```bash
sudo env "PATH=$PATH" deploy/install_application.sh          # root-owned /opt/paperwatch (read-only rules)
sudo deploy/setup_service_accounts.sh /srv/paperwatch          # paperwatch-collector, -worker, group -state
deploy/build_extractor_image.sh                                # pinned pdftotext image -> collector's store
#   then set [extractor] image = "<printed id>" in /srv/paperwatch/config.toml
sudo deploy/apply_egress_rules.sh                              # nftables: worker -> 127.0.0.1:8080 only,
                                                               #   collector -> DNS + public HTTPS only; enables
                                                               #   paperwatch-egress.service to reload at boot
sudo deploy/check_access_as_service_accounts.sh /srv/paperwatch   # access-check as each account
```

Do not persist the rules through the stock `nftables.service`: its `/etc/nftables.conf` begins with `flush ruleset`,
which would also wipe UFW's rules. `paperwatch-egress.service` loads only the paperwatch tables.

Run llama-server without `--tools`, `--agent`, `--mcp-servers-*`, `--ui-mcp-proxy`, `-hf`, or `--model-url`.
Add `--offline` so it can never download at run time.

## Layout

```text
src/paperwatch/
  cli.py, app_context.py        # argument parsing, shared config + DB + model client
  config.py, templates/         # typed config loading; default config.toml/topics.toml
  db.py, db_migrations.py       # SQLite (WAL, foreign keys, busy timeout); ordered schema migrations
  scholarly_identifiers.py      # DOI / arXiv / OpenAlex normalization, edition keys
  catalog.py, possible_matches.py # canonical papers, aliases, editions; flag-only duplicate detection
  review_inputs.py, readings.py # reading packet + key; when a (re)reading is queued
  review_store.py               # immutable reviews + separate topic assessments
  jobs.py, lease_keeper.py      # queue with short leases, renewal thread, lease-checked completion
  research_profile.py           # per-topic input mode (keywords / seeds / both), active seeds, profile hash
  seeds/                        # Zotero + CSL JSON seed parsers, seed cache, complete-snapshot sync
  research_graph.py             # citation edges and comparison keys (used from part 8)
  source_checkpoints.py         # per-search cursor, budget, coverage (used from part 5)
  public_query_catalog.py       # approved public queries with stable ids: the only searchable terms
  source_cache.py               # downloaded source bytes, named by internal ids and hashes only
  sandboxed_extractor.py        # rootless Podman runner for the pinned pdftotext/XML extractor
  network/                      # collector only: provider rules, gateway, secure fetcher, ledger, credentials,
                                #   and the per-process egress guard
  llm/                          # model client contract, llama.cpp client, fake client, lock, outage guard
  roles/                        # healthcheck + provisional abstract-only reviewer; output guards
  sources/                      # PaperSource protocol + recorded fixture source
  rendering/                    # Markdown escaping, notes, daily report, atomic writes
  commands/                     # one module per CLI command
settings/                       # operator settings (gitignored) + committed *.example.toml; see settings/README.md
deploy/                         # install, service accounts, nftables egress rules, extractor image, access checks,
                                #   apply_settings.sh (install settings/ into the paperwatch home)
tests/                          # step 1 vertical slice; step 2 identity/readings and worker crash/race tests;
                                #   step 3 prompt injection, provider gateway, sandbox and egress tests;
                                #   step 4 input modes, Zotero/CSL seed sync, outbound-parameter checks
```

## Tests

```bash
uv run pytest                               # offline, uses the fake model
PAPERWATCH_LIVE_MODEL=1 uv run pytest -m live_model   # also reviews the fixture with the real server
PAPERWATCH_TEST_EXTRACTOR_IMAGE=<image id> uv run pytest -k real_podman   # real sandbox (needs podman + image)
```
