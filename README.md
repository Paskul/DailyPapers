# DailyPapers (`paperwatch`)

A local reading service that answers **"Which papers are worth my time today, and why?"**
It collects papers, has the local llama.cpp model (Qwen, `qwen-local` at `http://127.0.0.1:8080/v1`) review them,
and publishes one Markdown report per day. The full design is in [`Local_Paper_Monitor_Plan.md`](Local_Paper_Monitor_Plan.md).

**Status:** parts 1-9 of 10 are implemented, and part 10 is under way (a one-day pilot; see below): the application skeleton and contracts; paper identity, editions,
reading identity, and crash-safe jobs; the access limits on what can reach the internet; the research profile
built from topics, Zotero, and CSL JSON seeds; collection of current papers plus a finite historical search; cheap
CPU screening of every candidate with ordered dispatch of the readings; and evidence-backed reading with
passage-level citations, a 0-100 reading-priority rubric per topic, and a permanent note per review; and following
the important references of reviewed papers plus comparisons of reviews that address the same question; and one daily Markdown report.

## Project decisions (confirmed by the owner)

These answer open questions from the plan. Later build steps must follow them.

| Decision | Detail | Affects |
| --- | --- | --- |
| **Many topics, all active** | `topics.toml` will hold several `[[topics]]` entries. Each `question` is a full sentence. Every topic is accounted for in every run. Collection, screening, review slots, and report selection must cover all topics; none is a "primary" topic. | Parts 4, 6, 7, 9 |
| **Output stays local for now** | `output_dir = "output"` inside the paperwatch home. Once the pipeline is trusted, `output_dir` will point into an Obsidian vault. Keep note links relative and Obsidian-safe so the move is only a config change. `state/` never moves into the vault. | Parts 7, 9 |
| **Zotero API and CSL JSON are equal seed sources** | A topic's seeds come from Zotero collections (live Web API, read-only key) and/or CSL JSON exports (`csl_json_files`). Both are first-class: same record type, same tables, same role in relevance, same fallback to the last good seed set. Per plan part 3, the Zotero key goes in the collector-only credentials file, never in `config.toml` or `topics.toml`. | Parts 3, 4, 6, 7 |
| **OS controls applied by the operator** | Service accounts, nftables egress rules, and the Podman extractor image ship as scripts in `deploy/`. The owner runs them with sudo. `paperwatch access-check` verifies them from each service account. | Parts 3, 10 |
| **llama-server stays a desktop process for now** | It keeps running as the desktop user. No model-account egress rule is loaded, because that UID is the desktop session. `access-check` reports this as a WARN. `deploy/nftables/paperwatch-model-egress.nft` is ready for a later move to its own account. | Parts 3, 10 |
| **Local drop folder for full text** | `collect` looks in `<home>/local_sources/` (`collection.local_sources_dir`) before downloading. Files are named by arXiv id or DOI (`2401.01234v2.pdf`, `10.1109_xyz.pdf`). | Part 5 |
| **Full text: download and extract, capped** | `collect` downloads full text for the papers screening promoted, at most `daily_source_text_budget` (30) per day, and extracts it in the sandbox when the image is set. | Parts 5, 6, 7 |
| **Historical search weighted toward recent work, back to 1940** | Candidates per publication year halve every 6 years (`historical_half_life_years`), down to 1940 (`historical_earliest_year`). The curve is cut into 8 windows of equal weight (about 1.2 years wide for the newest, 1940-2008 for the oldest); each window gets an equal share, and the least-served window is paged next. Windows at least 5 years old take the most-cited papers with the exact phrase in title or abstract, so old influential papers still appear. Replaces fixed 5-year bands (owner request, 2026-11-19). | Part 5 |
| **OpenAlex works without a key, but less** | Without a free key, OpenAlex serves only the first page of each search, and every client on the IP address shares $0.10/day. paperwatch then requests only first pages and budgets 80 requests/day. A key in the credentials file unlocks every page (900/day). | Parts 3, 5 |
| **Screening model = BAAI/bge-base-en-v1.5 on ONNX** | One pinned sentence-embedding model (768 dimensions), run in-process by onnxruntime on the CPU (no torch, no extra service or socket). Its files are pinned by revision and sha256 in the package and installed by `deploy/install_embedding_model.sh`; nothing is downloaded at run time. | Parts 6, 10 |
| **No minimum screening score yet** | `minimum_screening_score = 0`: the 30 daily slots go by rank, divided across topics. Embedding cosines depend on the model (bge rates unrelated texts around 0.2-0.4), so part 10 calibrates a real floor from labeled samples. | Parts 6, 10 |
| **Signals ranked within each topic before combining** | Each signal becomes its percentile among the topic's unpromoted candidates, so the 0/1 phrase signal cannot outweigh the cosines and one topic's higher raw cosines do not win the spare and near-cutoff slots. The minimum score is therefore a relative position. Owner request after review, 2026-11-19. | Parts 6, 10 |
| **Every topic is assessed** | Each completed reading is rated against every topic that has a question, terms, or seeds (topics with no input are skipped), in one topic-assessment call. Owner decision, 2026-11-19. | Parts 7, 9 |
| **Seeds and earlier reviews are comparison papers** | The rubric's "value relative to nearby saved papers" uses the topic's closest seeds from your library AND earlier reviews on the same question. It is omitted ("comparison unavailable") only when a topic has neither. Owner decision, 2026-11-19. | Parts 7, 9 |
| **Deep reading at the server's default reasoning** | Up to 10 reading sections per paper; every reading call (section reader, reviewer, topic assessor) uses the server's default reasoning effort (xhigh). Owner decision, 2026-11-19; measure in part 10. | Parts 7, 10 |
| **Reference readings have their own daily limit** | Up to 20 reference readings per UTC day (`references.daily_reference_readings`), on top of screening's 30. They sit in priority class 3, so they run only when no new release or older paper is waiting. Owner decision, 2026-11-20. | Parts 8, 10 |
| **Reference readings wait for their full text** | A chosen reference is held up to 6 h (`references.reference_reading_hold_minutes`) so the next `collect` can fetch its PDF from the same 30/day download budget; it is read from the abstract only when no full text exists. Owner decision, 2026-11-20. | Parts 5, 8 |
| **Comparisons get their own note** | Each comparison is written once to `output/comparisons/<comparison id>.md`, linking both reading notes relatively; `report` restores a missing one. Part 9 links to it. Owner decision, 2026-11-20. | Parts 8, 9 |
| **References are followed for relevant reviews only** | A reviewed paper's references are fetched only if its review rated question_fit >= 2 ("addresses part of the question") for some topic, the same bar as comparison anchors. Owner decision, 2026-11-20. | Part 8 |
| **Report writer at medium reasoning, 30-minute deadline** | The writer role uses reasoning `medium` (`[roles.writer]`). Getting the inference slot, the call, its one repair, and validation must finish within `report.writer_deadline_minutes` (30), or the report is rendered from the saved summaries and labelled as a fallback. Owner decision, 2026-11-20. | Parts 9, 10 |
| **Updates are decided by a deterministic rule** | A reported paper comes back as an "update" only if its newer review read different text AND its numeric results, its supported assumptions, its insufficient-evidence verdict, or (by at least 10 points) its best score changed. No model call; a purely verbal change is not detected. Owner decision, 2026-11-20. | Part 9 |
| **Pilot timers run once** | Plan part 10's 24-hour pilot uses one-shot systemd timers with absolute dates (collections the evening and night of the pilot day, the report at 07:00 the next morning). They can never run again: absolute times only, an expiry condition in every service, no unit enabled for boot, and a finish step that deletes every pilot unit. No recurring timers are installed until the owner reviews the pilot. Owner decision, 2026-11-20. | Part 10 |
| **Section reader at medium reasoning** | `[roles.section_reader] reasoning_effort = "medium"` so long papers cost less; the reviewer synthesis and the topic assessor stay at the server default (xhigh). Owner decision, 2026-11-20 (pilot). | Parts 7, 10 |
| **llama-server runs with `--offline`** | The model server (the owner's user unit `qwen.service`) never downloads or fetches anything at run time. It still runs as the desktop user. Owner decision, 2026-11-20. | Parts 3, 10 |
| **Extractor = poppler `pdftotext`** | PDF text comes from `pdftotext` (reading order, form feeds between pages) in a Debian-slim image pinned by image id. XML (JATS/TEI) comes from a small stdlib dumper in the same image. | Parts 3, 5, 7 |

## Setup

```bash
uv sync                                   # Python 3.12 + httpx + jsonschema + numpy/onnxruntime/tokenizers
uv run paperwatch --home ~/paperwatch init     # writes config.toml, topics.toml, state/, output/
deploy/install_embedding_model.sh ~/paperwatch # pinned screening model (~440 MB) -> ~/paperwatch/models/
$EDITOR ~/paperwatch/topics.toml               # add your research questions
uv run paperwatch --home ~/paperwatch doctor   # checks config, DB, screening model, and the llama.cpp server
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
| `doctor` | PASS/WARN/FAIL per component: config, paths, database, the screening model (pinned hashes, loads, ranks a probe sensibly), model health, alias, context budget, reasoning levels, a schema-constrained request with reasoning. |
| `collect [--fixture FILE] [--skip-searches]` | Sync seed papers, then collect current releases (arXiv feeds, missed-release recovery, recent OpenAlex windows) and each topic's finite historical search, run requested gap searches, fetch the references of relevant reviewed papers, screen every candidate and promote today's papers to review jobs, acquire their full text, and print each topic's input mode. `--fixture` replays recorded records instead of the live searches; `--skip-searches` runs no arXiv/OpenAlex requests. |
| `work --once` / `work` | Run one job / loop until SIGTERM. A review job reads the paper in bounded sections, validates every citation, scores it per topic, commits, then writes its note (and queues its comparison). A `select_references` job lets the research role choose which cited works to read; a `compare_reviews` job compares two saved reviews and writes `output/comparisons/<id>.md`. Idle = zero inference calls. Several workers may run at once. |
| `reread PAPER_ID [--reason TEXT]` | Operator request for a fresh reading; gets its own generation id and a separate review. |
| `report [--date YYYY-MM-DD] [--early]` | Publish `output/daily/<date>.md` (default: the latest date whose 07:00 cutoff has passed). Freezes the eligible reviews at the cutoff, lets the writer explain them (or labels a fallback), restores missing notes, then writes atomically. One report per date and one publishing process at a time; reruns produce identical bytes. `--early` publishes today's report before its cutoff and says so. |
| `status` | Topics with their input mode (keywords / seeds / both) and what is missing, seed sources and their last sync, papers by status, editions/aliases, unresolved possible duplicates, jobs by kind/status (incl. expired leases), queued jobs per priority class with the oldest one's age, the screening model and today's promotions by reason and topic, reviews by coverage, model pause state, last model call, new candidates today, each feed / recent window / historical search with its coverage and stopping reason, open coverage gaps, source text, withdrawals, outbound requests today and paused providers, last report. |
| `access-check --component collector\|worker` | Run as a service account. Verifies the account and group; the read-only provider rules; the private credentials; the output, state, WAL, and cache permissions; the egress probes; llama-server flags and account; and the extractor sandbox (smoke test). |

Failures always print `paperwatch: FAILED [<component>] ...` with a component-specific exit code
(2 config, 3 model-server unavailable, 4 database, 5 model call/output, 6 inference lock, 7 source, 8 output, 9 worker,
10 source-access: refused by the access rules, 11 extractor, 12 screening: the embedding model is missing or altered,
13 report: not due yet, or another `report` is publishing).

## How the model is used

- One fresh two-message context per model call (a long paper takes several); the model gets plain data and **no tools**.
- Each role (`roles/`) has its own prompt, prompt version, JSON output schema, host-side checks, and hard
  time/token limits (`[roles.<name>]` in `config.toml`).
- Requests are schema-constrained (`response_format: json_schema`); the final answer is parsed separately from
  `reasoning_content`. Model identity, server build, and prompt version are stored with every call and review.
- One inference request at a time across all roles and processes (`state/inference.lock`). At the report cutoff the
  report reserves the next slot (`inference_reservations`); the worker finishes its current call, then waits.
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

## Collecting papers (plan part 5)

`collect` runs these stages in order (`collection/`), so current releases always get the day's capacity first:

| Route | What it does | Checkpoint |
| --- | --- | --- |
| **arXiv feeds** | Polls `rss.arxiv.org/atom/<category>` for every approved category, with `If-None-Match`, so an unchanged feed costs one HTTP 304. Each item is labelled `new`, `cross_list`, or `revision`. | `arxiv-feed:<category>` |
| **Missed releases** | A release can be missed only if no feed poll completed for over 24 h, or if the budget cut a poll short. Either case records a dated arXiv API search over submission dates, from 4 days before the last complete poll up to today. It is an open coverage gap until it is exhausted, and it pages from saved cursors across runs. | `arxiv-recovery:<category>:<first>..<last>` |
| **Recent OpenAlex** | Journal and conference work. Each approved phrase is searched over publication dates from 14 days before the last completed sweep up to today, at most every 12 h. The overlap catches works indexed late. | `openalex-recent:<query id>` |
| **Historical search** | One finite plan per topic, created on first use or when its approved terms or schedule settings change (a new question alone does not count). Keyword topics search their phrases on arXiv and OpenAlex; seeds-only topics run category-and-date queries on arXiv. Recency-weighted windows (`collection/recency_schedule.py`), up to 300 candidates per topic; the next page always goes to the window furthest below its share, and a page admits at most that window's fair quota (real pages hold 100 records), so a window that runs dry passes its share on. arXiv is not asked about windows before August 1991. | `history-plan:<topic>:<hash>`, `history:<query id>:<first>..<last>` |
| **Source text** | For queued papers it tries, in order: a file in `local_sources/`, then a versioned arXiv PDF from `export.arxiv.org`, then an OpenAlex content PDF (needs a key). Downloads are capped per day. The text is then extracted in the sandbox into `state/cache/sources/<edition>/<hash>.txt`. Otherwise the review uses the abstract. | `edition_source_texts` |

- **Every page commits with its checkpoint** in one transaction (`collection/page_ingestion.py`). A crash or a failed
  page resumes at the first page that was not stored. Failures follow fixed rules:
  - An outage, a refused request, or a malformed page marks the search `blocked` with its reason and keeps the cursor.
  - A budget stop marks the search `capped` and keeps the cursor.
  - A search with no more pages is `exhausted`.
- **Budgets** (`[collection]` in config.toml):
  - The day's limit is 500 new metadata candidates (papers new to the catalog), counted from the catalog itself.
  - The historical search may use only what is left after the 200 reserved for current releases.
  - Known papers never cost budget, so their revisions and new routes are always recorded.
  - arXiv requests are serialized across processes, 3 s apart.
- **Dates keep their precision**: `2018` (year), `2018-06` (month), `2018-06-15` (day); see `publication_dates.py`.
  - A revision announced in the feed keeps the month in its id (`1806.01234` gives `2018-06`) as its first
    publication.
  - First publication, latest revision, and discovery time are stored separately.
  - A paper first published in 2018 but found today is older literature (priority class 2), not a new release.
  - Each route's report is kept in `paper_discoveries` (announcement kind, reported date).
- **Identity across providers**: an OpenAlex work merges with its arXiv paper only through identifiers OpenAlex
  states: an `arxiv.org/abs/...` location or arXiv's DataCite DOI.
- **Withdrawals and retractions**:
  - An arXiv comment or abstract that says the paper was withdrawn, or OpenAlex `is_retracted`, is recorded in
    `paper_withdrawals`.
  - Such papers are never queued or recommended.
  - A paper an earlier report recommended gets one correction in the next report. The old file is never changed.
- **Ingestion never decides relevance**: a new paper waits for screening (part 6, below). Only a paper that was
  already promoted gets its new text (a new version, an abstract appearing) queued during ingestion.

Put a PDF you already have into `local_sources/`, named after the work's identifier. `collect` uses it instead of
downloading. Under the service accounts, the folder belongs to group `paperwatch-state` (setgid) so the collector
can read it.

## Screening and dispatch (plan part 6)

After collection, `collect` scores every **candidate** (a catalog paper with no promotion and no job yet, other than
an active seed or a withdrawn work) for every topic, on the CPU, and promotes the day's best to a reading
(`screening/`). No model call is made and nothing leaves the machine.

- **Embeddings:** the pinned bge-base-en-v1.5 model (`screening/embedding_model_registry.py`) embeds
  "title. abstract" of the edition a reading would use (title only for a title-only record, which is recorded).
  - A topic question gets the model's retrieval instruction prefix.
  - Vectors are cached in `text_embeddings` by model identity and text hash.
  - Measured on this workstation: 425 real abstracts take about 30 s at 16 threads (`embedding_threads`); a warm
    rerun takes 0.3 s.
- **Score per topic (0-1)**: the weighted mean of the available signals, renormalized by the active weights
  (`screening/screening_score.py`):

  | Signal | Weight | Value |
  | --- | ---: | --- |
  | Question similarity | 0.5 | `max(0, cosine)` to the question (or the joined phrases when a keyword topic has no question) |
  | Seed similarity | 0.3 | mean `max(0, cosine)` of the 3 closest available seeds |
  | Phrase match | 0.2 | fraction of the topic's approved phrases found verbatim (case, hyphens, and simple plurals ignored), so an exact term like "Dafny" counts |

  A signal the topic lacks is omitted, not counted as zero; a seeds-only topic is scored by its seeds alone.
- **One scale for all signals and topics** (`screening/signal_normalization.py`): raw cosines bunch up in a narrow,
  topic-dependent band (live: 0.53-0.75 for one topic, 0.43-0.47 for another), while the phrase signal is mostly 0
  or 1. So before combining, promotion replaces each signal by its mid-rank percentile among the topic's unpromoted
  candidates in this run. The median candidate gets about 0.5, and ties share one value.
  - Within one signal the ranking is unchanged.
  - Across signals and topics the scores compare fairly.
  - The normalized score depends on the pool, so it is never cached; the raw signals are.

  A paper's screening score is its best topic's normalized score, and `minimum_screening_score` is a pool percentile.
  Every raw component, the closest seed ids, and the matched phrases are kept in `candidate_topic_scores`. Each
  promotion row keeps the normalized score, the raw score, and the pool size for calibration. The screening score only selects papers for reading; the report uses the
  evidence-based score of part 7.
- **Rescoring:** each score stores the hash of its inputs (text, question, seeds, phrases, weights, model). Only
  missing or stale scores are recomputed, so a changed profile rescores the unpromoted candidates locally, and an
  unchanged catalog embeds nothing. A promoted paper keeps the scores it was promoted with.
- **Promotion** (`screening/promotion_selection.py`, `screening/review_promotion.py`), up to 30 per UTC day:
  1. The 27 regular slots are divided evenly across the active topics. Topics pick in turns, each taking its best
     candidate for that topic.
  2. Unused shares and the remainder are spare slots, filled by best score.
  3. Then 3 more go to the best candidates just below the cutoff, to show what the filter misses.

  The selection and its queueing commit in one transaction, so concurrent collectors cannot exceed the limit.
  Unselected candidates become `screened_out`: they stay searchable, are rescored on profile changes, and can be
  promoted on a later day. Source text is acquired for the promoted papers in the same run.
- **Worker priority classes** (`dispatch_order.py`, claimed by `jobs.claim_next_due_job`):

  | Class | Job | Order within the class |
  | ---: | --- | --- |
  | 0 | Operator reread | queue age |
  | 1 | New release inside its 7-day fresh window (from the original release date) | screening score, then queue age |
  | 2 | Older paper (also a new release whose window ended while it waited) | screening score, then queue age |
  | 3 | Important reference (queued from part 8) | importance to the parent's argument, then topic score |
  | 4 | Comparison (queued from part 8) | shared evidence dimensions, then queue age |

  A paper that qualifies for several classes runs in the highest one: a queued job only ever moves up. Each claim is
  one bounded job, so a higher-priority arrival runs right after the current job. `status` shows the count and the
  oldest age per class. With no due job, the worker sleeps and makes zero inference calls.
- **A missing or altered model** fails visibly. Every other stage of `collect` still runs, candidates wait unscored
  (nothing is promoted on partial signals), and `collect` exits with code 12. `doctor`, `status`, and `access-check`
  name the problem.

## Reading papers (plan part 7)

The worker reads each queued paper from text the collector already saved; it never downloads anything
(`reading/`, roles in `roles/`).

- **Which text**: the sandboxed extractor's full text when it is usable (at least 1500 characters), else the
  abstract (`readings.select_reading_input`). `collect` holds the readings it promotes until its source-text stage
  has run (`collection.source_text_hold_minutes`), so a PDF fetched in the same run is read instead of the abstract.
  If full text is extracted while an abstract reading is still queued, the queued job is retired and the full-text
  reading takes its place. If the cached text has vanished or changed, the worker marks it failed and queues the
  abstract reading, without a model call.
- **Passages** (`reading/passages.py`): stable ids (`A1..` abstract sentences, `T1` the title of a title-only record,
  `P1..` full text), with the PDF page (pdftotext's form feeds) and the section heading. Line-end hyphenation is
  undone; ligatures and soft hyphens are normalized. Each section gets a kind (method, results, discussion,
  references, ...).
- **Reading plan** (`reading/reading_plan.py`): passages are cut into reading sections of at most
  `reading.section_max_characters` (one model call each). Up to `reading.max_sections_per_paper` are read, method and
  results first, then discussion/conclusion, introduction, background, front matter, appendices. References and
  acknowledgments are never read.
  - `full_text` only if every section with substantive content was read and a closing section (conclusion,
    discussion, references) exists; otherwise `partial_text`, with the reasons saved (e.g. "the extraction may be
    truncated").
  - When the config loads, one section plus its prompt plus the role's whole output limit must fit
    `model.required_context_tokens`; before each call the actual prompt is checked again.
- **Model calls** (`reading/reading_pipeline.py`), each a fresh context with plain data and no tools:
  1. `section_reader`, once per reading section of a long paper. Each validated result is saved
     (`review_stage_results`), so a timeout or crash resumes at the first missing section.
  2. `reviewer`: the review, from the passages (one section) or from the section notes. Saved as a stage too.
  3. `topic_assessor`: rubric ratings for every topic, with comparison anchors.
- **Evidence checks** (`reading/evidence_checks.py`, `roles/evidence_fields.py`), before anything is committed:
  - every cited passage id must be in the packet;
  - a finding `{claim, conditions, passage_id, excerpt}` quotes 3-60 contiguous words of its passage (case,
    whitespace, hyphenation, quotes and dashes ignored);
  - every number in a claim, its conditions, or a statement must occur in a cited passage, so an invented metric is
    rejected ("Table 2", "Section 3" are exempt);
  - the authors' claims (`results`, limitations stated by the authors) are kept apart from the model's
    (`reviewer_inferences`, limitations stated by the reviewer). Statements can say "insufficient evidence";
    ambiguous passages and garbled tables or equations are flagged.
  - An invalid answer gets **one repair attempt** (sent back with its problems); a second invalid answer fails the
    job with both reasons kept, and the worker moves on.
- **Reading priority** (`reading/reading_priority.py`): the model proposes an integer 0-4 per dimension and explains
  it; the code computes `round(100 * sum(weight * rating / 4) / sum(active weights))`, half up.

  | Dimension | Weight |
  | --- | ---: |
  | Fit to the research question | 40 |
  | Fit of method or assumptions | 25 |
  | Strength of evidence available to this review (capped at 1 for an abstract-only review) | 20 |
  | Value relative to nearby saved papers (omitted when no anchor exists: "comparison unavailable") | 15 |

  Anchors (`reading/comparison_anchors.py`) are the topic's closest seeds (from screening, else cached embeddings)
  and earlier reviews whose assessment rated question fit at least 2, nearest first. The worker never embeds
  anything. The score is a reading priority for your question, not a novelty claim.
- **Saved and rendered**: the review row stores the reading record (sections read and not read, partial reasons,
  table and equation recovery); the cited passages are copied into `review_evidence_passages`, so the evidence
  survives cache pruning; each topic assessment keeps the ratings, explanations, applied caps, and the topic profile
  hash. Right after the commit the worker writes `output/papers/<paper>/<review>.md`. It shows every field with its
  evidence pointers (`P12` (p. 4, 3 Results)) and quotes, the scoring table per topic, matched seeds, the coverage,
  and the cited passages. `report` restores a missing note from the saved review.

## References and comparisons (plan part 8)

The model never composes a request: it names ids from the list it was given, and host code checks every id, the hop
limit, the daily limits, and cycles before anything is queued or searched (`references/`, `comparisons/`).

- **Fetching references** (`collect`, `references/reference_fetch.py`): for each reviewed paper that addresses some
  topic's question and is less than `max_citation_hops` (2) from a screened paper, at most `reference_fetches_per_run`
  (10) per run:
  1. Its OpenAlex work is found by its OpenAlex id, its DOI, or arXiv's DataCite DOI of its arXiv id. The record must
     link back through an identifier the paper already has; then only the OpenAlex alias is added (never a new
     edition, which could change which abstract is read).
  2. Each id in its `referenced_works` becomes a citation edge (`openalex:referenced_works`). The cited works are
     looked up 100 at a time (`filter=openalex:W1|W2|...`, the new fixed `get_works` operation, which refuses any id
     that is not a stored edge) and catalogued under the route `reference:<citing paper>`. They are then screened
     like every other candidate.
  3. When the graph is incomplete: DOIs and arXiv ids in the paper's own bibliography that no edge covers are looked up
     (at most `bibliography_lookups_per_paper`). One becomes an edge (`bibliography:confirmed_by_provider`) only if
     the provider record's title matches the bibliography entry; otherwise it stays in `bibliography_identifiers` as
     an unresolved reference (`mismatch`, `not_found`).
  OpenAlex documents no filter for "the works W cites" (checked 2026-11-20: `cites:` is the other direction), so the
  list comes from the work's own `referenced_works`.
- **Choosing references** (worker, `research` role, class 3 ahead of reference readings): one call per parent
  review. The role sees the parent's validated review, the questions it addresses, up to 30 candidates (cited works
  not yet reviewed, queued, or promoted, not seeds or withdrawn; best screening score first), each with its citation
  contexts: the sentences of the parent that cite it (`references/citation_contexts.py`: numeric labels like
  "[3, 12]" or "[10-14]", or author-year). It picks at most 3, each with its role in the parent's argument (method,
  theorem, data, baseline), an importance 1-4, an explanation, and the context passages it relied on.
- **Admitting choices** (`references/reference_selection.py`): hop depth = parent's + 1; a choice that is already
  reviewed, queued, promoted, a seed, or withdrawn is skipped (this ends cycles: a paper's ancestors are reviewed).
  At most `daily_reference_readings` (20) are admitted per UTC day, counted from the table so the limit holds across
  restarts; the rest wait (`waiting_daily_cap`) and are admitted on a later day, most important first. An admitted
  choice records a `reference` promotion and queues a class-3 reading ordered by importance, then topic score,
  held until the next `collect` has tried to fetch its PDF. A paper reached from two parents is read once.
- **Gap searches** (`references/research_branches.py`): the research role may also pick one unused branch of an
  approved historical search (one catalog query over one recency window that has admitted nothing yet) and state the
  gap it fills. At most `research_branches_per_day` (3); a branch is never requested twice. `collect` runs one page
  of it through that window's own checkpoint under the route `research-branch:<query id>` (historical allowance) and
  records the gap, branch, parent review, and returned candidates.
- **Comparisons** (`comparisons/`, `comparison` role, class 4): when a review commits, its closest neighbour (by cached
  screening embeddings) among the newest reviews of other papers that also rated question_fit >= 2 for the same
  topic is chosen if the two share at least one evidence dimension (assumptions, guarantees, evaluation setting, by
  word overlap of supported statements). One comparison is started per review, at most `daily_comparison_jobs` (5)
  per UTC day; `collect` queues the ones the limit held back for reviews of the last 14 days. The key is the question
  plus the sorted pair of review ids, so a pair is compared once. The role reads the two saved reviews and the
  passages they cite (no rereading) and states the condition differences, agreements, differences, an optional
  ranking, and what the evidence permits the reader to conclude. Every evidence item `{review_id, passage_id,
  excerpt}` is checked against that review's saved passages, agreements and the conclusion must cite both reviews,
  and numbers must occur in the cited passages. A ranking under conditions that are not comparable is rejected
  unless the condition differences are listed and a caveat explains them.
- `status` shows fetched references, citation edges, unresolved bibliography identifiers, choices admitted / waiting /
  skipped against today's limit, gap searches, and comparisons against today's limit.

## The daily report (plan part 9)

`paperwatch report` runs in four recoverable stages (`commands/report_command.py`, `reporting/`):

1. **Freeze** (`reporting/report_snapshot.py`). One short transaction reads and saves everything the report needs;
   the report row becomes `writing`.
   - **Cutoff** (`report_schedule.py`): `report.publish_time` (07:00) on the reporting date in `report.timezone`
     (America/New_York), converted with that day's daylight-saving offset. Reviews completed later belong to a later
     report.
   - **Eligible** (`report_selection.py`): the latest review per paper completed by the cutoff, if its best topic
     score (assessed by the cutoff) is at least `report.minimum_reading_priority` (65). Withdrawn papers and papers
     in your active seed library are left out (seeds stay comparison anchors). So is any paper an earlier report
     recommended, unless the newer review is an update (`revision_changes.py`: different text, and a changed result,
     assumption, verdict, or score).
   - **Order**: score, then coverage (full > partial > abstract), then paper id. Up to `max_papers_per_report` (15)
     are chosen; the rest stay for later days.
   - **Also saved**: the opening's statistics since the previous report (papers found, screened, reviewed; backlog;
     provider, search, and model outages; hours since the previous report), the comparisons involving a selected
     review, and the corrections for earlier recommendations that were withdrawn.
2. **Synthesize** (`report_synthesis.py`, `roles/writer.py`). The report reserves the next inference slot until its
   deadline; the worker checks for that reservation before each call. The writer sees the frozen reviews, the
   passages they cite (bounded to the context budget), and the comparisons. It returns an overview, up to three
   connected findings, what to read first, and one reason per paper. Every claim is `{review_id, passage_id,
   excerpt}`-backed, and the host checks every one: the review must be selected, the passage must be that review's,
   the excerpt must be a contiguous quotation, any number must occur in a cited passage, and comparison ids must come
   from the input. The writer cannot add, drop, or reorder papers, or state scores. If the server is down, the slot
   is not free in time, the deadline (30 min) passes, or the answer is still invalid after one repair, the same
   selection is rendered from the saved summaries and topic rationales, with a "Fallback report" label and the
   reason. With nothing selected the writer is not called: a quiet day is told apart from an outage.
3. **Render** (`rendering/daily_report.py`). The renderer has four sections: opening, what matters today, ranked
   papers, and comparisons and corrections.
   - It supplies ranks, dates, scores, coverage, labels (new release / older paper / reference / update), source
     links from trusted identifiers, and relative note links.
   - A citation reads `[#2 · P12 (p. 4, Results)](<../papers/<paper>/<review>.md>)`.
   - An update links the earlier review's note and lists what changed.
   - All paper and model text is escaped.
   - The exact bytes and the synthesis are saved (`pending`).
4. **Publish**. Missing reading and comparison notes are restored from the database first. Then the file is written
   through a temp file and an atomic rename, and the row becomes `published`.

A rerun resumes:
- `writing`: synthesizes again over the same frozen selection.
- `pending`: writes the saved bytes again. A kill between the rename and the database update gives identical bytes.
- `published`: changes nothing, or restores a deleted file from the saved bytes.

A second `report` while one runs exits with code 13 (`state/report-publication.lock`). `--date D` for a missed date
publishes it with its own scheduled cutoff. A date whose cutoff is still ahead is refused, unless `--early` (today
only) publishes it with "now" as the cutoff and says so in the opening.

## Deployment and the one-day pilot (plan part 10)

The first unattended run is a **one-day pilot** that starts and ends by itself:

```bash
sudo env "PATH=$PATH" deploy/pilot/install_one_day_pilot.sh      # reinstall /opt, migrate, access-check, doctor,
                                                                 # render + enable the one-shot timers, collect now
paperwatch --home /srv/paperwatch status                         # progress (as yourself)
journalctl -u 'paperwatch-pilot-*' -f                            # live service logs (redacted)
sudo deploy/pilot/stop_one_day_pilot.sh                          # optional: end the pilot early
```

| Unit | When | What |
| --- | --- | --- |
| `paperwatch-pilot-collect` (collector account) | at install, then each `schedule.collect_times` still ahead before the cutoff (e.g. 18:30, 00:30, 06:30) | `collect`; also starts the worker |
| `paperwatch-pilot-worker` (worker account) | from the first collection until the report cutoff (07:00 the next day) | `work` |
| `paperwatch-pilot-report` (worker account) | 07:00:30 the next day; stops the worker first; retried until 07:35 | `report --date <next day>` |
| `paperwatch-pilot-finish` (root) | 08:40 the next day | stops the worker, disables every pilot timer, **deletes every pilot unit file** |

**Why it runs once** (`src/paperwatch/deployment/one_day_pilot.py`, tested in
`tests/test_step10_one_day_pilot_schedule.py`):
1. every `OnCalendar=` is one absolute instant with its timezone; `systemd-analyze calendar` finds no second elapse;
2. no relative or boot trigger (`OnBootSec=`, `OnUnitActiveSec=`, ...), and no service has an `[Install]` section;
3. every service has an `ExecCondition=` with a fixed Unix-time deadline, so it is skipped afterwards even if started
   by hand (the test runs the condition and checks that it skips);
4. the finish step deletes all seven unit files and the timer stamps, then reloads systemd.

**Sandbox of the processes that talk to the model** (worker and report units), on top of the nftables rule and the
in-process egress guard: `IPAddressDeny=any` + `IPAddressAllow=localhost`, `ProtectSystem=strict` with write access
only to `state/` and `output/`, `ProtectHome=yes`, private `/tmp` and `/dev`, no capabilities, no new privileges,
no namespaces. The collector keeps a lighter sandbox (`ProtectSystem=full`) because rootless Podman needs setuid id
mapping and `/run/user`; it relies on nftables (DNS + public HTTPS only), the egress guard, and file permissions.

**Service logs are redacted** (`service_log_redaction.py`): credential-like URL parameters and headers become
`[redacted]`, long quotes of paper or model text become `[text omitted]`, lines are bounded, tracebacks are dropped,
and the HTTP libraries never log request URLs (an OpenAlex key travels as `?api_key=`). Full details stay in the
database.

**Adversarial fixture day** (`tests/test_step10_hostile_model_day.py`): a paper tells the AI to fetch an invented
work, upload a private note, search a private phrase, read `/etc/passwd`, run a shell command, and award full
marks; the model is assumed fully compromised in every role. Over a whole day (`collect` -> worker -> `collect` ...
-> `report`) the test proves: every provider request is a GET built by host code from approved queries,
provider-supplied citation ids, and catalogued editions, with nothing from the paper or the model in it; stored
research choices are exactly ids the host offered (also rechecked where they are persisted:
`reference_selection.only_offered_choices`); no socket, no program, files only in `state/` and `output/` under
internal-id names; hostile text is inert in notes and the report. 1,500 random hostile research answers never get an
unoffered id past the checks.

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

- **Fixed operations** (`network/provider_gateway.py`): `search(topic_query_id, cursor)`,
  `poll_category_feed(arxiv_category)`, `get_work(provider_id)`, `get_text(known_edition_id)`,
  `list_seed_items(collection_key, start)`. Code builds every URL from validated ids
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
deploy/install_embedding_model.sh /srv/paperwatch              # pinned screening model -> models/ (root:paperwatch-state)
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
  collection/                   # part 5: current releases, missed-release recovery, recent OpenAlex windows,
                                #   historical search over year bands, candidate budget, atomic page ingestion,
                                #   source text (local files, capped downloads, sandboxed extraction)
  collection/recency_schedule.py  # exponential recency curve -> equal-mass date windows (historical search)
  publication_dates.py          # partial ISO dates (precision = form), release class (new vs older literature)
  seeds/                        # Zotero + CSL JSON seed parsers, seed cache, complete-snapshot sync
  screening/                    # part 6: pinned embedding model + ONNX embedder, embedding cache, phrase signal,
                                #   scoring formula, per-topic profiles, candidate scoring, slot selection, promotion
  dispatch_order.py             # worker priority classes and the order within each class
  research_graph.py             # citation edges and comparison keys
  references/                   # part 8: reference fetch, bibliography identifiers, citation contexts, hop depth,
                                #   candidates, research-role selection and admission, gap-search branches
  comparisons/                  # part 8: shared evidence dimensions, comparison queueing, comparison jobs
  source_checkpoints.py         # per-search cursor, budget, coverage, stop reason, last completed interval
  public_query_catalog.py       # approved public queries with stable ids: the only searchable terms
  source_cache.py               # downloaded source bytes, named by internal ids and hashes only
  sandboxed_extractor.py        # rootless Podman runner for the pinned pdftotext/XML extractor
  network/                      # collector only: provider rules, gateway, secure fetcher, ledger, credentials,
                                #   and the per-process egress guard
  llm/                          # model client contract, llama.cpp client, fake client, lock, slot reservation,
                                #   outage guard
  reading/                      # part 7: passages, reading plan, parser recovery, review packet, evidence checks,
                                #   saved stage results, comparison anchors, rubric, the reading pipeline
  roles/                        # healthcheck, section_reader, reviewer, topic_assessor, research, comparison,
                                #   writer;
                                #   shared evidence fields;
                                #   output guards
  sources/                      # record types, discovery route names, parsers: arXiv Atom (feed + API), OpenAlex
                                #   works JSON, recorded fixtures
  reporting/                    # part 9: cutoff schedule, report selection and update rule, frozen snapshot,
                                #   comparisons shown, writer synthesis with deadline and fallback, publication lock
  rendering/                    # Markdown escaping, evidence notes, comparison notes, daily report, atomic writes
  commands/                     # one module per CLI command
  deployment/                   # part 10: systemd units of the one-day pilot (one-shot timers)
  service_log_redaction.py      # part 10: no keys, signed URLs, or paper/model text in service logs
settings/                       # operator settings (gitignored) + committed *.example.toml; see settings/README.md
deploy/                         # install, service accounts, nftables egress rules, extractor image, embedding model,
                                #   access checks,
                                #   apply_settings.sh (install settings/ into the paperwatch home),
                                #   pilot/ (start and stop the one-day pilot)
tests/                          # step 1 vertical slice; step 2 identity/readings and worker crash/race tests;
                                #   step 3 prompt injection, provider gateway, sandbox and egress tests;
                                #   step 4 input modes, Zotero/CSL seed sync, outbound-parameter checks;
                                #   step 5 feed replay, mid-page failure, downtime recovery, late indexing, revisions,
                                #   historical cap, budgets, source text, withdrawals (fake_paper_providers.py);
                                #   step 6 500 candidates with known nearest seeds, fair promotion, claim order of all
                                #   four job types, zero inference while idle, rescoring, missing model;
                                #   step 7 full / truncated / abstract-only through one schema, invented metric and
                                #   unknown passage rejected, no second reading, resume after a timeout, rubric,
                                #   anchors, held readings, switch to full text, fallback when the cache is gone;
                                #   step 8 mutual citations with a shared reference (search ends, one review each,
                                #   one comparison), hop limit, daily limit, bibliography confirmation, citation
                                #   contexts, research-role id checks, gap search, incompatible-ranking rejection;
                                #   step 9 0 / 6 / 20 qualifying papers, late completion, missing note restored,
                                #   model outage fallback, kill between rename and DB update, crash before render,
                                #   updates vs re-reads, writer evidence checks, slot reservation, writer deadline,
                                #   DST cutoffs, early publication, one publication process
```

## Tests

```bash
uv run pytest                               # offline: the fake model and the word-hashing test embedder
PAPERWATCH_LIVE_MODEL=1 uv run pytest -m live_model   # also reviews the fixture with the real server
PAPERWATCH_TEST_EXTRACTOR_IMAGE=<image id> uv run pytest -k real_podman   # real sandbox (needs podman + image)
PAPERWATCH_TEST_EMBEDDING_MODEL_DIR=<home>/models/bge-base-en-v1.5 uv run pytest -k real_onnx   # real screening model
```
