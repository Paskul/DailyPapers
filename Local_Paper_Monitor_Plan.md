# Local paper monitor

## High-level idea

Build a local reading service that answers **“Which papers are worth my time today, and why?”** Use Zotero collections, research questions, or both to define interests. Search recent releases and older literature. Score hundreds of titles and abstracts on the CPU. Give the strongest candidates to Qwen 3.8 27B for a closer read. Save each review once. At a fixed time, publish one Markdown report with up to 15 papers and links to the evidence behind each recommendation.

Use the same local model in distinct roles. The research role selects promising search branches from an approved list. The reviewer extracts claims and evidence. The comparison role checks related papers against the same question. The writer explains the day's findings from completed reviews. Give each role its own input, output schema, and prompt. Start a fresh model context for each job. Ordinary Python code owns network requests, scheduling, scores, database changes, and file writes.

| Job | When it runs | What it produces |
| --- | --- | --- |
| Collect papers | Four times per day by default | New or updated records, search checkpoints, candidate scores, and review inputs |
| Review papers | Whenever a job is ready | A saved review with source passages, a reason for its score, and a permanent note |
| Write the report | Once per day, by local date | An overview, up to 15 ranked papers, and links to their notes |

**Build choice.** Start a small Python package beside the existing llama.cpp installation. The reference project retrieves recent papers, ranks abstracts against Zotero, summarizes the selected papers, and sends an email in one run. Its collection filters and abstract matching are useful examples. Implement the new workflow independently around SQLite and Markdown. This gives each stage its own restart point. [Reference project](https://github.com/TideDra/zotero-arxiv-daily) · [Upstream executor](https://github.com/TideDra/zotero-arxiv-daily/blob/main/src/zotero_arxiv_daily/executor.py)

The Markdown file is the delivery channel. Omit email, SMTP settings, and notification accounts.

The ten parts below are in build order. Each ends with a check that a coding agent can run. Numeric limits are starting settings. Measure them on the target workstation before enabling the schedule.

## 1. Establish the local application and its contracts

Use Python 3.12 on Ubuntu, one package named `paperwatch`, one SQLite database on local disk, and one output directory. Keep deployment to the application, the existing llama.cpp server, and systemd. Start with Python's `sqlite3`, `tomllib`, and `argparse`. Use a small HTTP client with explicit timeouts, a schema validator, a local embedding model, and a fixed PDF extractor as needed.

Expose `paperwatch init`, `doctor`, `collect`, `work --once`, `work`, `report`, and `status`. Make the first vertical test possible with a recorded paper and a fake model. Use the following layout. Generate internal IDs for filenames. Keep the database and source cache outside any synced notes folder.

```text
config.toml                       # Paths, model address, limits, schedule
topics.toml                       # Research questions and public search terms
state/paperwatch.sqlite3          # Catalog, jobs, reviews, report snapshots
state/cache/                      # Source bytes and extracted passages
output/papers/<id>/<revision>.md  # Stable, version-specific reading note
output/daily/YYYY-MM-DD.md        # One report per reporting date
```

Use `qwen-local` at `http://127.0.0.1:8080/v1` by default. Keep reasoning enabled on the existing server and run one inference request at a time across roles. Have `doctor` check the model alias, a short schema-constrained request to `/v1/chat/completions` with reasoning enabled, and the available context budget. Parse the final answer separately from reasoning output. Record the server build, model identity, and prompt version. Set a hard time and output limit for each role. If the server is down, pause model jobs together and back off the health check. Keep the papers queued. Never switch to a hosted model on failure. [llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)

**Check.** Run the fixture through the CLI in a temporary directory. Repeat it with the model endpoint unavailable. The CLI must report which component failed and preserve the input.

## 2. Give each paper an identity and each job a finish condition

Use a small set of SQLite tables. Store variable review fields as validated JSON. Keep fields used for joins, ordering, uniqueness, and recovery in indexed columns.

| Record | Contents and uniqueness rule |
| --- | --- |
| Papers and aliases | One internal paper ID. Map each trusted provider ID to one paper. Store title, authors, dates, source version, discovery routes, seed membership, and origin. |
| Jobs | Kind, input version, priority, status, attempts, due time, lease token, and lease expiry. Give each logical job a unique key. |
| Reviews | Immutable evidence, source fingerprint, coverage, model and prompt versions, completion time. Unique by paper and review input. Store topic assessments separately from the underlying reading. |
| References and comparisons | Directed citation edges with provenance. Key a comparison by question and sorted pair of review IDs. |
| Source checkpoints and reports | A cursor and budget per search. One report per local date, with cutoff, frozen selection, exact rendered bytes, and publish status. |

Normalize DOI forms and arXiv IDs. Keep the arXiv version separately. Merge a preprint and a journal record only when a trusted source supplies a clear identifier link. Keep distinct texts as editions. Treat similar titles and author names as possible matches. Do not silently merge papers on that evidence alone.

Expose a paper status such as `screened_out`, `waiting_input`, `queued`, `reviewing`, `complete`, or `failed`. Record `full_text`, `partial_text`, or `abstract_only` on a completed review. Reporting a paper is a separate fact. For jobs, use `queued → running → done`, with delayed retries and a terminal failure state. Start with three attempts for a paper-specific transient error. Pause the affected service on a provider-wide outage. Claim a job in a short transaction. Commit the review and mark the job done in the same transaction, and only if the lease token still matches. Renew leases on long calls. Reclaim expired leases. Use WAL mode, foreign keys, and a busy timeout. Keep external calls outside write transactions. [SQLite WAL](https://www.sqlite.org/wal.html)

A changed title, another discovery route, a restart, or an empty queue must not trigger another reading. Key a reading by canonical paper, edition, and normalized source-text hash. A verified text change, newly available full text, or an explicit operator request can create a new review. Give explicit rereads their own generation ID. Reuse the old review for a topic or scoring change. A model upgrade alone must not enqueue the archive.

**Check.** Import one paper by two routes. Kill a worker after claim and again after commit. Start two workers on the same fixture. Require one canonical paper, one accepted review for that input, and no stale worker overwriting the result.

## 3. Restrict what can reach the internet

Make the collector the only application component that can contact providers. Give it fixed operations such as `search(topic_query_id, cursor)`, `get_work(provider_id)`, and `get_text(known_edition_id)`. Code constructs the URL, validates the ID and the request budget, and attaches the credential for that provider. Leave general web search, URL fetching, and shell execution outside the model interface.

| Component | Access to grant |
| --- | --- |
| Collector | Read the approved query catalog and provider credentials. Make HTTPS requests on fixed hosts, methods, and paths. Write source state and the cache. |
| Reviewer, research role, and report writer | Their host process can read job inputs and call the local model. The host validates and persists outputs. The model receives plain data and has zero tools. |
| llama.cpp | Read its local weights and serve the local inference endpoint. Disable agent mode, server tools, MCP proxy, and server-side MCP tools. Keep provider keys out of its environment. |
| PDF or XML extractor | Run a pinned parser in rootless Podman with the network disabled, a read-only root, one read-only input, and bounded temporary space. Return text on stdout. Give it no model socket, keys, home directory, or database. |

Use distinct Linux accounts for the collector, the worker and writer together, and the model server. Keep keys in a collector-only credential file. Use group permissions for the shared SQLite directory, including its WAL files. Grant output-directory write access only to the worker and writer account. Enforce network limits outside Python. For the worker account, use nftables to allow TCP to `127.0.0.1:8080` and deny other IPv4 and IPv6 traffic. Match the service UID in the output chain. Block new outbound connections from the model service account while allowing replies to local clients. Limit filesystem access to the directories each service needs. Preinstall models and parser images. Disable runtime downloads. Test the rules from the actual service accounts. [llama.cpp server options](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) · [nftables UID matching](https://netfilter.org/projects/nftables/manpage.html) · [Podman run options](https://docs.podman.io/en/latest/markdown/podman-run.1.html)

Use strict TLS verification. Reject redirects by default. If a provider requires a content delivery redirect, allow only its documented destinations and strip provider credentials on a host change. Check the actual connection target against private, loopback, link-local, and reserved addresses. Apply the same checks on each allowed hop. Bound response bytes, decompressed size, parser time, and memory. Register the collector's host and path rules in read-only configuration. Recheck queued IDs against source records at dispatch. Treat titles, abstracts, PDFs, and extracted passages as untrusted text. Keep them out of commands, filenames, tool arguments, and raw Markdown. These controls limit the effects of a misleading passage. A model can still misread it or give a poor recommendation.

**MCP option.** The first release can use direct Python functions. If a second client needs MCP, expose the same **local, read-only** functions over stdio, such as `get_review(review_id)` and `list_ranked(topic_id, limit)`. Validate every ID and cap every result. Keep the adapter outside the autonomous review loop. If the research role needs to suggest another search, accept only a catalog ID or a verified citation edge. Apply the budget in host code. MCP supplies a protocol. Service permissions and input checks enforce the access limits. [MCP security guidance](https://modelcontextprotocol.io/specification/2025-11-25/basic/security_best_practices)

**Check.** Feed in a synthetic paper that asks the reviewer to upload a local note, open a link, and award it full marks. Check outbound traffic and file changes. Reject invented actions and references. Test a separate parser timeout and a blocked redirect. The next valid paper must still complete.

## 4. Build a research profile from Zotero, topics, or both

Make the question the unit of relevance. A robotics paper should be able to match a focused question even if most of the library covers another field. Keep the private question separate from the terms the collector may send to providers.

```toml
[[topics]]
id = "team-safety"
question = "Can a runtime monitor reject a joint robot plan before it blocks a human exit?"
public_queries = ["multi-robot runtime assurance", "multi-agent plan monitoring",
                  "runtime verification mobile robots"]
arxiv_categories = ["cs.RO", "cs.MA", "cs.LO"]
zotero_collections = []  # Optional Zotero collection keys
```

Support two seed inputs. For a live library, page through the selected Zotero collections with a read-only key. For a disconnected library, import a local CSL JSON export and let the user assign it to a topic. Read bibliographic metadata. Skip attachments and private notes by default. Handle missing abstracts with title-only matching and record that limitation. Cache the last successful seed set. Retain Zotero item keys or export IDs so that a review can identify the specific seed that matched. For live sync, use library versions, handle deleted items and collection moves, and commit the new checkpoint only after a complete sync. [Zotero sync API](https://www.zotero.org/support/dev/web_api/v3/syncing)

Support keywords alone, seeds alone, and both together. For seeds alone, search the configured public subject categories and rank their results against the seeds locally. Use category-and-year queries to start historical search. Make lookup by a private library's seed DOI an explicit setting because it discloses interest in that paper. Never form outbound queries from private notes, collection names, or arbitrary model text. If the library is empty or offline, keep processing configured topics. If both inputs are empty, `status` should explain what to add.

**Check.** Run all three input modes, including a missing Zotero key and a title-only seed. Inspect recorded outgoing query parameters. Only approved public terms, categories, and permitted IDs may leave the machine.

## 5. Collect current papers and run a finite historical search

| Route | Implementation |
| --- | --- |
| New papers | Poll arXiv category feeds for announcements. Record whether each item is new, cross-listed, or revised. Also query OpenAlex by approved topic and recent publication date to find journal and conference work outside arXiv. Label each result with its source and date precision. |
| Older papers | On first use or a material topic change, create a finite search per topic. Run the approved terms through arXiv and OpenAlex across year bands. Page through results and alternate bands to include older work. Save each query, cursor, stopping reason, and count. |
| Missed releases | Track each source's last completed interval. Use a dated arXiv API search to recover first submissions after downtime. Recheck an overlapping recent publication window in OpenAlex for late indexing. Record coverage gaps and resume incomplete pages on the next run. |
| Source text | For shortlisted work, try a local permitted file, a versioned arXiv text/PDF through its programmatic access route, or an available OpenAlex content record. Respect source terms and the configured free budget. Use the abstract when full text is unavailable. |

Store first publication, latest revision, and local discovery time independently. A paper from 2018 first found today belongs in the older literature. Record the original route even if another route later promotes its priority. Track retractions and withdrawals when providers report them. Exclude known withdrawn works from recommendations. For previously reported works, add a correction to the next report without altering the historical file.

Start with a budget of 500 unique new metadata candidates per day and 300 historical candidates per topic per initial search. Reserve capacity for current releases first. Persist overflow cursors. Mark a search as capped, exhausted, or temporarily blocked so its coverage is visible. Serialize arXiv legacy requests across the application and wait at least three seconds between them. Store all timestamps in UTC and convert them at the reporting boundary. Cache results, honor provider retry instructions, and stop a blocked or unaffordable route. A feed updated once per day will yield the same records on later polls. [arXiv feeds](https://info.arxiv.org/help/rss.html) · [arXiv API and rate limits](https://info.arxiv.org/help/api/tou.html) · [arXiv access routes](https://info.arxiv.org/help/bulk_data.html) · [OpenAlex API](https://help.openalex.org/api/) · [OpenAlex full text](https://help.openalex.org/access/fulltext/)

**Check.** Replay a feed, a mid-page failure, a week of downtime, a late-indexed work, a revision, and a historical search that reaches its cap. Check the recovered IDs, preserved cursors, and date labels.

## 6. Score every candidate cheaply and dispatch work in order

Compute title-and-abstract embeddings on CPU with one pinned local sentence embedding model. Cache embeddings by model and text hash. A simple NumPy matrix is sufficient for hundreds of candidates. Add a lexical signal for exact terms such as “Dafny” that can get lost in a broad semantic match.

For each topic, compute a screening score from three signals on a 0–1 scale. Start with 50% question or topic similarity, 30% average similarity to the three closest available seeds, and 20% fraction of approved phrases matched. Use a documented normalization for cosine similarity, for example `max(0, cosine)`. Omit unavailable signals and divide by the sum of active weights. For seeds-only mode, omit an absent question. Take the best topic score and keep its component values and closest seed IDs. This score selects papers for reading. The report uses the evidence-based score in part 7.

As a starting limit, promote up to 30 candidates per day, including up to three near the cutoff to test what the filter misses. Divide available review slots across active topics before filling spare slots by score. Apply a configurable minimum score and calibrate it in part 10. Keep the rest searchable in SQLite. They can be rescored locally when the profile changes. Schedule source acquisition for promoted papers during collection so the local worker can read them without requesting a URL.

| Worker priority | Job | Ordering within the class |
| --- | --- | --- |
| 1 | Highly relevant new release | Use the original release date and a seven-day fresh window. Order by screening score, then queue age. |
| 2 | Older paper from the initial topic search | Highest topic score, then queue age |
| 3 | Important reference from a completed review | Importance to the parent paper's argument, then topic score |
| 4 | Comparison on the same question | Pairs with shared evidence dimensions, then queue age |

Assign a paper that qualifies for several classes to the highest applicable class. Allow a higher-priority arrival to run after the current bounded job. Show age and count per class so the user can see when sustained new arrivals delay older work. The daily report has its own deadline. When no job is due, wait for new work or the next retry time. During that wait, issue zero inference calls.

**Check.** Feed in 500 candidates with known nearest seeds. Inspect scores and promotions. Put all four job types in the queue and verify claim order. Drain the queue and check that the model request count stays constant.

## 7. Read the selected papers and save evidence-backed notes

For each promoted paper, create an edition-specific packet. Assign stable passage IDs and page numbers or section names to the extracted text. Use the title and abstract for orientation, then read methods, results, and limitations. Process long files in bounded sections. Reserve context space for reasoning and the final answer. Save completed section results so a timeout does not restart a long paper. Record which sections were actually read and whether the parser could recover tables or equations. Use `full_text` only when the reviewer processed all substantive sections. A downloaded PDF reviewed in part gets `partial_text`.

Require fields for the research question, method, results, assumptions, evaluation setting, limitations, and supporting passage IDs. A finding has the shape `{claim, passage_id, excerpt}`. A topic assessment has the shape `{topic_id, ratings, rationale, matched_seed_ids}`. Reject unknown fields and IDs outside the input packet. Keep the authors' claims separate from the model's inference. Require each substantive finding to have a passage ID and a short matching excerpt. Validate IDs and excerpts against the saved edition. Require a numeric result and its conditions to occur in the cited evidence. Exact text matching checks attribution. It cannot establish that the interpretation is sound. Flag ambiguous passages and let the reviewer say “insufficient evidence.”

For each matched topic, attach the closest seed records and available saved reviews. Keep their IDs distinct from the passages in the candidate paper. Score the topic with the rubric below. The model proposes integer ratings and explains each one. The application validates the range and computes the total. Use the best qualifying topic in the report. For paper-specific assessments, keep the topic profile hash with the saved ratings.

| Dimension | Weight | Meaning of 0, 2, and 4 |
| --- | ---: | --- |
| Fit to the research question | 40 | Unrelated / addresses part of it / studies the question directly |
| Fit of method or assumptions | 25 | Incompatible / useful with changes / directly applicable under stated assumptions |
| Strength of evidence available to this review | 20 | Unsupported / partial support / direct proof or evaluation under stated conditions |
| Value relative to nearby saved papers | 15 | Repeats a known point / adds a useful difference / changes a concrete design choice or supplies a missing comparison |

Use intermediate ratings 1 and 3 when justified. Compute `round(100 × sum(weight × rating / 4) / sum(active weights))`. For example, ratings of 4, 3, 2, and 3 give **80/100**. If comparable saved papers are unavailable, omit the last dimension and show “comparison unavailable.” Cap evidence strength at 1 for an abstract-only note. Mark a partially read paper as partial. Display coverage beside the score. Describe the score as reading priority for the user's question. Avoid claims of field-wide novelty.

Validate JSON, evidence, and attribution before committing the review. Allow one bounded repair attempt. On repeated failure, retain a reason and move on. Render the permanent note from validated fields and trusted metadata. Include the scoring breakdown, matched topic and seeds, source link, version, evidence pointers, and review date. Link a report to the specific review revision it used.

**Check.** Run a full paper, a truncated extraction, and an abstract-only paper through the same schema. Reject an invented metric and an unknown passage ID. Confirm that an unchanged second run reads none of them again.

## 8. Follow important references and compare papers on the same question

After a review, the collector can fetch the paper's outgoing OpenAlex reference IDs. Identify the cited works used for the method, theorem, data, or baseline. Give the research role those candidate IDs and citation contexts. Ask it to choose up to three and explain the choice. Verify each choice against the supplied list. When the graph is incomplete, accept an identifier found in the paper's bibliography only after a provider record confirms the match. Store unresolved references locally. [OpenAlex citations](https://help.openalex.org/data/works/citations/) · [OpenAlex citation queries](https://help.openalex.org/how-to/api-recipes/)

Let the research role select an unused approved query or year band when reviews expose a gap. Record the gap, selected branch, parent review, and returned candidates. This lets Qwen direct a deeper search without composing arbitrary outbound requests. Start with two citation hops, three children per paper, and 20 newly admitted reference jobs per day, within the overall review budget. Enforce caps across restarts. Skip cycles, repeated searches, and previously reviewed inputs. Add references to the same screening process and queue them at priority 3.

Queue one comparison when two completed reviews address the same question and have comparable assumptions, guarantees, or evaluation settings. Prefer the closest reviewed neighbor and cap new comparison jobs at five per day. Ask what agrees, what differs, and what the evidence permits the reader to conclude. Cite both review IDs and their passages. For example, a theorem for exact robot positions and a simulation with noisy positions may address the same failure, but their guarantees differ. Store the comparison as a separate object. Link it to both papers without rereading either paper.

**Check.** Use a citation graph in which two papers cite each other and share a third reference. Verify that the search ends, the third paper has one review, and the same comparison is written once. Reject a comparison that ranks two results obtained under incompatible conditions without explaining the difference.

## 9. Publish one coherent daily Markdown report

Default to 07:00 in `America/New_York`. At the cutoff, take a short database snapshot of completed, eligible reviews. Exclude withdrawn papers and items already in the user's active seed library from the recommendation list. Use seeds as comparison anchors. Choose up to 15 unreported works above a configurable reading-priority floor, initially 65/100. Keep unselected eligible works for later days. Use the latest eligible review per canonical paper. Sort by final score, then evidence coverage, then stable paper ID. Include a previously reported paper again only for a revision that changes a result, assumption, or conclusion. Link the old and new evidence and label the entry as an update.

Give the writer the frozen reviews and existing comparisons. Ask it for one short overview of the findings and a grounded explanation of what to read first. Require review and passage IDs for its claims. The renderer supplies ranks, dates, numeric scores, URLs, and relative note links. Escape untrusted Markdown, HTML, image links, and Obsidian embeds. Use internal IDs for note paths and construct external links from trusted bibliographic identifiers.

| Report section | Required content |
| --- | --- |
| Opening | Reporting date, cutoff and timezone, counts screened and reviewed, number selected, backlog, and relevant source outages |
| What matters today | Two or three connected findings when supported, each traceable to a selected review |
| Ranked papers | Title, authors, publication date or year, new/older/reference/update label, score and coverage, a short summary, and a specific reason to read it. Link the source and the detailed note. |
| Comparisons and corrections | Include completed comparisons that help interpret the selected work. Identify withdrawals or corrections to previously reported papers. |

A synthetic example follows.

> **1. [Paper title] — 80/100 · older paper · full text.** “The authors check a joint plan before dispatch. Their guarantee assumes bounded travel time. Read it for the assumption your exit-clearance monitor would need to enforce.”

A real entry must link to the paper, the cited section, and the saved note. When only six papers qualify, publish six. With zero, write a short status report and distinguish a quiet day from a source or model outage.

Reserve the next inference slot for the report at cutoff. Set a short deadline for obtaining the slot and validating the synthesis. If Qwen is unavailable or misses the deadline, render the same selections from saved summaries and label this fallback. Use a shared local inference lock for the worker and writer. The worker checks for a pending report before it takes the lock for another call. Save the snapshot and the final Markdown bytes before publishing. Render or restore missing note files from the saved reviews first. Write the report to a temporary file on the output filesystem, then rename atomically. On restart, finish any pending publication from the saved bytes. Enforce one date key and one publication process per date. New completions after cutoff belong to a later report.

**Check.** Test 0, 6, and 20 qualifying papers, a late completion, a missing note file, and a model outage. Restore a missing note from its saved review. Kill the publisher between file rename and the database update. A rerun must produce the same bytes and no duplicate recommendations.

## 10. Deploy, measure, and verify unattended operation

Install one systemd worker service and two persistent calendar timers. Start collection at 00:30, 06:30, 12:30, and 18:30 local time. Trigger the report at 07:00. Use explicit timezones. Resume missed collection intervals from their checkpoints. After a long shutdown, publish the latest due report and disclose the gap. Use its scheduled cutoff for eligibility. Show the time since the preceding report in the opening. [systemd timer behavior](https://man7.org/linux/man-pages/man5/systemd.timer.5.html)

Use `status` to show the last successful fetch per source, oldest queued job per class, counts by coverage, terminal errors, last successful model call, and the published report path. Redact keys, signed URLs, private queries, and raw paper text from service logs. Bound disk growth with a source-cache limit. Keep validated reviews, cited evidence, and report snapshots when removing replaceable downloads. Back up SQLite with its online backup API and retain the config and output. Restore to a fresh directory once. [SQLite backup API](https://www.sqlite.org/backup.html)

Measure capacity on the two RTX A5000s with the actual Qwen model, reasoning settings, and context size. Time short, median, and long papers. Include parsing, section calls, grading, and retries. Estimate daily capacity as available inference minutes divided by measured minutes per review, after reserving time for research, comparisons, and the report. Reduce the deep-review limit if the queue grows over consecutive days. Keep CPU screening independent so the system can still score hundreds. A target of 15 recommendations per day is useful only when enough qualified reviews finish.

| Release check | Acceptance evidence |
| --- | --- |
| Ranking quality | Label a fixed sample of at least 40 candidates across active topics, including near misses. Mark “would read” and its reason before seeing system scores. Tune on one subset. Compare held-out precision at 15 and recall among labeled relevant candidates against keyword-only ranking. Keep the simpler ranker if it does better. |
| Evidence quality | Manually trace ten sampled report claims to the saved passages. Include an abstract-only result and a numeric claim. Correct unsupported claim types before enabling unattended publication. |
| Recovery and quiet operation | Replay duplicate pages, timeouts, a process crash, two simultaneous launches, and a daylight-saving time change. Confirm stable notes, one report per date, and zero model calls for an empty queue. |
| Access limits | Observe service-account egress, parser isolation, provider budgets, and output paths with ordinary and adversarial fixture text. Confirm that only the collector contacts public providers. |
| End to end | Run a local fixture day first. Then run a 24-hour pilot on one real topic. Count candidates screened, papers reviewed, recommendations the user would read, runtime, and backlog. Compare the results with the labeled sample and capacity estimate before enabling the regular timers. |

**Completion condition.** A rebooted workstation resumes unfinished work. A completed paper stays complete until its evidence changes or the operator requests another review. At the next cutoff, the user can open one dated Markdown file, understand why each paper earned its place, and follow each material finding to its source.
