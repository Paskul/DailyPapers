"""One `paperwatch collect` run after the seed sync. Only the collector process uses this package.

Stages, in order, so current releases always get the day's candidate capacity first:
  1. current releases (current_releases.py): arXiv category announcement feeds; dated arXiv API searches that recover
     releases missed during downtime; OpenAlex searches over a recent, overlapping publication window;
  2. the finite historical search of each topic (historical_search.py): approved terms across year bands, alternating
     bands, capped per topic and resumed from saved cursors;
  then the screening stage (paperwatch.screening) scores the candidates and promotes today's papers;
  3. source text (source_text.py): for papers queued for review, a local file, a versioned arXiv PDF, or an OpenAlex
     content PDF, within a daily download budget; then text extraction in the sandbox when it is set up.
Every page of results commits together with its checkpoint (page_ingestion.py), so a crash or failure mid-search
resumes at the first page that was not stored. Every stop is visible in source_checkpoints: active, capped,
exhausted, or blocked, with its reason.
"""
