"""A fixed sample of candidates for the owner to label by hand (plan part 10, release check "Ranking quality").

"Label a fixed sample of at least 40 candidates across active topics, including near misses. Mark 'would read' and
its reason before seeing system scores."

The sample is drawn per topic from three bands of that topic's screened candidates (seeds excluded), so it covers
what screening chose, what it nearly chose, and what it left out:
  chosen       papers screening promoted for the topic;
  near miss    the highest-ranked candidates that were NOT promoted (ranked by the raw screening score);
  the rest     a random draw from everything below the near misses.
The CSV shows title, authors, year, a link built from trusted identifiers, and the abstract, and NO score, rank,
or band; the rows are shuffled, so their order reveals nothing either. Fill in `would_read` (yes/no) and `reason`.
The draw is deterministic for a given --seed, so the same sample can be rebuilt. Scores are compared only later,
from the database, by paper id.
"""

from __future__ import annotations

import csv
import json
import math
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from paperwatch.config import AppConfig
from paperwatch.rendering.markdown_safety import build_trusted_source_url
from paperwatch.research_profile import load_active_seed_library_paper_ids

LABEL_COLUMNS = ("sample_number", "topic_id", "topic_question", "paper_id", "title", "authors", "year", "link",
                 "abstract", "would_read", "reason")
CHOSEN_SHARE = 0.3
NEAR_MISS_SHARE = 0.3  # the remaining share is drawn from the rest
NEAR_MISS_BAND_WIDTH_FACTOR = 3  # near misses come from the next (3 x their share) unpromoted ranks
MINIMUM_SAMPLE_SIZE = 40


@dataclass(frozen=True)
class SampledCandidate:
    topic_id: str
    paper_id: str
    band: str  # chosen | near_miss | rest; never written to the CSV


def draw_label_sample(connection: sqlite3.Connection, app_config: AppConfig, *, sample_size: int,
                      seed: int) -> list[SampledCandidate]:
    generator = random.Random(seed)
    active_topics = [topic for topic in app_config.topics]
    if not active_topics:
        return []
    per_topic_size = math.ceil(max(sample_size, MINIMUM_SAMPLE_SIZE) / len(active_topics))
    seed_paper_ids = load_active_seed_library_paper_ids(connection, app_config)
    sample: list[SampledCandidate] = []
    for topic in active_topics:
        ranked_paper_ids = [row["paper_id"] for row in connection.execute(
            """SELECT paper_id FROM candidate_topic_scores WHERE topic_id = ?
               ORDER BY unnormalized_score DESC, paper_id""", (topic.topic_id,))
            if row["paper_id"] not in seed_paper_ids]
        promoted_paper_ids = {row["paper_id"] for row in connection.execute(
            "SELECT paper_id FROM paper_promotions WHERE topic_id = ? AND promotion_reason != 'direct'",
            (topic.topic_id,))}
        chosen = [paper_id for paper_id in ranked_paper_ids if paper_id in promoted_paper_ids]
        unchosen = [paper_id for paper_id in ranked_paper_ids if paper_id not in promoted_paper_ids]
        chosen_count = min(len(chosen), round(per_topic_size * CHOSEN_SHARE))
        near_miss_count = round(per_topic_size * NEAR_MISS_SHARE)
        near_miss_band = unchosen[:near_miss_count * NEAR_MISS_BAND_WIDTH_FACTOR]
        near_misses = generator.sample(near_miss_band, min(near_miss_count, len(near_miss_band)))
        rest_pool = unchosen[len(near_miss_band):]
        rest_count = per_topic_size - chosen_count - len(near_misses)
        drawn_bands = (("chosen", generator.sample(chosen, chosen_count)), ("near_miss", near_misses),
                       ("rest", generator.sample(rest_pool, min(max(0, rest_count), len(rest_pool)))))
        sample += [SampledCandidate(topic.topic_id, paper_id, band)
                   for band, paper_ids in drawn_bands for paper_id in paper_ids]
    generator.shuffle(sample)
    return sample


def write_label_sheet(connection: sqlite3.Connection, app_config: AppConfig, sample: list[SampledCandidate],
                      output_path: Path) -> None:
    """The CSV the owner fills in. Written only if the file does not exist yet (labels are never overwritten)."""
    question_by_topic = {topic.topic_id: topic.question_for_local_model for topic in app_config.topics}
    with output_path.open("x", newline="", encoding="utf-8") as label_file:
        writer = csv.DictWriter(label_file, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        for sample_number, candidate in enumerate(sample, start=1):
            paper_row = connection.execute("SELECT * FROM papers WHERE paper_id = ?",
                                           (candidate.paper_id,)).fetchone()
            writer.writerow({
                "sample_number": sample_number, "topic_id": candidate.topic_id,
                "topic_question": question_by_topic.get(candidate.topic_id, ""), "paper_id": candidate.paper_id,
                "title": paper_row["title"], "authors": ", ".join(json.loads(paper_row["authors_json"])[:6]),
                "year": (paper_row["first_published_date"] or "")[:4],
                "link": _first_trusted_link(connection, candidate.paper_id),
                "abstract": paper_row["abstract"] or "(no abstract)", "would_read": "", "reason": ""})
    output_path.chmod(0o600)  # holds your private topic questions


def _first_trusted_link(connection: sqlite3.Connection, paper_id: str) -> str:
    for alias_row in connection.execute(
            """SELECT provider, provider_paper_id FROM paper_aliases WHERE paper_id = ?
               ORDER BY CASE provider WHEN 'arxiv' THEN 0 WHEN 'doi' THEN 1 ELSE 2 END, provider_paper_id""",
            (paper_id,)):
        trusted_url = build_trusted_source_url(alias_row["provider"], alias_row["provider_paper_id"])
        if trusted_url:
            return trusted_url
    return ""
