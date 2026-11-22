"""Plan part 6: score every candidate cheaply and dispatch work in order.

Check from the plan: feed in 500 candidates with known nearest seeds, inspect scores and promotions; put all four job
types in the queue and verify claim order; drain the queue and check that the model request count stays constant.
Also covered: the formula (omitted signals, renormalized weights, max(0, cosine)), exact phrase matching, topic
shares, spare slots, and near-cutoff picks; the daily limit across runs and days; rescoring only when the profile
changes; the seven-day fresh window; moving a job up (never down) a class; new text of a promoted paper; a missing
embedding model; and, when the pinned model is installed, the real ONNX embedder.

Embeddings with KNOWN geometry come from LookupTableEmbedder: every text maps to a chosen unit vector, so the nearest
seed of each candidate is known in advance.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from paperwatch.app_context import AppContext
from paperwatch.catalog import record_collected_paper
from paperwatch.collection.page_ingestion import ingest_records_with_checkpoints
from paperwatch.commands.collect_command import run_collect_command
from paperwatch.commands.work_command import WorkCycleOutcome, run_single_work_cycle
from paperwatch.config import load_app_config
from paperwatch.db import write_transaction
from paperwatch.dispatch_order import (
    JobDispatchOrder,
    JobPriorityClass,
    dispatch_order_for_comparison,
    dispatch_order_for_important_reference,
    dispatch_order_for_older_paper,
    dispatch_order_for_operator_request,
    dispatch_order_for_promoted_paper,
)
from paperwatch.errors import ConfigurationError, ScreeningUnavailable
from paperwatch.jobs import (
    COMPARE_REVIEWS_JOB_KIND,
    REVIEW_PAPER_JOB_KIND,
    claim_next_due_job,
    enqueue_job_if_absent,
    finish_job_with_writes,
    summarize_queued_jobs_by_class,
    upgrade_queued_job_dispatch_order_if_higher,
)
from paperwatch.research_profile import load_active_seed_library_paper_ids
from paperwatch.screening.embedding_model_registry import PINNED_EMBEDDING_MODELS
from paperwatch.screening.phrase_matching import find_matched_phrases, split_into_match_words
from paperwatch.screening.promotion_selection import (
    PromotionCandidate,
    PromotionReason,
    compute_slots_available,
    select_papers_for_promotion,
)
from paperwatch.screening.review_promotion import (
    count_screening_promotions_today,
    promote_paper_directly_and_queue_review,
)
from paperwatch.screening.screening_score import (
    ScreeningSignals,
    SignalWeights,
    combine_screening_signals,
    normalize_cosine_similarity,
)
from paperwatch.screening.screening_stage import run_screening_and_promotion
from paperwatch.screening.signal_normalization import compute_pool_normalized_scores, midrank_percentiles
from paperwatch.screening.text_embedders import load_text_embedder, verify_pinned_model_files
from paperwatch.screening.topic_screening_profiles import build_screening_document_text
from paperwatch.seeds.seed_items import build_csl_json_file_key
from paperwatch.seeds.seed_store import SeedSourceReference
from paperwatch.seeds.seed_sync import import_csl_json_seed_file
from paperwatch.sources.types import CollectedPaperRecord

from conftest import RECORDED_PAPER_FIXTURE, PaperwatchHome, build_in_process_app

SCREENING_MOMENT = datetime(2026, 11, 19, 12, 0, tzinfo=timezone.utc)
VECTOR_DIMENSIONS = 64
SEED_COUNT = 10
CANDIDATE_COUNT = 500
ALPHA_SEED_FILE = "seeds/alpha.csl.json"
BETA_SEED_FILE = "seeds/beta.csl.json"
BETA_QUESTION = "Which verified Dafny programs check robot plans?"
BETA_PHRASE = "verified Dafny programs"
TWO_TOPICS_WITH_SEEDS = f"""
[[topics]]
id = "alpha"
arxiv_categories = ["cs.RO"]
csl_json_files = ["{ALPHA_SEED_FILE}"]

[[topics]]
id = "beta"
question = "{BETA_QUESTION}"
public_queries = ["{BETA_PHRASE}"]
arxiv_categories = ["cs.LO"]
csl_json_files = ["{BETA_SEED_FILE}"]
"""


# --------------------------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------------------------


class LookupTableEmbedder:
    """Returns a chosen unit vector per exact text (zero for unknown text) and counts what it was asked to embed."""

    identity = "lookup-table-test"
    dimensions = VECTOR_DIMENSIONS

    def __init__(self, vector_by_text: dict[str, np.ndarray]) -> None:
        self.vector_by_text = vector_by_text
        self.embedded_texts: list[str] = []

    def prepare_query_text(self, query_text: str) -> str:
        return query_text

    def embed_texts(self, model_input_texts: list[str]) -> np.ndarray:
        self.embedded_texts += model_input_texts
        return np.array(
            [self.vector_by_text.get(text, np.zeros(VECTOR_DIMENSIONS)) for text in model_input_texts], dtype=np.float32
        )


class EmbedderFactory:
    """Counts how often screening had to build an embedder (zero when every score is current)."""

    def __init__(self, embedder) -> None:
        self.embedder = embedder
        self.build_count = 0

    def __call__(self):
        self.build_count += 1
        return self.embedder


def unit(vector: np.ndarray) -> np.ndarray:
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def write_csl_seeds(home_dir: Path, relative_path: str, seed_numbers: range) -> list[tuple[str, str]]:
    """Write a CSL JSON file of synthetic seeds; returns [(seed id, document text)] for the vector table."""
    seed_items = [
        {
            "id": f"seed-{seed_number}",
            "type": "article-journal",
            "title": f"Seed paper {seed_number}",
            "abstract": f"Synthetic seed abstract number {seed_number}.",
            "issued": {"date-parts": [[2020]]},
        }
        for seed_number in seed_numbers
    ]
    seed_path = home_dir / relative_path
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    seed_path.write_text(json.dumps(seed_items))
    file_key = build_csl_json_file_key(relative_path)
    return [
        (f"csl_json:{file_key}#{item['id']}", build_screening_document_text(item["title"], item["abstract"]))
        for item in seed_items
    ]


def candidate_record(
    candidate_number: int, *, abstract: str, first_published_date: str, provider_paper_id: str | None = None
) -> CollectedPaperRecord:
    return CollectedPaperRecord(
        provider="fixture",
        provider_paper_id=provider_paper_id or f"candidate-{candidate_number}",
        title=f"Candidate paper {candidate_number}",
        authors=("A. Author",),
        abstract=abstract,
        first_published_date=first_published_date,
        latest_revision_date=None,
        source_version=None,
        discovery_route="fixture:step6",
    )


@pytest.fixture
def five_hundred_candidates(tmp_path):
    """Two seed topics (alpha: seeds 0-4; beta: seeds 5-9 plus a question and a phrase) and 500 candidates whose
    nearest seed is seed (n % 10), at varying strength. Every 20th beta candidate contains the exact phrase."""
    app = build_in_process_app(tmp_path / "home", topics_toml=TWO_TOPICS_WITH_SEEDS)
    random_generator = np.random.default_rng(6)
    seed_directions, _ = np.linalg.qr(random_generator.normal(size=(VECTOR_DIMENSIONS, VECTOR_DIMENSIONS)))
    seed_directions = seed_directions[: SEED_COUNT + 1]  # orthonormal rows; the last one is a pure-noise direction
    vector_by_text: dict[str, np.ndarray] = {}
    seed_id_by_number: dict[int, str] = {}
    for relative_path, seed_numbers in ((ALPHA_SEED_FILE, range(0, 5)), (BETA_SEED_FILE, range(5, 10))):
        for seed_number, (seed_id, seed_text) in zip(
            seed_numbers, write_csl_seeds(app.config.home_dir, relative_path, seed_numbers)
        ):
            vector_by_text[seed_text] = unit(seed_directions[seed_number])
            seed_id_by_number[seed_number] = seed_id
        import_csl_json_seed_file(app.connection, app.config, SeedSourceReference.for_csl_json_file(relative_path))
    vector_by_text[BETA_QUESTION] = unit(seed_directions[5:10].sum(axis=0))

    expected = {}
    for candidate_number in range(CANDIDATE_COUNT):
        nearest_seed_number = candidate_number % SEED_COUNT
        signal_strength = 0.55 + 0.45 * (candidate_number / CANDIDATE_COUNT)
        has_phrase = nearest_seed_number >= 5 and candidate_number % 20 == 5
        abstract = f"Synthetic candidate abstract {candidate_number}." + (
            " It studies verified Dafny programs." if has_phrase else ""
        )
        published = "2026-11-17" if candidate_number % 50 == 0 else "2018-06"  # a few fresh releases
        record = candidate_record(candidate_number, abstract=abstract, first_published_date=published)
        paper_id = record_collected_paper(app.connection, record).paper_id
        noise = unit(random_generator.normal(size=VECTOR_DIMENSIONS) * 0.2 + seed_directions[SEED_COUNT])
        vector_by_text[build_screening_document_text(record.title, abstract)] = unit(
            signal_strength * seed_directions[nearest_seed_number] + (1 - signal_strength) * noise
        )
        expected[paper_id] = (seed_id_by_number[nearest_seed_number], nearest_seed_number, has_phrase, published)
    yield app, LookupTableEmbedder(vector_by_text), expected
    app.close()


def screen(app: AppContext, embedder_factory, moment: datetime = SCREENING_MOMENT):
    return run_screening_and_promotion(
        app.connection,
        app.config,
        active_seed_paper_ids=load_active_seed_library_paper_ids(app.connection, app.config),
        moment=moment,
        build_embedder=embedder_factory,
    )


def expected_pool_normalized_scores(score_rows) -> dict[str, dict[str, float]]:
    """An independent re-implementation of the rule: per topic and signal, the mid-rank percentile of each raw value
    among the pool's values; then the weighted mean of the available percentiles (weights 0.5 / 0.3 / 0.2)."""
    weights = {"question_similarity": 0.5, "seed_similarity": 0.3, "phrase_match_fraction": 0.2}
    percentiles: dict[tuple[str, str, str], float] = {}
    for topic_id in {row["topic_id"] for row in score_rows}:
        topic_rows = [row for row in score_rows if row["topic_id"] == topic_id]
        for signal_name in weights:
            values = [row[signal_name] for row in topic_rows if row[signal_name] is not None]
            for row in topic_rows:
                if row[signal_name] is not None:
                    lower = sum(value < row[signal_name] for value in values)
                    equal = sum(value == row[signal_name] for value in values)
                    percentiles[(row["paper_id"], topic_id, signal_name)] = (lower + 0.5 * equal) / len(values)
    normalized: dict[str, dict[str, float]] = {}
    for row in score_rows:
        available = [
            (weight, percentiles[(row["paper_id"], row["topic_id"], name)])
            for name, weight in weights.items()
            if row[name] is not None
        ]
        normalized.setdefault(row["paper_id"], {})[row["topic_id"]] = sum(
            weight * value for weight, value in available
        ) / sum(weight for weight, _value in available)
    return normalized


# --------------------------------------------------------------------------------------------------------------------
# The formula and the lexical signal
# --------------------------------------------------------------------------------------------------------------------


def test_unavailable_signals_are_omitted_and_the_weights_renormalized():
    weights = SignalWeights(question_similarity=0.5, seed_similarity=0.3, phrase_match=0.2)
    every_signal = ScreeningSignals(question_similarity=0.8, seed_similarity=0.6, phrase_match_fraction=0.5)
    assert combine_screening_signals(every_signal, weights) == pytest.approx(0.5 * 0.8 + 0.3 * 0.6 + 0.2 * 0.5)
    seeds_only = ScreeningSignals(question_similarity=None, seed_similarity=0.6, phrase_match_fraction=None)
    assert combine_screening_signals(seeds_only, weights) == pytest.approx(0.6)  # not 0.3 * 0.6
    no_question = ScreeningSignals(question_similarity=None, seed_similarity=0.6, phrase_match_fraction=1.0)
    assert combine_screening_signals(no_question, weights) == pytest.approx((0.3 * 0.6 + 0.2 * 1.0) / 0.5)
    assert combine_screening_signals(ScreeningSignals(None, None, None), weights) is None
    assert normalize_cosine_similarity(-0.4) == 0.0 and normalize_cosine_similarity(0.7) == pytest.approx(0.7)


def test_phrases_match_exact_words_in_order_ignoring_case_hyphens_and_plurals():
    text_words = split_into_match_words("Verified DAFNY programs for Multi-Robot runtime monitors.")
    assert find_matched_phrases(("dafny",), text_words) == ("dafny",)
    assert find_matched_phrases(("multi robot runtime monitor",), text_words) == ("multi robot runtime monitor",)
    assert find_matched_phrases(("verified dafny program", "dafny verified"), text_words) == ("verified dafny program",)
    assert find_matched_phrases(("runtime assurance",), text_words) == ()
    assert split_into_match_words("class glass") == ("class", "glass")  # "ss" is not a plural


def test_percentiles_use_mid_ranks_so_ties_share_one_value():
    assert midrank_percentiles([0.2, 0.5, 0.5, 0.9]) == [0.125, 0.5, 0.5, 0.875]
    assert midrank_percentiles([0.37]) == [0.5]  # a lone candidate sits in the middle
    assert midrank_percentiles([0.0] * 9 + [1.0]) == [0.45] * 9 + [0.95]  # the typical phrase signal


def test_topics_whose_raw_cosines_run_higher_do_not_outrank_other_topics():
    # "high" scores every paper 0.70-0.79, "low" 0.40-0.49: same spread, different band (as bge does live).
    signals = {f"h{n}": {"high": ScreeningSignals(0.70 + n / 100, None, None)} for n in range(10)}
    signals.update({f"l{n}": {"low": ScreeningSignals(0.40 + n / 100, None, None)} for n in range(10)})
    weights = SignalWeights(question_similarity=0.5, seed_similarity=0.3, phrase_match=0.2)
    normalized = compute_pool_normalized_scores(signals, weights)
    for rank in range(10):
        assert normalized[f"h{rank}"]["high"] == pytest.approx(normalized[f"l{rank}"]["low"])
    # So the best "low" paper ties the best "high" paper instead of losing to all ten of them.
    assert normalized["l9"]["low"] == pytest.approx(normalized["h9"]["high"])


def test_a_phrase_match_no_longer_outweighs_the_question_signal():
    # Ten papers on one topic; question cosines bunch up in 0.60-0.69. Only p0, the WORST question match, has the
    # phrase. With raw values the 0/1 phrase signal made p0 the best paper; on one scale p9 (best question match)
    # wins, and the phrase lifts p0 by a bounded amount.
    signals = {f"p{n}": {"topic": ScreeningSignals(0.60 + n / 100, None, 1.0 if n == 0 else 0.0)} for n in range(10)}
    weights = SignalWeights(question_similarity=0.5, seed_similarity=0.3, phrase_match=0.2)
    raw_scores = {
        paper_id: combine_screening_signals(by_topic["topic"], weights) for paper_id, by_topic in signals.items()
    }
    assert max(raw_scores, key=raw_scores.get) == "p0"
    normalized = {
        paper_id: scores["topic"] for paper_id, scores in compute_pool_normalized_scores(signals, weights).items()
    }
    assert max(normalized, key=normalized.get) == "p9"
    # The match moves p0's phrase percentile from 0.45 to 0.95, worth 0.2/0.5 x 0.5 = 20 question percentile points:
    # exactly two places among ten papers, so p0 passes p1 and ties p2.
    assert normalized["p0"] > normalized["p1"] and normalized["p0"] == pytest.approx(normalized["p2"])


# --------------------------------------------------------------------------------------------------------------------
# 500 candidates with known nearest seeds
# --------------------------------------------------------------------------------------------------------------------


def test_five_hundred_candidates_get_known_nearest_seeds_and_thirty_fair_promotions(five_hundred_candidates):
    app, embedder, expected = five_hundred_candidates
    embedder_factory = EmbedderFactory(embedder)
    stage_result = screen(app, embedder_factory)
    assert stage_result.embedder_failure is None
    assert "500 candidate(s) across 2 topic(s); 1000 topic score(s) computed now" in stage_result.report_lines[0]

    score_rows = app.connection.execute("SELECT * FROM candidate_topic_scores").fetchall()
    assert len(score_rows) == 2 * CANDIDATE_COUNT  # seeds are not candidates
    seed_paper_ids = load_active_seed_library_paper_ids(app.connection, app.config)
    assert not seed_paper_ids & {row["paper_id"] for row in score_rows}
    for score_row in score_rows:
        nearest_seed_id, nearest_seed_number, has_phrase, _published = expected[score_row["paper_id"]]
        assert 0.0 <= score_row["unnormalized_score"] <= 1.0
        owning_topic = "alpha" if nearest_seed_number < 5 else "beta"
        if score_row["topic_id"] == owning_topic:
            assert json.loads(score_row["closest_seed_ids_json"])[0] == nearest_seed_id
        if score_row["topic_id"] == "alpha":  # seeds-only: the question and phrase signals are omitted
            assert score_row["question_similarity"] is None and score_row["phrase_match_fraction"] is None
            assert score_row["unnormalized_score"] == pytest.approx(score_row["seed_similarity"])
        else:
            assert score_row["phrase_match_fraction"] == (1.0 if has_phrase else 0.0)
            assert (json.loads(score_row["matched_phrases_json"]) == [BETA_PHRASE]) == has_phrase
            assert score_row["unnormalized_score"] == pytest.approx(
                0.5 * score_row["question_similarity"]
                + 0.3 * score_row["seed_similarity"]
                + 0.2 * score_row["phrase_match_fraction"]
            )

    # The seed signal is the mean of the three closest seeds, recomputed here by hand for one candidate.
    some_row = next(row for row in score_rows if row["topic_id"] == "beta")
    candidate_text = build_screening_document_text(
        *app.connection.execute(
            "SELECT title, abstract FROM paper_editions WHERE paper_id = ?", (some_row["paper_id"],)
        ).fetchone()
    )
    beta_seed_vectors = [
        embedder.vector_by_text[f"Seed paper {n}. Synthetic seed abstract number {n}."] for n in range(5, 10)
    ]
    cosines = sorted(
        (max(0.0, float(embedder.vector_by_text[candidate_text] @ seed)) for seed in beta_seed_vectors), reverse=True
    )
    assert some_row["seed_similarity"] == pytest.approx(sum(cosines[:3]) / 3, abs=1e-5)

    # Promotions use each signal's percentile within the topic's pool (recomputed here independently).
    normalized = expected_pool_normalized_scores(score_rows)
    # Seeds-only "alpha" has a single signal, so its percentile ranking is exactly its raw ranking.
    alpha_by_raw = sorted(
        (row for row in score_rows if row["topic_id"] == "alpha"),
        key=lambda row: (-row["unnormalized_score"], row["paper_id"]),
    )
    assert [row["paper_id"] for row in alpha_by_raw] == sorted(
        (row["paper_id"] for row in alpha_by_raw), key=lambda paper_id: (-normalized[paper_id]["alpha"], paper_id)
    )

    # Promotions: 27 regular slots = 13 per topic share + 1 spare by score; then 3 near the cutoff.
    promotions = app.connection.execute("SELECT * FROM paper_promotions").fetchall()
    reasons = [row["promotion_reason"] for row in promotions]
    assert len(promotions) == 30
    assert reasons.count("topic_share") == 26 and reasons.count("spare_slot") == 1 and reasons.count("near_cutoff") == 3
    for topic_id in ("alpha", "beta"):
        share_ids = {
            row["paper_id"]
            for row in promotions
            if row["promotion_reason"] == "topic_share" and row["topic_id"] == topic_id
        }
        assert len(share_ids) == 13
        topic_ranking = sorted(
            normalized, key=lambda paper_id, topic_id=topic_id: (-normalized[paper_id][topic_id], paper_id)
        )
        other_shares = {
            row["paper_id"]
            for row in promotions
            if row["promotion_reason"] == "topic_share" and row["topic_id"] != topic_id
        }
        assert share_ids == set([paper_id for paper_id in topic_ranking if paper_id not in other_shares][:13])
    best_scores = {paper_id: max(scores.values()) for paper_id, scores in normalized.items()}
    regular_ids = {row["paper_id"] for row in promotions if row["promotion_reason"] != "near_cutoff"}
    best_unselected = sorted(
        (paper_id for paper_id in best_scores if paper_id not in regular_ids),
        key=lambda paper_id: (-best_scores[paper_id], paper_id),
    )
    assert {row["paper_id"] for row in promotions if row["promotion_reason"] == "near_cutoff"} == set(
        best_unselected[:3]
    )

    # Each promoted paper has one queued reading in its class; fresh releases are class 1, the rest class 2.
    job_rows = app.connection.execute("SELECT * FROM jobs").fetchall()
    assert len(job_rows) == 30 and {row["status"] for row in job_rows} == {"queued"}
    for job_row in job_rows:
        published = expected[job_row["paper_id"]][3]
        assert job_row["priority"] == (1 if published == "2026-11-17" else 2)
        assert job_row["within_class_order_primary"] == pytest.approx(best_scores[job_row["paper_id"]])
    for promotion_row in promotions:
        assert promotion_row["screening_score"] == pytest.approx(best_scores[promotion_row["paper_id"]])
        assert promotion_row["normalization_pool_size"] == CANDIDATE_COUNT
        best_raw = app.connection.execute(
            "SELECT unnormalized_score FROM candidate_topic_scores WHERE paper_id = ? AND topic_id = ?",
            (promotion_row["paper_id"], promotion_row["best_topic_id"]),
        ).fetchone()[0]
        assert promotion_row["unnormalized_score"] == pytest.approx(best_raw)
    status_counts = dict(
        app.connection.execute(
            "SELECT status, COUNT(*) FROM papers WHERE paper_id NOT IN (SELECT paper_id FROM seed_items) "
            "GROUP BY status"
        ).fetchall()
    )
    assert status_counts == {"queued": 30, "screened_out": 470}

    # The same day again: every score is current (no embedder built) and the daily limit is used up.
    embedded_before = len(embedder.embedded_texts)
    repeat_result = screen(app, embedder_factory, SCREENING_MOMENT + timedelta(hours=6))
    assert embedder_factory.build_count == 1 and len(embedder.embedded_texts) == embedded_before
    assert "Promotion: 0 paper(s) queued" in repeat_result.report_lines[1]

    # The next UTC day: 30 more from the remaining pool, still without embedding anything.
    screen(app, embedder_factory, SCREENING_MOMENT + timedelta(days=1))
    assert embedder_factory.build_count == 1
    assert app.connection.execute("SELECT COUNT(*) FROM paper_promotions").fetchone()[0] == 60
    assert app.connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 60


def test_a_changed_profile_rescores_unpromoted_candidates_only(five_hundred_candidates):
    app, embedder, _expected = five_hundred_candidates
    screen(app, EmbedderFactory(embedder))
    promoted_rows_before = {
        row["paper_id"]: row["scored_at"]
        for row in app.connection.execute(
            "SELECT paper_id, scored_at FROM candidate_topic_scores WHERE paper_id IN "
            "(SELECT paper_id FROM paper_promotions)"
        )
    }

    topics_path = app.config.home_dir / "topics.toml"
    changed_question = "Which verified Dafny programs check robot team plans?"
    topics_path.write_text(topics_path.read_text().replace(BETA_QUESTION, changed_question))
    embedder.vector_by_text[changed_question] = embedder.vector_by_text[BETA_QUESTION]
    changed_app = AppContext.load(app.config.home_dir, model_client=app.model_client)
    embedded_before = len(embedder.embedded_texts)
    rescore_result = screen(changed_app, EmbedderFactory(embedder), SCREENING_MOMENT + timedelta(minutes=5))

    # Only the new question was embedded: candidate and seed vectors came from the cache.
    assert embedder.embedded_texts[embedded_before:] == [changed_question]
    assert f"{CANDIDATE_COUNT - 30} topic score(s) computed now" in rescore_result.report_lines[0]
    assert {
        row["paper_id"]: row["scored_at"]
        for row in changed_app.connection.execute(
            "SELECT paper_id, scored_at FROM candidate_topic_scores WHERE paper_id IN "
            "(SELECT paper_id FROM paper_promotions)"
        )
    } == promoted_rows_before
    changed_app.close()


def test_promotion_selection_divides_slots_across_topics_before_score():
    # Topic "loud" has the best scores everywhere; "quiet" still gets its share.
    candidates = [
        PromotionCandidate(f"p{n:02d}", {"loud": 0.9 - n * 0.01, "quiet": 0.3 - n * 0.01}) for n in range(10)
    ] + [PromotionCandidate("q-only", {"quiet": 0.25})]
    slots = compute_slots_available(
        active_topic_ids=("loud", "quiet"),
        daily_promotion_limit=7,
        near_cutoff_promotions=1,
        regular_promotions_today=0,
        near_cutoff_promotions_today=0,
        topic_share_promotions_today_by_topic_id={},
    )
    decisions = select_papers_for_promotion(candidates, slots, minimum_screening_score=0.0)
    by_reason = {reason: [d.paper_id for d in decisions if d.promotion_reason == reason] for reason in PromotionReason}
    assert by_reason[PromotionReason.TOPIC_SHARE] == ["p00", "p01", "p02", "p03", "p04", "p05"]  # turns: loud, quiet
    assert [d.selecting_topic_id for d in decisions][:6] == ["loud", "quiet"] * 3
    assert by_reason[PromotionReason.SPARE_SLOT] == [] and by_reason[PromotionReason.NEAR_CUTOFF] == ["p06"]

    # A minimum score leaves a topic without qualifying candidates; its share becomes spare slots by best score.
    strict = select_papers_for_promotion(candidates, slots, minimum_screening_score=0.5)
    assert [d.promotion_reason for d in strict].count(PromotionReason.TOPIC_SHARE) == 3
    assert [d.promotion_reason for d in strict].count(PromotionReason.SPARE_SLOT) == 3
    assert all(d.selecting_topic_id == "loud" for d in strict if d.promotion_reason != PromotionReason.NEAR_CUTOFF)

    # Earlier promotions today reduce each topic's share and the regular slots.
    later_slots = compute_slots_available(
        active_topic_ids=("loud", "quiet"),
        daily_promotion_limit=7,
        near_cutoff_promotions=1,
        regular_promotions_today=5,
        near_cutoff_promotions_today=1,
        topic_share_promotions_today_by_topic_id={"loud": 3, "quiet": 2},
    )
    assert later_slots.remaining_share_by_topic_id == {"loud": 0, "quiet": 1}
    later = select_papers_for_promotion(candidates, later_slots, minimum_screening_score=0.0)
    assert [(d.paper_id, d.promotion_reason) for d in later] == [("p00", PromotionReason.TOPIC_SHARE)]


# --------------------------------------------------------------------------------------------------------------------
# Dispatch order
# --------------------------------------------------------------------------------------------------------------------


def test_release_class_uses_the_original_release_date_and_a_seven_day_window():
    fresh = dispatch_order_for_promoted_paper("2026-11-17", 0.6, SCREENING_MOMENT)
    assert fresh.priority_class == JobPriorityClass.NEW_RELEASE
    assert fresh.new_release_fresh_until == "2026-11-25T00:00:00Z"  # through the 7th day after release
    assert (
        dispatch_order_for_promoted_paper("2026-11-12", 0.6, SCREENING_MOMENT).priority_class
        == JobPriorityClass.NEW_RELEASE
    )
    for older_date in ("2026-11-11", "2018-06", "2026-11", None):  # "2026-11" counts from 2026-11-01
        older = dispatch_order_for_promoted_paper(older_date, 0.6, SCREENING_MOMENT)
        assert older.priority_class == JobPriorityClass.HISTORICAL_SEARCH and older.new_release_fresh_until is None


def test_all_four_job_types_are_claimed_in_class_order(tmp_path, monkeypatch):
    app = build_in_process_app(tmp_path / "home")
    clock_moment = [SCREENING_MOMENT]
    monkeypatch.setattr("paperwatch.jobs.utc_now", lambda: clock_moment[0])
    monkeypatch.setattr("paperwatch.jobs.utc_now_timestamp", lambda: clock_moment[0].strftime("%Y-%m-%dT%H:%M:%SZ"))
    past_window = (SCREENING_MOMENT - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    future_window = (SCREENING_MOMENT + timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    queued_jobs = [  # (job key, kind, dispatch order), enqueued in this order, 5 s apart
        ("comparison-2-dims", COMPARE_REVIEWS_JOB_KIND, dispatch_order_for_comparison(2)),
        ("reference-low-importance", REVIEW_PAPER_JOB_KIND, dispatch_order_for_important_reference(1.0, 0.9)),
        ("older-0.5", REVIEW_PAPER_JOB_KIND, dispatch_order_for_older_paper(0.5)),
        (
            "new-0.6-first",
            REVIEW_PAPER_JOB_KIND,
            JobDispatchOrder(JobPriorityClass.NEW_RELEASE, 0.6, new_release_fresh_until=future_window),
        ),
        ("comparison-5-dims", COMPARE_REVIEWS_JOB_KIND, dispatch_order_for_comparison(5)),
        (
            "reference-high-importance-low-topic",
            REVIEW_PAPER_JOB_KIND,
            dispatch_order_for_important_reference(3.0, 0.2),
        ),
        (
            "reference-high-importance-high-topic",
            REVIEW_PAPER_JOB_KIND,
            dispatch_order_for_important_reference(3.0, 0.8),
        ),
        ("older-0.8", REVIEW_PAPER_JOB_KIND, dispatch_order_for_older_paper(0.8)),
        (
            "new-0.7-window-over",
            REVIEW_PAPER_JOB_KIND,
            JobDispatchOrder(JobPriorityClass.NEW_RELEASE, 0.7, new_release_fresh_until=past_window),
        ),
        (
            "new-0.9",
            REVIEW_PAPER_JOB_KIND,
            JobDispatchOrder(JobPriorityClass.NEW_RELEASE, 0.9, new_release_fresh_until=future_window),
        ),
        (
            "new-0.6-second",
            REVIEW_PAPER_JOB_KIND,
            JobDispatchOrder(JobPriorityClass.NEW_RELEASE, 0.6, new_release_fresh_until=future_window),
        ),
        ("operator-reread", REVIEW_PAPER_JOB_KIND, dispatch_order_for_operator_request()),
    ]
    with write_transaction(app.connection):
        for job_key, kind, dispatch_order in queued_jobs:
            enqueue_job_if_absent(
                app.connection,
                job_key=job_key,
                kind=kind,
                paper_id=None,
                input_version="v",
                input_payload={},
                dispatch_order=dispatch_order,
                max_attempts=3,
            )
            clock_moment[0] += timedelta(seconds=5)
    summaries = {
        summary.priority_class: summary.queued_count
        for summary in summarize_queued_jobs_by_class(app.connection, clock_moment[0])
    }
    assert summaries == {
        JobPriorityClass.OPERATOR_REQUEST: 1,
        JobPriorityClass.NEW_RELEASE: 3,
        JobPriorityClass.HISTORICAL_SEARCH: 3,
        JobPriorityClass.IMPORTANT_REFERENCE: 3,
        JobPriorityClass.COMPARISON: 2,
    }

    claim_order = []
    while (claimed_job := claim_next_due_job(app.connection, lease_duration_seconds=60)) is not None:
        claim_order.append(claimed_job.job_key)
        assert finish_job_with_writes(app.connection, claimed_job, lambda connection: None)
    assert claim_order == [
        "operator-reread",
        "new-0.9",
        "new-0.6-first",
        "new-0.6-second",  # score, then queue age
        "older-0.8",
        "new-0.7-window-over",
        "older-0.5",  # a release past its window competes as older work
        "reference-high-importance-high-topic",
        "reference-high-importance-low-topic",
        "reference-low-importance",
        "comparison-5-dims",
        "comparison-2-dims",
    ]
    app.close()


def test_a_queued_job_moves_up_to_a_higher_class_but_never_down(tmp_path):
    app = build_in_process_app(tmp_path / "home")
    with write_transaction(app.connection):
        enqueue_job_if_absent(
            app.connection,
            job_key="paper-job",
            kind=REVIEW_PAPER_JOB_KIND,
            paper_id=None,
            input_version="v",
            input_payload={},
            max_attempts=3,
            dispatch_order=dispatch_order_for_important_reference(2.0, 0.4),
        )
        assert not upgrade_queued_job_dispatch_order_if_higher(
            app.connection, "paper-job", dispatch_order_for_comparison(9)
        )
        assert upgrade_queued_job_dispatch_order_if_higher(
            app.connection, "paper-job", dispatch_order_for_older_paper(0.7)
        )
        assert not upgrade_queued_job_dispatch_order_if_higher(
            app.connection, "paper-job", dispatch_order_for_important_reference(5.0, 1.0)
        )
    job_row = app.connection.execute("SELECT priority, within_class_order_primary FROM jobs").fetchone()
    assert (job_row["priority"], job_row["within_class_order_primary"]) == (2, pytest.approx(0.7))
    app.close()


def test_draining_the_queue_then_idling_makes_no_further_model_requests(tmp_path):
    app = build_in_process_app(tmp_path / "home")
    run_collect_command(app, argparse.Namespace(fixture_paths=[RECORDED_PAPER_FIXTURE]))
    assert app.connection.execute("SELECT COUNT(*) FROM jobs WHERE status = 'queued'").fetchone()[0] == 1
    while run_single_work_cycle(app, probe_paused_model_now=True).outcome != WorkCycleOutcome.NO_JOB_DUE:
        pass
    model_requests_after_drain = len(app.model_client.received_requests)
    assert model_requests_after_drain == 2  # reviewer + topic assessor for the one paper
    run_collect_command(app, argparse.Namespace(fixture_paths=[RECORDED_PAPER_FIXTURE]))  # nothing new to read
    for _idle_cycle in range(5):
        assert run_single_work_cycle(app, probe_paused_model_now=True).outcome == WorkCycleOutcome.NO_JOB_DUE
    assert len(app.model_client.received_requests) == model_requests_after_drain
    app.close()


# --------------------------------------------------------------------------------------------------------------------
# Ingestion, direct promotion, and a missing model
# --------------------------------------------------------------------------------------------------------------------


def test_new_text_of_a_promoted_paper_is_queued_without_spending_a_slot(tmp_path):
    def arxiv_version(arxiv_id: str, abstract: str) -> CollectedPaperRecord:
        return CollectedPaperRecord(
            provider="arxiv",
            provider_paper_id=arxiv_id,
            title=f"Paper {arxiv_id[:10]}",
            authors=("A. Author",),
            abstract=abstract,
            first_published_date="2024-01-02",
            latest_revision_date=None,
            source_version=None,
            discovery_route="fixture:step6",
        )

    app = build_in_process_app(tmp_path / "home")
    promoted_id = record_collected_paper(app.connection, arxiv_version("2401.00001v1", "First text.")).paper_id
    waiting_id = record_collected_paper(app.connection, arxiv_version("2401.00002v1", "Other text.")).paper_id
    assert promote_paper_directly_and_queue_review(app.connection, promoted_id, max_attempts=3)
    assert count_screening_promotions_today(app.connection, datetime.now(timezone.utc))["regular_count"] == 0

    page_result = ingest_records_with_checkpoints(
        app.connection,
        [
            arxiv_version("2401.00001v2", "First text, revised with proofs."),
            arxiv_version("2401.00002v2", "Other text, revised."),
        ],
        admission_limit=None,
        active_seed_paper_ids=frozenset(),
        max_attempts_per_job=3,
    )
    assert page_result.new_papers_admitted == 0 and page_result.new_editions_of_known_papers == 2
    assert page_result.queued_paper_ids == [promoted_id]  # the unpromoted paper waits for screening
    assert app.connection.execute("SELECT COUNT(*) FROM jobs WHERE paper_id = ?", (promoted_id,)).fetchone()[0] == 2
    assert app.connection.execute("SELECT COUNT(*) FROM jobs WHERE paper_id = ?", (waiting_id,)).fetchone()[0] == 0
    app.close()


def test_missing_embedding_model_fails_screening_visibly_and_keeps_candidates_waiting(
    initialized_home: PaperwatchHome,
) -> None:
    initialized_home.replace_in_config('backend = "llama_cpp"', 'backend = "fake"')  # keeps the ONNX embedder
    initialized_home.add_example_topic()
    collect_result = initialized_home.run_cli("collect", "--fixture", str(RECORDED_PAPER_FIXTURE))
    assert collect_result.exit_code == 12
    assert "FAILED [screening]" in collect_result.stderr and "is missing" in collect_result.stderr
    assert "install_embedding_model.sh" in collect_result.stderr
    assert "WAIT unscored" in collect_result.stdout
    assert initialized_home.query("SELECT COUNT(*) FROM jobs")[0][0] == 0
    assert initialized_home.query("SELECT status FROM papers")[0]["status"] == "waiting_input"
    assert "MISSING: run deploy/install_embedding_model.sh" in initialized_home.run_cli("status").stdout
    doctor_result = initialized_home.run_cli("doctor")
    assert "FAIL  screening" in doctor_result.stdout and doctor_result.exit_code == 12

    initialized_home.use_hashing_test_embedder()
    recovered_result = initialized_home.run_cli("collect", "--fixture", str(RECORDED_PAPER_FIXTURE))
    assert recovered_result.exit_code == 0, recovered_result.stderr
    assert "Promotion: 1 paper(s) queued for review now" in recovered_result.stdout
    status_output = initialized_home.run_cli("status").stdout
    # The fixture's release date is fixed, so its class depends on today's date: class 1 inside the window, else 2.
    assert re.search(
        r"Queue by class \(claimed in this order\): (1 new release|2 older paper) 1, oldest", status_output
    )
    assert "promoted today (UTC): 1 of 30" in status_output


def test_altered_model_file_is_refused(tmp_path):
    pinned_model = PINNED_EMBEDDING_MODELS["bge-base-en-v1.5"]
    for pinned_file in pinned_model.pinned_files:
        (tmp_path / pinned_file.file_name).write_bytes(b"not the pinned model")
    with pytest.raises(ScreeningUnavailable, match="does not match its pinned sha256"):
        verify_pinned_model_files(pinned_model, tmp_path)


def test_screening_settings_are_validated(tmp_path):
    for bad_line, good_line in (
        ("near_cutoff_promotions = 3", "near_cutoff_promotions = 31"),
        ('embedding_backend = "onnx"', 'embedding_backend = "remote"'),
        ("minimum_screening_score = 0.0", "minimum_screening_score = 1.5"),
    ):
        app = build_in_process_app(tmp_path / bad_line[:10], use_onnx_embedder=True)
        config_path = app.config.home_dir / "config.toml"
        config_path.write_text(config_path.read_text().replace(bad_line, good_line))
        with pytest.raises(ConfigurationError):
            load_app_config(app.config.home_dir)
        app.close()


# --------------------------------------------------------------------------------------------------------------------
# The real pinned model (runs only where it is installed)
# --------------------------------------------------------------------------------------------------------------------

REAL_MODEL_DIR = os.environ.get("PAPERWATCH_TEST_EMBEDDING_MODEL_DIR")


@pytest.mark.skipif(not REAL_MODEL_DIR, reason="set PAPERWATCH_TEST_EMBEDDING_MODEL_DIR to the installed model")
def test_real_onnx_embedder_ranks_a_related_abstract_first(tmp_path):
    app = build_in_process_app(
        tmp_path / "home",
        use_onnx_embedder=True,
        config_replacements={
            'embedding_model_dir = "models/bge-base-en-v1.5"': f'embedding_model_dir = "{REAL_MODEL_DIR}"'
        },
    )
    embedder = load_text_embedder(app.config)
    assert embedder.identity == PINNED_EMBEDDING_MODELS["bge-base-en-v1.5"].identity
    for number, abstract in enumerate(
        (
            "Runtime monitors reject unsafe joint plans of multi-robot teams before "
            "dispatch, keeping human exits clear.",
            "We sequence the genome of a desert beetle and annotate its genes.",
        )
    ):
        record_collected_paper(app.connection, candidate_record(number, abstract=abstract, first_published_date="2019"))
    run_screening_and_promotion(app.connection, app.config, active_seed_paper_ids=frozenset(), moment=SCREENING_MOMENT)
    ranked = app.connection.execute(
        """SELECT edition.abstract FROM candidate_topic_scores AS score
           JOIN paper_editions AS edition USING (edition_id) ORDER BY score.unnormalized_score DESC"""
    ).fetchall()
    assert ranked[0]["abstract"].startswith("Runtime monitors")
    app.close()
