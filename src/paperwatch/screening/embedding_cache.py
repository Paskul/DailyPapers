"""Embeddings cached in SQLite by model identity and the sha256 of the exact model input (plan part 6).

A topic question, a seed, or a paper is embedded once per pinned model. A changed text has a new hash, and a changed
model has a new identity, so a stale vector is never reused. Embedding runs OUTSIDE any write transaction (it is the
slow part); the new vectors are then stored in one short transaction.
"""

from __future__ import annotations

import hashlib
import sqlite3

import numpy as np

from paperwatch.clock import utc_now_timestamp
from paperwatch.db import write_transaction
from paperwatch.screening.text_embedders import TextEmbedder

SQLITE_MAXIMUM_PARAMETERS_PER_QUERY = 500


def hash_model_input_text(model_input_text: str) -> str:
    return hashlib.sha256(model_input_text.encode("utf-8")).hexdigest()


def embed_texts_with_cache(
    connection: sqlite3.Connection, embedder: TextEmbedder, model_input_texts: list[str]
) -> np.ndarray:
    """One L2-normalized row per text, in input order; only texts missing from the cache reach the model."""
    text_hashes = [hash_model_input_text(model_input_text) for model_input_text in model_input_texts]
    cached_vectors = load_cached_vectors(connection, embedder.identity, set(text_hashes))
    missing_text_by_hash = {
        text_hash: model_input_text
        for text_hash, model_input_text in zip(text_hashes, model_input_texts)
        if text_hash not in cached_vectors
    }
    if missing_text_by_hash:
        missing_hashes = list(missing_text_by_hash)
        new_vectors = embedder.embed_texts([missing_text_by_hash[text_hash] for text_hash in missing_hashes])
        store_vectors(connection, embedder.identity, dict(zip(missing_hashes, new_vectors)))
        cached_vectors.update(zip(missing_hashes, new_vectors))
    embeddings = np.zeros((len(model_input_texts), embedder.dimensions), dtype=np.float32)
    for row_index, text_hash in enumerate(text_hashes):
        embeddings[row_index] = cached_vectors[text_hash]
    return embeddings


def load_cached_vectors(
    connection: sqlite3.Connection, model_identity: str, text_hashes: set[str]
) -> dict[str, np.ndarray]:
    cached_vectors: dict[str, np.ndarray] = {}
    hash_list = sorted(text_hashes)
    for chunk_start in range(0, len(hash_list), SQLITE_MAXIMUM_PARAMETERS_PER_QUERY):
        hash_chunk = hash_list[chunk_start : chunk_start + SQLITE_MAXIMUM_PARAMETERS_PER_QUERY]
        placeholders = ", ".join("?" * len(hash_chunk))
        for vector_row in connection.execute(
            f"""SELECT text_sha256, vector_float32 FROM text_embeddings
                    WHERE embedding_model_identity = ? AND text_sha256 IN ({placeholders})""",  # noqa: S608  # only ? placeholders are interpolated
            (model_identity, *hash_chunk),
        ):
            cached_vectors[vector_row["text_sha256"]] = np.frombuffer(vector_row["vector_float32"], dtype="<f4")
    return cached_vectors


def store_vectors(connection: sqlite3.Connection, model_identity: str, vectors_by_hash: dict[str, np.ndarray]) -> None:
    created_at = utc_now_timestamp()
    with write_transaction(connection):
        connection.executemany(
            """INSERT OR IGNORE INTO text_embeddings (embedding_model_identity, text_sha256, dimensions, vector_float32,
                   created_at) VALUES (?, ?, ?, ?, ?)""",
            [
                (model_identity, text_hash, int(vector.shape[0]), vector.astype("<f4").tobytes(), created_at)
                for text_hash, vector in vectors_by_hash.items()
            ],
        )
