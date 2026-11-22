"""Sentence embedders for screening: L2-normalized vectors, so a dot product is the cosine similarity.

    OnnxSentenceEmbedder  the pinned model (embedding_model_registry.py) run in-process by onnxruntime on the CPU.
                          Its files are verified against their pinned sha256 hashes before the first use; a missing
                          or altered file raises ScreeningUnavailable instead of screening with something unverified.
    HashingTestEmbedder   a deterministic word-hashing stand-in for offline tests (screening.embedding_backend =
                          "hashing"). Texts sharing words get similar vectors; it has no notion of meaning.

Both take the exact MODEL INPUT text. `prepare_query_text` adds a model's query instruction to a short query (a topic
question), so the embedding cache can key on the exact input. onnxruntime and tokenizers are imported only when the
ONNX embedder is built, so commands that never screen do not load them.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Protocol

import numpy as np

from paperwatch.config import AppConfig
from paperwatch.errors import ScreeningUnavailable
from paperwatch.screening.embedding_model_registry import (
    POOLING_CLS_TOKEN,
    POOLING_MEAN_OF_TOKENS,
    PinnedEmbeddingModel,
    look_up_pinned_embedding_model,
)

EMBEDDING_BACKEND_ONNX = "onnx"
EMBEDDING_BACKEND_HASHING = "hashing"
HASHING_TEST_EMBEDDER_DIMENSIONS = 256
HASHING_TEST_EMBEDDER_IDENTITY = f"hashing-test-v1:{HASHING_TEST_EMBEDDER_DIMENSIONS}"
WORD_PATTERN = re.compile(r"[a-z0-9]+")
FILE_HASH_CHUNK_BYTES = 4 * 1024 * 1024


class TextEmbedder(Protocol):
    identity: str
    dimensions: int

    def prepare_query_text(self, query_text: str) -> str:
        """The exact model input for a short query (e.g. with the model's retrieval instruction)."""

    def embed_texts(self, model_input_texts: list[str]) -> np.ndarray:
        """float32 matrix, one L2-normalized row per text, in input order."""


def configured_embedder_identity(app_config: AppConfig) -> str:
    """The identity scores are stored under, known WITHOUT loading the model (so freshness checks are cheap)."""
    if app_config.screening.embedding_backend == EMBEDDING_BACKEND_HASHING:
        return HASHING_TEST_EMBEDDER_IDENTITY
    return look_up_pinned_embedding_model(app_config.screening.embedding_model).identity


def load_text_embedder(app_config: AppConfig) -> TextEmbedder:
    """Build the configured embedder. Raises ScreeningUnavailable if the pinned model cannot be used."""
    screening_settings = app_config.screening
    if screening_settings.embedding_backend == EMBEDDING_BACKEND_HASHING:
        return HashingTestEmbedder()
    return OnnxSentenceEmbedder.load(
        look_up_pinned_embedding_model(screening_settings.embedding_model),
        app_config.embedding_model_dir,
        thread_count=screening_settings.embedding_threads,
        batch_size=screening_settings.embedding_batch_size,
    )


# --------------------------------------------------------------------------------------------------------------------
# The pinned ONNX model
# --------------------------------------------------------------------------------------------------------------------


def verify_pinned_model_files(pinned_model: PinnedEmbeddingModel, model_dir: Path) -> None:
    """Every pinned file must exist and match its sha256. Raises ScreeningUnavailable naming the first problem."""
    install_hint = f"run deploy/install_embedding_model.sh to install {pinned_model.name} into {model_dir}"
    for pinned_file in pinned_model.pinned_files:
        file_path = model_dir / pinned_file.file_name
        try:
            file_hash = hashlib.sha256()
            with file_path.open("rb") as model_file:
                while chunk := model_file.read(FILE_HASH_CHUNK_BYTES):
                    file_hash.update(chunk)
        except FileNotFoundError:
            raise ScreeningUnavailable(
                f"embedding model file {file_path} is missing", operator_hint=install_hint
            ) from None
        except OSError as read_error:
            raise ScreeningUnavailable(
                f"embedding model file {file_path} cannot be read: {read_error}",
                operator_hint="the collector account needs read access to the model directory",
            ) from None
        if file_hash.hexdigest() != pinned_file.sha256:
            raise ScreeningUnavailable(
                f"embedding model file {file_path} does not match its pinned sha256", operator_hint=install_hint
            )


class OnnxSentenceEmbedder:
    def __init__(self, pinned_model: PinnedEmbeddingModel, inference_session, tokenizer, batch_size: int) -> None:
        self.pinned_model = pinned_model
        self.identity = pinned_model.identity
        self.dimensions = pinned_model.dimensions
        self._inference_session = inference_session
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._input_names = {session_input.name for session_input in inference_session.get_inputs()}

    @classmethod
    def load(
        cls, pinned_model: PinnedEmbeddingModel, model_dir: Path, *, thread_count: int, batch_size: int
    ) -> OnnxSentenceEmbedder:
        verify_pinned_model_files(pinned_model, model_dir)
        try:
            import onnxruntime
            from tokenizers import Tokenizer
        except ImportError as import_error:
            raise ScreeningUnavailable(
                f"the ONNX runtime is not installed: {import_error}",
                operator_hint="reinstall paperwatch with its dependencies (uv sync)",
            ) from None
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = thread_count
        session_options.inter_op_num_threads = 1
        session_options.log_severity_level = 3  # errors only
        try:
            inference_session = onnxruntime.InferenceSession(
                str(model_dir / pinned_model.onnx_model_file.file_name),
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
            tokenizer = Tokenizer.from_file(str(model_dir / pinned_model.tokenizer_file.file_name))
        except Exception as load_error:  # onnxruntime raises its own exception types
            raise ScreeningUnavailable(
                f"the embedding model {pinned_model.name} failed to load: {load_error}"
            ) from None
        tokenizer.enable_truncation(max_length=pinned_model.maximum_tokens)
        tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]") or 0, pad_token="[PAD]")
        return cls(pinned_model, inference_session, tokenizer, batch_size)

    def prepare_query_text(self, query_text: str) -> str:
        return self.pinned_model.query_instruction + query_text

    def embed_texts(self, model_input_texts: list[str]) -> np.ndarray:
        embeddings = np.zeros((len(model_input_texts), self.dimensions), dtype=np.float32)
        # Batches of similar length waste less work on padding; results go back to their input positions.
        positions_by_length = sorted(
            range(len(model_input_texts)), key=lambda position: len(model_input_texts[position])
        )
        for batch_start in range(0, len(positions_by_length), self._batch_size):
            batch_positions = positions_by_length[batch_start : batch_start + self._batch_size]
            embeddings[batch_positions] = self._embed_one_batch(
                [model_input_texts[position] for position in batch_positions]
            )
        return embeddings

    def _embed_one_batch(self, batch_texts: list[str]) -> np.ndarray:
        encodings = self._tokenizer.encode_batch(batch_texts)
        attention_mask = np.array([encoding.attention_mask for encoding in encodings], dtype=np.int64)
        model_inputs = {
            "input_ids": np.array([encoding.ids for encoding in encodings], dtype=np.int64),
            "attention_mask": attention_mask,
        }
        if "token_type_ids" in self._input_names:
            model_inputs["token_type_ids"] = np.array([encoding.type_ids for encoding in encodings], dtype=np.int64)
        last_hidden_states = self._inference_session.run(None, model_inputs)[0]
        if self.pinned_model.pooling == POOLING_CLS_TOKEN:
            pooled = last_hidden_states[:, 0, :]
        elif self.pinned_model.pooling == POOLING_MEAN_OF_TOKENS:
            token_weights = attention_mask[:, :, None].astype(np.float32)
            pooled = (last_hidden_states * token_weights).sum(axis=1) / np.maximum(token_weights.sum(axis=1), 1.0)
        else:
            raise ScreeningUnavailable(f"unknown pooling '{self.pinned_model.pooling}' for {self.pinned_model.name}")
        return normalize_rows(pooled.astype(np.float32))


# --------------------------------------------------------------------------------------------------------------------
# The offline test stand-in
# --------------------------------------------------------------------------------------------------------------------


class HashingTestEmbedder:
    """Each lower-cased word adds +1 or -1 to one of 256 dimensions chosen by its hash; rows are then normalized."""

    identity = HASHING_TEST_EMBEDDER_IDENTITY
    dimensions = HASHING_TEST_EMBEDDER_DIMENSIONS

    def prepare_query_text(self, query_text: str) -> str:
        return query_text

    def embed_texts(self, model_input_texts: list[str]) -> np.ndarray:
        embeddings = np.zeros((len(model_input_texts), self.dimensions), dtype=np.float32)
        for row_index, model_input_text in enumerate(model_input_texts):
            for word in WORD_PATTERN.findall(model_input_text.lower()):
                word_digest = hashlib.sha256(word.encode("utf-8")).digest()
                dimension = int.from_bytes(word_digest[:4], "little") % self.dimensions
                embeddings[row_index, dimension] += 1.0 if word_digest[4] % 2 == 0 else -1.0
        return normalize_rows(embeddings)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row; an all-zero row (no words) stays zero, so its cosine with anything is 0."""
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, row_norms, out=np.zeros_like(matrix), where=row_norms > 0)
