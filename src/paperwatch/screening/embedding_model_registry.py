"""The sentence-embedding models paperwatch can screen with, each pinned to exact file hashes (plan part 6).

config.toml picks a model by NAME only (`screening.embedding_model`); the files, their sha256 hashes, the pooling, and
the query instruction live here, in the installed package, so a config edit cannot swap in an unverified file. The
files are installed ahead of time by deploy/install_embedding_model.sh (which downloads the pinned revision and checks
the same hashes); at run time they are read-only local files and nothing is downloaded.

To add a model: add an entry with its pinned revision and the sha256 of every file (the install script reads them
from here). Its `identity` differs, so cached embeddings and scores of the old model are not reused.
"""

from __future__ import annotations

from dataclasses import dataclass

from paperwatch.errors import ConfigurationError

POOLING_CLS_TOKEN = "cls"  # the first token's last hidden state (BGE models)
POOLING_MEAN_OF_TOKENS = "mean"  # attention-masked mean of the last hidden states (MiniLM-style models)


@dataclass(frozen=True)
class PinnedModelFile:
    file_name: str  # name inside the model directory
    source_path: str  # path inside the source repository at the pinned revision
    sha256: str


@dataclass(frozen=True)
class PinnedEmbeddingModel:
    name: str
    source_repository: str  # where deploy/install_embedding_model.sh downloads it from (operator step, not run time)
    pinned_revision: str  # exact commit of the source repository
    onnx_model_file: PinnedModelFile
    tokenizer_file: PinnedModelFile
    dimensions: int
    maximum_tokens: int  # longer inputs are truncated (title + abstract rarely exceed it)
    pooling: str
    query_instruction: str  # prefixed to a short query (a topic question); documents get no prefix

    @property
    def identity(self) -> str:
        """Stored with every cached embedding and score: a changed file or revision is a different model."""
        return f"onnx:{self.name}@{self.pinned_revision[:12]}:{self.onnx_model_file.sha256[:16]}"

    @property
    def pinned_files(self) -> tuple[PinnedModelFile, ...]:
        return (self.onnx_model_file, self.tokenizer_file)


PINNED_EMBEDDING_MODELS: dict[str, PinnedEmbeddingModel] = {
    "bge-base-en-v1.5": PinnedEmbeddingModel(
        name="bge-base-en-v1.5",
        source_repository="https://huggingface.co/BAAI/bge-base-en-v1.5",
        pinned_revision="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
        onnx_model_file=PinnedModelFile(
            "model.onnx", "onnx/model.onnx", "9bc579acdba21c253c62a9bf866891355a63ffa3442b52c8a37d75b2ccb91848"
        ),
        tokenizer_file=PinnedModelFile(
            "tokenizer.json", "tokenizer.json", "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"
        ),
        dimensions=768,
        maximum_tokens=512,
        pooling=POOLING_CLS_TOKEN,
        query_instruction="Represent this sentence for searching relevant passages: ",
    ),
}


def look_up_pinned_embedding_model(model_name: str) -> PinnedEmbeddingModel:
    try:
        return PINNED_EMBEDDING_MODELS[model_name]
    except KeyError:
        raise ConfigurationError(
            f"screening.embedding_model '{model_name}' is not a pinned model",
            operator_hint=f"use one of {sorted(PINNED_EMBEDDING_MODELS)}",
        ) from None


def print_install_manifest(model_name: str) -> None:
    """For deploy/install_embedding_model.sh: one line per file, "<file name> <download URL> <sha256>", so the
    pinned values live only in this module."""
    pinned_model = look_up_pinned_embedding_model(model_name)
    for pinned_file in pinned_model.pinned_files:
        download_url = (
            f"{pinned_model.source_repository}/resolve/{pinned_model.pinned_revision}/{pinned_file.source_path}"
        )
        print(f"{pinned_file.file_name} {download_url} {pinned_file.sha256}")


if __name__ == "__main__":
    import sys

    print_install_manifest(sys.argv[1] if len(sys.argv) > 1 else "bge-base-en-v1.5")
