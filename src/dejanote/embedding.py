"""The local embedding model: one explicit, verified download, then offline inference."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import truststore

from dejanote import config
from dejanote.privacy import enable_offline_mode, minimize_download_footprint


@dataclass(frozen=True)
class ModelSpec:
    """A pinned embedding model: which commit, and the sha256 of every file we use from it."""

    name: str
    repo_id: str
    revision: str
    files: dict[str, str]
    dimension: int
    download_mb: int
    query_prefix: str = ""  # instruction some retrieval models expect in front of searches

    @property
    def model_id(self) -> str:
        """Identifies the exact weights; the index records it so vectors are never mixed."""
        return f"{self.repo_id}@{self.revision}"


# Every file is pinned, not just the weights: the config files decide which
# code sentence-transformers runs, so a swapped config is as dangerous as
# swapped weights. Hashes were checked against the git objects of the commit.
DEFAULT_MODEL = ModelSpec(
    name="all-MiniLM-L6-v2",
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    files={
        "1_Pooling/config.json": "4be450dde3b0273bb9787637cfbd28fe04a7ba6ab9d36ac48e92b11e350ffc23",
        "README.md": "dcd602d2fd35c203a247304a06fec6654a12f7941b739f9221a064fe8dc3b7f0",
        "config.json": "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41",
        "config_sentence_transformers.json": "061ca9d39661d6c6d6de5ba27f79a1cd5770ea247f8d46412a68a498dc5ac9f3",
        "model.safetensors": "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
        "modules.json": "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf",
        "sentence_bert_config.json": "fc1993fde0a95c24ec6c022539d41cf6e2f7c9721e5415d6fb6897472a9cd4b7",
        "special_tokens_map.json": "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3",
        "tokenizer.json": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
        "tokenizer_config.json": "acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b",
        "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    },
    dimension=384,
    download_mb=92,
)


class ModelNotFoundError(FileNotFoundError):
    """The embedding model has not been downloaded yet."""


class ModelIntegrityError(RuntimeError):
    """Model files are missing or the weights do not match the pinned hash."""


def default_model_dir(spec: ModelSpec = DEFAULT_MODEL) -> Path:
    return config.models_dir() / spec.name


def is_model_downloaded(model_dir: Path, spec: ModelSpec = DEFAULT_MODEL) -> bool:
    return all((model_dir / name).is_file() for name in spec.files)


def verify_model_files(model_dir: Path, spec: ModelSpec = DEFAULT_MODEL) -> None:
    missing = [name for name in spec.files if not (model_dir / name).is_file()]
    if missing:
        raise ModelIntegrityError(f"model files missing from {model_dir}: {', '.join(missing)}")
    for name, expected in spec.files.items():
        actual = _sha256(model_dir / name)
        if actual != expected:
            raise ModelIntegrityError(
                f"{name} does not match its pinned sha256 (expected {expected}, got {actual})"
            )


def download_model(spec: ModelSpec = DEFAULT_MODEL, model_dir: Path | None = None) -> Path:
    """Fetch the pinned model files from Hugging Face.

    This is the only network access dejanote ever makes. Files land in a
    staging folder and move into place only after every hash matches, so an
    interrupted or tampered download is never used. Must run before anything
    else in the process imports huggingface_hub, which reads its settings
    from the environment at import time.
    """
    target = model_dir or default_model_dir(spec)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=target.parent, prefix=f".{spec.name}-download-"))
    try:
        minimize_download_footprint(hf_home=staging / "hf-home")
        from huggingface_hub import snapshot_download

        # Verify TLS against the operating system's certificate store, as pip does,
        # so the download also works behind TLS-inspecting corporate proxies.
        truststore.inject_into_ssl()
        files = staging / "model"
        snapshot_download(
            repo_id=spec.repo_id,
            revision=spec.revision,
            allow_patterns=list(spec.files),
            local_dir=files,
        )
        shutil.rmtree(files / ".cache", ignore_errors=True)  # download bookkeeping, not part of the model
        verify_model_files(files, spec)
        if target.exists():
            shutil.rmtree(target)
        files.rename(target)
    finally:
        truststore.extract_from_ssl()
        shutil.rmtree(staging, ignore_errors=True)
    return target


class Embedder:
    """Turns text into unit-length vectors using a model loaded from local files only.

    Creating one only checks that the model is on disk. The model itself (and
    torch) loads on first use, so a run with nothing to embed stays fast.
    """

    def __init__(self, spec: ModelSpec = DEFAULT_MODEL, model_dir: Path | None = None):
        self.spec = spec
        self.model_dir = model_dir or default_model_dir(spec)
        if not is_model_downloaded(self.model_dir, spec):
            raise ModelNotFoundError(
                f"The embedding model is not on this machine yet (looked in {self.model_dir}).\n"
                f"Run `dejanote setup` once to download it (about {spec.download_mb} MB)."
            )
        self.model_id = spec.model_id
        self.dimension = spec.dimension
        self._model = None

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """One row per text."""
        return self._encode(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self.spec.query_prefix + text])[0]

    def _loaded(self):
        if self._model is None:
            enable_offline_mode()
            from sentence_transformers import SentenceTransformer
            from transformers.utils import logging as transformers_logging

            transformers_logging.disable_progress_bar()  # keep library progress bars out of our output
            model = SentenceTransformer(str(self.model_dir), device="cpu", local_files_only=True)
            dimension = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
            if dimension() != self.dimension:
                raise ModelIntegrityError(
                    f"{self.model_dir} produces {dimension()}-d vectors, expected {self.dimension}"
                )
            self._model = model
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._loaded().encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
