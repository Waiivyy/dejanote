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


# Chosen with scripts/evaluate.py: on the 46 example queries it ranked the right
# note first 45 times, against 42 for all-MiniLM-L6-v2 (see the README).
#
# Every file is pinned, not just the weights: the config files decide which
# code sentence-transformers runs, so a swapped config is as dangerous as
# swapped weights. Hashes were checked against the git objects of the commit.
DEFAULT_MODEL = ModelSpec(
    name="bge-small-en-v1.5",
    repo_id="BAAI/bge-small-en-v1.5",
    revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    files={
        "1_Pooling/config.json": "d1caf60c96f5fba2157c0c26b76d80818fad6cf0b8eb5e73ec372ff9818eba5c",
        "README.md": "ddb964361a55c6e5dfca6361615854b260c9c960205d04c7520151aaa1d75837",
        "config.json": "094f8e891b932f2000c92cfc663bac4c62069f5d8af5b5278c4306aef3084750",
        "config_sentence_transformers.json": "940d5f50db195fa6e5e6a4f122c095f77880de259d74b14a65779ed48bdd7c56",
        "model.safetensors": "3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad",
        "modules.json": "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf",
        "sentence_bert_config.json": "84e39fda68ccbff05bfa723ae9c0e70e23e2ec373b76e0f8c6e71af72a693cbf",
        "special_tokens_map.json": "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
        "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
        "tokenizer_config.json": "9261e7d79b44c8195c1cada2b453e55b00aeb81e907a6664974b4d7776172ab3",
        "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    },
    dimension=384,
    download_mb=135,
    query_prefix="Represent this sentence for searching relevant passages: ",
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
