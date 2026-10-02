from __future__ import annotations

import os
import zlib

import numpy as np
import pytest

from dejanote.embedding import DEFAULT_MODEL, Embedder, default_model_dir, is_model_downloaded


class FakeEmbedder:
    """Stand-in for the real model in pipeline tests: hashes words into a small unit vector.

    Records every batch it is asked to embed, so tests can check what was embedded.
    """

    model_id = "fake-model@1"
    dimension = 16

    def __init__(self):
        self.batches: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self.batches.append(list(texts))
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.lower().split():
                vectors[row, zlib.crc32(word.encode()) % self.dimension] += 1
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1, norms)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    @property
    def embedded_texts(self) -> list[str]:
        return [text for batch in self.batches for text in batch]


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture(scope="session")
def model_dir():
    """The real embedding model on disk; tests that need it skip until `dejanote setup` has run."""
    path = default_model_dir(DEFAULT_MODEL)
    if not is_model_downloaded(path, DEFAULT_MODEL):
        if os.environ.get("DEJANOTE_REQUIRE_MODEL"):
            pytest.fail(f"embedding model missing at {path}; run `dejanote setup --yes` first")
        pytest.skip("embedding model not downloaded; run `dejanote setup` to enable these tests")
    return path


@pytest.fixture(scope="session")
def embedder(model_dir):
    return Embedder(model_dir=model_dir)
