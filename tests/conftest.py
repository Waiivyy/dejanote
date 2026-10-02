from __future__ import annotations

import os

import pytest

from dejanote.embedding import DEFAULT_MODEL, Embedder, default_model_dir, is_model_downloaded


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
