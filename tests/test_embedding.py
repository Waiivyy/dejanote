"""Embedding model: verified local files, offline loading, vectors that capture meaning."""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import numpy as np
import pytest

from dejanote.embedding import (
    DEFAULT_MODEL,
    Embedder,
    ModelIntegrityError,
    ModelNotFoundError,
    verify_model_files,
)
from dejanote.privacy import block_network

NOTE = Path(__file__).parent / "fixtures" / "sourdough-starter.md"


def _fake_model(root: Path, weights: bytes = b"genuine weights", config: bytes = b"{}"):
    """A two-file model pinned to the hashes of b'genuine weights' and b'{}'."""
    (root / "config.json").write_bytes(config)
    (root / "model.safetensors").write_bytes(weights)
    return dataclasses.replace(
        DEFAULT_MODEL,
        files={
            "config.json": hashlib.sha256(b"{}").hexdigest(),
            "model.safetensors": hashlib.sha256(b"genuine weights").hexdigest(),
        },
    )


def test_verification_accepts_files_matching_their_pinned_hashes(tmp_path):
    spec = _fake_model(tmp_path)
    verify_model_files(tmp_path, spec)


def test_verification_rejects_tampered_weights(tmp_path):
    spec = _fake_model(tmp_path, weights=b"tampered weights")
    with pytest.raises(ModelIntegrityError, match="model.safetensors"):
        verify_model_files(tmp_path, spec)


def test_verification_rejects_a_tampered_config_file(tmp_path):
    # Config files decide which code sentence-transformers loads, so they are pinned too.
    spec = _fake_model(tmp_path, config=b'{"tampered": true}')
    with pytest.raises(ModelIntegrityError, match="config.json"):
        verify_model_files(tmp_path, spec)


def test_verification_rejects_missing_files(tmp_path):
    spec = _fake_model(tmp_path)
    (tmp_path / "config.json").unlink()
    with pytest.raises(ModelIntegrityError, match="config.json"):
        verify_model_files(tmp_path, spec)


def test_missing_model_points_to_setup(tmp_path):
    with pytest.raises(ModelNotFoundError, match="dejanote setup"):
        Embedder(model_dir=tmp_path / "absent")


@pytest.mark.model
def test_a_note_embeds_to_a_unit_vector(embedder):
    vector = embedder.embed_documents([NOTE.read_text()])[0]
    assert vector.shape == (384,)
    assert vector.dtype == np.float32
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.model
def test_a_paraphrase_scores_far_above_an_unrelated_query(embedder):
    # The paraphrase shares no content words with the note, so it can only match on meaning.
    note = embedder.embed_documents([NOTE.read_text()])[0]
    paraphrase = embedder.embed_query("maintaining wild yeast for homemade loaves")
    unrelated = embedder.embed_query("quarterly VAT filing deadline")
    assert float(note @ paraphrase) > float(note @ unrelated) + 0.2


@pytest.mark.model
def test_loading_the_model_prints_nothing(model_dir, capfd):
    # Library progress bars would clutter every search's output.
    Embedder(model_dir=model_dir)
    assert capfd.readouterr() == ("", "")


@pytest.mark.model
def test_loading_and_embedding_make_no_network_attempts(model_dir):
    with block_network() as guard:
        Embedder(model_dir=model_dir).embed_query("anything at all")
    assert guard.attempts == []
