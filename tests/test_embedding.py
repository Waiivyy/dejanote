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


def test_a_new_model_revision_gets_a_new_model_id():
    # The index records the model id, so vectors from another revision are never mixed in.
    newer = dataclasses.replace(DEFAULT_MODEL, revision="0123456789abcdef")
    assert newer.model_id != DEFAULT_MODEL.model_id


def test_missing_model_points_to_setup(tmp_path):
    with pytest.raises(ModelNotFoundError, match="dejanote setup"):
        Embedder(model_dir=tmp_path / "absent")


def test_creating_an_embedder_does_not_load_the_model(tmp_path):
    # Every file is present but none is a real model: only an attempt to load them can fail.
    for name in DEFAULT_MODEL.files:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("not a model")
    embedder = Embedder(model_dir=tmp_path)
    assert embedder.dimension == 384
    with block_network() as guard:
        with pytest.raises(Exception):
            embedder.embed_query("first use is when the model loads")
    assert guard.attempts == []


@pytest.mark.model
def test_the_embedder_reports_its_vector_size_and_model(embedder):
    assert embedder.dimension == 384
    assert embedder.model_id == DEFAULT_MODEL.model_id


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
def test_queries_are_embedded_with_the_models_query_prefix(model_dir):
    # Some retrieval models (bge, e5) expect searches to be marked with an instruction.
    embedder = Embedder(dataclasses.replace(DEFAULT_MODEL, query_prefix="search for: "), model_dir=model_dir)
    np.testing.assert_allclose(
        embedder.embed_query("sourdough"),
        embedder.embed_documents(["search for: sourdough"])[0],
        atol=1e-6,
    )


@pytest.mark.model
def test_loading_the_model_prints_nothing(model_dir, capfd):
    # Library progress bars would clutter every search's output.
    Embedder(model_dir=model_dir).embed_query("loading happens on first use")
    assert capfd.readouterr() == ("", "")


@pytest.mark.model
def test_loading_and_embedding_make_no_network_attempts(model_dir):
    with block_network() as guard:
        Embedder(model_dir=model_dir).embed_query("anything at all")
    assert guard.attempts == []
