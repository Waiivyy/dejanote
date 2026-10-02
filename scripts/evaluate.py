"""How well does search find the right note? Scores embedding models on the example queries.

For each model the example notes are indexed from scratch, then every query in
examples/queries.json is searched and scored (see dejanote.evaluation).

Candidates other than the default model are downloaded into the dejanote data
folder on first use, which needs the network. Each candidate is pinned to a
commit with a sha256 for every file, exactly like the default model.

    python scripts/evaluate.py                   # the default model
    python scripts/evaluate.py --all --misses    # every candidate, listing misses
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

from dejanote.embedding import DEFAULT_MODEL, Embedder, ModelSpec, default_model_dir, download_model, is_model_downloaded
from dejanote.evaluation import load_cases, run_case, score
from dejanote.indexer import index_folder
from dejanote.privacy import block_network
from dejanote.store import Store

ROOT = Path(__file__).resolve().parent.parent
NOTES = ROOT / "examples" / "notes"
QUERIES = ROOT / "examples" / "queries.json"

CANDIDATES = {
    spec.name: spec
    for spec in (
        DEFAULT_MODEL,
        ModelSpec(
            name="multi-qa-MiniLM-L6-cos-v1",
            repo_id="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
            revision="b207367332321f8e44f96e224ef15bc607f4dbf0",
            files={
                "1_Pooling/config.json": "4be450dde3b0273bb9787637cfbd28fe04a7ba6ab9d36ac48e92b11e350ffc23",
                "README.md": "516c81b58be758ff4a521ec86d6f30668c68644e2d55d5dcbba3e60f4ca5efe0",
                "config.json": "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41",
                "config_sentence_transformers.json": "061ca9d39661d6c6d6de5ba27f79a1cd5770ea247f8d46412a68a498dc5ac9f3",
                "model.safetensors": "7bec4fd9eba43073d5c5dcf1b79b0a3397608fa063e6f626d6f8fd70a81f2d8c",
                "modules.json": "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf",
                "sentence_bert_config.json": "ec8e29d6dcb61b611b7d3fdd2982c4524e6ad985959fa7194eacfb655a8d0d51",
                "special_tokens_map.json": "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3",
                "tokenizer.json": "7fa9272f7ef1ebd1666bb3bfd9d4707660ff0076ca9d1671cd9a9c6e18e03331",
                "tokenizer_config.json": "857c5db35e9664bd0ea1db3a5ee62c0ce86d79e3ec85861ca1680bd7aaff12f8",
                "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
            },
            dimension=384,
            download_mb=92,
        ),
        ModelSpec(
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
        ),
    )
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", action="append", choices=sorted(CANDIDATES), help="model to score (repeatable)")
    parser.add_argument("--all", action="store_true", help="score every candidate model")
    parser.add_argument("--misses", action="store_true", help="list the queries whose note was not ranked first")
    args = parser.parse_args()
    names = sorted(CANDIDATES) if args.all else (args.model or [DEFAULT_MODEL.name])
    cases = load_cases(QUERIES)

    # Downloads first: huggingface_hub reads its settings once, at import, and
    # loading any model switches it to offline mode.
    for name in names:
        spec = CANDIDATES[name]
        if not is_model_downloaded(default_model_dir(spec), spec):
            print(f"downloading {spec.repo_id} (about {spec.download_mb} MB, needs the network)...")
            download_model(spec)

    print(f"{len(cases)} queries over {NOTES.relative_to(ROOT)}\n")
    print(f"{'model':<28}{'note@1':>8}{'note@3':>8}{'sect@1':>8}{'sect@3':>8}{'MRR':>7}{'index':>8}{'query':>8}")
    for name in names:
        spec = CANDIDATES[name]
        embedder = Embedder(spec)
        with tempfile.TemporaryDirectory() as tmp, block_network() as guard:
            with Store(Path(tmp) / "index.db", embedder.model_id, embedder.dimension) as store:
                embedder.embed_query("warm up")  # keep model loading out of the timings
                started = time.perf_counter()
                index_folder(NOTES, store, embedder)
                indexed = time.perf_counter() - started
                started = time.perf_counter()
                outcomes = [run_case(store, embedder, NOTES, case) for case in cases]
                per_query = (time.perf_counter() - started) / len(cases)
        assert not guard.attempts, guard.attempts
        s = score(outcomes)
        print(
            f"{name:<28}{s.note_at_1:>8.2f}{s.note_at_3:>8.2f}{s.section_at_1:>8.2f}{s.section_at_3:>8.2f}"
            f"{s.mrr:>7.2f}{indexed:>7.1f}s{per_query * 1000:>6.0f}ms"
        )
        if args.misses:
            for o in outcomes:
                if o.note_rank != 1:
                    rank = f"rank {o.note_rank}" if o.note_rank else "not found"
                    print(f"    miss ({rank}): {o.case.query!r} -> wanted {o.case.note}, got {o.top_note}")


if __name__ == "__main__":
    main()
