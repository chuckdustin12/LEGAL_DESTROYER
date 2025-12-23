import argparse
import json
import time
from pathlib import Path

import numpy as np


def read_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}


def iter_metadata(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def merge_stores(case_dir: Path, research_dir: Path, out_dir: Path) -> None:
    case_embeddings = np.load(case_dir / "embeddings.npy")
    research_embeddings = np.load(research_dir / "embeddings.npy")

    out_dir.mkdir(parents=True, exist_ok=True)
    embeddings = (
        np.vstack([case_embeddings, research_embeddings])
        if case_embeddings.size and research_embeddings.size
        else case_embeddings
    )

    metadata_path = out_dir / "metadata.jsonl"
    if metadata_path.exists():
        metadata_path.unlink()

    offset = int(case_embeddings.shape[0])
    vector_id = 0

    with metadata_path.open("a", encoding="utf-8") as f:
        for record in iter_metadata(case_dir / "metadata.jsonl"):
            record["vector_id"] = vector_id
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
            vector_id += 1

        for record in iter_metadata(research_dir / "metadata.jsonl"):
            record["vector_id"] = offset + (record.get("vector_id") or 0)
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
            vector_id += 1

    np.save(out_dir / "embeddings.npy", embeddings)

    case_manifest = read_manifest(case_dir / "manifest.json")
    research_manifest = read_manifest(research_dir / "manifest.json")
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_stores": [
            {
                "path": str(case_dir),
                "manifest": case_manifest,
            },
            {
                "path": str(research_dir),
                "manifest": research_manifest,
            },
        ],
        "total_chunks": int(embeddings.shape[0]) if embeddings.size else 0,
        "embedding_dim": int(embeddings.shape[1]) if embeddings.size else 0,
        "output_root": str(out_dir),
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge two vector stores into a single combined store."
    )
    parser.add_argument("--case-store", default="vector_store")
    parser.add_argument("--research-store", default="vector_store_research")
    parser.add_argument("--output", default="vector_store_all")
    args = parser.parse_args()

    merge_stores(
        case_dir=Path(args.case_store),
        research_dir=Path(args.research_store),
        out_dir=Path(args.output),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
