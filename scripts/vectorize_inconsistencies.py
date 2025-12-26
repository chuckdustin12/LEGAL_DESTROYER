"""Vectorize PDFs in INCONSISTENCIES and build a merged vector store."""

from __future__ import annotations

import argparse
import json
import sys
import re
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vectorize_case_docs import merge_vector_stores, vectorize_document  # noqa: E402


def _slugify(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    return cleaned or "document"


def _iter_pdfs(input_dir: Path) -> List[Path]:
    return sorted(input_dir.glob("*.pdf"))


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _store_has_chunks(store_dir: Path) -> bool:
    manifest = store_dir / "manifest.json"
    if manifest.exists():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            total = int(payload.get("total_chunks", 0))
            return total > 0
        except (ValueError, OSError, json.JSONDecodeError):
            return False
    embeddings = store_dir / "embeddings.npy"
    if embeddings.exists():
        try:
            data = np.load(embeddings)
            return data.shape[0] > 0
        except Exception:
            return False
    return False


def _rewrite_metadata_source(metadata_path: Path, pdf_path: Path) -> None:
    records = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pdf_rel = _relative_path(pdf_path)
    for record in records:
        record["source_pdf"] = pdf_rel
        record["source_exists"] = True
    metadata_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=True) for record in records),
        encoding="utf-8",
    )


def _rewrite_manifest_source(manifest_path: Path, pdf_path: Path) -> None:
    if not manifest_path.exists():
        return
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["input_pdf_root"] = _relative_path(pdf_path.parent)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vectorize PDFs under INCONSISTENCIES.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("INCONSISTENCIES"),
        help="Directory containing inconsistency PDFs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("vector_store_inconsistencies"),
        help="Merged vector store output directory.",
    )
    parser.add_argument(
        "--sources-dir",
        type=Path,
        default=Path("vector_store_inconsistencies_sources"),
        help="Directory for per-document vector stores.",
    )
    parser.add_argument(
        "--text-root",
        type=Path,
        default=Path("extracted_text_full/inconsistencies"),
        help="Root directory for extracted text/JSON outputs.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name.",
    )
    parser.add_argument("--max-chars", type=int, default=2000, help="Maximum characters per chunk.")
    parser.add_argument("--overlap", type=int, default=200, help="Character overlap between chunks.")
    parser.add_argument("--min-chars", type=int, default=50, help="Minimum characters per chunk.")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild per-document stores even if they already exist.",
    )
    parser.add_argument(
        "--use-ocr",
        action="store_true",
        help="Prefer existing OCR text outputs when available.",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Skip merging per-document stores into the merged output.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    sources_dir = args.sources_dir.expanduser().resolve()
    text_root = args.text_root.expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Missing input directory: {input_dir}")

    sources_dir.mkdir(parents=True, exist_ok=True)
    text_root.mkdir(parents=True, exist_ok=True)

    store_paths: List[Path] = []
    for pdf_path in _iter_pdfs(input_dir):
        slug = _slugify(pdf_path.stem)
        store_dir = sources_dir / slug
        text_dir = text_root / slug
        text_path = text_dir / f"{slug}.txt"
        embeddings_path = store_dir / "embeddings.npy"
        if embeddings_path.exists() and not args.force:
            print(f"Skipping {pdf_path.name}; store exists at {store_dir}")
            if _store_has_chunks(store_dir):
                store_paths.append(store_dir)
            else:
                print(f"Skipping merge for {pdf_path.name}; no chunks available.")
            continue

        use_ocr = args.use_ocr and text_path.exists() and text_path.stat().st_size > 0
        input_path = text_path if use_ocr else pdf_path
        result = vectorize_document(
            input_path,
            store_dir,
            text_output_dir=text_dir if not use_ocr else None,
            base_name=slug if not use_ocr else None,
            model_name=args.model,
            max_chars=args.max_chars,
            overlap=args.overlap,
            min_chars=args.min_chars,
            batch_size=args.batch_size,
        )
        if use_ocr:
            _rewrite_metadata_source(store_dir / "metadata.jsonl", pdf_path)
            _rewrite_manifest_source(store_dir / "manifest.json", pdf_path)
        if result.total_chunks > 0:
            store_paths.append(result.output_dir)
            print(f"Saved {result.total_chunks} chunks for {pdf_path.name}")
        else:
            print(f"No text chunks extracted for {pdf_path.name}; skipping merge.")

    if args.no_merge:
        print("Skipping merge step.")
        return

    if not store_paths:
        raise ValueError("No PDFs found to vectorize.")

    merge_vector_stores(store_paths, output_dir)
    print(f"Merged {len(store_paths)} stores into {output_dir}")


if __name__ == "__main__":
    main()
