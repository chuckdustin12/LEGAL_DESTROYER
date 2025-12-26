"""Vectorize CASE DOCS by filer folder and merge per-filer stores."""

from __future__ import annotations

import argparse
import json
import re
import sys
from hashlib import sha1
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vectorize_case_docs import merge_vector_stores, vectorize_document  # noqa: E402


def _slugify(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    return cleaned or "document"


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _iter_filer_dirs(root: Path, include_root: bool) -> List[Path]:
    dirs = [entry for entry in root.iterdir() if entry.is_dir()]
    if include_root:
        dirs.append(root)
    return sorted(dirs, key=lambda p: p.name.lower())


def _iter_pdfs(root: Path) -> List[Path]:
    return sorted(root.rglob("*.pdf"))


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


def _unique_slug(base: str, seen: Dict[str, int], path: Path) -> str:
    if base not in seen:
        seen[base] = 1
        return base
    digest = sha1(str(path).encode("utf-8")).hexdigest()[:8]
    slug = f"{base}_{digest}"
    seen[base] += 1
    return slug


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vectorize CASE DOCS by filer folder.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("CASE DOCS"),
        help="Root directory containing filer subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("vector_store_case_docs_by_filer"),
        help="Root directory for merged per-filer vector stores.",
    )
    parser.add_argument(
        "--sources-dir",
        type=Path,
        default=Path("vector_store_case_docs_by_filer_sources"),
        help="Root directory for per-document vector stores.",
    )
    parser.add_argument(
        "--text-root",
        type=Path,
        default=Path("extracted_text_full/case_docs_by_filer"),
        help="Root directory for extracted text/JSON outputs.",
    )
    parser.add_argument(
        "--include-root",
        action="store_true",
        help="Also process PDFs directly under the CASE DOCS root.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional list of filer folder names to process (case-insensitive).",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=None,
        help="Optional list of filer folder names to skip (case-insensitive).",
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
        help="Skip merging per-document stores into per-filer outputs.",
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
    output_dir.mkdir(parents=True, exist_ok=True)

    filer_dirs = _iter_filer_dirs(input_dir, args.include_root)
    if not filer_dirs:
        raise ValueError(f"No filer directories found under {input_dir}")

    only = {_slugify(name) for name in args.only} if args.only else None
    skip = {_slugify(name) for name in args.skip} if args.skip else set()

    for filer_dir in filer_dirs:
        if filer_dir == input_dir and not args.include_root:
            continue
        filer_name = "combined" if filer_dir == input_dir else filer_dir.name
        filer_slug = _slugify(filer_name)
        if only is not None and filer_slug not in only:
            continue
        if filer_slug in skip:
            continue
        pdfs = _iter_pdfs(filer_dir)
        if filer_dir == input_dir and not args.include_root:
            pdfs = [path for path in pdfs if path.parent != input_dir]
        if not pdfs:
            continue

        print(f"\n== Filer: {filer_name} ({len(pdfs)} PDFs) ==")
        filer_sources_dir = sources_dir / filer_slug
        filer_text_root = text_root / filer_slug
        filer_sources_dir.mkdir(parents=True, exist_ok=True)
        filer_text_root.mkdir(parents=True, exist_ok=True)

        seen_slugs: Dict[str, int] = {}
        store_paths: List[Path] = []
        for pdf_path in pdfs:
            base_slug = _slugify(pdf_path.stem)
            slug = _unique_slug(base_slug, seen_slugs, pdf_path)
            store_dir = filer_sources_dir / slug
            text_dir = filer_text_root / slug
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
            continue

        if not store_paths:
            print(f"No stores to merge for filer {filer_name}.")
            continue

        merge_vector_stores(store_paths, output_dir / filer_slug)
        print(f"Merged {len(store_paths)} stores into {output_dir / filer_slug}")


if __name__ == "__main__":
    main()
