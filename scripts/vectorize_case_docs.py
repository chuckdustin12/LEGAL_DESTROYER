"""Vectorize case documents into embeddings and JSONL metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ingest_merged_case import ingest_file  # noqa: E402


@dataclass
class VectorizationResult:
    """Captures outputs for a vectorization run."""

    output_dir: Path
    total_chunks: int
    embedding_dim: int


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _chunk_text(text: str, max_chars: int, overlap: int, min_chars: int) -> List[Tuple[int, str]]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive.")
    if overlap < 0:
        raise ValueError("overlap must be non-negative.")
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars.")

    chunks: List[Tuple[int, str]] = []
    text_len = len(text)
    start = 0
    chunk_index = 0
    while start < text_len:
        end = min(start + max_chars, text_len)
        chunk = text[start:end]
        if len(chunk) >= min_chars:
            chunks.append((chunk_index, chunk))
            chunk_index += 1
        if end == text_len:
            break
        start = end - overlap
    return chunks


def _hash_id(source_txt: str, chunk_index: int, text: str) -> str:
    payload = f"{source_txt}|{chunk_index}|{text}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _write_jsonl(records: Iterable[dict], output_path: Path) -> None:
    lines = [json.dumps(record, ensure_ascii=True) for record in records]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_manifest(
    output_path: Path,
    *,
    model_name: str,
    text_mode: str,
    max_chars: int,
    overlap: int,
    min_chars: int,
    batch_size: int,
    total_chunks: int,
    embedding_dim: int,
    input_text_root: str,
    input_pdf_root: str,
    output_root: str,
) -> None:
    manifest = {
        "created_at": _timestamp(),
        "model": model_name,
        "text_mode": text_mode,
        "max_chars": max_chars,
        "overlap": overlap,
        "min_chars": min_chars,
        "batch_size": batch_size,
        "total_chunks": total_chunks,
        "embedding_dim": embedding_dim,
        "input_text_root": input_text_root,
        "input_pdf_root": input_pdf_root,
        "output_root": output_root,
    }
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _vectorize_text(
    text: str,
    *,
    model_name: str,
    batch_size: int,
    max_chars: int,
    overlap: int,
    min_chars: int,
) -> Tuple[List[Tuple[int, str]], np.ndarray]:
    chunks = _chunk_text(text, max_chars=max_chars, overlap=overlap, min_chars=min_chars)
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        [chunk for _, chunk in chunks],
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    return chunks, embeddings


def _build_records(
    chunks: List[Tuple[int, str]],
    *,
    source_txt: str,
    source_pdf: str,
    source_exists: bool,
) -> List[dict]:
    records: List[dict] = []
    for vector_id, (chunk_index, text) in enumerate(chunks):
        records.append(
            {
                "id": _hash_id(source_txt, chunk_index, text),
                "vector_id": vector_id,
                "source_txt": source_txt,
                "source_pdf": source_pdf,
                "source_exists": source_exists,
                "page": None,
                "chunk_index": chunk_index,
                "char_len": len(text),
                "text": text,
            }
        )
    return records


def vectorize_document(
    input_path: Path,
    output_dir: Path,
    *,
    text_output_dir: Path | None,
    base_name: str | None,
    model_name: str,
    max_chars: int,
    overlap: int,
    min_chars: int,
    batch_size: int,
) -> VectorizationResult:
    input_path = input_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = input_path.suffix.lower()
    if suffix == ".pdf":
        if text_output_dir is None:
            text_output_dir = ROOT / "extracted_text_full" / input_path.stem.lower()
        result = ingest_file(input_path, text_output_dir, base_name=base_name)
        text_path = result.text_path
        source_pdf = input_path
    elif suffix == ".txt":
        text_path = input_path
        source_pdf = input_path.with_suffix(".pdf")
    else:
        raise ValueError(f"Unsupported input extension: {suffix}")

    text = text_path.read_text(encoding="utf-8")
    chunks, embeddings = _vectorize_text(
        text,
        model_name=model_name,
        batch_size=batch_size,
        max_chars=max_chars,
        overlap=overlap,
        min_chars=min_chars,
    )

    source_txt_rel = _relative_path(text_path)
    source_pdf_rel = _relative_path(source_pdf)
    source_exists = source_pdf.exists()
    records = _build_records(
        chunks,
        source_txt=source_txt_rel,
        source_pdf=source_pdf_rel,
        source_exists=source_exists,
    )

    np.save(output_dir / "embeddings.npy", embeddings)
    _write_jsonl(records, output_dir / "metadata.jsonl")
    _write_manifest(
        output_dir / "manifest.json",
        model_name=model_name,
        text_mode="both" if suffix == ".pdf" else "text",
        max_chars=max_chars,
        overlap=overlap,
        min_chars=min_chars,
        batch_size=batch_size,
        total_chunks=len(records),
        embedding_dim=embeddings.shape[1] if embeddings.size else 0,
        input_text_root=_relative_path(text_path.parent),
        input_pdf_root=_relative_path(source_pdf.parent),
        output_root=_relative_path(output_dir),
    )

    return VectorizationResult(
        output_dir=output_dir,
        total_chunks=len(records),
        embedding_dim=embeddings.shape[1] if embeddings.size else 0,
    )


def merge_vector_stores(store_paths: List[Path], output_dir: Path) -> None:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records: List[dict] = []
    embeddings_list: List[np.ndarray] = []
    source_stores: List[dict] = []
    total_chunks = 0
    embedding_dim = None

    for store_path in store_paths:
        store_path = store_path.expanduser().resolve()
        manifest_path = store_path / "manifest.json"
        metadata_path = store_path / "metadata.jsonl"
        embeddings_path = store_path / "embeddings.npy"

        if not (manifest_path.exists() and metadata_path.exists() and embeddings_path.exists()):
            raise FileNotFoundError(f"Missing vector store artifacts under {store_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_stores.append({"path": _relative_path(store_path), "manifest": manifest})

        records = [
            json.loads(line)
            for line in metadata_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        embeddings = np.load(embeddings_path)

        if embeddings.shape[0] != len(records):
            raise ValueError(f"Embeddings and metadata length mismatch for {store_path}")

        if embedding_dim is None:
            embedding_dim = embeddings.shape[1]
        elif embeddings.shape[1] != embedding_dim:
            raise ValueError(f"Embedding dimensions differ for {store_path}")

        for record in records:
            record["vector_id"] = total_chunks
            total_chunks += 1
            all_records.append(record)
        embeddings_list.append(embeddings)

    if embedding_dim is None:
        raise ValueError("No vector stores provided for merge.")

    combined = np.vstack(embeddings_list) if embeddings_list else np.zeros((0, embedding_dim))
    combined = combined.astype(np.float32, copy=False)

    np.save(output_dir / "embeddings.npy", combined)
    _write_jsonl(all_records, output_dir / "metadata.jsonl")

    manifest = {
        "created_at": _timestamp(),
        "source_stores": source_stores,
        "total_chunks": total_chunks,
        "embedding_dim": embedding_dim,
        "output_root": _relative_path(output_dir),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vectorize a case PDF or text file.")
    parser.add_argument("--input", required=True, type=Path, help="PDF or text file to vectorize.")
    parser.add_argument("--output", required=True, type=Path, help="Directory to write embeddings/metadata.")
    parser.add_argument(
        "--text-output-dir",
        type=Path,
        default=None,
        help="Directory to write extracted text/JSON if the input is a PDF.",
    )
    parser.add_argument("--base-name", type=str, default=None, help="Override base name for outputs.")
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
        "--merge-into",
        type=Path,
        default=None,
        help="Optional output directory to merge this store with others.",
    )
    parser.add_argument(
        "--merge-sources",
        nargs="*",
        type=Path,
        default=None,
        help="Vector store directories to merge; defaults to vector_store, vector_store_research, and the new output.",
    )
    return parser.parse_args()


def _resolve_merge_sources(
    output_dir: Path, merge_sources: List[Path] | None
) -> List[Path]:
    if merge_sources:
        sources = list(merge_sources)
    else:
        defaults = [ROOT / "vector_store", ROOT / "vector_store_research", output_dir]
        sources = [path for path in defaults if path.exists()]
    if output_dir not in sources:
        sources.append(output_dir)
    return sources


def main() -> None:
    args = _parse_args()
    result = vectorize_document(
        args.input,
        args.output,
        text_output_dir=args.text_output_dir,
        base_name=args.base_name,
        model_name=args.model,
        max_chars=args.max_chars,
        overlap=args.overlap,
        min_chars=args.min_chars,
        batch_size=args.batch_size,
    )
    print(f"Saved {result.total_chunks} chunks to {result.output_dir}")

    if args.merge_into:
        sources = _resolve_merge_sources(result.output_dir, args.merge_sources)
        merge_vector_stores(sources, args.merge_into)
        print(f"Merged stores into {args.merge_into}")


if __name__ == "__main__":
    main()
