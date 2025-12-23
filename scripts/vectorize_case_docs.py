import argparse
import json
import re
import time
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Generator, Iterable

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class Chunk:
    source_txt: str
    source_pdf: str
    source_exists: bool
    page: int | None
    chunk_index: int
    text: str


def iter_text_files(root: Path) -> list[Path]:
    return sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".txt"],
        key=lambda p: str(p).lower(),
    )


def parse_extracted_text(path: Path) -> Generator[tuple[int | None, str, str], None, None]:
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        yield (None, "", "")
        return

    lines = content.splitlines()
    current_page = None
    section = None
    extracted = ""
    ocr = ""
    saw_page = False

    for line in lines:
        if line.startswith("=== PAGE "):
            if current_page is not None:
                yield (current_page, extracted, ocr)
            match = re.match(r"=== PAGE (\\d+)", line)
            current_page = int(match.group(1)) if match else None
            extracted = ""
            ocr = ""
            section = None
            saw_page = True
            continue

        if line.startswith("--- EXTRACTED TEXT ---"):
            section = "extracted"
            continue
        if line.startswith("--- OCR TEXT ---"):
            section = "ocr"
            continue

        if section == "extracted":
            extracted += line + "\n"
        elif section == "ocr":
            ocr += line + "\n"

    if saw_page:
        yield (current_page, extracted, ocr)
    else:
        yield (None, content, "")


def select_text(extracted: str, ocr: str, mode: str) -> str:
    extracted = extracted.strip()
    ocr = ocr.strip()
    if mode == "ocr":
        return ocr or extracted
    if mode == "extract":
        return extracted or ocr
    # both
    if extracted and ocr:
        return extracted + "\n" + ocr
    return extracted or ocr


def chunk_text(text: str, max_chars: int, overlap: int) -> list[str]:
    text = re.sub(r"\\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    if max_chars <= 0:
        return [text]
    chunks = []
    start = 0
    length = len(text)
    overlap = max(0, min(overlap, max_chars - 1))
    while start < length:
        end = min(start + max_chars, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = end - overlap
    return chunks


def iter_chunks(
    text_root: Path,
    pdf_root: Path,
    text_mode: str,
    max_chars: int,
    overlap: int,
    min_chars: int,
    skip_patterns: list[re.Pattern],
) -> Generator[Chunk, None, None]:
    for txt_path in iter_text_files(text_root):
        rel = txt_path.relative_to(text_root)
        source_pdf = pdf_root / rel.with_suffix(".pdf")
        if skip_patterns:
            path_str = str(source_pdf) if source_pdf.exists() else str(txt_path)
            if any(pat.search(path_str) for pat in skip_patterns):
                continue

        for page, extracted, ocr in parse_extracted_text(txt_path):
            selected = select_text(extracted, ocr, text_mode)
            if not selected:
                continue
            parts = chunk_text(selected, max_chars=max_chars, overlap=overlap)
            for idx, chunk in enumerate(parts):
                if len(chunk) < min_chars:
                    continue
                yield Chunk(
                    source_txt=str(txt_path),
                    source_pdf=str(source_pdf),
                    source_exists=source_pdf.exists(),
                    page=page,
                    chunk_index=idx,
                    text=chunk,
                )


def write_manifest(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Vectorize extracted PDF text for semantic search."
    )
    parser.add_argument("--input", default="extracted_text", help="Input text root.")
    parser.add_argument("--pdf-root", default="CASE DOCS", help="PDF root.")
    parser.add_argument("--output", default="vector_store", help="Output folder.")
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name or path.",
    )
    parser.add_argument(
        "--text-mode",
        choices=["both", "extract", "ocr"],
        default="both",
        help="Which text to embed per page.",
    )
    parser.add_argument(
        "--max-chars", type=int, default=2000, help="Max chars per chunk."
    )
    parser.add_argument(
        "--overlap", type=int, default=200, help="Chunk overlap in chars."
    )
    parser.add_argument(
        "--min-chars", type=int, default=50, help="Minimum chunk size."
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Embed batch size.")
    parser.add_argument(
        "--skip-pattern",
        action="append",
        default=[],
        help="Regex to skip files (case-insensitive). Can be repeated.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Limit number of text files for a dry run (0 means no limit).",
    )
    args = parser.parse_args()

    text_root = Path(args.input)
    pdf_root = Path(args.pdf_root)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    skip_patterns = [re.compile(pat, re.IGNORECASE) for pat in args.skip_pattern]

    model = SentenceTransformer(args.model)

    all_embeddings = []
    metadata_path = out_root / "metadata.jsonl"

    if metadata_path.exists():
        metadata_path.unlink()

    vector_id = 0
    batch_texts: list[str] = []
    batch_meta: list[Chunk] = []

    txt_files = iter_text_files(text_root)
    if args.max_files:
        txt_files = txt_files[: args.max_files]

    chunk_iter = iter_chunks(
        text_root=text_root,
        pdf_root=pdf_root,
        text_mode=args.text_mode,
        max_chars=args.max_chars,
        overlap=args.overlap,
        min_chars=args.min_chars,
        skip_patterns=skip_patterns,
    )

    def flush_batch() -> None:
        nonlocal vector_id
        if not batch_texts:
            return
        vectors = model.encode(
            batch_texts,
            batch_size=args.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        all_embeddings.append(vectors.astype(np.float32))
        with metadata_path.open("a", encoding="utf-8") as f:
            for meta, text in zip(batch_meta, batch_texts):
                uid_source = f"{meta.source_txt}|{meta.page}|{meta.chunk_index}"
                uid = sha1(uid_source.encode("utf-8")).hexdigest()
                record = {
                    "id": uid,
                    "vector_id": vector_id,
                    "source_txt": meta.source_txt,
                    "source_pdf": meta.source_pdf,
                    "source_exists": meta.source_exists,
                    "page": meta.page,
                    "chunk_index": meta.chunk_index,
                    "char_len": len(text),
                    "text": text,
                }
                f.write(json.dumps(record, ensure_ascii=True) + "\n")
                vector_id += 1
        batch_texts.clear()
        batch_meta.clear()

    total_chunks = 0
    for chunk in chunk_iter:
        batch_texts.append(chunk.text)
        batch_meta.append(chunk)
        total_chunks += 1
        if len(batch_texts) >= args.batch_size:
            flush_batch()

    flush_batch()

    embeddings = np.vstack(all_embeddings) if all_embeddings else np.zeros((0, 0))
    np.save(out_root / "embeddings.npy", embeddings)

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "text_mode": args.text_mode,
        "max_chars": args.max_chars,
        "overlap": args.overlap,
        "min_chars": args.min_chars,
        "batch_size": args.batch_size,
        "total_chunks": total_chunks,
        "embedding_dim": int(embeddings.shape[1]) if embeddings.size else 0,
        "input_text_root": str(text_root),
        "input_pdf_root": str(pdf_root),
        "output_root": str(out_root),
    }
    write_manifest(out_root / "manifest.json", manifest)

    print(f"Chunks: {total_chunks}")
    print(f"Embeddings: {embeddings.shape}")
    print(f"Metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
