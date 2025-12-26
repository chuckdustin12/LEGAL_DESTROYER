"""OCR case-doc PDFs with empty vector stores into per-filer text/JSON outputs."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from hashlib import sha1
from pathlib import Path
from typing import Dict, List

import numpy as np
import fitz
import pytesseract
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _ensure_tesseract() -> None:
    current = os.environ.get("TESSDATA_PREFIX")
    if current and (Path(current) / "eng.traineddata").exists():
        return
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tessdata",
        r"C:\Program Files (x86)\Tesseract-OCR\tessdata",
    ]
    for candidate in candidates:
        if (Path(candidate) / "eng.traineddata").exists():
            os.environ["TESSDATA_PREFIX"] = candidate
            return


def _slugify(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    return cleaned or "document"


def _unique_slug(base: str, seen: Dict[str, int], path: Path) -> str:
    if base not in seen:
        seen[base] = 1
        return base
    digest = sha1(str(path).encode("utf-8")).hexdigest()[:8]
    slug = f"{base}_{digest}"
    seen[base] += 1
    return slug


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


def _write_text_output(pages: List[str], output_path: Path) -> None:
    normalized_pages = [page.strip() for page in pages]
    combined = "\n\n".join(normalized_pages).strip()
    output_path.write_text(combined, encoding="utf-8")


def _write_json_output(
    pages: List[str], source: Path, base_name: str, output_path: Path, *, dpi: int, lang: str
) -> None:
    payload = {
        "source": str(source),
        "base_name": base_name,
        "page_count": len(pages),
        "ocr": {"dpi": dpi, "lang": lang, "created_at": _timestamp()},
        "pages": [
            {"page_number": idx + 1, "text": page, "char_length": len(page)}
            for idx, page in enumerate(pages)
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _ocr_pdf(pdf_path: Path, output_dir: Path, *, base_name: str, dpi: int, lang: str) -> int:
    doc = fitz.open(str(pdf_path))
    scale = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)
    pages: List[str] = []
    for page in doc:
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        image = ImageOps.autocontrast(image.convert("L"))
        text = pytesseract.image_to_string(image, lang=lang)
        pages.append(text)

    text_path = output_dir / f"{base_name}.txt"
    json_path = output_dir / f"{base_name}.json"
    _write_text_output(pages, text_path)
    _write_json_output(pages, pdf_path, base_name, json_path, dpi=dpi, lang=lang)
    return len(pages)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OCR CASE DOCS PDFs with empty vector stores."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("CASE DOCS"),
        help="Root directory containing filer subfolders.",
    )
    parser.add_argument(
        "--sources-dir",
        type=Path,
        default=Path("vector_store_case_docs_by_filer_sources"),
        help="Root directory for per-document vector stores.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("extracted_text_full/case_docs_by_filer"),
        help="Root directory for OCR outputs.",
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
    parser.add_argument("--dpi", type=int, default=300, help="Render DPI for OCR.")
    parser.add_argument("--lang", type=str, default="eng", help="OCR language.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run OCR even if outputs already exist.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="OCR every PDF, not just those with empty vector stores.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _ensure_tesseract()
    input_dir = args.input_dir.expanduser().resolve()
    sources_dir = args.sources_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Missing input directory: {input_dir}")

    filer_dirs = _iter_filer_dirs(input_dir, args.include_root)
    if not filer_dirs:
        raise ValueError(f"No filer directories found under {input_dir}")

    only = {_slugify(name) for name in args.only} if args.only else None
    skip = {_slugify(name) for name in args.skip} if args.skip else set()

    total = 0
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

        seen_slugs: Dict[str, int] = {}
        for pdf_path in pdfs:
            base_slug = _slugify(pdf_path.stem)
            slug = _unique_slug(base_slug, seen_slugs, pdf_path)
            store_dir = sources_dir / filer_slug / slug
            if not args.all:
                if not store_dir.exists():
                    continue
                if _store_has_chunks(store_dir):
                    continue

            output_dir = output_root / filer_slug / slug
            output_dir.mkdir(parents=True, exist_ok=True)
            text_path = output_dir / f"{slug}.txt"
            json_path = output_dir / f"{slug}.json"
            if text_path.exists() and json_path.exists() and not args.force:
                if text_path.stat().st_size > 0:
                    print(f"Skipping OCR for {pdf_path.name}; outputs exist.")
                    continue
                print(f"OCR needed for {pdf_path.name}; existing text is empty.")

            pages = _ocr_pdf(pdf_path, output_dir, base_name=slug, dpi=args.dpi, lang=args.lang)
            total += 1
            print(f"OCR complete: {pdf_path.name} ({pages} pages)")

    print(f"OCR finished. Documents processed: {total}")


if __name__ == "__main__":
    main()
