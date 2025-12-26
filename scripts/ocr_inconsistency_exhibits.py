"""OCR exhibit PDFs in INCONSISTENCIES into text + JSON outputs."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List

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
            {
                "page_number": idx + 1,
                "text": page,
                "char_length": len(page),
            }
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
    parser = argparse.ArgumentParser(description="OCR exhibit PDFs under INCONSISTENCIES.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("INCONSISTENCIES"),
        help="Directory containing exhibit PDFs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("extracted_text_full/inconsistencies"),
        help="Root directory for OCR outputs.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*EXHIBIT*.pdf",
        help="Glob pattern for exhibit PDFs.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Render DPI for OCR.")
    parser.add_argument("--lang", type=str, default="eng", help="OCR language.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run OCR even if outputs already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _ensure_tesseract()
    input_dir = args.input_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(input_dir.glob(args.pattern))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs matched {args.pattern} under {input_dir}")

    for pdf_path in pdfs:
        slug = _slugify(pdf_path.stem)
        output_dir = output_root / slug
        output_dir.mkdir(parents=True, exist_ok=True)
        text_path = output_dir / f"{slug}.txt"
        json_path = output_dir / f"{slug}.json"
        if text_path.exists() and json_path.exists() and not args.force:
            print(f"Skipping OCR for {pdf_path.name}; outputs exist.")
            continue
        pages = _ocr_pdf(pdf_path, output_dir, base_name=slug, dpi=args.dpi, lang=args.lang)
        print(f"OCR complete: {pdf_path.name} ({pages} pages)")


if __name__ == "__main__":
    main()
