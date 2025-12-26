"""Utilities for ingesting merged case PDFs into text and JSON.

This module extracts page-level text from PDFs (or raw text files),
combines the content into newline-delimited text, and writes both
plain-text and JSON outputs that downstream tooling can vectorize.

Usage:
    python scripts/ingest_merged_case.py --input "CASE DOCS/28B_merged.pdf" \
        --output extracted_text_full/28b_merged

The script is intentionally small and dependency-light. The functions
are separated to make unit testing straightforward.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from pypdf import PdfReader


@dataclass
class IngestionResult:
    """Represents the saved artifacts for a single ingested file."""

    text_path: Path
    json_path: Path
    page_count: int


class UnsupportedFileTypeError(ValueError):
    """Raised when an unsupported file extension is provided."""


SUPPORTED_SUFFIXES = {".pdf", ".txt"}


def _extract_pages_from_pdf(pdf_path: Path) -> List[str]:
    """Return a list of page texts extracted from a PDF.

    Empty strings are allowed if a page has no extractable text, but we
    always return an entry per page to keep indices aligned.
    """

    reader = PdfReader(str(pdf_path))
    pages: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text.strip())
    return pages


def _extract_pages_from_text(txt_path: Path) -> List[str]:
    """Treat a raw text file as a single-page document."""

    content = txt_path.read_text(encoding="utf-8")
    return [content.strip()]


def _iter_pages(input_path: Path) -> List[str]:
    """Dispatch to the appropriate extractor based on file suffix."""

    suffix = input_path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pages_from_pdf(input_path)
    if suffix == ".txt":
        return _extract_pages_from_text(input_path)
    raise UnsupportedFileTypeError(
        f"Unsupported file type {suffix}; supported types: {sorted(SUPPORTED_SUFFIXES)}"
    )


def _write_text_output(pages: Iterable[str], output_path: Path) -> None:
    """Write concatenated pages to a UTF-8 text file with spacing."""

    normalized_pages = [page.strip() for page in pages]
    combined = "\n\n".join(normalized_pages).strip()
    output_path.write_text(combined, encoding="utf-8")


def _write_json_output(
    pages: List[str], source: Path, base_name: str, output_path: Path
) -> None:
    """Persist structured metadata about the ingested document."""

    payload = {
        "source": str(source),
        "base_name": base_name,
        "page_count": len(pages),
        "pages": [
            {
                "page_number": idx + 1,
                "text": page,
                "char_length": len(page),
            }
            for idx, page in enumerate(pages)
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ingest_file(input_path: Path, output_dir: Path, base_name: str | None = None) -> IngestionResult:
    """Extract text from `input_path` and write JSON + text outputs.

    Args:
        input_path: Path to the PDF or plain-text document.
        output_dir: Directory to contain the extracted outputs.
        base_name: Optional override for the output file names (without extension).

    Returns:
        IngestionResult describing the saved artifact paths and page count.
    """

    input_path = input_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = base_name or input_path.stem
    pages = _iter_pages(input_path)

    text_path = output_dir / f"{base_name}.txt"
    json_path = output_dir / f"{base_name}.json"

    _write_text_output(pages, text_path)
    _write_json_output(pages, input_path, base_name, json_path)

    return IngestionResult(text_path=text_path, json_path=json_path, page_count=len(pages))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest a merged case PDF into text/JSON outputs.")
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the merged PDF or text file (e.g., 'CASE DOCS/28B_merged.pdf').",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory to write the extracted outputs (e.g., extracted_text_full/28b_merged).",
    )
    parser.add_argument(
        "--base-name",
        dest="base_name",
        type=str,
        default=None,
        help="Optional base name for output files; defaults to the input stem.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = ingest_file(args.input, args.output, base_name=args.base_name)
    print(f"Saved text to {result.text_path}")
    print(f"Saved JSON to {result.json_path}")
    print(f"Page count: {result.page_count}")


if __name__ == "__main__":
    main()
