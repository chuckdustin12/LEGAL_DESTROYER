from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fpdf import FPDF

# Ensure repository root is on the import path for local modules.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ingest_merged_case import ingest_file, UnsupportedFileTypeError


def _create_sample_pdf(tmp_path: Path) -> Path:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 10, "First page text with numbers 123.")
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 10, "Second page text with symbols !@#.")
    pdf_path = tmp_path / "sample.pdf"
    pdf.output(pdf_path)
    return pdf_path


def test_ingest_pdf_creates_outputs(tmp_path: Path) -> None:
    pdf_path = _create_sample_pdf(tmp_path)
    output_dir = tmp_path / "out"

    result = ingest_file(pdf_path, output_dir, base_name="sample_output")

    assert result.page_count == 2
    assert result.text_path.exists()
    assert result.json_path.exists()

    combined_text = result.text_path.read_text(encoding="utf-8")
    assert "First page text" in combined_text
    assert "Second page text" in combined_text

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["page_count"] == 2
    assert payload["pages"][0]["page_number"] == 1
    assert payload["pages"][1]["page_number"] == 2
    assert payload["pages"][0]["char_length"] > 0
    assert payload["pages"][1]["char_length"] > 0


def test_ingest_plain_text(tmp_path: Path) -> None:
    text_path = tmp_path / "note.txt"
    text_content = "Lone page content for ingestion."
    text_path.write_text(text_content, encoding="utf-8")

    result = ingest_file(text_path, tmp_path)

    assert result.page_count == 1
    saved_text = result.text_path.read_text(encoding="utf-8")
    assert saved_text == text_content

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["page_count"] == 1
    assert payload["pages"][0]["text"] == text_content


def test_rejects_unsupported_extension(tmp_path: Path) -> None:
    bogus = tmp_path / "fake.docx"
    bogus.write_text("not real", encoding="utf-8")

    with pytest.raises(UnsupportedFileTypeError):
        ingest_file(bogus, tmp_path)
