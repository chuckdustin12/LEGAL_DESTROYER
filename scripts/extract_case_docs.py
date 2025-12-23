import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
from PIL import Image


def find_tesseract() -> str | None:
    candidates = [
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Users\chuck\anaconda3\envs\video_notes\Library\bin\tesseract.exe"),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return shutil.which("tesseract")


def find_tessdata_dir(tesseract_cmd: str | None) -> str | None:
    if not tesseract_cmd:
        return None
    base = Path(tesseract_cmd).parent
    tessdata = base / "tessdata"
    if tessdata.exists():
        return str(tessdata)
    return None


def iter_pdfs(root: Path) -> list[Path]:
    return sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"],
        key=lambda p: str(p).lower(),
    )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_log_row(log_path: Path, row: dict) -> None:
    ensure_parent(log_path)
    file_exists = log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "input_path",
                "output_txt",
                "output_meta",
                "pages",
                "ocr_pages",
                "ocr_mode",
                "dpi",
                "text_chars",
                "ocr_chars",
                "duration_sec",
                "status",
                "error",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def render_page_image(page: fitz.Page, dpi: int) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return image


def process_pdf(
    input_path: Path,
    output_txt: Path,
    output_meta: Path,
    log_path: Path,
    ocr_mode: str,
    min_text_chars: int,
    dpi: int,
    lang: str,
    tess_config: str,
    resume: bool,
    log_skips: bool,
    skip_reason: str | None = None,
) -> None:
    if resume and output_txt.exists() and output_txt.stat().st_size > 0:
        if log_skips:
            write_log_row(
                log_path,
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "input_path": str(input_path),
                    "output_txt": str(output_txt),
                    "output_meta": str(output_meta),
                    "pages": "",
                    "ocr_pages": "",
                    "ocr_mode": ocr_mode,
                    "dpi": dpi,
                    "text_chars": "",
                    "ocr_chars": "",
                    "duration_sec": "",
                    "status": "skipped",
                    "error": "",
                },
            )
        return

    start = time.time()
    text_chars_total = 0
    ocr_chars_total = 0
    ocr_pages = 0
    pages = 0
    status = "ok"
    error = ""
    error_details: list[str] = []

    ensure_parent(output_txt)
    ensure_parent(output_meta)

    try:
        if skip_reason:
            if log_skips:
                write_log_row(
                    log_path,
                    {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "input_path": str(input_path),
                        "output_txt": str(output_txt),
                        "output_meta": str(output_meta),
                        "pages": "",
                        "ocr_pages": "",
                        "ocr_mode": ocr_mode,
                        "dpi": dpi,
                        "text_chars": "",
                        "ocr_chars": "",
                        "duration_sec": "",
                        "status": "skipped",
                        "error": skip_reason,
                    },
                )
            return
        doc = fitz.open(str(input_path))
        pages = doc.page_count
        with output_txt.open("w", encoding="utf-8", errors="ignore") as out:
            out.write(f"FILE: {input_path}\n")
            out.write(f"PAGES: {pages}\n")
            out.write(f"OCR_MODE: {ocr_mode}\n")
            out.write(f"DPI: {dpi}\n")
            out.write("=" * 80 + "\n\n")

            for page_index in range(pages):
                page = doc.load_page(page_index)
                extracted = page.get_text("text") or ""
                text_chars_total += len(extracted)

                do_ocr = ocr_mode == "always" or (
                    ocr_mode == "auto" and len(extracted.strip()) < min_text_chars
                )
                ocr_text = ""
                if do_ocr:
                    ocr_pages += 1
                    try:
                        image = render_page_image(page, dpi=dpi)
                        ocr_text = pytesseract.image_to_string(
                            image, lang=lang, config=tess_config
                        )
                        ocr_chars_total += len(ocr_text)
                    except Exception as exc:
                        status = "partial"
                        error_details.append(f"page {page_index + 1}: {exc}")
                        ocr_text = ""

                out.write(f"=== PAGE {page_index + 1} ===\n")
                if extracted.strip():
                    out.write("--- EXTRACTED TEXT ---\n")
                    out.write(extracted)
                    if not extracted.endswith("\n"):
                        out.write("\n")
                if do_ocr:
                    out.write("--- OCR TEXT ---\n")
                    out.write(ocr_text)
                    if not ocr_text.endswith("\n"):
                        out.write("\n")
                out.write("\n")
        doc.close()
    except Exception as exc:
        status = "error"
        error = str(exc)

    if error_details and not error:
        error = " | ".join(error_details[:5])

    duration_sec = round(time.time() - start, 2)

    meta = {
        "input_path": str(input_path),
        "output_txt": str(output_txt),
        "pages": pages,
        "ocr_pages": ocr_pages,
        "ocr_mode": ocr_mode,
        "dpi": dpi,
        "lang": lang,
        "text_chars": text_chars_total,
        "ocr_chars": ocr_chars_total,
        "duration_sec": duration_sec,
        "status": status,
        "error": error,
    }
    with output_meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    write_log_row(
        log_path,
        {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input_path": str(input_path),
            "output_txt": str(output_txt),
            "output_meta": str(output_meta),
            "pages": pages,
            "ocr_pages": ocr_pages,
            "ocr_mode": ocr_mode,
            "dpi": dpi,
            "text_chars": text_chars_total,
            "ocr_chars": ocr_chars_total,
            "duration_sec": duration_sec,
            "status": status,
            "error": error,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract text and OCR all PDFs for narrative analysis."
    )
    parser.add_argument(
        "--input",
        default="CASE DOCS",
        help="Input root folder containing PDFs.",
    )
    parser.add_argument(
        "--output",
        default="extracted_text",
        help="Output root folder for extracted text files.",
    )
    parser.add_argument(
        "--log",
        default="",
        help="Optional log path (CSV). Defaults to <output>/extraction_log.csv.",
    )
    parser.add_argument(
        "--ocr-mode",
        choices=["always", "auto"],
        default="always",
        help="OCR every page or only pages with low extracted text.",
    )
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=50,
        help="Minimum extracted characters before skipping OCR in auto mode.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Render DPI for OCR images.",
    )
    parser.add_argument(
        "--lang",
        default="eng",
        help="Tesseract language(s), e.g., eng or eng+spa.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip PDFs that already have non-empty output text.",
    )
    parser.add_argument(
        "--log-skips",
        action="store_true",
        help="Log skipped files when using --resume.",
    )
    parser.add_argument(
        "--skip-pattern",
        action="append",
        default=[],
        help="Regex pattern to skip files (case-insensitive). Can be repeated.",
    )
    parser.add_argument(
        "--include-pattern",
        action="append",
        default=[],
        help="Regex pattern to include files (case-insensitive). If set, only matching files are processed.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Process at most N files (0 means no limit).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="1-based index in the sorted PDF list to start from.",
    )
    args = parser.parse_args()

    tesseract_cmd = find_tesseract()
    if not tesseract_cmd or not Path(tesseract_cmd).exists():
        print("Tesseract not found. Install it and try again.", file=sys.stderr)
        return 1
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    tessdata_dir = find_tessdata_dir(tesseract_cmd)
    if tessdata_dir:
        os.environ["TESSDATA_PREFIX"] = tessdata_dir

    input_root = Path(args.input)
    if not input_root.exists():
        print(f"Input folder not found: {input_root}", file=sys.stderr)
        return 1

    output_root = Path(args.output)
    log_path = Path(args.log) if args.log else output_root / "extraction_log.csv"

    all_pdfs = iter_pdfs(input_root)
    total_all = len(all_pdfs)
    start_index = max(args.start_index, 1)
    if start_index > total_all:
        print("Start index is beyond available files.", file=sys.stderr)
        return 1
    pdfs = all_pdfs[start_index - 1 :]
    if args.max_files:
        pdfs = pdfs[: args.max_files]

    include_patterns = [re.compile(pat, re.IGNORECASE) for pat in args.include_pattern]
    if include_patterns:
        pdfs = [
            p
            for p in pdfs
            if any(pat.search(str(p)) for pat in include_patterns)
        ]

    skip_patterns = [re.compile(pat, re.IGNORECASE) for pat in args.skip_pattern]
    log_skips = args.log_skips or bool(skip_patterns)
    tess_config = "--oem 1 --psm 3"

    total = len(pdfs)
    start_offset = start_index - 1
    for idx, pdf_path in enumerate(pdfs, start=1):
        rel = pdf_path.relative_to(input_root)
        output_txt = output_root / rel.with_suffix(".txt")
        output_meta = output_root / rel.with_suffix(".json")

        global_idx = start_offset + idx
        print(f"[{global_idx}/{total_all}] {pdf_path}")
        skip_reason = None
        if skip_patterns:
            path_str = str(pdf_path)
            for pat in skip_patterns:
                if pat.search(path_str):
                    skip_reason = f"skip_pattern:{pat.pattern}"
                    break
        process_pdf(
            input_path=pdf_path,
            output_txt=output_txt,
            output_meta=output_meta,
            log_path=log_path,
            ocr_mode=args.ocr_mode,
            min_text_chars=args.min_text_chars,
            dpi=args.dpi,
            lang=args.lang,
            tess_config=tess_config,
            resume=args.resume,
            log_skips=log_skips,
            skip_reason=skip_reason,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
