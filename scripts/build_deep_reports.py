import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path


DATE_PATTERNS = [
    re.compile(
        r"(?P<month>\d{1,2})[./_-](?P<day>\d{1,2})[./_-](?P<year>\d{2,4})"
    ),
    re.compile(
        r"(?P<year>\d{4})[./_-](?P<month>\d{1,2})[./_-](?P<day>\d{1,2})"
    ),
]

CASE_PATTERNS = [
    re.compile(r"\b\d{2}-\d{2}-\d{5}-CV\b", re.IGNORECASE),
    re.compile(r"\b\d{2}-\d{4}\b"),
    re.compile(r"\b\d{3}-\d{6}-\d{2}\b"),
    re.compile(r"\b\d{2}-CV-\d{5}-[A-Z]\b", re.IGNORECASE),
    re.compile(r"\b\d{1,2}-\d{2}-cv-\d{5}\b", re.IGNORECASE),
]

TAG_KEYWORDS = [
    ("mandamus", ["mandamus"]),
    ("motion", ["motion"]),
    ("order", ["order"]),
    ("notice", ["notice"]),
    ("petition", ["petition"]),
    ("affidavit", ["affidavit"]),
    ("hearing", ["hearing"]),
    ("recusal", ["recusal"]),
    ("emergency", ["emergency"]),
    ("exhibit", ["exhibit"]),
    ("objection", ["objection"]),
    ("response", ["response"]),
    ("reply", ["reply"]),
    ("brief", ["brief"]),
    ("memorandum", ["memorandum", "memo"]),
    ("summary_judgment", ["summary judgment"]),
    ("discovery", ["discovery"]),
    ("sanctions", ["sanctions"]),
    ("transfer", ["transfer"]),
    ("record", ["record"]),
    ("report", ["report"]),
    ("declaration", ["declaration"]),
    ("complaint", ["complaint"]),
    ("appeal", ["appeal"]),
    ("injunction", ["injunction"]),
    ("removal", ["removal"]),
    ("remand", ["remand"]),
    ("federal", ["federal"]),
]

SIGNAL_PATTERNS = {
    "federal": re.compile(r"\bfederal\b|\bunited states\b|\bu\.s\.\b", re.IGNORECASE),
    "district_court": re.compile(
        r"\bdistrict court\b|\bu\.s\. district\b", re.IGNORECASE
    ),
    "removal": re.compile(r"\bremoval\b", re.IGNORECASE),
    "remand": re.compile(r"\bremand\b", re.IGNORECASE),
    "civil_rights_1983": re.compile(r"\b1983\b|\b42 u\.s\.c\.\s*1983\b", re.IGNORECASE),
    "rico": re.compile(
        r"\brico\b|\bracketeering\b|\benterprise\b", re.IGNORECASE
    ),
    "due_process": re.compile(
        r"\bdue process\b|\bfourteenth amendment\b|\bconstitutional\b", re.IGNORECASE
    ),
    "jurisdiction": re.compile(
        r"\bjurisdiction\b|\bdiversity\b|\b1332\b|\bfederal question\b",
        re.IGNORECASE,
    ),
    "abstention": re.compile(
        r"\byounger\b|\brooker\b|\bfeldman\b|\babstention\b", re.IGNORECASE
    ),
    "injunction": re.compile(r"\binjunction\b|\binjunctive\b", re.IGNORECASE),
    "mandamus": re.compile(r"\bmandamus\b", re.IGNORECASE),
}

WORD_RE = re.compile(r"\b\w+\b")


def iter_text_files(root: Path) -> list[Path]:
    return sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".txt"],
        key=lambda p: str(p).lower(),
    )


def normalize_year(year: int) -> int:
    if year < 100:
        return 2000 + year
    return year


def parse_date_from_name(name: str) -> date | None:
    candidates = []
    for pattern in DATE_PATTERNS:
        for match in pattern.finditer(name):
            try:
                year = normalize_year(int(match.group("year")))
                month = int(match.group("month"))
                day = int(match.group("day"))
                if year < 1990 or year > 2100:
                    continue
                candidates.append((match.start(), date(year, month, day)))
            except Exception:
                continue
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def extract_case_numbers(name: str) -> list[str]:
    found = []
    for pattern in CASE_PATTERNS:
        for match in pattern.finditer(name):
            found.append(match.group(0))
    deduped = []
    seen = set()
    for item in found:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def extract_tags(name: str) -> list[str]:
    name_lower = name.lower()
    name_norm = re.sub(r"[_\\-]+", " ", name_lower)
    tags = []
    for tag, keywords in TAG_KEYWORDS:
        if any(keyword in name_norm for keyword in keywords):
            tags.append(tag)
    return tags


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def strip_headers(content: str) -> str:
    lines = content.splitlines()
    filtered = []
    for line in lines:
        if line.startswith(("FILE:", "PAGES:", "OCR_MODE:", "DPI:")):
            continue
        if line.startswith("===") or line.startswith("--- EXTRACTED TEXT ---"):
            continue
        if line.startswith("--- OCR TEXT ---"):
            continue
        filtered.append(line)
    return "\n".join(filtered)


def load_text_body(txt_path: Path) -> str:
    try:
        content = txt_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    return strip_headers(content)


def load_meta(meta_path: Path) -> dict:
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            filtered = {field: row.get(field, "") for field in fieldnames}
            writer.writerow(filtered)


def summarize_months(entries: list[dict]) -> list[tuple[str, int]]:
    counter = Counter()
    for entry in entries:
        if not entry["date"]:
            continue
        month = entry["date"][:7]
        counter[month] += 1
    return counter.most_common(10)


def summarize_days(entries: list[dict]) -> list[tuple[str, int]]:
    counter = Counter()
    for entry in entries:
        if not entry["date"]:
            continue
        counter[entry["date"]] += 1
    return counter.most_common(10)


def summarize_tags(entries: list[dict]) -> list[tuple[str, int]]:
    counter = Counter()
    for entry in entries:
        tags = entry["doc_tags"].split(";") if entry["doc_tags"] else []
        for tag in tags:
            counter[tag] += 1
    return counter.most_common(20)


def build_reports(
    case_text_root: Path,
    case_pdf_root: Path,
    research_text_root: Path,
    research_pdf_root: Path,
    out_dir: Path,
    min_signal: int,
    skip_patterns: list[re.Pattern],
) -> None:
    corpora = [
        ("case", case_text_root, case_pdf_root),
        ("research", research_text_root, research_pdf_root),
    ]

    doc_entries = []
    signal_rows = []
    signal_totals = Counter()

    for corpus, text_root, pdf_root in corpora:
        for txt_path in iter_text_files(text_root):
            rel = txt_path.relative_to(text_root)
            source_pdf = pdf_root / rel.with_suffix(".pdf")
            if skip_patterns:
                path_str = f"{source_pdf} {txt_path}"
                if any(pat.search(path_str) for pat in skip_patterns):
                    continue
            meta = load_meta(txt_path.with_suffix(".json"))
            pages = meta.get("pages", 0) or 0
            text_chars = meta.get("text_chars", 0) or 0
            ocr_chars = meta.get("ocr_chars", 0) or 0
            status = meta.get("status", "")

            name_for_parse = str(rel)
            parsed_date = parse_date_from_name(name_for_parse)
            case_numbers = extract_case_numbers(name_for_parse)
            tags = extract_tags(name_for_parse)

            body = load_text_body(txt_path)
            words = count_words(body)

            entry = {
                "corpus": corpus,
                "source_pdf": str(source_pdf),
                "source_txt": str(txt_path),
                "source_exists": "yes" if source_pdf.exists() else "no",
                "date": parsed_date.isoformat() if parsed_date else "",
                "year": parsed_date.year if parsed_date else "",
                "month": f"{parsed_date.month:02d}" if parsed_date else "",
                "day": f"{parsed_date.day:02d}" if parsed_date else "",
                "case_numbers": ";".join(case_numbers),
                "doc_tags": ";".join(tags),
                "pages": pages,
                "text_chars": text_chars,
                "ocr_chars": ocr_chars,
                "word_count": words,
                "status": status,
            }
            doc_entries.append(entry)

            if body:
                signal_counts = {}
                total = 0
                for key, pattern in SIGNAL_PATTERNS.items():
                    count = len(pattern.findall(body))
                    signal_counts[key] = count
                    total += count
                    if count:
                        signal_totals[key] += count
                if total >= min_signal:
                    signal_row = {
                        "corpus": corpus,
                        "source_pdf": str(source_pdf),
                        "source_txt": str(txt_path),
                        "date": parsed_date.isoformat() if parsed_date else "",
                        "case_numbers": ";".join(case_numbers),
                        "signal_total": total,
                    }
                    signal_row.update(signal_counts)
                    signal_rows.append(signal_row)

    doc_index_path = out_dir / "doc_index.csv"
    doc_fields = [
        "corpus",
        "source_pdf",
        "source_txt",
        "source_exists",
        "date",
        "year",
        "month",
        "day",
        "case_numbers",
        "doc_tags",
        "pages",
        "text_chars",
        "ocr_chars",
        "word_count",
        "status",
    ]
    write_csv(doc_index_path, doc_entries, doc_fields)

    case_entries = [e for e in doc_entries if e["corpus"] == "case"]
    research_entries = [e for e in doc_entries if e["corpus"] == "research"]

    timeline_rows = [
        row
        for row in case_entries
        if row["date"]
    ]
    timeline_rows.sort(key=lambda r: (r["date"], r["source_pdf"]))
    timeline_path = out_dir / "filing_timeline.csv"
    timeline_fields = ["date", "source_pdf", "doc_tags", "case_numbers", "pages"]
    write_csv(timeline_path, timeline_rows, timeline_fields)

    signal_fields = [
        "corpus",
        "source_pdf",
        "source_txt",
        "date",
        "case_numbers",
        "signal_total",
    ] + list(SIGNAL_PATTERNS.keys())
    signal_rows.sort(key=lambda r: (-int(r["signal_total"]), r["source_pdf"]))
    signal_path = out_dir / "federal_signal_map.csv"
    write_csv(signal_path, signal_rows, signal_fields)

    case_pdf_count = len(
        {e["source_pdf"] for e in case_entries if e["source_exists"] == "yes"}
    )
    research_pdf_count = len(
        {e["source_pdf"] for e in research_entries if e["source_exists"] == "yes"}
    )
    case_pages = sum(int(e["pages"]) for e in case_entries if e["pages"])
    research_pages = sum(int(e["pages"]) for e in research_entries if e["pages"])
    case_words = sum(int(e["word_count"]) for e in case_entries)
    research_words = sum(int(e["word_count"]) for e in research_entries)

    top_months = summarize_months(case_entries)
    top_days = summarize_days(case_entries)
    top_tags = summarize_tags(case_entries)

    largest_filings = sorted(
        [e for e in case_entries if e["pages"]],
        key=lambda r: int(r["pages"]),
        reverse=True,
    )[:15]

    orphan_txt = len([e for e in case_entries if e["source_exists"] == "no"])

    stats_path = out_dir / "effort_stats_deep.md"
    with stats_path.open("w", encoding="utf-8") as f:
        f.write("# Effort Statistics (Deep Scan)\n\n")
        f.write("Corpus size (PDFs and pages):\n")
        f.write(f"- Case docs: {case_pdf_count} PDFs, {case_pages} pages.\n")
        f.write(f"- Research: {research_pdf_count} PDFs, {research_pages} pages.\n")
        f.write(
            f"- Total: {case_pdf_count + research_pdf_count} PDFs, {case_pages + research_pages} pages.\n\n"
        )
        f.write("Text volume processed:\n")
        f.write(f"- Case text files: {len(case_entries)}, approx {case_words} words.\n")
        f.write(
            f"- Research text files: {len(research_entries)}, approx {research_words} words.\n\n"
        )

        f.write("Filing volume by month (case docs, top 10):\n")
        for month, count in top_months:
            f.write(f"- {month}: {count} files.\n")
        f.write("\n")

        f.write("Most active filing days (case docs, top 10):\n")
        for day, count in top_days:
            f.write(f"- {day}: {count} files.\n")
        f.write("\n")

        f.write("Filing type tags by filename (case docs, top 20):\n")
        for tag, count in top_tags:
            f.write(f"- {tag}: {count}.\n")
        f.write("\n")

        f.write("Largest filings by page count (case docs, top 15):\n")
        for entry in largest_filings:
            f.write(f"- {entry['pages']} pages: {entry['source_pdf']}.\n")
        f.write("\n")

        f.write("Federal-signal keyword totals across all text (case + research):\n")
        for key, count in signal_totals.most_common():
            f.write(f"- {key}: {count}.\n")
        f.write("\n")

        f.write("Coverage notes:\n")
        f.write(
            f"- Case docs: {orphan_txt} text files without a matching PDF (likely removed/renamed).\n"
        )
        f.write("- Research: extraction now covers all PDFs.\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build deep reports across case and research corpora."
    )
    parser.add_argument("--case-text", default="extracted_text")
    parser.add_argument("--case-pdf", default="CASE DOCS")
    parser.add_argument("--research-text", default="extracted_research")
    parser.add_argument("--research-pdf", default="RESEARCH")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--min-signal", type=int, default=1)
    parser.add_argument(
        "--skip-pattern",
        action="append",
        default=[],
        help="Regex pattern to skip files (case-insensitive). Can be repeated.",
    )
    args = parser.parse_args()

    skip_patterns = [re.compile(pat, re.IGNORECASE) for pat in args.skip_pattern]

    build_reports(
        case_text_root=Path(args.case_text),
        case_pdf_root=Path(args.case_pdf),
        research_text_root=Path(args.research_text),
        research_pdf_root=Path(args.research_pdf),
        out_dir=Path(args.out_dir),
        min_signal=args.min_signal,
        skip_patterns=skip_patterns,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
