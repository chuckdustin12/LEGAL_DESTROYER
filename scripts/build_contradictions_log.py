import argparse
import csv
import re
from collections import defaultdict
from datetime import date
from pathlib import Path


DATE_NUMERIC = re.compile(
    r"\b(?P<month>\d{1,2})[/-](?P<day>\d{1,2})[/-](?P<year>\d{2,4})\b"
)
DATE_MONTHNAME = re.compile(
    r"\b(?P<month_name>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)\s+(?P<day>\d{1,2})(?:,)?\s+(?P<year>\d{4})\b",
    re.IGNORECASE,
)

MONTH_MAP = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

EVENT_PATTERNS = [
    ("remand", [r"\bremand\b"]),
    ("removal", [r"\bremoval\b", r"\bremove(d)? to federal\b"]),
    ("recusal", [r"\brecusal\b", r"\brecuse\b", r"disqualif"]),
    ("assignment", [r"order of assignment", r"assigned judge", r"assignment"]),
    ("order", [r"\border\b", r"order of referral", r"proposed order"]),
    ("hearing", [r"\bhearing\b", r"show cause", r"conference", r"oral argument"]),
    ("trial", [r"\btrial\b", r"trial setting", r"final trial"]),
    ("notice", [r"\bnotice\b", r"notification"]),
    ("mandamus", [r"\bmandamus\b", r"original proceeding", r"writ of mandamus"]),
    ("appeal", [r"\bappeal\b", r"appellate", r"court of appeals", r"supreme court"]),
    ("petition", [r"\bpetition\b"]),
    ("injunction", [r"\binjunction\b", r"\btro\b"]),
    ("temporary_orders", [r"temporary order", r"temporary orders"]),
    ("de_novo", [r"\bde novo\b"]),
]


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            filtered = {field: row.get(field, "") for field in fieldnames}
            writer.writerow(filtered)


def normalize_year(year: int) -> int:
    if year < 100:
        return 2000 + year
    return year


def parse_dates(text: str) -> list[date]:
    dates = []
    for match in DATE_NUMERIC.finditer(text):
        try:
            year = normalize_year(int(match.group("year")))
            month = int(match.group("month"))
            day = int(match.group("day"))
            dates.append(date(year, month, day))
        except Exception:
            continue
    for match in DATE_MONTHNAME.finditer(text):
        try:
            name = match.group("month_name")[:3].lower()
            month = MONTH_MAP.get(name)
            if not month:
                continue
            year = int(match.group("year"))
            day = int(match.group("day"))
            dates.append(date(year, month, day))
        except Exception:
            continue
    deduped = []
    seen = set()
    for item in dates:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


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


def load_text_lines(path: Path) -> list[str]:
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    content = strip_headers(content)
    return [line.strip() for line in content.splitlines() if line.strip()]


def normalize_context(line: str) -> str:
    line = DATE_NUMERIC.sub("<DATE>", line)
    line = DATE_MONTHNAME.sub("<DATE>", line)
    line = re.sub(r"[^a-zA-Z0-9\s<>]", " ", line)
    line = re.sub(r"\s+", " ", line).strip().lower()
    tokens = line.split()
    return " ".join(tokens[:12])


def ascii_safe(value: str) -> str:
    return value.encode("ascii", errors="ignore").decode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a potential contradictions/conflicts log from case docs."
    )
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--case-text", default="extracted_text")
    parser.add_argument("--max-groups", type=int, default=200)
    parser.add_argument("--min-docs", type=int, default=2)
    args = parser.parse_args()

    reports_dir = Path(args.reports)
    doc_index = read_csv(reports_dir / "doc_index.csv")
    txt_map = {row.get("source_txt", ""): row for row in doc_index if row.get("corpus") == "case"}

    event_regex = [
        (event_type, [re.compile(pat, re.IGNORECASE) for pat in patterns])
        for event_type, patterns in EVENT_PATTERNS
    ]

    groups = {}

    for txt_path in sorted(Path(args.case_text).rglob("*.txt")):
        meta = txt_map.get(str(txt_path), {})
        source_pdf = meta.get("source_pdf", "")
        case_numbers = meta.get("case_numbers", "")
        lines = load_text_lines(txt_path)
        if not lines:
            continue

        for line in lines:
            event_type = None
            for evt, patterns in event_regex:
                if any(pat.search(line) for pat in patterns):
                    event_type = evt
                    break
            if not event_type:
                continue
            date_hits = parse_dates(line)
            if not date_hits:
                continue

            context = normalize_context(line)
            key = (event_type, case_numbers, context)
            if key not in groups:
                groups[key] = {
                    "event_type": event_type,
                    "case_numbers": case_numbers,
                    "context": context,
                    "dates": defaultdict(list),
                    "sources": set(),
                }
            group = groups[key]
            group["sources"].add(source_pdf)

            for dt in date_hits:
                if len(group["dates"][dt]) < 1:
                    snippet = ascii_safe(line)
                    snippet = snippet[:200] + "..." if len(snippet) > 200 else snippet
                    sample = f"{dt.isoformat()} | {ascii_safe(source_pdf)} | {snippet}"
                    group["dates"][dt].append(sample)

    rows = []
    for key, group in groups.items():
        date_values = sorted(group["dates"].keys())
        if len(date_values) < 2:
            continue
        doc_count = len(group["sources"])
        if doc_count < args.min_docs:
            continue

        samples = []
        for dt in date_values:
            samples.extend(group["dates"][dt][:1])
        rows.append(
            {
                "event_type": group["event_type"],
                "case_numbers": group["case_numbers"],
                "context": group["context"],
                "distinct_dates": len(date_values),
                "date_values": ";".join(d.isoformat() for d in date_values),
                "doc_count": doc_count,
                "sources": ";".join(sorted(group["sources"])[:10]),
                "sample_lines": " || ".join(samples),
            }
        )

    rows.sort(key=lambda r: (-int(r["distinct_dates"]), -int(r["doc_count"])))
    rows = rows[: args.max_groups]

    out_csv = reports_dir / "contradictions_log.csv"
    write_csv(
        out_csv,
        rows,
        [
            "event_type",
            "case_numbers",
            "context",
            "distinct_dates",
            "date_values",
            "doc_count",
            "sources",
            "sample_lines",
        ],
    )

    out_md = reports_dir / "contradictions_log.md"
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Potential Conflicts Log (Document-Derived)\n\n")
        f.write("This log flags potential date conflicts for review. It is not legal advice.\n\n")
        f.write(f"Groups shown: {len(rows)} (min docs per group: {args.min_docs}).\n\n")
        for row in rows[:50]:
            f.write(f"- {row['event_type']} | cases: {row['case_numbers']} | dates: {row['date_values']} | docs: {row['doc_count']}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
