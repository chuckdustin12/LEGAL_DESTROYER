import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


PROCEDURAL_TAGS = {
    "motion",
    "order",
    "notice",
    "hearing",
    "recusal",
    "mandamus",
    "appeal",
    "petition",
    "affidavit",
    "response",
    "objection",
    "brief",
    "memorandum",
    "emergency",
    "transfer",
    "discovery",
    "sanctions",
    "injunction",
    "report",
    "removal",
    "remand",
    "record",
    "trial",
    "ruling",
}


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


def parse_date(date_str: str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a procedural docket map.")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--min-year", type=int, default=1990)
    parser.add_argument("--max-year", type=int, default=2030)
    args = parser.parse_args()

    reports_dir = Path(args.reports)
    doc_index = read_csv(reports_dir / "doc_index.csv")
    issue_doc_map = read_csv(reports_dir / "issue_doc_map.csv")

    issue_map = defaultdict(set)
    for row in issue_doc_map:
        if row.get("corpus") != "case":
            continue
        source = row.get("source_pdf", "")
        issue = row.get("issue", "")
        if source and issue:
            issue_map[source].add(issue)

    docket_rows = []
    tag_counts = Counter()
    month_tag_counts = defaultdict(Counter)
    case_tag_counts = defaultdict(Counter)

    for row in doc_index:
        if row.get("corpus") != "case":
            continue
        date_val = parse_date(row.get("date", ""))
        if not date_val or not (args.min_year <= date_val.year <= args.max_year):
            continue
        doc_tags = [t for t in (row.get("doc_tags", "") or "").split(";") if t]
        proc_tags = [t for t in doc_tags if t in PROCEDURAL_TAGS]
        if not proc_tags:
            continue

        source_pdf = row.get("source_pdf", "")
        case_numbers = row.get("case_numbers", "")
        issues = sorted(issue_map.get(source_pdf, set()))

        docket_rows.append(
            {
                "date": date_val.isoformat(),
                "case_numbers": case_numbers,
                "doc_tags": ";".join(sorted(proc_tags)),
                "issue_tags": ";".join(issues),
                "pages": row.get("pages", ""),
                "source_pdf": source_pdf,
            }
        )

        month = date_val.strftime("%Y-%m")
        for tag in proc_tags:
            tag_counts[tag] += 1
            month_tag_counts[month][tag] += 1
            if case_numbers:
                for case_num in case_numbers.split(";"):
                    case_num = case_num.strip()
                    if case_num:
                        case_tag_counts[case_num][tag] += 1

    docket_rows.sort(key=lambda r: (r["date"], r["source_pdf"]))
    write_csv(
        reports_dir / "procedural_docket_map.csv",
        docket_rows,
        ["date", "case_numbers", "doc_tags", "issue_tags", "pages", "source_pdf"],
    )

    month_rows = []
    for month in sorted(month_tag_counts.keys()):
        for tag, count in month_tag_counts[month].items():
            month_rows.append({"month": month, "tag": tag, "doc_count": count})
    write_csv(
        reports_dir / "procedural_docket_monthly.csv",
        month_rows,
        ["month", "tag", "doc_count"],
    )

    tag_rows = [{"tag": tag, "doc_count": count} for tag, count in tag_counts.most_common()]
    write_csv(
        reports_dir / "procedural_docket_tag_counts.csv",
        tag_rows,
        ["tag", "doc_count"],
    )

    case_rows = []
    for case_num, counts in case_tag_counts.items():
        for tag, count in counts.items():
            case_rows.append({"case_number": case_num, "tag": tag, "doc_count": count})
    write_csv(
        reports_dir / "procedural_docket_by_case.csv",
        case_rows,
        ["case_number", "tag", "doc_count"],
    )

    summary_path = reports_dir / "procedural_docket_map.md"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("# Procedural Docket Map (Document-Derived)\n\n")
        f.write("This map indexes procedural filings by date and tag. It is not legal advice.\n\n")
        f.write(f"- Rows: {len(docket_rows)}\n")
        f.write(f"- Year bounds: {args.min_year} to {args.max_year}\n\n")
        f.write("Top procedural tags:\n")
        for tag, count in tag_counts.most_common(10):
            f.write(f"- {tag}: {count}\n")
        f.write("\nOutputs:\n")
        f.write("- reports/procedural_docket_map.csv\n")
        f.write("- reports/procedural_docket_monthly.csv\n")
        f.write("- reports/procedural_docket_tag_counts.csv\n")
        f.write("- reports/procedural_docket_by_case.csv\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
