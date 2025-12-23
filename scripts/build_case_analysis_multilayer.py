import argparse
import csv
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def to_int(value: str) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def parse_date(date_str: str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except Exception:
        return None


def ascii_safe(value: str) -> str:
    return value.encode("ascii", errors="ignore").decode("ascii")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            filtered = {field: row.get(field, "") for field in fieldnames}
            writer.writerow(filtered)


def build_analysis(
    reports_dir: Path,
    output_path: Path,
    timeline_out: Path,
    top_issue_count: int,
    top_issue_months: int,
    top_pairs: int,
    top_docs: int,
    min_year: int,
    max_year: int,
    skip_patterns: list[re.Pattern],
) -> None:
    issue_month_rows = read_csv(reports_dir / "issue_month_map.csv")
    issue_doc_rows = read_csv(reports_dir / "issue_doc_map.csv")
    timeline_rows = read_csv(reports_dir / "filing_timeline.csv")
    co_rows = read_csv(reports_dir / "issue_cooccurrence.csv")
    effort_stats = (reports_dir / "effort_stats_deep.md").read_text(
        encoding="utf-8", errors="ignore"
    )

    def should_skip(row: dict) -> bool:
        if not skip_patterns:
            return False
        path_str = f"{row.get('source_pdf', '')} {row.get('source_txt', '')}"
        return any(pat.search(path_str) for pat in skip_patterns)

    issue_month_rows = [
        row
        for row in issue_month_rows
        if row.get("corpus") == "case" and not should_skip(row)
    ]
    issue_doc_rows = [
        row
        for row in issue_doc_rows
        if row.get("corpus") == "case" and not should_skip(row)
    ]
    timeline_rows = [row for row in timeline_rows if not should_skip(row)]

    dates = []
    for row in timeline_rows:
        date_val = parse_date(row.get("date", ""))
        if date_val and min_year <= date_val.year <= max_year:
            dates.append(date_val)
    dates.sort()
    date_range = (dates[0], dates[-1]) if dates else (None, None)
    unique_dates = len({d.isoformat() for d in dates})

    month_filings = Counter()
    day_filings = Counter()
    for row in timeline_rows:
        date_val = parse_date(row.get("date", ""))
        if not date_val or not (min_year <= date_val.year <= max_year):
            continue
        month_filings[date_val.strftime("%Y-%m")] += 1
        day_filings[date_val.strftime("%Y-%m-%d")] += 1

    issue_docs = defaultdict(set)
    issue_hits = Counter()
    for row in issue_doc_rows:
        issue = row.get("issue", "")
        source = row.get("source_pdf", "")
        if not issue or not source:
            continue
        issue_docs[issue].add(source)
        issue_hits[issue] += to_int(row.get("hit_count", "0"))

    issue_doc_counts = {issue: len(sources) for issue, sources in issue_docs.items()}
    top_issues = [
        issue
        for issue, _ in sorted(
            issue_doc_counts.items(), key=lambda x: x[1], reverse=True
        )[:top_issue_count]
    ]

    issue_month_counts = defaultdict(lambda: Counter())
    for row in issue_month_rows:
        issue = row.get("issue", "")
        month = row.get("month", "")
        if not issue or not month:
            continue
        issue_month_counts[issue][month] += to_int(row.get("doc_count", "0"))

    issue_top_months = {}
    for issue in top_issues:
        months = issue_month_counts.get(issue, {})
        top = sorted(months.items(), key=lambda x: x[1], reverse=True)[:top_issue_months]
        issue_top_months[issue] = top

    monthly_issue_totals = Counter()
    for row in issue_month_rows:
        month = row.get("month", "")
        monthly_issue_totals[month] += to_int(row.get("doc_count", "0"))

    co_rows_sorted = sorted(co_rows, key=lambda r: -to_int(r.get("doc_count", "0")))

    issue_docs_sorted = defaultdict(list)
    for row in issue_doc_rows:
        issue = row.get("issue", "")
        if issue not in top_issues:
            continue
        issue_docs_sorted[issue].append(row)
    for issue in issue_docs_sorted:
        issue_docs_sorted[issue].sort(
            key=lambda r: -to_int(r.get("hit_count", "0"))
        )

    # Build monthly pivot for top issues
    months = sorted(month_filings.keys())
    timeline_rows_out = []
    for month in months:
        row = {"month": month, "filings": month_filings.get(month, 0)}
        row["issue_docs_total"] = monthly_issue_totals.get(month, 0)
        for issue in top_issues:
            row[issue] = issue_month_counts.get(issue, {}).get(month, 0)
        timeline_rows_out.append(row)

    fieldnames = ["month", "filings", "issue_docs_total"] + top_issues
    write_csv(timeline_out, timeline_rows_out, fieldnames)

    with output_path.open("w", encoding="utf-8") as f:
        f.write("# Multilayer Case Analysis (Document-Derived)\n\n")
        f.write(
            "This analysis maps issues over time using keyword hits and filing metadata. It is not legal advice.\n\n"
        )

        if skip_patterns:
            f.write("Exclusions:\n")
            for pat in skip_patterns:
                f.write(f"- skipped files matching: {pat.pattern}\n")
            f.write("\n")

        if date_range[0] and date_range[1]:
            f.write("## Coverage\n\n")
            f.write(f"- Filing date range (from filenames): {date_range[0]} to {date_range[1]}.\n")
            f.write(f"- Unique dated days: {unique_dates}.\n")
            f.write(f"- Timeline rows (within year bounds): {sum(month_filings.values())}.\n")
            f.write(f"- Year bounds applied: {min_year} to {max_year}.\n\n")

        f.write("## Layer 1: Filing Intensity Over Time\n\n")
        f.write("Top months by filing volume:\n")
        for month, count in month_filings.most_common(10):
            f.write(f"- {month}: {count} filings\n")
        f.write("\nTop individual filing days:\n")
        for day, count in day_filings.most_common(10):
            f.write(f"- {day}: {count} filings\n")
        f.write("\n")

        f.write("## Layer 2: Issue Volume (Document Counts)\n\n")
        for issue in top_issues:
            f.write(
                f"- {issue}: {issue_doc_counts.get(issue, 0)} docs, {issue_hits.get(issue, 0)} hits\n"
            )
        f.write("\n")

        f.write("## Layer 3: Issue Arcs by Month (Top Months)\n\n")
        for issue in top_issues:
            months = issue_top_months.get(issue, [])
            if not months:
                continue
            month_str = ", ".join([f"{m} ({c})" for m, c in months])
            f.write(f"- {issue}: {month_str}\n")
        f.write("\n")

        f.write("## Layer 4: Cross-Issue Clusters (Top Co-occurrences)\n\n")
        for row in co_rows_sorted[:top_pairs]:
            a = row.get("issue_a", "")
            b = row.get("issue_b", "")
            count = row.get("doc_count", "0")
            f.write(f"- {a} + {b}: {count} docs\n")
        f.write("\n")

        f.write("## Layer 5: Anchor Documents by Issue (Top Hits)\n\n")
        for issue in top_issues:
            f.write(f"### {issue}\n\n")
            rows = issue_docs_sorted.get(issue, [])[:top_docs]
            if not rows:
                f.write("No documents found for this issue.\n\n")
                continue
            for row in rows:
                date_val = row.get("date") or "undated"
                source = ascii_safe(row.get("source_pdf", ""))
                hits = row.get("hit_count", "0")
                f.write(f"- {date_val}: {source} (hits: {hits})\n")
            f.write("\n")

        f.write("## Layer 6: Time-Series CSV for Top Issues\n\n")
        f.write(f"- {timeline_out}\n\n")

        if effort_stats:
            f.write("## Reference Stats\n\n")
            f.write(effort_stats.strip() + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a multilayer case analysis mapping issues over time."
    )
    parser.add_argument("--reports", default="reports")
    parser.add_argument(
        "--output", default="reports/case_analysis_multilayer.md"
    )
    parser.add_argument(
        "--timeline-out", default="reports/issue_timeline_top_issues.csv"
    )
    parser.add_argument("--top-issues", type=int, default=12)
    parser.add_argument("--top-months", type=int, default=3)
    parser.add_argument("--top-pairs", type=int, default=20)
    parser.add_argument("--top-docs", type=int, default=5)
    parser.add_argument("--min-year", type=int, default=1990)
    parser.add_argument("--max-year", type=int, default=2030)
    parser.add_argument(
        "--skip-pattern",
        action="append",
        default=[],
        help="Regex pattern to skip files (case-insensitive). Can be repeated.",
    )
    args = parser.parse_args()

    skip_patterns = [re.compile(pat, re.IGNORECASE) for pat in args.skip_pattern]

    build_analysis(
        reports_dir=Path(args.reports),
        output_path=Path(args.output),
        timeline_out=Path(args.timeline_out),
        top_issue_count=args.top_issues,
        top_issue_months=args.top_months,
        top_pairs=args.top_pairs,
        top_docs=args.top_docs,
        min_year=args.min_year,
        max_year=args.max_year,
        skip_patterns=skip_patterns,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
