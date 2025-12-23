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


def ascii_safe(value: str) -> str:
    return value.encode("ascii", errors="ignore").decode("ascii")


def top_entries(rows: list[dict], key: str, top_n: int) -> list[dict]:
    def to_int(val: str) -> int:
        try:
            return int(float(val))
        except Exception:
            return 0

    return sorted(rows, key=lambda r: -to_int(r.get(key, "0")))[:top_n]


def summarize_timeline(timeline_rows: list[dict]) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    month_counts = Counter()
    day_counts = Counter()
    for row in timeline_rows:
        date_str = row.get("date", "")
        if not date_str:
            continue
        try:
            date_val = datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            continue
        month_counts[date_val.strftime("%Y-%m")] += 1
        day_counts[date_val.strftime("%Y-%m-%d")] += 1
    return month_counts.most_common(8), day_counts.most_common(8)


def build_outline(
    reports_dir: Path,
    out_path: Path,
    top_docs_per_issue: int,
    top_pairs: int,
    top_signals: int,
) -> None:
    issue_doc_rows = read_csv(reports_dir / "issue_doc_map.csv")
    co_rows = read_csv(reports_dir / "issue_cooccurrence.csv")
    timeline_rows = read_csv(reports_dir / "filing_timeline.csv")
    signal_rows = read_csv(reports_dir / "federal_signal_map.csv")

    issues = sorted({row["issue"] for row in issue_doc_rows if row.get("issue")})
    issue_groups = defaultdict(lambda: {"case": [], "research": []})
    for row in issue_doc_rows:
        issue = row.get("issue", "")
        corpus = row.get("corpus", "")
        if issue and corpus in ("case", "research"):
            issue_groups[issue][corpus].append(row)

    top_months, top_days = summarize_timeline(timeline_rows)

    signal_case = [row for row in signal_rows if row.get("corpus") == "case"]
    signal_research = [row for row in signal_rows if row.get("corpus") == "research"]
    signal_case = top_entries(signal_case, "signal_total", top_signals)
    signal_research = top_entries(signal_research, "signal_total", top_signals)

    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Document-Derived Argument Outline\n\n")
        f.write("This outline is built from the record and research text. It is not legal advice.\n\n")

        if top_months:
            f.write("## Activity Concentration (Case Filings)\n\n")
            f.write("Top months by filing volume:\n")
            for month, count in top_months:
                f.write(f"- {month}: {count} filings\n")
            f.write("\nTop individual filing days:\n")
            for day, count in top_days:
                f.write(f"- {day}: {count} filings\n")
            f.write("\n")

        f.write("## Issue Anchors in the Record (Top Documents)\n\n")
        for issue in issues:
            f.write(f"### {issue}\n\n")
            case_docs = top_entries(issue_groups[issue]["case"], "hit_count", top_docs_per_issue)
            research_docs = top_entries(
                issue_groups[issue]["research"], "hit_count", max(3, top_docs_per_issue - 2)
            )
            if case_docs:
                f.write("Case docs:\n")
                for row in case_docs:
                    date_val = row.get("date") or "undated"
                    source_pdf = ascii_safe(row.get("source_pdf", ""))
                    hits = row.get("hit_count", "0")
                    f.write(f"- {date_val}: {source_pdf} (hits: {hits})\n")
                f.write("\n")
            else:
                f.write("Case docs: no keyword hits found.\n\n")

            if research_docs:
                f.write("Research docs:\n")
                for row in research_docs:
                    date_val = row.get("date") or "undated"
                    source_pdf = ascii_safe(row.get("source_pdf", ""))
                    hits = row.get("hit_count", "0")
                    f.write(f"- {date_val}: {source_pdf} (hits: {hits})\n")
                f.write("\n")
            else:
                f.write("Research docs: no keyword hits found.\n\n")

        if co_rows:
            f.write("## Issue Co-occurrence Clusters (Top Pairs)\n\n")
            for row in top_entries(co_rows, "doc_count", top_pairs):
                a = row.get("issue_a", "")
                b = row.get("issue_b", "")
                count = row.get("doc_count", "0")
                f.write(f"- {a} + {b}: {count} documents\n")
            f.write("\n")

        if signal_case or signal_research:
            f.write("## Federal Signal Highlights (Keyword Density)\n\n")
            if signal_case:
                f.write("Case docs (top signal totals):\n")
                for row in signal_case:
                    source = ascii_safe(row.get("source_pdf", ""))
                    total = row.get("signal_total", "0")
                    f.write(f"- {source} (signal_total: {total})\n")
                f.write("\n")
            if signal_research:
                f.write("Research docs (top signal totals):\n")
                for row in signal_research:
                    source = ascii_safe(row.get("source_pdf", ""))
                    total = row.get("signal_total", "0")
                    f.write(f"- {source} (signal_total: {total})\n")
                f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a document-derived argument outline from issue mappings."
    )
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--output", default="reports/argument_outline.md")
    parser.add_argument("--top-docs", type=int, default=6)
    parser.add_argument("--top-pairs", type=int, default=15)
    parser.add_argument("--top-signals", type=int, default=10)
    args = parser.parse_args()

    build_outline(
        reports_dir=Path(args.reports),
        out_path=Path(args.output),
        top_docs_per_issue=args.top_docs,
        top_pairs=args.top_pairs,
        top_signals=args.top_signals,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
