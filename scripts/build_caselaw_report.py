import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path


REPORTER_PATTERN = re.compile(
    r"\b\d{1,4}\s+(S\.W\.3d|S\.W\.2d|S\.W\.|U\.S\.|F\.3d|F\.2d|F\. Supp\. 2d|F\. Supp\.|S\. Ct\.|L\. Ed\. 2d|L\. Ed\.)\s+\d+\b"
)
CASE_NAME_PATTERN = re.compile(r"\b[A-Z][A-Za-z.&'-]{2,}\s+v\.?\s+[A-Z][A-Za-z.&'-]{2,}\b")
IN_RE_PATTERN = re.compile(r"\bIn re\s+[A-Z][A-Za-z.&'-]{2,}\b")
EX_PARTE_PATTERN = re.compile(r"\bEx parte\s+[A-Z][A-Za-z.&'-]{2,}\b")


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


def ascii_safe(value: str) -> str:
    return value.encode("ascii", errors="ignore").decode("ascii")


def normalize_case_name(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip(" ,.;:)("))
    return text


def add_citation(stats: dict, key: str, source_pdf: str, corpus: str, issues: set, line: str, max_examples: int) -> None:
    if key not in stats:
        stats[key] = {
            "doc_set": set(),
            "doc_set_case": set(),
            "doc_set_research": set(),
            "hit_count": 0,
            "issue_counts": Counter(),
            "examples": [],
        }
    entry = stats[key]
    entry["hit_count"] += 1
    if source_pdf not in entry["doc_set"]:
        entry["doc_set"].add(source_pdf)
        if corpus == "case":
            entry["doc_set_case"].add(source_pdf)
        else:
            entry["doc_set_research"].add(source_pdf)
        for issue in issues:
            entry["issue_counts"][issue] += 1
    if len(entry["examples"]) < max_examples:
        snippet = ascii_safe(line)
        snippet = snippet[:200] + "..." if len(snippet) > 200 else snippet
        entry["examples"].append(f"{ascii_safe(source_pdf)} | {snippet}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a caselaw report with contexts.")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--case-text", default="extracted_text")
    parser.add_argument("--research-text", default="extracted_research")
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()

    reports_dir = Path(args.reports)
    doc_index = read_csv(reports_dir / "doc_index.csv")
    issue_doc_map = read_csv(reports_dir / "issue_doc_map.csv")

    meta_map = {row.get("source_txt", ""): row for row in doc_index}
    issue_map = defaultdict(set)
    for row in issue_doc_map:
        source = row.get("source_pdf", "")
        issue = row.get("issue", "")
        if source and issue:
            issue_map[source].add(issue)

    case_stats = {}
    reporter_stats = {}
    pair_stats = {}

    for root in [Path(args.case_text), Path(args.research_text)]:
        for txt_path in sorted(root.rglob("*.txt")):
            meta = meta_map.get(str(txt_path), {})
            source_pdf = meta.get("source_pdf", "")
            corpus = meta.get("corpus", "case")
            issues = issue_map.get(source_pdf, set())
            lines = load_text_lines(txt_path)
            if not lines:
                continue

            for line in lines:
                reporters = [m.group(0).strip(" ,.;:)(") for m in REPORTER_PATTERN.finditer(line)]
                case_names = [normalize_case_name(m.group(0)) for m in CASE_NAME_PATTERN.finditer(line)]
                case_names += [normalize_case_name(m.group(0)) for m in IN_RE_PATTERN.finditer(line)]
                case_names += [normalize_case_name(m.group(0)) for m in EX_PARTE_PATTERN.finditer(line)]

                for case_name in case_names:
                    add_citation(case_stats, case_name, source_pdf, corpus, issues, line, args.examples)
                for reporter in reporters:
                    add_citation(reporter_stats, reporter, source_pdf, corpus, issues, line, args.examples)
                if case_names and reporters:
                    for case_name in case_names:
                        for reporter in reporters:
                            key = f"{case_name} | {reporter}"
                            add_citation(pair_stats, key, source_pdf, corpus, issues, line, args.examples)

    def to_rows(stats: dict) -> list[dict]:
        rows = []
        for citation, meta in stats.items():
            issues_sorted = [k for k, _ in meta["issue_counts"].most_common(5)]
            rows.append(
                {
                    "citation": citation,
                    "doc_count": len(meta["doc_set"]),
                    "doc_count_case": len(meta["doc_set_case"]),
                    "doc_count_research": len(meta["doc_set_research"]),
                    "total_hits": meta["hit_count"],
                    "top_issues": ";".join(issues_sorted),
                    "sample_context": meta["examples"][0] if meta["examples"] else "",
                }
            )
        rows.sort(key=lambda r: (-int(r["doc_count"]), -int(r["total_hits"]), r["citation"]))
        return rows

    case_rows = to_rows(case_stats)
    reporter_rows = to_rows(reporter_stats)
    pair_rows = to_rows(pair_stats)

    write_csv(
        reports_dir / "caselaw_cases_all.csv",
        case_rows,
        ["citation", "doc_count", "doc_count_case", "doc_count_research", "total_hits", "top_issues", "sample_context"],
    )
    write_csv(
        reports_dir / "caselaw_reporters_all.csv",
        reporter_rows,
        ["citation", "doc_count", "doc_count_case", "doc_count_research", "total_hits", "top_issues", "sample_context"],
    )
    write_csv(
        reports_dir / "caselaw_case_reporter_pairs.csv",
        pair_rows,
        ["citation", "doc_count", "doc_count_case", "doc_count_research", "total_hits", "top_issues", "sample_context"],
    )

    # Case-by-issue mapping
    issue_case = defaultdict(Counter)
    for citation, meta in case_stats.items():
        for issue, count in meta["issue_counts"].items():
            issue_case[issue][citation] += count

    issue_rows = []
    for issue, counts in issue_case.items():
        for citation, count in counts.most_common(args.top):
            issue_rows.append({"issue": issue, "citation": citation, "doc_count": count})
    write_csv(
        reports_dir / "caselaw_by_issue.csv",
        issue_rows,
        ["issue", "citation", "doc_count"],
    )

    out_md = reports_dir / "caselaw_report.md"
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Caselaw Report (Document-Derived)\n\n")
        f.write("This report lists cited cases and how they are referenced in the corpus. It is not legal advice.\n\n")
        f.write(f"- Unique case names: {len(case_rows)}\n")
        f.write(f"- Unique reporter citations: {len(reporter_rows)}\n")
        f.write(f"- Case+reporter pairs: {len(pair_rows)}\n\n")

        f.write("## Top Case Names (by document count)\n\n")
        for row in case_rows[: args.top]:
            f.write(
                f"- {row['citation']} | docs: {row['doc_count']} | hits: {row['total_hits']} | issues: {row['top_issues']}\n"
            )
            if row["sample_context"]:
                f.write(f"  - {row['sample_context']}\n")
        f.write("\n")

        f.write("## Top Reporter Citations (by document count)\n\n")
        for row in reporter_rows[: args.top]:
            f.write(
                f"- {row['citation']} | docs: {row['doc_count']} | hits: {row['total_hits']} | issues: {row['top_issues']}\n"
            )
            if row["sample_context"]:
                f.write(f"  - {row['sample_context']}\n")
        f.write("\n")

        f.write("## Top Case + Reporter Pairs (by document count)\n\n")
        for row in pair_rows[: args.top]:
            f.write(
                f"- {row['citation']} | docs: {row['doc_count']} | hits: {row['total_hits']} | issues: {row['top_issues']}\n"
            )
            if row["sample_context"]:
                f.write(f"  - {row['sample_context']}\n")
        f.write("\n")

        f.write("Outputs:\n")
        f.write("- reports/caselaw_cases_all.csv\n")
        f.write("- reports/caselaw_reporters_all.csv\n")
        f.write("- reports/caselaw_case_reporter_pairs.csv\n")
        f.write("- reports/caselaw_by_issue.csv\n")
        f.write("- reports/caselaw_report.md\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
