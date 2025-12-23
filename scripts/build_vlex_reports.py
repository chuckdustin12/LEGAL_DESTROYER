import argparse
import csv
import itertools
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


USC_PATTERN = re.compile(
    r"\b(?P<title>\d+)\s+U\.?S\.?C\.?\s*(?:§+)?\s*(?P<section>[\w\.\-]+)"
)
TEX_CODE_PATTERN = re.compile(
    r"\bTex\.?\s*(?P<code>Gov't|Govt|Fam\.|Family|Penal|Civ\.|Admin\.|Const\.)\s*Code\.?\s*(?:§+)?\s*(?P<section>[\w\.\-]+)"
)
TRCP_PATTERN = re.compile(r"\bTex\.?\s*R\.?\s*Civ\.?\s*P\.?\s*(?P<section>\d+[A-Za-z]?)")
TRCP_SHORT_PATTERN = re.compile(r"\bTRCP\s*(?P<section>\d+[A-Za-z]?)")
TAC_PATTERN = re.compile(r"\bTex\.?\s*Admin\.?\s*Code\s*(?:§+)?\s*(?P<section>[\w\.\-]+)")

REPORTER_PATTERN = re.compile(
    r"\b\d{1,4}\s+(S\.W\.3d|S\.W\.2d|S\.W\.|U\.S\.|F\.3d|F\.2d|F\. Supp\. 2d|F\. Supp\.|S\. Ct\.|L\. Ed\. 2d|L\. Ed\.)\s+\d+\b"
)
CASE_NAME_PATTERN = re.compile(r"\b[A-Z][A-Za-z.&'-]{2,}\s+v\.?\s+[A-Z][A-Za-z.&'-]{2,}\b")
IN_RE_PATTERN = re.compile(r"\bIn re\s+[A-Z][A-Za-z.&'-]{2,}\b")
EX_PARTE_PATTERN = re.compile(r"\bEx parte\s+[A-Z][A-Za-z.&'-]{2,}\b")

WORD_PATTERN = re.compile(r"\b[a-zA-Z]{3,}\b")

STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "this",
    "from",
    "were",
    "have",
    "has",
    "had",
    "not",
    "are",
    "was",
    "but",
    "you",
    "your",
    "they",
    "their",
    "them",
    "his",
    "her",
    "she",
    "him",
    "our",
    "out",
    "into",
    "about",
    "over",
    "under",
    "between",
    "after",
    "before",
    "while",
    "where",
    "when",
    "what",
    "which",
    "there",
    "here",
    "because",
    "also",
    "case",
    "court",
    "order",
    "orders",
    "motion",
    "notice",
    "hearing",
    "filed",
    "file",
    "document",
    "records",
    "record",
    "evidence",
    "law",
    "legal",
    "texas",
    "state",
    "federal",
    "judge",
    "trial",
    "district",
    "plaintiff",
    "defendant",
    "respondent",
    "petitioner",
    "relator",
    "appellant",
    "appellee",
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


def load_text_body(path: Path) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    return strip_headers(content)


def normalize_us_code(match: re.Match) -> str:
    title = match.group("title")
    section = match.group("section").strip(" ,.;:)(")
    return f"{title} U.S.C. {section}"


def normalize_tex_code(match: re.Match) -> str:
    code = match.group("code")
    section = match.group("section").strip(" ,.;:)(")
    code = code.replace("Govt", "Gov't")
    code = code.replace("Family", "Fam.")
    return f"Tex. {code} Code {section}"


def normalize_trcp(match: re.Match) -> str:
    section = match.group("section").strip(" ,.;:)(")
    return f"Tex. R. Civ. P. {section}"


def normalize_tac(match: re.Match) -> str:
    section = match.group("section").strip(" ,.;:)(")
    return f"Tex. Admin. Code {section}"


def ascii_safe(value: str) -> str:
    return value.encode("ascii", errors="ignore").decode("ascii")


def add_citation(stats: dict, key: str, source_pdf: str, date_val: str, line: str, max_examples: int) -> None:
    if key not in stats:
        stats[key] = {"count": 0, "docs": set(), "examples": []}
    entry = stats[key]
    entry["count"] += 1
    entry["docs"].add(source_pdf)
    if len(entry["examples"]) < max_examples:
        snippet = ascii_safe(line)
        snippet = snippet[:200] + "..." if len(snippet) > 200 else snippet
        entry["examples"].append(f"{date_val} | {ascii_safe(source_pdf)} | {snippet}")


def parse_date(date_str: str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Build reports for vLex research corpus.")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--research-text", default="extracted_research")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--vlex-pattern", default="vlex")
    args = parser.parse_args()

    reports_dir = Path(args.reports)
    doc_index = read_csv(reports_dir / "doc_index.csv")
    issue_doc_map = read_csv(reports_dir / "issue_doc_map.csv")

    vlex_pattern = args.vlex_pattern.lower()
    vlex_docs = [
        row
        for row in doc_index
        if row.get("corpus") == "research"
        and vlex_pattern in (row.get("source_pdf", "").lower())
    ]
    vlex_pdf_set = {row.get("source_pdf", "") for row in vlex_docs if row.get("source_pdf")}
    vlex_txt_set = {row.get("source_txt", "") for row in vlex_docs if row.get("source_txt")}

    total_docs = len(vlex_docs)
    total_pages = sum(int(row.get("pages") or 0) for row in vlex_docs)
    total_words = sum(int(row.get("word_count") or 0) for row in vlex_docs)

    dates = [parse_date(row.get("date", "")) for row in vlex_docs if row.get("date")]
    dates = [d for d in dates if d]
    date_range = (min(dates), max(dates)) if dates else (None, None)

    write_csv(
        reports_dir / "vlex_doc_index.csv",
        vlex_docs,
        [
            "source_pdf",
            "source_txt",
            "source_exists",
            "date",
            "case_numbers",
            "doc_tags",
            "pages",
            "word_count",
            "status",
        ],
    )

    # Issue radar and co-occurrence for vLex
    issue_doc_counts = defaultdict(set)
    issue_hit_counts = Counter()
    doc_issues = defaultdict(set)
    for row in issue_doc_map:
        if row.get("corpus") != "research":
            continue
        source = row.get("source_pdf", "")
        if source not in vlex_pdf_set:
            continue
        issue = row.get("issue", "")
        if not issue:
            continue
        issue_doc_counts[issue].add(source)
        issue_hit_counts[issue] += int(float(row.get("hit_count", 0) or 0))
        doc_issues[source].add(issue)

    issue_radar_rows = []
    for issue, docs in issue_doc_counts.items():
        issue_radar_rows.append(
            {
                "issue": issue,
                "docs_with_hits": len(docs),
                "total_hits": issue_hit_counts.get(issue, 0),
                "pct_docs": round(len(docs) / max(total_docs, 1), 4),
            }
        )
    issue_radar_rows.sort(key=lambda r: (-r["docs_with_hits"], r["issue"]))
    write_csv(
        reports_dir / "vlex_issue_radar.csv",
        issue_radar_rows,
        ["issue", "docs_with_hits", "total_hits", "pct_docs"],
    )

    co_counts = Counter()
    for issues in doc_issues.values():
        for a, b in itertools.combinations(sorted(issues), 2):
            co_counts[(a, b)] += 1
    co_rows = [
        {"issue_a": a, "issue_b": b, "doc_count": count}
        for (a, b), count in co_counts.items()
    ]
    co_rows.sort(key=lambda r: -r["doc_count"])
    write_csv(
        reports_dir / "vlex_issue_cooccurrence.csv",
        co_rows,
        ["issue_a", "issue_b", "doc_count"],
    )

    # Citations and top terms for vLex
    statute_stats = {}
    caselaw_stats = {}
    case_name_stats = {}
    term_counts = Counter()

    meta_map = {row.get("source_txt", ""): row for row in vlex_docs}
    for txt_path in sorted(Path(args.research_text).rglob("*.txt")):
        if str(txt_path) not in vlex_txt_set:
            continue
        meta = meta_map.get(str(txt_path), {})
        source_pdf = meta.get("source_pdf", "")
        date_val = meta.get("date", "undated")

        body = load_text_body(txt_path)
        if body:
            for token in WORD_PATTERN.findall(body.lower()):
                if token in STOPWORDS:
                    continue
                term_counts[token] += 1

        for line in load_text_lines(txt_path):
            for match in USC_PATTERN.finditer(line):
                add_citation(statute_stats, normalize_us_code(match), source_pdf, date_val, line, args.examples)
            for match in TEX_CODE_PATTERN.finditer(line):
                add_citation(statute_stats, normalize_tex_code(match), source_pdf, date_val, line, args.examples)
            for match in TRCP_PATTERN.finditer(line):
                add_citation(statute_stats, normalize_trcp(match), source_pdf, date_val, line, args.examples)
            for match in TRCP_SHORT_PATTERN.finditer(line):
                add_citation(statute_stats, normalize_trcp(match), source_pdf, date_val, line, args.examples)
            for match in TAC_PATTERN.finditer(line):
                add_citation(statute_stats, normalize_tac(match), source_pdf, date_val, line, args.examples)

            for match in REPORTER_PATTERN.finditer(line):
                add_citation(caselaw_stats, match.group(0).strip(" ,.;:)("), source_pdf, date_val, line, args.examples)
            for match in CASE_NAME_PATTERN.finditer(line):
                add_citation(case_name_stats, match.group(0).strip(" ,.;:)("), source_pdf, date_val, line, args.examples)
            for match in IN_RE_PATTERN.finditer(line):
                add_citation(case_name_stats, match.group(0).strip(" ,.;:)("), source_pdf, date_val, line, args.examples)
            for match in EX_PARTE_PATTERN.finditer(line):
                add_citation(case_name_stats, match.group(0).strip(" ,.;:)("), source_pdf, date_val, line, args.examples)

    def sorted_stats(stats: dict) -> list[tuple[str, dict]]:
        return sorted(stats.items(), key=lambda x: (-len(x[1]["docs"]), -x[1]["count"]))

    statute_sorted = sorted_stats(statute_stats)[: args.top]
    caselaw_sorted = sorted_stats(caselaw_stats)[: args.top]
    case_name_sorted = sorted_stats(case_name_stats)[: args.top]

    write_csv(
        reports_dir / "vlex_citation_pack_statutes.csv",
        [
            {
                "citation": citation,
                "doc_count": len(meta["docs"]),
                "total_hits": meta["count"],
                "sample_source": meta["examples"][0] if meta["examples"] else "",
            }
            for citation, meta in statute_sorted
        ],
        ["citation", "doc_count", "total_hits", "sample_source"],
    )
    write_csv(
        reports_dir / "vlex_citation_pack_caselaw.csv",
        [
            {
                "citation": citation,
                "doc_count": len(meta["docs"]),
                "total_hits": meta["count"],
                "sample_source": meta["examples"][0] if meta["examples"] else "",
            }
            for citation, meta in caselaw_sorted
        ],
        ["citation", "doc_count", "total_hits", "sample_source"],
    )
    write_csv(
        reports_dir / "vlex_citation_pack_case_names.csv",
        [
            {
                "citation": citation,
                "doc_count": len(meta["docs"]),
                "total_hits": meta["count"],
                "sample_source": meta["examples"][0] if meta["examples"] else "",
            }
            for citation, meta in case_name_sorted
        ],
        ["citation", "doc_count", "total_hits", "sample_source"],
    )

    top_terms = term_counts.most_common(args.top)
    write_csv(
        reports_dir / "vlex_top_terms.csv",
        [{"term": term, "count": count} for term, count in top_terms],
        ["term", "count"],
    )

    out_md = reports_dir / "vlex_summary.md"
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# vLex Research Summary (Document-Derived)\n\n")
        f.write("This report summarizes vLex research files only. It is not legal advice.\n\n")
        f.write(f"- vLex docs found (by filename match): {total_docs}\n")
        f.write(f"- Total pages (sum of metadata): {total_pages}\n")
        f.write(f"- Total words (sum of metadata): {total_words}\n")
        if date_range[0] and date_range[1]:
            f.write(f"- Date range (from filenames): {date_range[0]} to {date_range[1]}\n")
        f.write("\nTop issues (by doc count):\n")
        for row in issue_radar_rows[:10]:
            f.write(
                f"- {row['issue']}: {row['docs_with_hits']} docs, {row['total_hits']} hits\n"
            )
        f.write("\nTop statutes (by doc count):\n")
        for citation, meta in statute_sorted[:10]:
            f.write(f"- {citation}: {len(meta['docs'])} docs\n")
        f.write("\nTop caselaw reporter citations (by doc count):\n")
        for citation, meta in caselaw_sorted[:10]:
            f.write(f"- {citation}: {len(meta['docs'])} docs\n")
        f.write("\nTop case names (by doc count):\n")
        for citation, meta in case_name_sorted[:10]:
            f.write(f"- {citation}: {len(meta['docs'])} docs\n")
        f.write("\nTop terms:\n")
        for term, count in top_terms[:10]:
            f.write(f"- {term}: {count}\n")
        f.write("\nOutputs:\n")
        f.write("- reports/vlex_doc_index.csv\n")
        f.write("- reports/vlex_issue_radar.csv\n")
        f.write("- reports/vlex_issue_cooccurrence.csv\n")
        f.write("- reports/vlex_citation_pack_statutes.csv\n")
        f.write("- reports/vlex_citation_pack_caselaw.csv\n")
        f.write("- reports/vlex_citation_pack_case_names.csv\n")
        f.write("- reports/vlex_top_terms.csv\n")
        f.write("- reports/vlex_summary.md\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
