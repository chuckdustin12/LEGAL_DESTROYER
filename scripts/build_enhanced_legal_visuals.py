import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


DATE_PATTERNS = [
    re.compile(
        r"(?P<month>\d{1,2})[./_-](?P<day>\d{1,2})[./_-](?P<year>\d{2,4})"
    ),
    re.compile(
        r"(?P<year>\d{4})[./_-](?P<month>\d{1,2})[./_-](?P<day>\d{1,2})"
    ),
]

MORGAN_PATTERN = re.compile(
    r"\bmorgan\s+michelle\s+myers\b|\bmorgan\s+m\.\s+myers\b|\bmorgan\s+myers\b",
    re.IGNORECASE,
)

REPORTER_PATTERN = re.compile(
    r"\b\d{1,4}\s+(S\.W\.3d|S\.W\.2d|S\.W\.|U\.S\.|F\.3d|F\.2d|F\. Supp\. 2d|F\. Supp\.|S\. Ct\.|L\. Ed\. 2d|L\. Ed\.)\s+\d+\b"
)

CASE_NAME_PATTERN = re.compile(r"\b[A-Z][A-Za-z.&'-]{2,}\s+v\.?\s+[A-Z][A-Za-z.&'-]{2,}\b")
IN_RE_PATTERN = re.compile(r"\bIn re\s+[A-Z][A-Za-z.&'-]{2,}\b")
EX_PARTE_PATTERN = re.compile(r"\bEx parte\s+[A-Z][A-Za-z.&'-]{2,}\b")

USC_PATTERN = re.compile(
    r"\b(?P<title>\d+)\s+U\.?S\.?C\.?\s*(?:§+)?\s*(?P<section>[\w\.\-]+)"
)
TEX_CODE_PATTERN = re.compile(
    r"\bTex\.?\s*(?P<code>Gov't|Govt|Fam\.|Family|Penal|Civ\.|Admin\.|Const\.)\s*Code\.?\s*(?:§+)?\s*(?P<section>[\w\.\-]+)"
)
TRCP_PATTERN = re.compile(r"\bTex\.?\s*R\.?\s*Civ\.?\s*P\.?\s*(?P<section>\d+[A-Za-z]?)")
TRCP_SHORT_PATTERN = re.compile(r"\bTRCP\s*(?P<section>\d+[A-Za-z]?)")
TAC_PATTERN = re.compile(
    r"\bTex\.?\s*Admin\.?\s*Code\s*(?:§+)?\s*(?P<section>[\w\.\-]+)"
)

PROCEDURAL_ISSUES = [
    "due_process",
    "notice",
    "hearing",
    "recusal",
    "jurisdiction",
    "associate_judge",
    "ex_parte",
    "removal_remand",
    "discovery",
    "sanctions_contempt",
    "temporary_orders",
]


def iter_text_files(root: Path) -> list[Path]:
    return sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".txt"],
        key=lambda p: str(p).lower(),
    )


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


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


def normalize_statute(match: re.Match) -> str:
    text = match.group(0)
    text = text.replace("§", " ")
    text = re.sub(r"\s+", " ", text).strip(" ,.;:)(")
    return text


def normalize_us_code(match: re.Match) -> str:
    title = match.group("title")
    section = match.group("section")
    section = section.strip(" ,.;:)(")
    return f"{title} U.S.C. {section}"


def normalize_tex_code(match: re.Match) -> str:
    code = match.group("code")
    section = match.group("section")
    section = section.strip(" ,.;:)(")
    code = code.replace("Govt", "Gov't")
    code = code.replace("Family", "Fam.")
    return f"Tex. {code} Code {section}"


def normalize_trcp(match: re.Match) -> str:
    section = match.group("section")
    section = section.strip(" ,.;:)(")
    return f"Tex. R. Civ. P. {section}"


def normalize_tac(match: re.Match) -> str:
    section = match.group("section")
    section = section.strip(" ,.;:)(")
    return f"Tex. Admin. Code {section}"


def to_int(value: str) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            filtered = {field: row.get(field, "") for field in fieldnames}
            writer.writerow(filtered)


def save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def bar_chart(labels: list[str], values: list[int], title: str, out_path: Path) -> None:
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels[::-1], values[::-1], color="#264653")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Count")
    save_plot(out_path)


def line_chart(x_values: list[str], y_values: list[int], title: str, out_path: Path) -> None:
    if not x_values:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_values, y_values, color="#1f6f78", linewidth=2)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel("Count")
    ax.set_xlabel("Month")
    ax.tick_params(axis="x", rotation=45)
    save_plot(out_path)


def heatmap(matrix: np.ndarray, x_labels: list[str], y_labels: list[str], title: str, out_path: Path) -> None:
    if matrix.size == 0:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrBr")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_title(title, fontsize=13)
    fig.colorbar(im, ax=ax, label="Documents")
    save_plot(out_path)


def issue_citation_network(
    edges: list[tuple[str, str, int]],
    title: str,
    out_path: Path,
) -> None:
    if not edges:
        return
    graph = nx.Graph()
    for issue, citation, count in edges:
        graph.add_edge(issue, citation, weight=count)

    weights = [graph[u][v]["weight"] for u, v in graph.edges]
    max_weight = max(weights) if weights else 1
    widths = [1 + (w / max_weight) * 3 for w in weights]

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(graph, seed=42, k=0.7)
    nx.draw_networkx_nodes(graph, pos, node_size=420, node_color="#2a9d8f", alpha=0.9, ax=ax)
    nx.draw_networkx_edges(graph, pos, width=widths, alpha=0.4, edge_color="#264653", ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=8, font_color="#1b1b1b", ax=ax)
    ax.set_title(title, fontsize=13)
    ax.axis("off")
    save_plot(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build enhanced visuals for citations, allegations, and procedural issues."
    )
    parser.add_argument("--case-text", default="extracted_text")
    parser.add_argument("--case-pdf", default="CASE DOCS")
    parser.add_argument("--research-text", default="extracted_research")
    parser.add_argument("--research-pdf", default="RESEARCH")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    reports_dir = Path(args.reports)
    visuals_dir = reports_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    issue_doc_rows = read_csv(reports_dir / "issue_doc_map.csv")
    issue_month_rows = read_csv(reports_dir / "issue_month_map.csv")

    issue_map_case = defaultdict(set)
    for row in issue_doc_rows:
        if row.get("corpus") != "case":
            continue
        issue = row.get("issue", "")
        source = row.get("source_pdf", "")
        if issue and source:
            issue_map_case[source].add(issue)

    statute_counts = Counter()
    statute_counts_case = Counter()
    statute_counts_research = Counter()
    reporter_counts = Counter()
    case_name_counts = Counter()

    doc_statutes = defaultdict(set)
    doc_reporters = defaultdict(set)
    doc_case_names = defaultdict(set)
    morgan_docs = set()

    corpora = [
        ("case", Path(args.case_text), Path(args.case_pdf)),
        ("research", Path(args.research_text), Path(args.research_pdf)),
    ]

    for corpus, text_root, pdf_root in corpora:
        for txt_path in iter_text_files(text_root):
            rel = txt_path.relative_to(text_root)
            source_pdf = str(pdf_root / rel.with_suffix(".pdf"))
            body = load_text_body(txt_path)
            if not body:
                continue

            if MORGAN_PATTERN.search(body):
                morgan_docs.add(source_pdf)

            statutes = set()
            for match in USC_PATTERN.finditer(body):
                statutes.add(normalize_us_code(match))
            for match in TEX_CODE_PATTERN.finditer(body):
                statutes.add(normalize_tex_code(match))
            for match in TRCP_PATTERN.finditer(body):
                statutes.add(normalize_trcp(match))
            for match in TRCP_SHORT_PATTERN.finditer(body):
                statutes.add(normalize_trcp(match))
            for match in TAC_PATTERN.finditer(body):
                statutes.add(normalize_tac(match))

            reporters = set(m.group(0).strip(" ,.;:)(") for m in REPORTER_PATTERN.finditer(body))
            case_names = set(m.group(0).strip(" ,.;:)(") for m in CASE_NAME_PATTERN.finditer(body))
            case_names |= set(m.group(0).strip(" ,.;:)(") for m in IN_RE_PATTERN.finditer(body))
            case_names |= set(m.group(0).strip(" ,.;:)(") for m in EX_PARTE_PATTERN.finditer(body))

            for stat in statutes:
                statute_counts[stat] += 1
                if corpus == "case":
                    statute_counts_case[stat] += 1
                else:
                    statute_counts_research[stat] += 1
                doc_statutes[source_pdf].add(stat)

            for rep in reporters:
                reporter_counts[rep] += 1
                doc_reporters[source_pdf].add(rep)

            for name in case_names:
                case_name_counts[name] += 1
                doc_case_names[source_pdf].add(name)

    # Statute and caselaw outputs
    statute_rows = [
        {"citation": citation, "count": count}
        for citation, count in statute_counts.most_common()
    ]
    write_csv(reports_dir / "statute_citations.csv", statute_rows, ["citation", "count"])

    reporter_rows = [
        {"citation": citation, "count": count}
        for citation, count in reporter_counts.most_common()
    ]
    write_csv(reports_dir / "caselaw_reporter_citations.csv", reporter_rows, ["citation", "count"])

    case_name_rows = [
        {"citation": citation, "count": count}
        for citation, count in case_name_counts.most_common()
    ]
    write_csv(reports_dir / "caselaw_case_names.csv", case_name_rows, ["citation", "count"])

    # Morgan issue mapping
    morgan_issue_hits = Counter()
    morgan_issue_docs = defaultdict(set)
    for row in issue_doc_rows:
        if row.get("corpus") != "case":
            continue
        source = row.get("source_pdf", "")
        if source not in morgan_docs:
            continue
        issue = row.get("issue", "")
        morgan_issue_hits[issue] += to_int(row.get("hit_count", "0"))
        morgan_issue_docs[issue].add(source)

    morgan_issue_rows = [
        {
            "issue": issue,
            "docs_with_hits": len(morgan_issue_docs.get(issue, set())),
            "total_hits": count,
        }
        for issue, count in morgan_issue_hits.most_common()
    ]
    write_csv(
        reports_dir / "morgan_issue_map.csv",
        morgan_issue_rows,
        ["issue", "docs_with_hits", "total_hits"],
    )

    morgan_docs_rows = [{"source_pdf": doc} for doc in sorted(morgan_docs)]
    write_csv(reports_dir / "morgan_docs.csv", morgan_docs_rows, ["source_pdf"])

    # Procedural issues summary (from issue maps)
    procedural_hits = Counter()
    procedural_docs = defaultdict(set)
    for row in issue_doc_rows:
        if row.get("corpus") != "case":
            continue
        issue = row.get("issue", "")
        if issue not in PROCEDURAL_ISSUES:
            continue
        source = row.get("source_pdf", "")
        procedural_hits[issue] += to_int(row.get("hit_count", "0"))
        if source:
            procedural_docs[issue].add(source)

    procedural_rows = [
        {
            "issue": issue,
            "docs_with_hits": len(procedural_docs.get(issue, set())),
            "total_hits": procedural_hits.get(issue, 0),
        }
        for issue in PROCEDURAL_ISSUES
    ]
    write_csv(
        reports_dir / "procedural_violations_summary.csv",
        procedural_rows,
        ["issue", "docs_with_hits", "total_hits"],
    )

    # Procedural timeline by month
    procedural_month = Counter()
    for row in issue_month_rows:
        if row.get("corpus") != "case":
            continue
        issue = row.get("issue", "")
        if issue not in PROCEDURAL_ISSUES:
            continue
        month = row.get("month", "")
        if month:
            procedural_month[month] += to_int(row.get("doc_count", "0"))

    procedural_month_rows = [
        {"month": month, "doc_count": count}
        for month, count in sorted(procedural_month.items())
    ]
    write_csv(
        reports_dir / "procedural_violations_timeline.csv",
        procedural_month_rows,
        ["month", "doc_count"],
    )

    # Issue-statute co-occurrence (case docs)
    issue_statute_counts = Counter()
    for source, issues in issue_map_case.items():
        statutes = doc_statutes.get(source, set())
        for issue in issues:
            for stat in statutes:
                issue_statute_counts[(issue, stat)] += 1

    issue_statute_rows = [
        {"issue": issue, "statute": stat, "doc_count": count}
        for (issue, stat), count in issue_statute_counts.items()
    ]
    issue_statute_rows.sort(key=lambda r: -r["doc_count"])
    write_csv(
        reports_dir / "issue_statute_cooccurrence.csv",
        issue_statute_rows,
        ["issue", "statute", "doc_count"],
    )

    # Issue-caselaw co-occurrence (case docs, reporter citations)
    issue_caselaw_counts = Counter()
    for source, issues in issue_map_case.items():
        reporters = doc_reporters.get(source, set())
        for issue in issues:
            for rep in reporters:
                issue_caselaw_counts[(issue, rep)] += 1

    issue_caselaw_rows = [
        {"issue": issue, "citation": rep, "doc_count": count}
        for (issue, rep), count in issue_caselaw_counts.items()
    ]
    issue_caselaw_rows.sort(key=lambda r: -r["doc_count"])
    write_csv(
        reports_dir / "issue_caselaw_cooccurrence.csv",
        issue_caselaw_rows,
        ["issue", "citation", "doc_count"],
    )

    # Visuals
    top_statutes = statute_counts.most_common(args.top_n)
    bar_chart(
        [s for s, _ in top_statutes],
        [c for _, c in top_statutes],
        "Statute Citations (Top)",
        visuals_dir / "statute_citations_top.png",
    )

    top_reporters = reporter_counts.most_common(args.top_n)
    bar_chart(
        [s for s, _ in top_reporters],
        [c for _, c in top_reporters],
        "Caselaw Reporter Citations (Top)",
        visuals_dir / "caselaw_citations_top.png",
    )

    top_case_names = case_name_counts.most_common(args.top_n)
    bar_chart(
        [s for s, _ in top_case_names],
        [c for _, c in top_case_names],
        "Case Name Citations (Top)",
        visuals_dir / "caselaw_case_names_top.png",
    )

    if morgan_issue_rows:
        bar_chart(
            [r["issue"] for r in morgan_issue_rows[:args.top_n]],
            [r["docs_with_hits"] for r in morgan_issue_rows[:args.top_n]],
            "Issue Mentions in Morgan-Referenced Docs (Top)",
            visuals_dir / "morgan_issue_map.png",
        )

    procedural_sorted = sorted(
        procedural_rows, key=lambda r: r["docs_with_hits"], reverse=True
    )
    bar_chart(
        [r["issue"] for r in procedural_sorted],
        [r["docs_with_hits"] for r in procedural_sorted],
        "Procedural Issue Mentions (Docs with Hits)",
        visuals_dir / "procedural_violations_categories.png",
    )

    if procedural_month_rows:
        line_chart(
            [row["month"] for row in procedural_month_rows],
            [row["doc_count"] for row in procedural_month_rows],
            "Procedural Issue Mentions Over Time",
            visuals_dir / "procedural_violations_timeline.png",
        )

    # Issue-statute heatmap (top issues + top statutes)
    issue_doc_counts = Counter()
    for row in issue_doc_rows:
        if row.get("corpus") != "case":
            continue
        issue = row.get("issue", "")
        if issue:
            issue_doc_counts[issue] += 1
    top_issues = [issue for issue, _ in issue_doc_counts.most_common(12)]
    top_statute_labels = [stat for stat, _ in top_statutes[:12]]
    if top_issues and top_statute_labels:
        matrix = np.zeros((len(top_issues), len(top_statute_labels)))
        for i, issue in enumerate(top_issues):
            for j, stat in enumerate(top_statute_labels):
                matrix[i, j] = issue_statute_counts.get((issue, stat), 0)
        heatmap(
            matrix,
            top_statute_labels,
            top_issues,
            "Issue vs Statute Co-occurrence (Top)",
            visuals_dir / "issue_statute_heatmap.png",
        )

    # Issue-caselaw network (top edges)
    top_edges = [
        (row["issue"], row["citation"], row["doc_count"])
        for row in issue_caselaw_rows[:30]
    ]
    issue_citation_network(
        top_edges,
        "Issue vs Caselaw Co-occurrence (Top)",
        visuals_dir / "issue_caselaw_network.png",
    )

    summary_path = reports_dir / "enhanced_visuals_summary.md"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("# Enhanced Visuals Summary (Document-Derived)\n\n")
        f.write("These outputs are keyword- and citation-based indexes, not legal advice.\n\n")
        f.write(f"- Morgan-referenced docs: {len(morgan_docs)}\n")
        f.write(f"- Statute citations found: {len(statute_counts)} unique\n")
        f.write(f"- Caselaw reporter citations found: {len(reporter_counts)} unique\n")
        f.write(f"- Case name citations found: {len(case_name_counts)} unique\n\n")
        f.write("Outputs:\n")
        f.write("- reports/statute_citations.csv\n")
        f.write("- reports/caselaw_reporter_citations.csv\n")
        f.write("- reports/caselaw_case_names.csv\n")
        f.write("- reports/morgan_issue_map.csv\n")
        f.write("- reports/procedural_violations_summary.csv\n")
        f.write("- reports/procedural_violations_timeline.csv\n")
        f.write("- reports/issue_statute_cooccurrence.csv\n")
        f.write("- reports/issue_caselaw_cooccurrence.csv\n")
        f.write("- reports/morgan_docs.csv\n")
        f.write("\nVisuals:\n")
        f.write("- reports/visuals/statute_citations_top.png\n")
        f.write("- reports/visuals/caselaw_citations_top.png\n")
        f.write("- reports/visuals/caselaw_case_names_top.png\n")
        f.write("- reports/visuals/morgan_issue_map.png\n")
        f.write("- reports/visuals/procedural_violations_categories.png\n")
        f.write("- reports/visuals/procedural_violations_timeline.png\n")
        f.write("- reports/visuals/issue_statute_heatmap.png\n")
        f.write("- reports/visuals/issue_caselaw_network.png\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
