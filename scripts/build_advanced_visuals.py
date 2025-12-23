import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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

PROCEDURAL_ISSUES = {
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


def to_int(value: str) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def parse_month(date_str: str) -> str:
    if not date_str:
        return ""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%Y-%m")
    except Exception:
        return ""


def stacked_area(months: list[str], series: dict[str, list[int]], title: str, out_path: Path) -> None:
    if not months or not series:
        return
    labels = list(series.keys())
    data = np.array([series[label] for label in labels])
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(months, data, labels=labels, alpha=0.8)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Month")
    ax.set_ylabel("Doc counts")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    save_plot(out_path)


def stacked_area_pct(months: list[str], series: dict[str, list[int]], title: str, out_path: Path) -> None:
    if not months or not series:
        return
    labels = list(series.keys())
    data = np.array([series[label] for label in labels], dtype=float)
    totals = data.sum(axis=0)
    totals[totals == 0] = 1.0
    data = data / totals
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(months, data, labels=labels, alpha=0.85)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Month")
    ax.set_ylabel("Share of issue mentions")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    save_plot(out_path)


def heatmap(matrix: np.ndarray, x_labels: list[str], y_labels: list[str], title: str, out_path: Path) -> None:
    if matrix.size == 0:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_title(title, fontsize=13)
    fig.colorbar(im, ax=ax, label="Documents")
    save_plot(out_path)


def grouped_bar(labels: list[str], series: list[list[float]], series_labels: list[str], title: str, out_path: Path) -> None:
    if not labels:
        return
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, data in enumerate(series):
        ax.bar(x + idx * width - (width / 2), data, width, label=series_labels[idx])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=8)
    save_plot(out_path)


def pareto_chart(labels: list[str], counts: list[int], title: str, out_path: Path) -> None:
    if not labels:
        return
    cumulative = np.cumsum(counts) / max(sum(counts), 1)
    x = np.arange(len(labels))

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(x, counts, color="#264653")
    ax1.set_ylabel("Count")
    ax1.set_title(title, fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(x, cumulative, color="#e76f51", linewidth=2)
    ax2.set_ylabel("Cumulative share")
    ax2.set_ylim(0, 1.05)
    save_plot(out_path)


def line_chart(months: list[str], series: dict[str, list[int]], title: str, out_path: Path) -> None:
    if not months:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, values in series.items():
        ax.plot(months, values, label=label, linewidth=2)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Month")
    ax.set_ylabel("Citation count")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8, ncol=2)
    save_plot(out_path)


def iter_text_files(root: Path) -> list[Path]:
    return sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".txt"],
        key=lambda p: str(p).lower(),
    )


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Build advanced visuals from case analysis data.")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--case-text", default="extracted_text")
    parser.add_argument("--top-issues", type=int, default=10)
    parser.add_argument("--top-cases", type=int, default=8)
    parser.add_argument("--top-citations", type=int, default=15)
    args = parser.parse_args()

    reports_dir = Path(args.reports)
    visuals_dir = reports_dir / "visuals_advanced"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    issue_month_rows = read_csv(reports_dir / "issue_month_map.csv")
    issue_doc_rows = read_csv(reports_dir / "issue_doc_map.csv")
    issue_radar_case = read_csv(reports_dir / "issue_radar_case.csv")
    issue_co_rows = read_csv(reports_dir / "issue_cooccurrence.csv")
    doc_index = read_csv(reports_dir / "doc_index.csv")
    morgan_issue_rows = read_csv(reports_dir / "morgan_issue_map.csv")
    morgan_docs_rows = read_csv(reports_dir / "morgan_docs.csv")
    statute_rows = read_csv(reports_dir / "statute_citations.csv")
    caselaw_rows = read_csv(reports_dir / "caselaw_reporter_citations.csv")

    # Month list and issue series (case docs only)
    month_issue = defaultdict(lambda: Counter())
    for row in issue_month_rows:
        if row.get("corpus") != "case":
            continue
        issue = row.get("issue", "")
        month = row.get("month", "")
        if not issue or not month:
            continue
        month_issue[month][issue] += to_int(row.get("doc_count", "0"))

    months = sorted(month_issue.keys())
    issue_doc_counts = {
        row["issue"]: to_int(row.get("docs_with_hits", "0")) for row in issue_radar_case
    }
    top_issues = [
        issue for issue, _ in sorted(issue_doc_counts.items(), key=lambda x: x[1], reverse=True)[: args.top_issues]
    ]

    issue_series = {}
    for issue in top_issues:
        issue_series[issue] = [month_issue[m].get(issue, 0) for m in months]

    stacked_area(
        months,
        issue_series,
        "Issue Mentions Over Time (Top Issues)",
        visuals_dir / "issue_stacked_area.png",
    )
    stacked_area_pct(
        months,
        issue_series,
        "Issue Share Over Time (Top Issues)",
        visuals_dir / "issue_stacked_area_pct.png",
    )

    # Procedural vs non-procedural
    procedural = []
    non_procedural = []
    for month in months:
        proc_count = sum(month_issue[month].get(issue, 0) for issue in PROCEDURAL_ISSUES)
        total = sum(month_issue[month].values())
        procedural.append(proc_count)
        non_procedural.append(max(total - proc_count, 0))
    stacked_area(
        months,
        {"procedural": procedural, "other": non_procedural},
        "Procedural vs Other Issue Mentions Over Time",
        visuals_dir / "procedural_vs_other.png",
    )

    # Issue co-occurrence heatmap (top issues)
    issue_index = {issue: idx for idx, issue in enumerate(top_issues)}
    matrix = np.zeros((len(top_issues), len(top_issues)))
    for row in issue_co_rows:
        a = row.get("issue_a", "")
        b = row.get("issue_b", "")
        if a not in issue_index or b not in issue_index:
            continue
        count = to_int(row.get("doc_count", "0"))
        i = issue_index[a]
        j = issue_index[b]
        matrix[i, j] = count
        matrix[j, i] = count
    heatmap(
        matrix,
        top_issues,
        top_issues,
        "Issue Co-occurrence Heatmap (Top Issues)",
        visuals_dir / "issue_cooccurrence_heatmap.png",
    )

    # Case number vs issue heatmap
    case_map = defaultdict(list)
    for row in doc_index:
        if row.get("corpus") != "case":
            continue
        case_numbers = row.get("case_numbers", "")
        if not case_numbers:
            continue
        nums = [n.strip() for n in case_numbers.split(";") if n.strip()]
        if not nums:
            continue
        case_map[row.get("source_pdf", "")] = nums

    case_issue_counts = Counter()
    case_totals = Counter()
    for row in issue_doc_rows:
        if row.get("corpus") != "case":
            continue
        source = row.get("source_pdf", "")
        issue = row.get("issue", "")
        if not source or not issue:
            continue
        for case_num in case_map.get(source, []):
            case_issue_counts[(case_num, issue)] += 1
            case_totals[case_num] += 1

    top_cases = [case for case, _ in case_totals.most_common(args.top_cases)]
    case_issue_matrix = np.zeros((len(top_cases), len(top_issues)))
    for i, case_num in enumerate(top_cases):
        for j, issue in enumerate(top_issues):
            case_issue_matrix[i, j] = case_issue_counts.get((case_num, issue), 0)
    heatmap(
        case_issue_matrix,
        top_issues,
        top_cases,
        "Case Number vs Issue (Top Cases/Issues)",
        visuals_dir / "case_issue_heatmap.png",
    )

    # Morgan vs overall issue distribution
    overall_counts = {row["issue"]: to_int(row.get("docs_with_hits", "0")) for row in issue_radar_case}
    morgan_counts = {row["issue"]: to_int(row.get("docs_with_hits", "0")) for row in morgan_issue_rows}
    morgan_doc_total = len(morgan_docs_rows) if morgan_docs_rows else 1
    overall_doc_total = sum(overall_counts.values()) if overall_counts else 1

    labels = [
        issue for issue, _ in sorted(morgan_counts.items(), key=lambda x: x[1], reverse=True)[: args.top_issues]
    ]
    morgan_pct = [morgan_counts.get(issue, 0) / morgan_doc_total for issue in labels]
    overall_pct = [overall_counts.get(issue, 0) / overall_doc_total for issue in labels]
    grouped_bar(
        labels,
        [morgan_pct, overall_pct],
        ["Morgan-referenced docs", "All case docs"],
        "Issue Share: Morgan-Referenced vs Overall",
        visuals_dir / "morgan_vs_overall_issue.png",
    )

    # Pareto charts for citations
    statute_labels = [row["citation"] for row in statute_rows[: args.top_citations]]
    statute_counts = [to_int(row.get("count", "0")) for row in statute_rows[: args.top_citations]]
    pareto_chart(
        statute_labels,
        statute_counts,
        "Statute Citations Pareto (Top)",
        visuals_dir / "statute_citations_pareto.png",
    )

    caselaw_labels = [row["citation"] for row in caselaw_rows[: args.top_citations]]
    caselaw_counts = [to_int(row.get("count", "0")) for row in caselaw_rows[: args.top_citations]]
    pareto_chart(
        caselaw_labels,
        caselaw_counts,
        "Caselaw Reporter Citations Pareto (Top)",
        visuals_dir / "caselaw_citations_pareto.png",
    )

    # Citations by month (case docs)
    doc_index_map = {row.get("source_txt", ""): row for row in doc_index if row.get("corpus") == "case"}
    statute_by_month = Counter()
    caselaw_by_month = Counter()
    for txt_path in iter_text_files(Path(args.case_text)):
        meta = doc_index_map.get(str(txt_path), {})
        month = parse_month(meta.get("date", ""))
        if not month:
            continue
        body = load_text_body(txt_path)
        if not body:
            continue
        stat_count = 0
        stat_count += len(USC_PATTERN.findall(body))
        stat_count += len(TEX_CODE_PATTERN.findall(body))
        stat_count += len(TRCP_PATTERN.findall(body))
        stat_count += len(TRCP_SHORT_PATTERN.findall(body))
        stat_count += len(TAC_PATTERN.findall(body))
        case_count = len(REPORTER_PATTERN.findall(body))
        if stat_count:
            statute_by_month[month] += stat_count
        if case_count:
            caselaw_by_month[month] += case_count

    citation_months = sorted(set(statute_by_month.keys()) | set(caselaw_by_month.keys()))
    rows = []
    for month in citation_months:
        rows.append(
            {
                "month": month,
                "statute_citations": statute_by_month.get(month, 0),
                "caselaw_citations": caselaw_by_month.get(month, 0),
            }
        )
    write_csv(
        reports_dir / "citations_by_month.csv",
        rows,
        ["month", "statute_citations", "caselaw_citations"],
    )
    line_chart(
        citation_months,
        {
            "statutes": [statute_by_month.get(m, 0) for m in citation_months],
            "caselaw": [caselaw_by_month.get(m, 0) for m in citation_months],
        },
        "Caselaw vs Statute Citations Over Time (Case Docs)",
        visuals_dir / "citations_over_time.png",
    )

    summary_path = reports_dir / "advanced_visuals_summary.md"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("# Advanced Visuals Summary (Document-Derived)\n\n")
        f.write("These visuals are keyword/citation indexes, not legal advice.\n\n")
        f.write("Outputs (visuals):\n")
        f.write("- reports/visuals_advanced/issue_stacked_area.png\n")
        f.write("- reports/visuals_advanced/issue_stacked_area_pct.png\n")
        f.write("- reports/visuals_advanced/procedural_vs_other.png\n")
        f.write("- reports/visuals_advanced/issue_cooccurrence_heatmap.png\n")
        f.write("- reports/visuals_advanced/case_issue_heatmap.png\n")
        f.write("- reports/visuals_advanced/morgan_vs_overall_issue.png\n")
        f.write("- reports/visuals_advanced/statute_citations_pareto.png\n")
        f.write("- reports/visuals_advanced/caselaw_citations_pareto.png\n")
        f.write("- reports/visuals_advanced/citations_over_time.png\n\n")
        f.write("Outputs (data):\n")
        f.write("- reports/citations_by_month.csv\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
