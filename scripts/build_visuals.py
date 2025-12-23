import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


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


def radar_chart(rows: list[dict], title: str, out_path: Path, top_n: int) -> None:
    rows = sorted(rows, key=lambda r: to_float(r.get("pct_docs", "0")), reverse=True)
    rows = rows[:top_n]
    labels = [r["issue"] for r in rows]
    values = [to_float(r.get("pct_docs", "0")) for r in rows]
    if not labels:
        return

    angles = np.linspace(0, 2 * math.pi, len(labels), endpoint=False)
    values += values[:1]
    angles = np.concatenate([angles, angles[:1]])

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color="#1f6f78", linewidth=2)
    ax.fill(angles, values, color="#1f6f78", alpha=0.25)
    ax.set_title(title, pad=20, fontsize=13)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels([])
    ax.set_ylim(0, max(values) * 1.1)
    save_plot(out_path)


def bar_chart(rows: list[dict], title: str, out_path: Path, top_n: int) -> None:
    rows = sorted(rows, key=lambda r: to_int(r.get("docs_with_hits", "0")), reverse=True)
    rows = rows[:top_n]
    labels = [r["issue"] for r in rows][::-1]
    values = [to_int(r.get("docs_with_hits", "0")) for r in rows][::-1]
    if not labels:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels, values, color="#264653")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Documents with hits")
    save_plot(out_path)


def heatmap(month_rows: list[dict], title: str, out_path: Path, top_n: int) -> None:
    month_rows = [row for row in month_rows if row.get("corpus") == "case"]
    if not month_rows:
        return
    issue_counts = {}
    for row in month_rows:
        issue = row["issue"]
        issue_counts[issue] = issue_counts.get(issue, 0) + to_int(row.get("doc_count", "0"))
    top_issues = [k for k, _ in sorted(issue_counts.items(), key=lambda x: -x[1])[:top_n]]

    months = sorted({row["month"] for row in month_rows})
    matrix = np.zeros((len(top_issues), len(months)))
    month_index = {month: idx for idx, month in enumerate(months)}

    for row in month_rows:
        issue = row["issue"]
        if issue not in top_issues:
            continue
        month = row["month"]
        i = top_issues.index(issue)
        j = month_index[month]
        matrix[i, j] += to_int(row.get("doc_count", "0"))

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrBr")
    ax.set_yticks(range(len(top_issues)))
    ax.set_yticklabels(top_issues, fontsize=9)
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=45, ha="right", fontsize=8)
    ax.set_title(title, fontsize=13)
    fig.colorbar(im, ax=ax, label="Documents with hits")
    save_plot(out_path)


def cooccurrence_network(
    co_rows: list[dict], radar_rows: list[dict], title: str, out_path: Path, top_n: int
) -> None:
    co_rows = sorted(co_rows, key=lambda r: -to_int(r.get("doc_count", "0")))
    co_rows = co_rows[:top_n]
    if not co_rows:
        return

    node_sizes = {}
    for row in radar_rows:
        issue = row["issue"]
        node_sizes[issue] = 300 + to_int(row.get("docs_with_hits", "0")) * 2

    graph = nx.Graph()
    for row in co_rows:
        a = row["issue_a"]
        b = row["issue_b"]
        weight = to_int(row.get("doc_count", "0"))
        graph.add_edge(a, b, weight=weight)

    sizes = [node_sizes.get(node, 300) for node in graph.nodes]
    weights = [graph[u][v]["weight"] for u, v in graph.edges]
    max_weight = max(weights) if weights else 1
    widths = [1 + (w / max_weight) * 3 for w in weights]

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(graph, seed=42, k=0.7)
    nx.draw_networkx_nodes(graph, pos, node_size=sizes, node_color="#2a9d8f", alpha=0.9, ax=ax)
    nx.draw_networkx_edges(graph, pos, width=widths, alpha=0.4, edge_color="#264653", ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=9, font_color="#1b1b1b", ax=ax)
    ax.set_title(title, fontsize=13)
    ax.axis("off")
    save_plot(out_path)


def parse_effort_stats(path: Path) -> dict:
    if not path.exists():
        return {}
    data = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = re.match(r"- Case docs:\s*(\d+.*PDFs.*)$", line)
            if match:
                data["case_docs"] = match.group(1).strip()
                continue
            match = re.match(r"- Research:\s*(\d+.*PDFs.*)$", line)
            if match:
                data["research_docs"] = match.group(1).strip()
                continue
            match = re.match(r"- Total:\s*(\d+.*PDFs.*)$", line)
            if match:
                data["total_docs"] = match.group(1).strip()
    return data


def build_dashboard(out_dir: Path, stats: dict) -> None:
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Case Analysis Dashboard</title>
  <style>
    :root {{
      --bg: #f3efe7;
      --ink: #1f1d1a;
      --accent: #1f6f78;
      --accent2: #c46d5e;
      --panel: #ffffff;
      --muted: #6c6a66;
    }}
    body {{
      margin: 0;
      font-family: "EB Garamond", "Garamond", "Georgia", serif;
      color: var(--ink);
      background: radial-gradient(circle at 20% 10%, #fce7c2 0%, transparent 45%),
                  radial-gradient(circle at 80% 20%, #d0f0f6 0%, transparent 45%),
                  var(--bg);
    }}
    header {{
      padding: 32px 40px 12px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 32px;
      letter-spacing: 0.5px;
    }}
    .subtitle {{
      color: var(--muted);
      font-size: 16px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 18px;
      padding: 20px 40px 40px;
    }}
    .card {{
      background: var(--panel);
      border-radius: 16px;
      padding: 16px;
      box-shadow: 0 8px 20px rgba(0,0,0,0.08);
      border: 1px solid rgba(0,0,0,0.05);
    }}
    .card h2 {{
      font-size: 18px;
      margin: 4px 0 12px;
      color: var(--accent);
    }}
    .stats {{
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      font-size: 14px;
      color: var(--muted);
    }}
    img {{
      width: 100%;
      border-radius: 12px;
    }}
    footer {{
      padding: 0 40px 40px;
      color: var(--muted);
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Issue Radar & Evidence Mapping</h1>
    <div class="subtitle">Document-derived visual mapping (navigation aid, not legal advice).</div>
    <div class="stats">
      Case docs: {stats.get("case_docs", "n/a")}<br/>
      Research docs: {stats.get("research_docs", "n/a")}<br/>
      Total: {stats.get("total_docs", "n/a")}
    </div>
  </header>
  <section class="grid">
    <div class="card">
      <h2>Case Issue Radar</h2>
      <img src="visuals/issue_radar_case.png" alt="Case Issue Radar" />
    </div>
    <div class="card">
      <h2>Research Issue Radar</h2>
      <img src="visuals/issue_radar_research.png" alt="Research Issue Radar" />
    </div>
    <div class="card">
      <h2>Issue Frequency (Case)</h2>
      <img src="visuals/issue_bar_case.png" alt="Issue Bar Case" />
    </div>
    <div class="card">
      <h2>Issue Frequency (Research)</h2>
      <img src="visuals/issue_bar_research.png" alt="Issue Bar Research" />
    </div>
    <div class="card">
      <h2>Issue Heatmap by Month</h2>
      <img src="visuals/issue_heatmap_case.png" alt="Issue Heatmap" />
    </div>
    <div class="card">
      <h2>Issue Co-occurrence Network</h2>
      <img src="visuals/issue_cooccurrence_network.png" alt="Issue Network" />
    </div>
  </section>
  <footer>
    Data sources: reports/issue_radar_*.csv, reports/issue_month_map.csv, reports/issue_cooccurrence.csv.
  </footer>
</body>
</html>
"""
    (out_dir / "dashboard.html").write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build visual mappings and dashboard.")
    parser.add_argument("--reports", default="reports")
    args = parser.parse_args()

    reports_dir = Path(args.reports)
    visuals_dir = reports_dir / "visuals"
    ensure_dir(visuals_dir)

    radar_case = read_csv(reports_dir / "issue_radar_case.csv")
    radar_research = read_csv(reports_dir / "issue_radar_research.csv")
    radar_overall = read_csv(reports_dir / "issue_radar_overall.csv")
    co_rows = read_csv(reports_dir / "issue_cooccurrence.csv")
    month_rows = read_csv(reports_dir / "issue_month_map.csv")

    radar_chart(
        radar_case, "Case Issue Radar (Top 12)", visuals_dir / "issue_radar_case.png", 12
    )
    radar_chart(
        radar_research,
        "Research Issue Radar (Top 12)",
        visuals_dir / "issue_radar_research.png",
        12,
    )
    bar_chart(
        radar_case,
        "Issue Frequency (Case, Top 15)",
        visuals_dir / "issue_bar_case.png",
        15,
    )
    bar_chart(
        radar_research,
        "Issue Frequency (Research, Top 15)",
        visuals_dir / "issue_bar_research.png",
        15,
    )
    heatmap(
        month_rows,
        "Issue Volume by Month (Case, Top 10)",
        visuals_dir / "issue_heatmap_case.png",
        10,
    )
    cooccurrence_network(
        co_rows,
        radar_overall,
        "Issue Co-occurrence (Top 30 pairs)",
        visuals_dir / "issue_cooccurrence_network.png",
        30,
    )

    stats = parse_effort_stats(reports_dir / "effort_stats_deep.md")
    build_dashboard(reports_dir, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
