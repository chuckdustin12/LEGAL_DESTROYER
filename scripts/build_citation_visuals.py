"""Create visuals from the citation spotting report."""

from __future__ import annotations

import argparse
import re
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NON_ASCII_MAP = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "--",
        "\u2026": "...",
        "\u00a0": " ",
        "\u2011": "-",
        "\u2212": "-",
        "\u00ad": "",
        "\u2022": "-",
        "\u00a7": "sec.",
    }
)

SECTION_RE = re.compile(r"^##\s+(.*)")
ITEM_RE = re.compile(r"^- (.+?) \(count:\s+(\d+);", re.IGNORECASE)


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _normalize_ascii(text: str) -> str:
    cleaned = text.translate(NON_ASCII_MAP)
    return cleaned.encode("ascii", "ignore").decode("ascii")


def _parse_citations(path: Path) -> Dict[str, List[Tuple[str, int]]]:
    sections: Dict[str, List[Tuple[str, int]]] = {}
    current: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        section_match = SECTION_RE.match(line)
        if section_match:
            current = _normalize_ascii(section_match.group(1).strip())
            sections.setdefault(current, [])
            continue
        if current is None:
            continue
        item_match = ITEM_RE.match(line)
        if not item_match:
            continue
        label = _normalize_ascii(item_match.group(1).strip())
        count = int(item_match.group(2))
        sections[current].append((label, count))
    return sections


def _wrap_labels(labels: List[str], width: int = 40) -> List[str]:
    wrapped = []
    for label in labels:
        wrapped.append("\n".join(textwrap.wrap(label, width=width)) or label)
    return wrapped


def _plot_bar(path: Path, items: List[Tuple[str, int]], title: str, xlabel: str) -> None:
    if not items:
        return
    labels = [label for label, _count in items]
    counts = [count for _label, count in items]
    labels = _wrap_labels(labels)
    height = max(4.0, 0.38 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(11, height))
    ax.barh(labels, counts, color="#355f8d")
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_category_summary(
    path: Path,
    items: List[Tuple[str, int]],
    title: str,
    xlabel: str,
) -> None:
    if not items:
        return
    labels = [label for label, _count in items]
    counts = [count for _label, count in items]
    labels = _wrap_labels(labels, width=24)
    height = max(3.5, 0.4 * len(labels) + 1.0)
    fig, ax = plt.subplots(figsize=(9, height))
    ax.barh(labels, counts, color="#6b8f71")
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "section"


def _write_index(output_path: Path, images: List[Tuple[str, str]]) -> None:
    lines = [
        "<!doctype html>",
        "<html lang=\"en\">",
        "<head>",
        "<meta charset=\"utf-8\" />",
        "<title>28B Citation Visuals</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 24px; background: #f5f6f8; }",
        "h1 { margin-bottom: 6px; }",
        ".grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; }",
        ".card { background: #fff; padding: 16px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }",
        ".card img { width: 100%; height: auto; border-radius: 6px; }",
        ".caption { margin-top: 8px; font-size: 14px; color: #444; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>28B Citation Visuals</h1>",
        f"<p>Generated: {_timestamp()}</p>",
        "<div class=\"grid\">",
    ]
    for filename, caption in images:
        lines.extend(
            [
                "<div class=\"card\">",
                f"<img src=\"{filename}\" alt=\"{caption}\" />",
                f"<div class=\"caption\">{caption}</div>",
                "</div>",
            ]
        )
    lines.extend(["</div>", "</body>", "</html>"])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build visuals from the citation report.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("reports/28b_citations.md"),
        help="Path to the citations markdown report.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/visuals_citations"),
        help="Directory to write images and HTML.",
    )
    parser.add_argument("--top", type=int, default=20, help="Top items per category.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sections = _parse_citations(input_path)

    unique_counts = [(section, len(items)) for section, items in sections.items()]
    mention_counts = [(section, sum(count for _label, count in items)) for section, items in sections.items()]
    unique_counts.sort(key=lambda item: item[1], reverse=True)
    mention_counts.sort(key=lambda item: item[1], reverse=True)

    images: List[Tuple[str, str]] = []

    unique_path = output_dir / "citation_unique_counts.png"
    _plot_category_summary(unique_path, unique_counts, "Unique citations by category", "Unique items")
    images.append((unique_path.name, "Unique citations by category"))

    mentions_path = output_dir / "citation_total_mentions.png"
    _plot_category_summary(mentions_path, mention_counts, "Total citation mentions by category", "Total mentions")
    images.append((mentions_path.name, "Total citation mentions by category"))

    for section, items in sections.items():
        sorted_items = sorted(items, key=lambda item: item[1], reverse=True)[: args.top]
        filename = f"citations_{_safe_slug(section)}.png"
        path = output_dir / filename
        _plot_bar(path, sorted_items, f"Top citations: {section}", "Count")
        images.append((filename, f"Top citations: {section}"))

    index_path = output_dir / "index.html"
    _write_index(index_path, images)

    print(f"Wrote citation visuals to {output_dir}")


if __name__ == "__main__":
    main()
