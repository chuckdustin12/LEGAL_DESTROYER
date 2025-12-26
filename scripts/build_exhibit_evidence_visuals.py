"""Build exhibit evidence marker visuals from OCR outputs."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts import analyze_exhibit_evidence as evidence


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _short_label(name: str, max_len: int = 30) -> str:
    if len(name) <= max_len:
        return name
    return name[: max_len - 3] + "..."


def _plot_marker_balance(output_path: Path, exhibits: List[evidence.ExhibitSummary]) -> None:
    labels = [_short_label(Path(exhibit.source).stem) for exhibit in exhibits]
    evidence_counts = [
        exhibit.marker_counts.get("affidavit_or_sworn", 0)
        + exhibit.marker_counts.get("notary_or_seal", 0)
        + exhibit.marker_counts.get("authentication", 0)
        for exhibit in exhibits
    ]
    signature_counts = [exhibit.marker_counts.get("signature_block", 0) for exhibit in exhibits]
    screenshot_counts = [exhibit.screenshot_hits for exhibit in exhibits]

    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar([i - 0.25 for i in x], evidence_counts, width=0.25, label="evidence markers")
    ax.bar([i for i in x], signature_counts, width=0.25, label="signature blocks")
    ax.bar([i + 0.25 for i in x], screenshot_counts, width=0.25, label="screenshot indicators")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("OCR keyword hits")
    ax.set_title("Exhibit Evidence Markers vs Informal Artifacts (OCR)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _write_index(output_path: Path, images: List[tuple[str, str]]) -> None:
    lines = [
        "<!doctype html>",
        "<html lang=\"en\">",
        "<head>",
        "<meta charset=\"utf-8\" />",
        "<title>Exhibit Evidence Visuals</title>",
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
        "<h1>Exhibit Evidence Visuals</h1>",
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
    parser = argparse.ArgumentParser(description="Build exhibit evidence visuals from OCR.")
    parser.add_argument(
        "--exhibit-root",
        type=Path,
        default=Path("extracted_text_full/inconsistencies"),
        help="Root directory containing OCRed exhibit outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/visuals_exhibits"),
        help="Output directory for images and HTML.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    exhibit_root = args.exhibit_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    exhibits = evidence._collect_exhibits(exhibit_root)
    if not exhibits:
        raise ValueError(f"No exhibit OCR outputs found under {exhibit_root}")

    chart_path = output_dir / "exhibit_evidence_markers.png"
    _plot_marker_balance(chart_path, exhibits)

    index_path = output_dir / "index.html"
    images = [
        ("exhibit_evidence_markers.png", "Evidence markers vs screenshot indicators (OCR)"),
    ]
    _write_index(index_path, images)

    print(f"Wrote visuals to {output_dir}")


if __name__ == "__main__":
    main()
