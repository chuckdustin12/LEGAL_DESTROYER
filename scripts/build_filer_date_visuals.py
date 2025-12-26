"""Build date-based filing visuals by filer using docket entries."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DOCKET_HEADER = "ALL TRANSACTIONS FOR A CASE"
DOCKET_ENTRY = re.compile(r"(\d{1,4})\s+(\d{2}/\d{2}/\d{4})\s+")

FILER_ORDER = [
    "charles_dustin_myers",
    "cooper_carter",
    "morgan_michelle_myers",
    "court",
    "clerk",
    "oag",
    "unknown",
]

FILER_LABELS = {
    "charles_dustin_myers": "Charles",
    "cooper_carter": "Counsel",
    "morgan_michelle_myers": "Petitioner",
    "court": "Court",
    "clerk": "Clerk",
    "oag": "OAG",
    "unknown": "Unknown",
}


@dataclass
class DocketEntry:
    filemark: str
    date: datetime
    description: str


def _normalize_ascii(text: str) -> str:
    return text.encode("ascii", "ignore").decode("ascii")


def _iter_pages(json_path: Path) -> Iterable[str]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    for page in payload.get("pages", []):
        yield page.get("text", "")


def _clean_description(raw: str) -> str:
    cleaned = re.sub(r"\s+", " ", raw).strip()
    cleaned = re.sub(
        r"\s+(?:UI|UIM|NUI|I|N|Y|NA)\s*[-]?\d+\.\d{2}\b.*$",
        "",
        cleaned,
    )
    return cleaned.strip()


def _parse_docket_entries(json_path: Path) -> List[DocketEntry]:
    entries: List[DocketEntry] = []
    for page_text in _iter_pages(json_path):
        if DOCKET_HEADER not in page_text:
            continue
        normalized = _normalize_ascii(page_text)
        normalized = re.sub(
            r"(\d+\.\d{2})(\d{1,4})\s+(\d{2}/\d{2}/\d{4})",
            r"\1 \2 \3",
            normalized,
        )
        marker = normalized.find("Filemark")
        if marker != -1:
            normalized = normalized[marker:]
        matches = list(DOCKET_ENTRY.finditer(normalized))
        for idx, match in enumerate(matches):
            filemark_raw = match.group(1)
            try:
                filemark = str(int(filemark_raw))
            except ValueError:
                continue
            if filemark == "0":
                continue
            date_text = match.group(2)
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(normalized)
            raw_desc = normalized[start:end]
            if not raw_desc.strip():
                continue
            if "Date Filed" in raw_desc or "Page" in raw_desc:
                continue
            try:
                date_value = datetime.strptime(date_text, "%m/%d/%Y")
            except ValueError:
                continue
            description = _clean_description(raw_desc)
            if not description:
                continue
            entries.append(
                DocketEntry(
                    filemark=filemark,
                    date=date_value,
                    description=description,
                )
            )
    return entries


def _load_docket_filer_map(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not path.exists():
        return mapping
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            filemark = row.get("filemark", "").strip()
            filer = row.get("filer", "").strip()
            if filemark and filer:
                mapping[filemark] = filer
    return mapping


def _month_key(date_value: datetime) -> str:
    return date_value.strftime("%Y-%m")


def _series_by_filer(
    entries: List[DocketEntry], docket_map: Dict[str, str]
) -> Tuple[List[str], Dict[str, List[int]]]:
    counts: Dict[str, Counter[str]] = defaultdict(Counter)
    all_months: set[str] = set()

    for entry in entries:
        filer = docket_map.get(entry.filemark, "unknown")
        month = _month_key(entry.date)
        counts[filer][month] += 1
        all_months.add(month)

    months = sorted(all_months)
    series: Dict[str, List[int]] = {}
    for filer in FILER_ORDER:
        if filer not in counts:
            continue
        series[filer] = [counts[filer].get(month, 0) for month in months]
    return months, series


def _plot_stacked_monthly(
    output_path: Path, months: List[str], series: Dict[str, List[int]]
) -> None:
    if not months or not series:
        return
    filers = [f for f in FILER_ORDER if f in series]
    data = np.vstack([series[filer] for filer in filers])
    x = np.arange(len(months))

    plt.figure(figsize=(12, 6))
    labels = [FILER_LABELS.get(filer, filer) for filer in filers]
    plt.stackplot(x, data, labels=labels, alpha=0.85)
    step = max(1, len(months) // 12)
    plt.xticks(x[::step], months[::step], rotation=45, ha="right")
    plt.ylabel("Filings")
    plt.title("Filings by Month (Stacked by Filer)")
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_cumulative(
    output_path: Path, months: List[str], series: Dict[str, List[int]]
) -> None:
    if not months or not series:
        return
    filers = [f for f in FILER_ORDER if f in series]
    x = np.arange(len(months))
    plt.figure(figsize=(12, 6))
    for filer in filers:
        cumulative = np.cumsum(series[filer])
        label = FILER_LABELS.get(filer, filer)
        plt.plot(x, cumulative, label=label, linewidth=2)
    step = max(1, len(months) // 12)
    plt.xticks(x[::step], months[::step], rotation=45, ha="right")
    plt.ylabel("Cumulative Filings")
    plt.title("Cumulative Filings by Month")
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_per_filer(
    output_dir: Path, months: List[str], series: Dict[str, List[int]]
) -> List[str]:
    if not months or not series:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(months))
    images: List[str] = []
    for filer in FILER_ORDER:
        if filer not in series:
            continue
        plt.figure(figsize=(11, 5))
        plt.bar(x, series[filer], color="#3465a4")
        step = max(1, len(months) // 12)
        plt.xticks(x[::step], months[::step], rotation=45, ha="right")
        plt.ylabel("Filings")
        title = f"Filings by Month - {FILER_LABELS.get(filer, filer)}"
        plt.title(title)
        plt.tight_layout()
        filename = f"filings_by_month_{filer}.png"
        plt.savefig(output_dir / filename, dpi=160)
        plt.close()
        images.append(filename)
    return images


def _write_index(output_path: Path, images: List[Tuple[str, str]]) -> None:
    lines = [
        "<!doctype html>",
        "<html lang=\"en\">",
        "<head>",
        "<meta charset=\"utf-8\" />",
        "<title>28B Filing Dates by Filer</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 24px; background: #f3f3f3; }",
        "h1 { margin-bottom: 8px; }",
        ".grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; }",
        ".card { background: #fff; padding: 16px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }",
        ".card img { width: 100%; height: auto; border-radius: 6px; }",
        ".caption { margin-top: 8px; font-size: 14px; color: #444; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>28B Filing Dates by Filer</h1>",
        f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
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


def _write_entries_csv(output_path: Path, rows: List[dict]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["date", "filemark", "filer", "description"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_summary(output_path: Path, rows: List[dict]) -> None:
    counts = Counter(row["filer"] for row in rows)
    if rows:
        dates = [row["date"] for row in rows]
        min_date = min(dates)
        max_date = max(dates)
    else:
        min_date = max_date = "unknown"
    lines = [
        "# 28B Filing Date Summary",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Total docket entries: {len(rows)}",
        f"Date range: {min_date} to {max_date}",
        "",
        "## Counts by filer",
        "",
    ]
    for filer in FILER_ORDER:
        if filer in counts:
            lines.append(f"- {filer}: {counts[filer]}")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build filing date visuals by filer.")
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("extracted_text_full/28b_merged/28B_merged.json"),
        help="Path to the extracted JSON (page-level) file.",
    )
    parser.add_argument(
        "--docket-filer-map",
        type=Path,
        default=Path("reports/28b_docket_filer_map.csv"),
        help="CSV map of filemark-to-filer.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/visuals_by_filer_dates"),
        help="Directory to write images and HTML.",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=Path("reports"),
        help="Directory to write summary and CSV outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    json_path = args.json.expanduser().resolve()
    docket_map_path = args.docket_filer_map.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    summary_dir = args.summary_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    entries = _parse_docket_entries(json_path)
    docket_map = _load_docket_filer_map(docket_map_path)

    rows = []
    for entry in entries:
        filer = docket_map.get(entry.filemark, "unknown")
        rows.append(
            {
                "date": entry.date.strftime("%Y-%m-%d"),
                "filemark": entry.filemark,
                "filer": filer,
                "description": _normalize_ascii(entry.description),
            }
        )

    _write_entries_csv(summary_dir / "28b_filer_filing_dates.csv", rows)
    _write_summary(summary_dir / "28b_filer_filing_dates_summary.md", rows)

    months, series = _series_by_filer(entries, docket_map)
    stacked_path = output_dir / "filings_by_filer_monthly.png"
    cumulative_path = output_dir / "cumulative_filings_by_filer.png"
    _plot_stacked_monthly(stacked_path, months, series)
    _plot_cumulative(cumulative_path, months, series)
    per_filer_images = _plot_per_filer(output_dir, months, series)

    images = [
        (stacked_path.name, "Filings by month (stacked by filer)"),
        (cumulative_path.name, "Cumulative filings by month"),
    ]
    for filename in per_filer_images:
        filer = filename.replace("filings_by_month_", "").replace(".png", "")
        label = FILER_LABELS.get(filer, filer)
        images.append((filename, f"{label} filings by month"))

    _write_index(output_dir / "index.html", images)
    print(f"Wrote filer date visuals to {output_dir}")


if __name__ == "__main__":
    main()
