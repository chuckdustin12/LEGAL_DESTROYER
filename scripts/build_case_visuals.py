"""Create advanced visualizations from the 28B page-level JSON extract."""

from __future__ import annotations

import argparse
import json
import math
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

MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

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

DATE_PATTERNS = [
    re.compile(
        r"\b("
        r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
        r"Dec(?:ember)?"
        r")\s+\d{1,2},?\s+\d{2,4}\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-](?:\d{2}|\d{4})\b"),
]

ISSUE_CATEGORIES = {
    "recusal": [
        "recusal",
        "recuse",
        "disqualify",
        "disqualification",
        "recusal hearing",
    ],
    "emergency relief": [
        "emergency relief",
        "emergency motion",
        "emergency order",
        "temporary restraining order",
        "tro",
        "ex parte",
    ],
    "temporary orders": [
        "temporary order",
        "temporary orders",
        "temporary injunction",
        "interim order",
    ],
    "notice/hearing": [
        "notice of hearing",
        "notice of trial",
        "notice of trial setting",
        "hearing set",
        "trial setting",
    ],
    "jurisdiction/void": [
        "jurisdiction",
        "subject matter",
        "void order",
        "lack of jurisdiction",
        "no jurisdiction",
    ],
    "mandamus/appeal": [
        "mandamus",
        "petition for writ",
        "appeal",
        "appellate",
    ],
    "sanctions/fees": [
        "sanctions",
        "attorney's fees",
        "attorneys fees",
        "fees",
        "costs",
        "contempt",
    ],
    "custody/child": [
        "custody",
        "conservatorship",
        "possession",
        "child support",
        "best interest",
    ],
    "property/financial": [
        "property",
        "bank",
        "account",
        "transfer",
        "wire",
        "fraud",
        "asset",
    ],
}

PROCEDURAL_FLAGS = {
    "no_notice": [
        "without notice",
        "no notice",
        "lack of notice",
        "notice not provided",
        "not served",
    ],
    "no_hearing": [
        "without hearing",
        "no hearing",
        "hearing denied",
        "denied a hearing",
        "refused to hear",
    ],
    "ex_parte": ["ex parte", "ex-parte"],
    "lack_of_consent": ["without consent", "no consent", "consent not present"],
    "jurisdiction": ["no jurisdiction", "lack of jurisdiction", "void order"],
    "due_process": ["due process", "notice and opportunity", "fundamental fairness"],
    "bias": ["bias", "impartiality", "recusal", "disqualify"],
}

OUTCOME_TERMS = [
    "granted",
    "denied",
    "dismissed",
    "vacated",
    "overruled",
    "sustained",
    "struck",
    "affirmed",
    "reversed",
]

EXHIBIT_PATTERN = re.compile(r"\bEXHIBIT\s+[A-Z0-9][A-Z0-9.-]*\b", re.IGNORECASE)
EMAIL_MARKERS = ("from:", "sent:", "to:", "subject:")


@dataclass
class PageRecord:
    page_number: int
    text: str


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _normalize_ascii(text: str) -> str:
    cleaned = text.translate(NON_ASCII_MAP)
    return cleaned.encode("ascii", "ignore").decode("ascii")


def _parse_date(date_text: str, min_year: int, max_year: int) -> datetime | None:
    date_text = date_text.strip()
    month_match = re.match(
        r"(?i)^(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
        r"Dec(?:ember)?)\s+(\d{1,2}),?\s+(\d{2,4})$",
        date_text,
    )
    if month_match:
        month_key = month_match.group(1).lower()
        day = int(month_match.group(2))
        year_text = month_match.group(3)
        if len(year_text) not in (2, 4):
            return None
        year = int(year_text)
        if len(year_text) == 2:
            year += 2000
        if year < min_year or year > max_year:
            return None
        month = MONTHS.get(month_key, 0)
        if month:
            return datetime(year, month, day)
        return None

    numeric_match = re.match(r"^(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})$", date_text)
    if numeric_match:
        month = int(numeric_match.group(1))
        day = int(numeric_match.group(2))
        year_text = numeric_match.group(3)
        if len(year_text) not in (2, 4):
            return None
        year = int(year_text)
        if len(year_text) == 2:
            year += 2000
        if year < min_year or year > max_year:
            return None
        try:
            return datetime(year, month, day)
        except ValueError:
            return None
    return None


def _iter_pages(json_path: Path) -> Iterable[PageRecord]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    for page in payload.get("pages", []):
        yield PageRecord(page_number=page["page_number"], text=page.get("text", ""))


def _issue_tags(text_lower: str) -> List[str]:
    tags = []
    for issue, keywords in ISSUE_CATEGORIES.items():
        if any(keyword in text_lower for keyword in keywords):
            tags.append(issue)
    return tags


def _count_keywords(text_lower: str, keywords: List[str]) -> int:
    return sum(text_lower.count(keyword) for keyword in keywords)


def _collect_metrics(
    pages: List[PageRecord],
    min_year: int,
    max_year: int,
) -> dict:
    issue_counts = {issue: 0 for issue in ISSUE_CATEGORIES}
    flag_counts_by_page = {flag: [0] * len(pages) for flag in PROCEDURAL_FLAGS}
    issue_counts_by_page = {issue: [0] * len(pages) for issue in ISSUE_CATEGORIES}
    outcome_counts = Counter()
    exhibit_counts_by_page = [0] * len(pages)
    correspondence_counts_by_page = [0] * len(pages)
    date_counts = Counter()
    issue_counts_by_month = {issue: defaultdict(int) for issue in ISSUE_CATEGORIES}

    outcome_re = re.compile(r"(?i)\b(" + "|".join(OUTCOME_TERMS) + r")\b")

    for idx, page in enumerate(pages):
        text_norm = _normalize_ascii(page.text)
        text_lower = text_norm.lower()

        for issue, keywords in ISSUE_CATEGORIES.items():
            count = _count_keywords(text_lower, keywords)
            issue_counts[issue] += count
            issue_counts_by_page[issue][idx] = count

        for flag, keywords in PROCEDURAL_FLAGS.items():
            if any(keyword in text_lower for keyword in keywords):
                flag_counts_by_page[flag][idx] = 1

        for match in outcome_re.findall(text_norm):
            outcome_counts[match.lower()] += 1

        exhibit_counts_by_page[idx] = len(EXHIBIT_PATTERN.findall(text_norm))

        correspondence_hits = 0
        for line in text_norm.splitlines():
            lowered = line.lower()
            if any(marker in lowered for marker in EMAIL_MARKERS) or "text message" in lowered:
                correspondence_hits += 1
        correspondence_counts_by_page[idx] = correspondence_hits

        dates_in_page: set[datetime] = set()
        for pattern in DATE_PATTERNS:
            for match in pattern.finditer(text_norm):
                parsed = _parse_date(match.group(0), min_year, max_year)
                if parsed:
                    dates_in_page.add(parsed)

        for parsed in dates_in_page:
            date_counts[(parsed.year, parsed.month)] += 1

        tags = _issue_tags(text_lower)
        for tag in tags:
            for parsed in dates_in_page:
                issue_counts_by_month[tag][(parsed.year, parsed.month)] += 1

    return {
        "issue_counts": issue_counts,
        "flag_counts_by_page": flag_counts_by_page,
        "issue_counts_by_page": issue_counts_by_page,
        "outcome_counts": outcome_counts,
        "exhibit_counts_by_page": exhibit_counts_by_page,
        "correspondence_counts_by_page": correspondence_counts_by_page,
        "date_counts": date_counts,
        "issue_counts_by_month": issue_counts_by_month,
    }


def _year_month_range(date_counts: Counter, fallback_start: int, fallback_end: int) -> List[Tuple[int, int]]:
    if not date_counts:
        return [(fallback_start, month) for month in range(1, 13)]
    years = [year for year, _ in date_counts.keys()]
    start_year = min(years)
    end_year = max(years)
    timeline = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            timeline.append((year, month))
    return timeline


def _bin_series(values: List[int], bin_size: int) -> Tuple[np.ndarray, np.ndarray]:
    count = len(values)
    bin_count = math.ceil(count / bin_size)
    binned = np.zeros(bin_count, dtype=int)
    for idx, value in enumerate(values):
        binned[idx // bin_size] += value
    x = np.arange(bin_count) * bin_size + 1
    return x, binned


def _plot_issue_distribution(output_path: Path, issue_counts: Dict[str, int]) -> None:
    labels = list(issue_counts.keys())
    values = [issue_counts[label] for label in labels]
    order = np.argsort(values)
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]

    plt.figure(figsize=(10, 6))
    plt.barh(labels, values, color="#3465a4")
    plt.title("Issue Keyword Distribution")
    plt.xlabel("Keyword Hits")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_timeline_heatmap(
    output_path: Path,
    date_counts: Counter,
    fallback_start: int,
    fallback_end: int,
) -> None:
    timeline = _year_month_range(date_counts, fallback_start, fallback_end)
    years = sorted({year for year, _ in timeline})
    year_index = {year: idx for idx, year in enumerate(years)}

    heat = np.zeros((12, len(years)), dtype=int)
    for year, month in timeline:
        heat[month - 1, year_index[year]] = date_counts.get((year, month), 0)

    plt.figure(figsize=(12, 6))
    im = plt.imshow(heat, aspect="auto", cmap="magma")
    plt.title("Date Mentions Heatmap (Year x Month)")
    plt.xlabel("Year")
    plt.ylabel("Month")

    xticks = np.arange(len(years))
    xtick_labels = [str(year) if (year - years[0]) % 2 == 0 else "" for year in years]
    plt.xticks(xticks, xtick_labels, rotation=45, ha="right")
    plt.yticks(np.arange(12), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    plt.colorbar(im, label="Mentions")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_issue_timeline(
    output_path: Path,
    issue_counts_by_month: Dict[str, Dict[Tuple[int, int], int]],
    date_counts: Counter,
    top_n: int,
    fallback_start: int,
    fallback_end: int,
) -> None:
    totals = {issue: sum(counts.values()) for issue, counts in issue_counts_by_month.items()}
    top_issues = sorted(totals, key=totals.get, reverse=True)[:top_n]
    timeline = _year_month_range(date_counts, fallback_start, fallback_end)
    x = np.arange(len(timeline))
    labels = [f"{year}-{month:02d}" for year, month in timeline]

    series = []
    for issue in top_issues:
        counts = issue_counts_by_month[issue]
        series.append([counts.get(key, 0) for key in timeline])

    plt.figure(figsize=(14, 6))
    plt.stackplot(x, series, labels=top_issues, alpha=0.85)
    plt.title("Issue Mentions Over Time (Top Issues)")
    plt.ylabel("Mentions")
    tick_idx = np.arange(0, len(x), max(1, len(x) // 12))
    plt.xticks(tick_idx, [labels[i] for i in tick_idx], rotation=45, ha="right")
    plt.legend(loc="upper left", ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_flags_by_page(
    output_path: Path,
    flag_counts_by_page: Dict[str, List[int]],
    bin_size: int,
    top_n: int,
) -> None:
    totals = {flag: sum(values) for flag, values in flag_counts_by_page.items()}
    top_flags = sorted(totals, key=totals.get, reverse=True)[:top_n]

    binned_series = []
    x = None
    for flag in top_flags:
        x, binned = _bin_series(flag_counts_by_page[flag], bin_size)
        binned_series.append(binned)

    plt.figure(figsize=(12, 6))
    bottom = np.zeros_like(binned_series[0])
    for flag, series in zip(top_flags, binned_series):
        plt.bar(x, series, bottom=bottom, width=bin_size * 0.9, label=flag)
        bottom += series
    plt.title("Procedural Flags by Page Range")
    plt.xlabel(f"Page Range (bin size = {bin_size})")
    plt.ylabel("Flag Mentions (binary per page)")
    plt.legend(loc="upper right", ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_outcome_frequency(output_path: Path, outcome_counts: Counter) -> None:
    labels = OUTCOME_TERMS
    values = [outcome_counts.get(term, 0) for term in labels]
    plt.figure(figsize=(10, 5))
    plt.bar(labels, values, color="#4e9a06")
    plt.title("Outcome Term Frequency")
    plt.ylabel("Occurrences")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_exhibit_correspondence(
    output_path: Path,
    exhibit_counts_by_page: List[int],
    correspondence_counts_by_page: List[int],
    bin_size: int,
) -> None:
    x, exhibits = _bin_series(exhibit_counts_by_page, bin_size)
    _, correspondence = _bin_series(correspondence_counts_by_page, bin_size)

    plt.figure(figsize=(12, 6))
    plt.plot(x, exhibits, label="Exhibits per bin", color="#75507b", linewidth=2)
    plt.plot(x, correspondence, label="Correspondence per bin", color="#c17d11", linewidth=2)
    plt.title("Exhibit and Correspondence Density by Page Range")
    plt.xlabel(f"Page Range (bin size = {bin_size})")
    plt.ylabel("Counts")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _write_index(output_path: Path, images: List[Tuple[str, str]]) -> None:
    lines = [
        "<!doctype html>",
        "<html lang=\"en\">",
        "<head>",
        "<meta charset=\"utf-8\" />",
        "<title>28B Case Visuals</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 24px; background: #f7f7f7; }",
        "h1 { margin-bottom: 8px; }",
        ".grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; }",
        ".card { background: #fff; padding: 16px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }",
        ".card img { width: 100%; height: auto; border-radius: 6px; }",
        ".caption { margin-top: 8px; font-size: 14px; color: #444; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>28B Case Visuals</h1>",
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
    parser = argparse.ArgumentParser(description="Build advanced visuals from the 28B JSON extract.")
    parser.add_argument(
        "--json",
        type=Path,
        required=True,
        help="Path to the extracted JSON (page-level) file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/visuals"),
        help="Directory to write images and HTML.",
    )
    parser.add_argument("--min-year", type=int, default=1800, help="Minimum year for dates.")
    parser.add_argument("--max-year", type=int, default=2100, help="Maximum year for dates.")
    parser.add_argument("--bin-size", type=int, default=100, help="Page bin size for density charts.")
    parser.add_argument("--top-issues", type=int, default=5, help="Top issues in stacked timeline.")
    parser.add_argument("--top-flags", type=int, default=5, help="Top flags in stacked bars.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    json_path = args.json.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pages = list(_iter_pages(json_path))
    metrics = _collect_metrics(pages, args.min_year, args.max_year)

    issue_dist_path = output_dir / "issue_keyword_distribution.png"
    timeline_heatmap_path = output_dir / "timeline_heatmap.png"
    issue_timeline_path = output_dir / "issue_mentions_over_time.png"
    flags_path = output_dir / "procedural_flags_by_page.png"
    outcomes_path = output_dir / "outcome_term_frequency.png"
    exhibit_path = output_dir / "exhibit_correspondence_density.png"

    _plot_issue_distribution(issue_dist_path, metrics["issue_counts"])
    _plot_timeline_heatmap(
        timeline_heatmap_path,
        metrics["date_counts"],
        args.min_year,
        args.max_year,
    )
    _plot_issue_timeline(
        issue_timeline_path,
        metrics["issue_counts_by_month"],
        metrics["date_counts"],
        args.top_issues,
        args.min_year,
        args.max_year,
    )
    _plot_flags_by_page(
        flags_path,
        metrics["flag_counts_by_page"],
        args.bin_size,
        args.top_flags,
    )
    _plot_outcome_frequency(outcomes_path, metrics["outcome_counts"])
    _plot_exhibit_correspondence(
        exhibit_path,
        metrics["exhibit_counts_by_page"],
        metrics["correspondence_counts_by_page"],
        args.bin_size,
    )

    index_path = output_dir / "index.html"
    images = [
        ("issue_keyword_distribution.png", "Issue keyword distribution (full record)"),
        ("timeline_heatmap.png", "Date mentions heatmap by year and month"),
        ("issue_mentions_over_time.png", "Issue mentions over time (top issues)"),
        ("procedural_flags_by_page.png", "Procedural flags by page range"),
        ("outcome_term_frequency.png", "Outcome term frequency"),
        ("exhibit_correspondence_density.png", "Exhibit + correspondence density"),
    ]
    _write_index(index_path, images)

    print(f"Wrote visuals to {output_dir}")


if __name__ == "__main__":
    main()
