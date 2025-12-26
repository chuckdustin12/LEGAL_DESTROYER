"""Generate advanced case insights from a page-level JSON extract."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
    "filing_refusal": ["refused to file", "rejected filing", "returned unfiled"],
    "bias": ["bias", "impartiality", "recusal", "disqualify"],
    "due_process": ["due process", "notice and opportunity", "fundamental fairness"],
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

TARGET_TERMS = [
    "motion",
    "petition",
    "application",
    "request",
    "plea",
    "mandamus",
    "appeal",
    "rehearing",
    "order",
    "relief",
    "injunction",
]

TITLE_PATTERNS = [
    re.compile(
        r"\b(Hon\.?|Judge|Justice|Chief Justice|Clerk|District Clerk|Presiding Judge|"
        r"Court Coordinator|Coordinator|Associate Judge|Magistrate|Attorney|Counsel|"
        r"Assistant Attorney General)\s+([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){0,4})"
    ),
    re.compile(
        r"\b([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){0,4}),\s*"
        r"(Judge|Justice|Clerk|District Clerk|Presiding Judge|Court Coordinator|"
        r"Associate Judge|Attorney|Counsel)\b"
    ),
]

EXHIBIT_PATTERN = re.compile(r"\bEXHIBIT\s+[A-Z0-9][A-Z0-9.-]*\b", re.IGNORECASE)
EMAIL_MARKERS = ("from:", "sent:", "to:", "subject:")

DOCKET_HEADER = "ALL TRANSACTIONS FOR A CASE"
DOCKET_LINE = re.compile(r"(\d{2}/\d{2}/\d{4})\s+(.+)")

STOP_NAMES = {"TARRANT", "COUNTY", "DISTRICT", "COURT", "OFFICE", "CLERK"}


@dataclass
class PageRecord:
    page_number: int
    text: str


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _normalize_ascii(text: str) -> str:
    cleaned = text.translate(NON_ASCII_MAP)
    return cleaned.encode("ascii", "ignore").decode("ascii")


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _snippet(text: str, max_len: int = 360) -> str:
    cleaned = _normalize_ascii(_normalize_ws(text))
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3].rstrip() + "..."


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


def _extract_dates(text: str, min_year: int, max_year: int) -> List[datetime]:
    results = []
    for pattern in DATE_PATTERNS:
        for match in pattern.finditer(text):
            parsed = _parse_date(match.group(0), min_year, max_year)
            if parsed:
                results.append(parsed)
    return results


def _iter_pages(json_path: Path) -> Iterable[PageRecord]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    for page in payload.get("pages", []):
        yield PageRecord(page_number=page["page_number"], text=page.get("text", ""))


def _score_keywords(text: str, keywords: List[str]) -> int:
    lowered = text.lower()
    score = 0
    for keyword in keywords:
        score += lowered.count(keyword)
    return score


def _first_keyword_span(text: str, keywords: List[str]) -> Tuple[int, int] | None:
    lowered = text.lower()
    for keyword in keywords:
        idx = lowered.find(keyword)
        if idx != -1:
            return idx, idx + len(keyword)
    return None


def _collect_hotspots(
    pages: Iterable[PageRecord],
    categories: Dict[str, List[str]],
    max_per_category: int,
) -> Dict[str, List[dict]]:
    hits: Dict[str, List[dict]] = {category: [] for category in categories}
    for page in pages:
        text = page.text
        for category, keywords in categories.items():
            score = _score_keywords(text, keywords)
            if score <= 0:
                continue
            span = _first_keyword_span(text, keywords)
            if span:
                start = max(0, span[0] - 80)
                end = min(len(text), span[1] + 220)
                snippet = _snippet(text[start:end])
            else:
                snippet = _snippet(text)
            hits[category].append(
                {"page": page.page_number, "score": score, "snippet": snippet}
            )
    for category in hits:
        hits[category].sort(key=lambda item: (-item["score"], item["page"]))
        hits[category] = hits[category][:max_per_category]
    return hits


def _collect_procedural_flags(
    pages: Iterable[PageRecord],
    flags: Dict[str, List[str]],
    max_hits: int,
) -> Dict[str, List[dict]]:
    results: Dict[str, List[dict]] = {flag: [] for flag in flags}
    for page in pages:
        lowered = page.text.lower()
        for flag, keywords in flags.items():
            if not any(keyword in lowered for keyword in keywords):
                continue
            span = _first_keyword_span(page.text, keywords)
            if span:
                start = max(0, span[0] - 80)
                end = min(len(page.text), span[1] + 220)
                snippet = _snippet(page.text[start:end])
            else:
                snippet = _snippet(page.text)
            results[flag].append(
                {"page": page.page_number, "snippet": snippet}
            )
    for flag in results:
        results[flag] = results[flag][:max_hits]
    return results


def _collect_outcomes(
    pages: Iterable[PageRecord], max_hits: int, max_per_outcome: int
) -> List[dict]:
    hits: List[dict] = []
    outcome_re = re.compile(r"(?i)\b(" + "|".join(OUTCOME_TERMS) + r")\b")
    seen = set()
    outcome_counts = Counter()
    for page in pages:
        normalized = _normalize_ascii(page.text)
        for match in outcome_re.finditer(normalized):
            outcome = match.group(1).lower()
            if outcome_counts[outcome] >= max_per_outcome:
                continue
            start = max(0, match.start() - 120)
            end = min(len(normalized), match.end() + 160)
            snippet = _snippet(normalized[start:end], 320)
            key = (page.page_number, snippet)
            if key in seen:
                continue
            seen.add(key)
            outcome_counts[outcome] += 1
            hits.append({"page": page.page_number, "snippet": snippet})
            if len(hits) >= max_hits:
                return hits
    return hits


def _collect_actors(pages: Iterable[PageRecord]) -> Dict[Tuple[str, str], dict]:
    actors: Dict[Tuple[str, str], dict] = defaultdict(lambda: {"count": 0, "pages": set()})
    for page in pages:
        text = _normalize_ascii(page.text)
        for pattern in TITLE_PATTERNS:
            for match in pattern.finditer(text):
                if pattern is TITLE_PATTERNS[0]:
                    role = match.group(1).strip()
                    name = match.group(2).strip()
                else:
                    name = match.group(1).strip()
                    role = match.group(2).strip()
                if not name or name.upper() in STOP_NAMES:
                    continue
                entry = actors[(name, role)]
                entry["count"] += 1
                entry["pages"].add(page.page_number)
    return actors


def _collect_exhibits(pages: Iterable[PageRecord], max_hits: int) -> Dict[str, List[int]]:
    exhibits: Dict[str, List[int]] = defaultdict(list)
    for page in pages:
        text = _normalize_ascii(page.text)
        for match in EXHIBIT_PATTERN.findall(text):
            label = _normalize_ascii(match.upper())
            if page.page_number not in exhibits[label]:
                exhibits[label].append(page.page_number)
    for label in list(exhibits.keys()):
        exhibits[label] = exhibits[label][:max_hits]
    return dict(sorted(exhibits.items(), key=lambda item: item[0]))


def _collect_correspondence(pages: Iterable[PageRecord], max_hits: int) -> List[dict]:
    hits: List[dict] = []
    for page in pages:
        lines = page.text.splitlines()
        for line in lines:
            lowered = line.strip().lower()
            if not lowered:
                continue
            if any(marker in lowered for marker in EMAIL_MARKERS) or "text message" in lowered:
                hits.append({"page": page.page_number, "snippet": _snippet(line)})
                if len(hits) >= max_hits:
                    return hits
    return hits


def _collect_docket_entries(pages: Iterable[PageRecord], max_entries: int) -> List[dict]:
    entries: List[dict] = []
    seen = set()
    for page in pages:
        if DOCKET_HEADER not in page.text:
            continue
        for line in page.text.splitlines():
            line = _normalize_ascii(line.strip())
            if not line:
                continue
            match = DOCKET_LINE.search(line)
            if not match:
                continue
            date_text = match.group(1)
            description = match.group(2).strip()
            if "Date Filed" in description or "Page" in description:
                continue
            key = (date_text, description)
            if key in seen:
                continue
            seen.add(key)
            entries.append(
                {
                    "date": date_text,
                    "description": _normalize_ascii(description),
                    "page": page.page_number,
                }
            )
            if len(entries) >= max_entries:
                return entries
    return entries


def _write_advanced_summary(
    output_path: Path,
    *,
    page_count: int,
    total_chars: int,
    date_stats: Dict[str, str],
    category_counts: Dict[str, int],
    hotspots: Dict[str, List[dict]],
) -> None:
    lines = [
        "# 28B Advanced Insights Summary",
        "",
        f"Generated: {_timestamp()}",
        "",
        f"Pages scanned: {page_count}",
        f"Total characters: {total_chars}",
        f"Date range: {date_stats.get('min_date', 'unknown')} to {date_stats.get('max_date', 'unknown')}",
        f"Unique dates: {date_stats.get('unique_dates', 0)}",
        "",
        "## Issue Hotspots (Top Pages by Keyword Hits)",
        "",
    ]
    for category, count in category_counts.items():
        lines.append(f"### {category} (total hits: {count})")
        for hit in hotspots.get(category, []):
            lines.append(f"- Page {hit['page']} | score {hit['score']}")
            lines.append(f"  Snippet: {hit['snippet']}")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_procedural_flags(output_path: Path, flags: Dict[str, List[dict]]) -> None:
    lines = [
        "# 28B Procedural Flags (Keyword Matches)",
        "",
        f"Generated: {_timestamp()}",
        "",
        "These are keyword matches, not legal conclusions.",
        "",
    ]
    for flag, hits in flags.items():
        lines.append(f"## {flag}")
        if not hits:
            lines.append("- No matches found.")
            lines.append("")
            continue
        for hit in hits:
            lines.append(f"- Page {hit['page']}")
            lines.append(f"  Snippet: {hit['snippet']}")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_outcomes(output_path: Path, outcomes: List[dict]) -> None:
    lines = [
        "# 28B Motion/Order Outcomes",
        "",
        f"Generated: {_timestamp()}",
        "",
    ]
    if not outcomes:
        lines.append("No outcome lines found.")
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return
    for item in outcomes:
        lines.append(f"- Page {item['page']}: {item['snippet']}")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_actor_map(output_path: Path, actors: Dict[Tuple[str, str], dict], max_actors: int) -> None:
    lines = [
        "# 28B Actor Map (Titles + Names)",
        "",
        f"Generated: {_timestamp()}",
        "",
    ]
    items = []
    for (name, role), stats in actors.items():
        pages = sorted(stats["pages"])
        items.append((stats["count"], name, role, pages))
    items.sort(key=lambda item: (-item[0], item[1], item[2]))
    for count, name, role, pages in items[:max_actors]:
        page_sample = ", ".join(str(p) for p in pages[:8])
        lines.append(f"- {name} | {role} | mentions: {count} | pages: {page_sample}")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_exhibits(output_path: Path, exhibits: Dict[str, List[int]]) -> None:
    lines = [
        "# 28B Exhibit Index",
        "",
        f"Generated: {_timestamp()}",
        "",
    ]
    if not exhibits:
        lines.append("No exhibits found.")
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return
    for label, pages in exhibits.items():
        page_sample = ", ".join(str(p) for p in pages[:10])
        lines.append(f"- {label} | pages: {page_sample}")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_correspondence(output_path: Path, hits: List[dict]) -> None:
    lines = [
        "# 28B Correspondence Index (Emails/Texts)",
        "",
        f"Generated: {_timestamp()}",
        "",
    ]
    if not hits:
        lines.append("No correspondence markers found.")
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return
    for hit in hits:
        lines.append(f"- Page {hit['page']}: {hit['snippet']}")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_docket_entries(output_path: Path, entries: List[dict]) -> None:
    lines = [
        "# 28B Docket Entries (Parsed from ALL TRANSACTIONS pages)",
        "",
        f"Generated: {_timestamp()}",
        "",
    ]
    if not entries:
        lines.append("No docket entries parsed.")
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return
    for entry in entries:
        lines.append(
            f"- {entry['date']} | Page {entry['page']} | {entry['description']}"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate advanced case insights.")
    parser.add_argument(
        "--json",
        type=Path,
        required=True,
        help="Path to the extracted JSON (page-level) file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory to write report files.",
    )
    parser.add_argument("--label", type=str, default="28b", help="Report filename label prefix.")
    parser.add_argument("--min-year", type=int, default=1800, help="Minimum year for date parsing.")
    parser.add_argument("--max-year", type=int, default=2100, help="Maximum year for date parsing.")
    parser.add_argument("--max-hotspots", type=int, default=15, help="Max hotspots per category.")
    parser.add_argument("--max-flag-hits", type=int, default=200, help="Max flag hits per category.")
    parser.add_argument("--max-outcomes", type=int, default=200, help="Max outcome lines.")
    parser.add_argument(
        "--max-outcome-per-term",
        type=int,
        default=40,
        help="Max outcome hits per outcome term.",
    )
    parser.add_argument("--max-actors", type=int, default=120, help="Max actors to list.")
    parser.add_argument("--max-exhibit-pages", type=int, default=25, help="Max pages per exhibit label.")
    parser.add_argument("--max-correspondence", type=int, default=200, help="Max correspondence hits.")
    parser.add_argument("--max-docket-entries", type=int, default=1500, help="Max docket entries.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    json_path = args.json.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pages = list(_iter_pages(json_path))

    total_chars = sum(len(page.text) for page in pages)
    all_dates: List[datetime] = []
    category_counts: Dict[str, int] = {category: 0 for category in ISSUE_CATEGORIES}

    for page in pages:
        all_dates.extend(_extract_dates(page.text, args.min_year, args.max_year))
        for category, keywords in ISSUE_CATEGORIES.items():
            category_counts[category] += _score_keywords(page.text, keywords)

    date_stats = {
        "min_date": min(all_dates).strftime("%Y-%m-%d") if all_dates else "unknown",
        "max_date": max(all_dates).strftime("%Y-%m-%d") if all_dates else "unknown",
        "unique_dates": len({dt.date() for dt in all_dates}) if all_dates else 0,
    }

    hotspots = _collect_hotspots(pages, ISSUE_CATEGORIES, args.max_hotspots)
    flags = _collect_procedural_flags(pages, PROCEDURAL_FLAGS, args.max_flag_hits)
    outcomes = _collect_outcomes(pages, args.max_outcomes, args.max_outcome_per_term)
    actors = _collect_actors(pages)
    exhibits = _collect_exhibits(pages, args.max_exhibit_pages)
    correspondence = _collect_correspondence(pages, args.max_correspondence)
    docket_entries = _collect_docket_entries(pages, args.max_docket_entries)

    _write_advanced_summary(
        output_dir / f"{args.label}_advanced_insights.md",
        page_count=len(pages),
        total_chars=total_chars,
        date_stats=date_stats,
        category_counts=category_counts,
        hotspots=hotspots,
    )
    _write_procedural_flags(output_dir / f"{args.label}_procedural_flags.md", flags)
    _write_outcomes(output_dir / f"{args.label}_motion_outcomes.md", outcomes)
    _write_actor_map(output_dir / f"{args.label}_actor_map.md", actors, args.max_actors)
    _write_exhibits(output_dir / f"{args.label}_exhibit_index.md", exhibits)
    _write_correspondence(
        output_dir / f"{args.label}_correspondence_index.md", correspondence
    )
    _write_docket_entries(output_dir / f"{args.label}_docket_entries.md", docket_entries)

    print(f"Wrote advanced reports to {output_dir}")


if __name__ == "__main__":
    main()
