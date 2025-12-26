"""Build a concise case memorandum from the 28B page-level JSON extract."""

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


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _snippet(text: str, max_len: int = 320) -> str:
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


def _iter_pages(json_path: Path) -> Iterable[PageRecord]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    for page in payload.get("pages", []):
        yield PageRecord(page_number=page["page_number"], text=page.get("text", ""))


def _extract_case_info(text: str) -> Dict[str, str]:
    normalized = _normalize_ascii(_normalize_ws(text))
    info = {}
    match = re.search(r"Cause Number:\s*([A-Z0-9-]+)", normalized)
    if match:
        info["cause_number"] = match.group(1)
    match = re.search(r"Date Filed:\s*(\d{2}/\d{2}/\d{4})", normalized)
    if match:
        info["date_filed"] = match.group(1)
    match = re.search(r"Case Status\.*:\s*([A-Z ]+)", normalized)
    if match:
        info["case_status"] = match.group(1).strip()
    match = re.search(r"Cause of Action:\s*([A-Z0-9 /-]+)", normalized)
    if match:
        info["cause_of_action"] = match.group(1).strip()
    match = re.search(r"([A-Z][A-Z ]+?)\s+v\s+([A-Z][A-Z ]+)", normalized)
    if match:
        info["party_a"] = match.group(1).strip()
        info["party_b"] = match.group(2).strip()
    return info


def _score_keywords(text: str, keywords: List[str]) -> int:
    lowered = text.lower()
    score = 0
    for keyword in keywords:
        score += lowered.count(keyword)
    return score


def _issue_tags(text: str) -> List[str]:
    lowered = text.lower()
    tags = []
    for category, keywords in ISSUE_CATEGORIES.items():
        if any(keyword in lowered for keyword in keywords):
            tags.append(category)
    return tags


def _collect_issue_counts(pages: Iterable[PageRecord]) -> Dict[str, int]:
    counts = {category: 0 for category in ISSUE_CATEGORIES}
    for page in pages:
        for category, keywords in ISSUE_CATEGORIES.items():
            counts[category] += _score_keywords(page.text, keywords)
    return counts


def _collect_hotspots(
    pages: Iterable[PageRecord], max_per_category: int
) -> Dict[str, List[dict]]:
    hits: Dict[str, List[dict]] = {category: [] for category in ISSUE_CATEGORIES}
    for page in pages:
        for category, keywords in ISSUE_CATEGORIES.items():
            score = _score_keywords(page.text, keywords)
            if score <= 0:
                continue
            snippet = _snippet(page.text)
            hits[category].append(
                {"page": page.page_number, "score": score, "snippet": snippet}
            )
    for category in hits:
        hits[category].sort(key=lambda item: (-item["score"], item["page"]))
        hits[category] = hits[category][:max_per_category]
    return hits


def _collect_events(
    pages: Iterable[PageRecord],
    min_year: int,
    max_year: int,
    max_events: int,
    max_per_date: int,
    max_per_page: int,
) -> List[dict]:
    events: List[dict] = []
    for page in pages:
        text = _normalize_ascii(page.text)
        tags = _issue_tags(text)
        outcomes = sum(term in text.lower() for term in OUTCOME_TERMS)
        page_events = 0
        for pattern in DATE_PATTERNS:
            for match in pattern.finditer(text):
                parsed = _parse_date(match.group(0), min_year, max_year)
                if not parsed:
                    continue
                start = max(0, match.start() - 80)
                end = min(len(text), match.end() + 200)
                snippet = _snippet(text[start:end], 260)
                score = 1 + (2 * len(tags)) + outcomes
                events.append(
                    {
                        "date": parsed,
                        "date_text": match.group(0),
                        "page": page.page_number,
                        "tags": tags,
                        "score": score,
                        "snippet": snippet,
                    }
                )
                page_events += 1
                if page_events >= max_per_page:
                    break
            if page_events >= max_per_page:
                break

    grouped: Dict[datetime, List[dict]] = defaultdict(list)
    for event in events:
        grouped[event["date"].date()].append(event)

    condensed: List[dict] = []
    for date_key, items in grouped.items():
        items.sort(key=lambda item: (-item["score"], item["page"]))
        condensed.extend(items[:max_per_date])

    condensed.sort(key=lambda item: (item["date"], -item["score"], item["page"]))
    return condensed[:max_events]


def _collect_outcomes(
    pages: Iterable[PageRecord], max_hits: int, max_per_term: int
) -> List[dict]:
    hits: List[dict] = []
    outcome_re = re.compile(r"(?i)\b(" + "|".join(OUTCOME_TERMS) + r")\b")
    seen = set()
    term_counts = Counter()
    for page in pages:
        normalized = _normalize_ascii(page.text)
        for match in outcome_re.finditer(normalized):
            term = match.group(1).lower()
            if term_counts[term] >= max_per_term:
                continue
            start = max(0, match.start() - 120)
            end = min(len(normalized), match.end() + 180)
            snippet = _snippet(normalized[start:end], 320)
            key = (page.page_number, snippet)
            if key in seen:
                continue
            seen.add(key)
            term_counts[term] += 1
            hits.append({"page": page.page_number, "snippet": snippet})
            if len(hits) >= max_hits:
                return hits
    return hits


def _collect_procedural_flags(
    pages: Iterable[PageRecord], max_hits_per_flag: int
) -> Dict[str, List[dict]]:
    results: Dict[str, List[dict]] = {flag: [] for flag in PROCEDURAL_FLAGS}
    for page in pages:
        lowered = page.text.lower()
        for flag, keywords in PROCEDURAL_FLAGS.items():
            if len(results[flag]) >= max_hits_per_flag:
                continue
            if not any(keyword in lowered for keyword in keywords):
                continue
            snippet = _snippet(page.text)
            results[flag].append({"page": page.page_number, "snippet": snippet})
    return results


def _collect_exhibits(pages: Iterable[PageRecord], max_pages: int) -> Dict[str, List[int]]:
    exhibits: Dict[str, List[int]] = defaultdict(list)
    for page in pages:
        text = _normalize_ascii(page.text)
        for match in EXHIBIT_PATTERN.findall(text):
            label = _normalize_ascii(match.upper())
            if page.page_number not in exhibits[label]:
                exhibits[label].append(page.page_number)
    for label in list(exhibits.keys()):
        exhibits[label] = exhibits[label][:max_pages]
    return dict(sorted(exhibits.items(), key=lambda item: item[0]))


def _collect_correspondence(pages: Iterable[PageRecord], max_hits: int) -> List[dict]:
    hits: List[dict] = []
    for page in pages:
        for line in page.text.splitlines():
            lowered = line.strip().lower()
            if not lowered:
                continue
            if any(marker in lowered for marker in EMAIL_MARKERS) or "text message" in lowered:
                hits.append({"page": page.page_number, "snippet": _snippet(line)})
                if len(hits) >= max_hits:
                    return hits
    return hits


def _collect_date_stats(pages: Iterable[PageRecord], min_year: int, max_year: int) -> Dict[str, str]:
    dates = []
    for page in pages:
        for pattern in DATE_PATTERNS:
            for match in pattern.finditer(page.text):
                parsed = _parse_date(match.group(0), min_year, max_year)
                if parsed:
                    dates.append(parsed)
    if not dates:
        return {"min_date": "unknown", "max_date": "unknown", "unique_dates": "0"}
    return {
        "min_date": min(dates).strftime("%Y-%m-%d"),
        "max_date": max(dates).strftime("%Y-%m-%d"),
        "unique_dates": str(len({dt.date() for dt in dates})),
    }


def _write_memo(
    output_path: Path,
    *,
    case_info: Dict[str, str],
    page_count: int,
    total_chars: int,
    date_stats: Dict[str, str],
    issue_counts: Dict[str, int],
    hotspots: Dict[str, List[dict]],
    events: List[dict],
    outcomes: List[dict],
    flags: Dict[str, List[dict]],
    exhibits: Dict[str, List[int]],
    correspondence: List[dict],
) -> None:
    title_parts = []
    if case_info.get("party_a") and case_info.get("party_b"):
        title_parts.append(f"{case_info['party_a']} v {case_info['party_b']}")
    if case_info.get("cause_number"):
        title_parts.append(f"Cause No. {case_info['cause_number']}")
    title = " - ".join(title_parts) if title_parts else "Case Memorandum"

    lines = [
        f"# {title}",
        "",
        f"Generated: {_timestamp()}",
        "",
        "## Case Overview",
        "",
    ]
    if case_info.get("cause_number"):
        lines.append(f"- Cause number: {case_info['cause_number']}")
    if case_info.get("date_filed"):
        lines.append(f"- Date filed: {case_info['date_filed']}")
    if case_info.get("case_status"):
        lines.append(f"- Case status (recorded): {case_info['case_status']}")
    if case_info.get("cause_of_action"):
        lines.append(f"- Cause of action (recorded): {case_info['cause_of_action']}")
    if case_info.get("party_a") and case_info.get("party_b"):
        lines.append(f"- Parties: {case_info['party_a']} v {case_info['party_b']}")
    if len(lines) == 4:
        lines.append("- Case header details not found in the record header.")

    lines.extend(
        [
            "",
            "## Record Scope",
            "",
            f"- Pages scanned: {page_count}",
            f"- Total characters: {total_chars}",
            f"- Date range (parsed): {date_stats.get('min_date')} to {date_stats.get('max_date')}",
            f"- Unique dates (parsed): {date_stats.get('unique_dates')}",
            "",
            "## Issue Distribution (keyword hits)",
            "",
        ]
    )
    for category, count in issue_counts.items():
        lines.append(f"- {category}: {count}")

    lines.extend(["", "## Issue Hotspots (top pages)", ""])
    for category, hits in hotspots.items():
        if not hits:
            continue
        lines.append(f"### {category}")
        for hit in hits:
            lines.append(f"- Page {hit['page']} | score {hit['score']}")
            lines.append(f"  Snippet: {hit['snippet']} (28B p. {hit['page']})")
        lines.append("")

    lines.extend(["## Key Timeline (condensed)", ""])
    for event in events:
        tag_label = ", ".join(event["tags"]) if event["tags"] else "none"
        date_label = event["date"].strftime("%Y-%m-%d")
        lines.append(
            f"- {date_label} | issues: {tag_label} | Page {event['page']}"
        )
        lines.append(f"  Snippet: {event['snippet']} (28B p. {event['page']})")
    lines.append("")

    lines.extend(["## Outcome Highlights (keyword matched)", ""])
    if outcomes:
        for item in outcomes:
            lines.append(
                f"- Page {item['page']}: {item['snippet']} (28B p. {item['page']})"
            )
    else:
        lines.append("- No outcome lines found.")
    lines.append("")

    lines.extend(["## Procedural Flags (keyword matched)", ""])
    for flag, hits in flags.items():
        if not hits:
            continue
        lines.append(f"### {flag}")
        for hit in hits:
            lines.append(
                f"- Page {hit['page']}: {hit['snippet']} (28B p. {hit['page']})"
            )
        lines.append("")

    lines.extend(["## Exhibit Index (sample)", ""])
    if exhibits:
        for label, pages in list(exhibits.items())[:25]:
            page_sample = ", ".join(str(page) for page in pages[:8])
            lines.append(f"- {label} | pages: {page_sample}")
    else:
        lines.append("- No exhibits found.")
    lines.append("")

    lines.extend(["## Correspondence Index (sample)", ""])
    if correspondence:
        for hit in correspondence:
            lines.append(
                f"- Page {hit['page']}: {hit['snippet']} (28B p. {hit['page']})"
            )
    else:
        lines.append("- No correspondence markers found.")
    lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- This memorandum is a record-based summary using keyword and date parsing.",
            "- Items above should be verified directly against the cited 28B page before filing.",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a case memorandum from the 28B JSON extract.")
    parser.add_argument(
        "--json",
        type=Path,
        required=True,
        help="Path to the extracted JSON (page-level) file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/28b_case_memorandum.md"),
        help="Output markdown path.",
    )
    parser.add_argument("--min-year", type=int, default=1800, help="Minimum year for dates.")
    parser.add_argument("--max-year", type=int, default=2100, help="Maximum year for dates.")
    parser.add_argument("--max-hotspots", type=int, default=5, help="Hotspots per category.")
    parser.add_argument("--max-events", type=int, default=80, help="Max timeline events.")
    parser.add_argument("--max-events-per-date", type=int, default=3, help="Events per date.")
    parser.add_argument("--max-events-per-page", type=int, default=2, help="Events per page.")
    parser.add_argument("--max-outcomes", type=int, default=60, help="Outcome hits.")
    parser.add_argument("--max-outcome-per-term", type=int, default=15, help="Outcome hits per term.")
    parser.add_argument("--max-flag-hits", type=int, default=8, help="Flag hits per category.")
    parser.add_argument("--max-exhibit-pages", type=int, default=8, help="Exhibit pages per label.")
    parser.add_argument("--max-correspondence", type=int, default=20, help="Correspondence hits.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    json_path = args.json.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pages = list(_iter_pages(json_path))
    total_chars = sum(len(page.text) for page in pages)
    case_info = _extract_case_info(pages[0].text if pages else "")
    date_stats = _collect_date_stats(pages, args.min_year, args.max_year)
    issue_counts = _collect_issue_counts(pages)
    hotspots = _collect_hotspots(pages, args.max_hotspots)
    events = _collect_events(
        pages,
        args.min_year,
        args.max_year,
        args.max_events,
        args.max_events_per_date,
        args.max_events_per_page,
    )
    outcomes = _collect_outcomes(pages, args.max_outcomes, args.max_outcome_per_term)
    flags = _collect_procedural_flags(pages, args.max_flag_hits)
    exhibits = _collect_exhibits(pages, args.max_exhibit_pages)
    correspondence = _collect_correspondence(pages, args.max_correspondence)

    _write_memo(
        output_path,
        case_info=case_info,
        page_count=len(pages),
        total_chars=total_chars,
        date_stats=date_stats,
        issue_counts=issue_counts,
        hotspots=hotspots,
        events=events,
        outcomes=outcomes,
        flags=flags,
        exhibits=exhibits,
        correspondence=correspondence,
    )

    print(f"Wrote case memorandum to {output_path}")


if __name__ == "__main__":
    main()
