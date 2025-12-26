"""Build a per-party action list from CASE DOCS vector stores."""

from __future__ import annotations

import argparse
import json
import re
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


@dataclass
class Party:
    label: str
    variants: List[str]
    patterns: List[re.Pattern]


@dataclass
class ActionHit:
    date_text: str | None
    date_value: datetime | None
    snippet: str
    source_pdf: str
    vector_id: int
    chunk_index: int
    filer: str


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


def _parse_date(date_text: str) -> datetime | None:
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
        if year < 1800 or year > 2100:
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
        if year < 1800 or year > 2100:
            return None
        try:
            return datetime(year, month, day)
        except ValueError:
            return None
    return None


def _extract_dates(text: str) -> List[Tuple[str, datetime | None]]:
    results: List[Tuple[str, datetime | None]] = []
    for pattern in DATE_PATTERNS:
        for match in pattern.finditer(text):
            date_text = match.group(0)
            results.append((date_text, _parse_date(date_text)))
    return results


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    chunks = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _load_party_list(names_path: Path | None, names: List[str]) -> List[Party]:
    raw_entries: List[str] = []
    if names_path:
        if not names_path.exists():
            raise FileNotFoundError(f"Missing names file: {names_path}")
        raw_entries.extend(
            line.strip()
            for line in names_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    raw_entries.extend([name.strip() for name in names if name.strip()])
    if not raw_entries:
        raise ValueError("Provide --names or --names-file with at least one name.")

    parties: List[Party] = []
    for entry in raw_entries:
        variants = [part.strip() for part in entry.split("|") if part.strip()]
        if not variants:
            continue
        label = variants[0]
        words = label.split()
        if len(words) >= 2:
            first_last = f"{words[0]} {words[-1]}"
            if first_last not in variants:
                variants.append(first_last)
        patterns = [
            re.compile(rf"\b{re.escape(variant)}\b", re.IGNORECASE) for variant in variants
        ]
        parties.append(Party(label=label, variants=variants, patterns=patterns))
    return parties


def _iter_records(store_root: Path) -> Iterable[Tuple[str, dict]]:
    for store_dir in sorted(store_root.iterdir(), key=lambda p: p.name.lower()):
        metadata_path = store_dir / "metadata.jsonl"
        if not metadata_path.exists():
            continue
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            yield store_dir.name, json.loads(line)


def _matches_party(text: str, party: Party) -> bool:
    return any(pattern.search(text) for pattern in party.patterns)


def _extract_party_hits(
    store_root: Path,
    parties: List[Party],
    max_per_party: int,
) -> Dict[str, List[ActionHit]]:
    results: Dict[str, List[ActionHit]] = {party.label: [] for party in parties}
    seen: Dict[str, set] = {party.label: set() for party in parties}

    for filer, record in _iter_records(store_root):
        text = record.get("text", "")
        if not text:
            continue
        normalized = _normalize_ascii(_normalize_ws(text))
        sentences = _split_sentences(normalized)

        for party in parties:
            if not _matches_party(normalized, party):
                continue
            for sentence in sentences:
                if not _matches_party(sentence, party):
                    continue
                key = (sentence, record.get("source_pdf", ""))
                if key in seen[party.label]:
                    continue
                seen[party.label].add(key)
                dates = _extract_dates(sentence)
                if dates:
                    date_text, date_value = dates[0]
                else:
                    date_text, date_value = (None, None)
                hit = ActionHit(
                    date_text=date_text,
                    date_value=date_value,
                    snippet=_snippet(sentence),
                    source_pdf=record.get("source_pdf", ""),
                    vector_id=int(record.get("vector_id", 0)),
                    chunk_index=int(record.get("chunk_index", 0)),
                    filer=filer,
                )
                results[party.label].append(hit)
                if len(results[party.label]) >= max_per_party:
                    break
            if len(results[party.label]) >= max_per_party:
                continue

    return results


def _format_filer(name: str) -> str:
    return name.replace("_", " ").title()


def _write_report(
    output_path: Path,
    store_root: Path,
    parties: List[Party],
    hits: Dict[str, List[ActionHit]],
    max_per_party: int,
) -> None:
    lines = [
        "# 28B Per-Party Action List (Court Documents)",
        "",
        f"Generated: {_timestamp()}",
        f"Source store: {store_root}",
        f"Max entries per party: {max_per_party}",
        "",
        "Notes:",
        "- These are document statements mentioning each party; verify in source PDFs.",
        "- Dates are extracted from the same sentence when present.",
        "",
    ]

    for party in parties:
        party_hits = hits.get(party.label, [])
        lines.append(f"## {party.label}")
        if not party_hits:
            lines.append("- No mentions found in CASE DOCS.")
            lines.append("")
            continue

        dated = [hit for hit in party_hits if hit.date_value]
        undated = [hit for hit in party_hits if not hit.date_value]
        dated.sort(key=lambda hit: hit.date_value)

        for hit in dated + undated:
            date_label = hit.date_value.strftime("%Y-%m-%d") if hit.date_value else "Undated"
            if hit.date_text:
                date_label = f"{date_label} ({hit.date_text})"
            lines.append(
                f"- {date_label} | {hit.snippet} "
                f"(source: {hit.source_pdf}; filer: {_format_filer(hit.filer)}; "
                f"vector_id: {hit.vector_id}; chunk: {hit.chunk_index})"
            )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a per-party action list from CASE DOCS vector stores."
    )
    parser.add_argument(
        "--store-root",
        type=Path,
        default=Path("vector_store_case_docs_by_filer"),
        help="Root directory containing merged per-filer vector stores.",
    )
    parser.add_argument(
        "--names-file",
        type=Path,
        default=Path("reports/28b_party_action_names.txt"),
        help="Text file listing party names and variants.",
    )
    parser.add_argument(
        "--names",
        nargs="*",
        default=[],
        help="Party names (optional); add | for variants.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/28b_party_actions.md"),
        help="Output markdown report path.",
    )
    parser.add_argument(
        "--max-per-party",
        type=int,
        default=80,
        help="Maximum entries per party.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    store_root = args.store_root.expanduser().resolve()
    if not store_root.exists():
        raise FileNotFoundError(f"Missing store root: {store_root}")

    names_path = args.names_file.expanduser()
    parties = _load_party_list(names_path, args.names)
    hits = _extract_party_hits(store_root, parties, args.max_per_party)
    _write_report(args.output, store_root, parties, hits, args.max_per_party)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
