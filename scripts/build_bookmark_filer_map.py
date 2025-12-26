"""Parse PDF bookmarks to infer filer, date, substance, and page ranges."""

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

import PyPDF2


DATE_PATTERNS = [
    re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b"),
    re.compile(
        r"\b("
        r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
        r"Dec(?:ember)?"
        r")\s+\d{1,2},?\s+\d{2,4}\b",
        re.IGNORECASE,
    ),
]

FILER_RULES = {
    "charles_dustin_myers": [
        (re.compile(r"/s/\s*charles\s+dustin\s+myers", re.IGNORECASE), 6),
        (re.compile(r"charles\s+dustin\s+myers", re.IGNORECASE), 5),
        (re.compile(r"charles\s+myers", re.IGNORECASE), 4),
        (re.compile(r"\brespondent'?s\b", re.IGNORECASE), 3),
        (re.compile(r"\brespondent\b", re.IGNORECASE), 2),
        (re.compile(r"\brelator\b", re.IGNORECASE), 3),
        (re.compile(r"\bdefendant\b", re.IGNORECASE), 2),
        (re.compile(r"\bcounterpetition\b", re.IGNORECASE), 3),
        (re.compile(r"\banswer\b", re.IGNORECASE), 2),
        (re.compile(r"\bobjection\b", re.IGNORECASE), 2),
        (re.compile(r"\bmandamus\b", re.IGNORECASE), 2),
        (re.compile(r"\bfather\b", re.IGNORECASE), 1),
    ],
    "morgan_michelle_myers": [
        (re.compile(r"/s/\s*morgan\s+michelle\s+myers", re.IGNORECASE), 6),
        (re.compile(r"morgan\s+michelle\s+myers", re.IGNORECASE), 5),
        (re.compile(r"morgan\s+myers", re.IGNORECASE), 4),
        (re.compile(r"\bpetitioner'?s\b", re.IGNORECASE), 3),
        (re.compile(r"\bpetitioner\b", re.IGNORECASE), 2),
        (re.compile(r"\bplaintiff\b", re.IGNORECASE), 2),
        (re.compile(r"original\s+petition", re.IGNORECASE), 3),
        (re.compile(r"application\s+for\s+protective\s+order", re.IGNORECASE), 3),
        (re.compile(r"\bmother\b", re.IGNORECASE), 1),
    ],
    "cooper_carter": [
        (re.compile(r"/s/\s*cooper\s+l\.?\s+carter", re.IGNORECASE), 6),
        (re.compile(r"cooper\s+l\.?\s+carter", re.IGNORECASE), 5),
        (re.compile(r"cooper\s+carter", re.IGNORECASE), 4),
        (re.compile(r"\bcooper\b", re.IGNORECASE), 3),
        (re.compile(r"\bcarter\b", re.IGNORECASE), 2),
    ],
    "court": [
        (re.compile(r"memorandum\s+opinion", re.IGNORECASE), 4),
        (re.compile(r"per\s+curiam", re.IGNORECASE), 4),
        (re.compile(r"court\s+of\s+appeals", re.IGNORECASE), 4),
        (re.compile(r"\bct/?appeals\b", re.IGNORECASE), 4),
        (re.compile(r"supreme\s+court", re.IGNORECASE), 4),
        (re.compile(r"\bsup/ct\b", re.IGNORECASE), 3),
        (re.compile(r"associate\s+judge", re.IGNORECASE), 3),
        (re.compile(r"aj'?s\s+report", re.IGNORECASE), 3),
        (re.compile(r"\bjudgment\b", re.IGNORECASE), 3),
        (re.compile(r"\bopinion\b", re.IGNORECASE), 3),
        (re.compile(r"\border\b", re.IGNORECASE), 3),
        (re.compile(r"\bnotice\s+of\s+hearing\b", re.IGNORECASE), 2),
        (re.compile(r"\bnotice\s+of\s+trial\b", re.IGNORECASE), 2),
        (re.compile(r"\bjudge\b", re.IGNORECASE), 2),
        (re.compile(r"\bsigned\b", re.IGNORECASE), 2),
        (re.compile(r"\bdenied\b", re.IGNORECASE), 1),
    ],
    "clerk": [
        (re.compile(r"district\s+clerk", re.IGNORECASE), 4),
        (re.compile(r"\bclerk\b", re.IGNORECASE), 3),
        (re.compile(r"\bcitation\b", re.IGNORECASE), 3),
        (re.compile(r"\bservice\b", re.IGNORECASE), 2),
        (re.compile(r"\breturn\b", re.IGNORECASE), 2),
        (re.compile(r"\bpayment\b", re.IGNORECASE), 2),
        (re.compile(r"\bfee\b", re.IGNORECASE), 2),
        (re.compile(r"\bcopies\b", re.IGNORECASE), 2),
        (re.compile(r"\bcertified\b", re.IGNORECASE), 2),
    ],
    "oag": [
        (re.compile(r"attorney\s+general", re.IGNORECASE), 4),
        (re.compile(r"\boag\b", re.IGNORECASE), 3),
        (re.compile(r"income\s+withholding", re.IGNORECASE), 3),
        (re.compile(r"\biwo\b", re.IGNORECASE), 3),
        (re.compile(r"child\s+support", re.IGNORECASE), 2),
    ],
}

FILER_PRIORITY = [
    "charles_dustin_myers",
    "morgan_michelle_myers",
    "cooper_carter",
    "court",
    "clerk",
    "oag",
    "unknown",
]

MANUAL_LABELS = {
    "charles dustin myers": "charles_dustin_myers",
    "charles myers": "charles_dustin_myers",
    "charles d myers": "charles_dustin_myers",
    "morgan michelle myers": "morgan_michelle_myers",
    "morgan myers": "morgan_michelle_myers",
    "cooper carter": "cooper_carter",
    "cooper l carter": "cooper_carter",
    "court": "court",
    "courts": "court",
    "district clerk": "clerk",
    "clerk": "clerk",
    "clerk s office": "clerk",
    "clerk office": "clerk",
    "oag": "oag",
    "office of the attorney general": "oag",
    "attorney general": "oag",
    "unknown": "unknown",
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

DOCKET_HEADER = "ALL TRANSACTIONS FOR A CASE"
DOCKET_ENTRY = re.compile(r"^\s*(\d{1,4})\s+(\d{2}/\d{2}/\d{4})\s+(.+)$")
FILEMARK_STAMP = re.compile(r"(?i)\bfilemark\b[^0-9]{0,6}(\d{1,4})")
DOCKET_LINE = re.compile(r"^\s*(\d{1,4})\s+\d{2}/\d{2}/\d{4}\b")

STOP_WORDS = {
    "the",
    "of",
    "for",
    "and",
    "a",
    "to",
    "on",
    "in",
    "from",
    "with",
    "without",
    "by",
    "or",
    "att",
    "attachment",
}


@dataclass
class BookmarkEntry:
    index: int
    depth: int
    title: str
    start_page: int | None
    end_page: int | None
    has_children: bool
    filer: str
    date: str | None
    substance: str
    score: int
    mapped_by: str


def _parse_outline(reader: PyPDF2.PdfReader, items: List[object], depth: int = 0) -> List[dict]:
    nodes: List[dict] = []
    i = 0
    while i < len(items):
        item = items[i]
        if isinstance(item, list):
            i += 1
            continue
        title = getattr(item, "title", None) or str(item)
        try:
            page = reader.get_destination_page_number(item)
            start_page = page + 1
        except Exception:
            start_page = None
        node = {
            "title": title,
            "depth": depth,
            "start_page": start_page,
            "children": [],
        }
        if i + 1 < len(items) and isinstance(items[i + 1], list):
            node["children"] = _parse_outline(reader, items[i + 1], depth + 1)
            i += 1
        nodes.append(node)
        i += 1
    return nodes


def _flatten(nodes: List[dict], flat: List[dict]) -> None:
    for node in nodes:
        flat.append(node)
        if node["children"]:
            _flatten(node["children"], flat)


def _extract_date(title: str) -> Tuple[str | None, str]:
    for pattern in DATE_PATTERNS:
        match = pattern.search(title)
        if not match:
            continue
        date_text = match.group(0)
        parsed = _parse_date(date_text)
        cleaned = _strip_date(title, date_text)
        return parsed, cleaned
    return None, title.strip()


def _parse_date(value: str) -> str | None:
    value = value.strip()
    if re.match(r"^\d{1,2}[./-]\d{1,2}[./-]\d{2,4}$", value):
        month, day, year = re.split(r"[./-]", value)
        try:
            year_num = int(year)
        except ValueError:
            return None
        if year_num < 100:
            year_num += 2000
        try:
            dt = datetime(int(year_num), int(month), int(day))
        except ValueError:
            return None
        return dt.strftime("%Y-%m-%d")
    match = re.match(
        r"(?i)^(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
        r"Dec(?:ember)?)\s+(\d{1,2}),?\s+(\d{2,4})$",
        value,
    )
    if match:
        month_name = match.group(1).lower()
        day = int(match.group(2))
        year_text = match.group(3)
        year_num = int(year_text)
        if year_num < 100:
            year_num += 2000
        month_map = {
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
        month = month_map.get(month_name, 0)
        if month == 0:
            return None
        try:
            dt = datetime(year_num, month, day)
        except ValueError:
            return None
        return dt.strftime("%Y-%m-%d")
    return None


def _strip_date(title: str, date_text: str) -> str:
    cleaned = title.replace(date_text, "").strip()
    cleaned = re.sub(r"^[\s\-–—:]+", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def _score_filer(title: str) -> Tuple[str, int, List[str]]:
    scores: Dict[str, int] = defaultdict(int)
    signals: Dict[str, List[str]] = defaultdict(list)
    for filer, rules in FILER_RULES.items():
        for pattern, weight in rules:
            if pattern.search(title):
                scores[filer] += weight
                signals[filer].append(pattern.pattern)
    if not scores:
        return "unknown", 0, []
    max_score = max(scores.values())
    winners = [filer for filer, score in scores.items() if score == max_score]
    for filer in FILER_PRIORITY:
        if filer in winners:
            return filer, max_score, signals.get(filer, [])
    return "unknown", 0, []


def _normalize_label_value(text: str) -> str:
    cleaned = _normalize_ascii(str(text)).lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _canonical_manual_label(value: str) -> str | None:
    if not value:
        return None
    normalized = _normalize_label_value(value)
    return MANUAL_LABELS.get(normalized)


def _normalize_ascii(text: str) -> str:
    cleaned = text.translate(NON_ASCII_MAP)
    return cleaned.encode("ascii", "ignore").decode("ascii")


def _normalize_for_match(text: str) -> str:
    cleaned = _normalize_ascii(text).lower()
    cleaned = re.sub(r"[^a-z0-9\s]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _load_page_text(json_path: Path) -> Dict[int, str]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    pages: Dict[int, str] = {}
    for page in payload.get("pages", []):
        page_number = page.get("page_number")
        if page_number is None:
            continue
        try:
            page_id = int(page_number)
        except (TypeError, ValueError):
            continue
        pages[page_id] = page.get("text", "") or ""
    return pages


def _load_docket_filer_map(path: Path | None) -> Dict[str, str]:
    if not path or not path.exists():
        return {}
    mapping: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            filemark = row.get("filemark")
            filer = row.get("filer")
            if filemark and filer:
                mapping[filemark.strip()] = filer.strip()
    return mapping


def _parse_index(value: str) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        try:
            return int(float(text))
        except ValueError:
            return None


def _load_manual_overrides(path: Path | None) -> Dict[int, Tuple[str, str]]:
    if not path or not path.exists():
        return {}
    overrides: Dict[int, Tuple[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return {}
        header_lower = [cell.strip().lower() for cell in header]
        index_col = header_lower.index("index") if "index" in header_lower else None

        for row in reader:
            if not row:
                continue
            idx_value = row[index_col] if index_col is not None and index_col < len(row) else row[0]
            idx = _parse_index(idx_value)
            if idx is None:
                continue

            candidates: List[Tuple[str, str]] = []
            for cell in row:
                label = _canonical_manual_label(cell)
                if label:
                    candidates.append((label, str(cell).strip()))
            if not candidates:
                continue

            non_unknown = [item for item in candidates if item[0] != "unknown"]
            chosen = None
            for filer in FILER_PRIORITY:
                for label, raw in non_unknown or candidates:
                    if label == filer:
                        chosen = (label, raw)
                        break
                if chosen:
                    break
            if not chosen:
                chosen = candidates[0]
            overrides[idx] = chosen
    return overrides


def _extract_docket_entries(pages: Dict[int, str]) -> List[dict]:
    entries: List[dict] = []
    for text in pages.values():
        if DOCKET_HEADER not in text:
            continue
        for line in text.splitlines():
            line = _normalize_ascii(line.strip())
            if not line:
                continue
            match = DOCKET_ENTRY.match(line)
            if not match:
                continue
            filemark = match.group(1)
            date = match.group(2)
            description = match.group(3).strip()
            if "Date Filed" in description or "Page" in description:
                continue
            entries.append(
                {
                    "filemark": filemark,
                    "date": date,
                    "description": description,
                }
            )
    return entries


def _build_match_key(description: str) -> str:
    tokens = re.findall(r"[a-z0-9']+", description.lower())
    tokens = [token for token in tokens if token not in STOP_WORDS]
    if len(tokens) < 3:
        return ""
    return " ".join(tokens[:6])


def _detect_filemark(text: str) -> str | None:
    match = FILEMARK_STAMP.search(text)
    if match:
        return match.group(1)
    for line in text.splitlines():
        match = DOCKET_LINE.match(line)
        if match:
            return match.group(1)
    return None


def _compute_ranges(flat: List[dict], total_pages: int) -> None:
    ordered = [
        (idx, entry["start_page"])
        for idx, entry in enumerate(flat)
        if entry["start_page"] is not None
    ]
    ordered.sort(key=lambda item: (item[1], item[0]))

    next_by_index: Dict[int, int] = {}
    for pos, (idx, start_page) in enumerate(ordered):
        next_page = total_pages
        for next_idx, next_start in ordered[pos + 1 :]:
            if next_start > start_page:
                next_page = next_start - 1
                break
        next_by_index[idx] = max(start_page, next_page)

    for idx, entry in enumerate(flat):
        start_page = entry["start_page"]
        if start_page is None:
            entry["end_page"] = None
            continue
        entry["end_page"] = next_by_index.get(idx, total_pages)


def _build_entries(flat: List[dict]) -> List[BookmarkEntry]:
    results: List[BookmarkEntry] = []
    for idx, item in enumerate(flat, start=1):
        title = item["title"]
        date, substance = _extract_date(title)
        filer, score, _signals = _score_filer(title)
        start_page = item["start_page"]
        end_page = item.get("end_page")
        results.append(
            BookmarkEntry(
                index=idx,
                depth=item["depth"],
                title=title,
                start_page=start_page,
                end_page=end_page,
                has_children=bool(item.get("children")),
                filer=filer,
                date=date,
                substance=substance,
                score=score,
                mapped_by="title" if filer != "unknown" else "unknown",
            )
        )
    return results


def _apply_manual_overrides(
    entries: List[BookmarkEntry],
    overrides: Dict[int, Tuple[str, str]],
    min_score: int,
) -> Tuple[List[BookmarkEntry], List[dict]]:
    if not overrides:
        return entries, []
    applied: List[dict] = []
    for entry in entries:
        if entry.index not in overrides:
            continue
        new_filer, raw_value = overrides[entry.index]
        old_filer = entry.filer
        if new_filer == old_filer:
            continue
        entry.filer = new_filer
        entry.mapped_by = "manual"
        entry.score = max(entry.score, min_score)
        applied.append(
            {
                "index": entry.index,
                "start_page": entry.start_page,
                "end_page": entry.end_page or "",
                "old_filer": old_filer,
                "new_filer": new_filer,
                "mapped_by": entry.mapped_by,
                "match_value": raw_value,
            }
        )
    return entries, applied


def _resolve_unknowns_by_context(
    entries: List[BookmarkEntry],
    pages: Dict[int, str],
    docket_map: Dict[str, str],
    docket_entries: List[dict],
    context_pages: int,
    min_score: int,
) -> Tuple[List[BookmarkEntry], List[dict]]:
    if not pages:
        return entries, []

    prepared_entries = []
    for entry in docket_entries:
        filer = docket_map.get(entry["filemark"])
        if not filer:
            continue
        key = _build_match_key(entry["description"])
        if not key:
            continue
        prepared_entries.append(
            {
                "filemark": entry["filemark"],
                "filer": filer,
                "key": key,
                "key_len": len(key),
            }
        )
    prepared_entries.sort(key=lambda item: item["key_len"], reverse=True)

    overrides: List[dict] = []
    for entry in entries:
        if entry.filer != "unknown":
            continue
        if not entry.start_page:
            continue
        end_page = entry.end_page or entry.start_page
        last_page = min(end_page, entry.start_page + max(context_pages, 1) - 1)
        sample_parts = []
        for page_num in range(entry.start_page, last_page + 1):
            text = pages.get(page_num)
            if text:
                sample_parts.append(text)
        if not sample_parts:
            continue
        sample_text = "\n".join(sample_parts)

        filemark = _detect_filemark(sample_text)
        if filemark and filemark in docket_map:
            old_filer = entry.filer
            entry.filer = docket_map[filemark]
            entry.mapped_by = "filemark"
            entry.score = max(entry.score, min_score)
            overrides.append(
                {
                    "index": entry.index,
                    "start_page": entry.start_page,
                    "end_page": entry.end_page or "",
                    "old_filer": old_filer,
                    "new_filer": entry.filer,
                    "mapped_by": entry.mapped_by,
                    "match_value": filemark,
                }
            )
            continue

        normalized = _normalize_for_match(sample_text)[:6000]
        matched = False
        for docket_entry in prepared_entries:
            if docket_entry["key"] in normalized:
                old_filer = entry.filer
                entry.filer = docket_entry["filer"]
                entry.mapped_by = "docket_description"
                entry.score = max(entry.score, min_score - 1)
                overrides.append(
                    {
                        "index": entry.index,
                        "start_page": entry.start_page,
                        "end_page": entry.end_page or "",
                        "old_filer": old_filer,
                        "new_filer": entry.filer,
                        "mapped_by": entry.mapped_by,
                        "match_value": docket_entry["filemark"],
                    }
                )
                matched = True
                break
        if matched:
            continue

        filer, score, signals = _score_filer(sample_text)
        if filer != "unknown" and score >= min_score:
            old_filer = entry.filer
            entry.filer = filer
            entry.mapped_by = "text"
            entry.score = max(entry.score, score)
            overrides.append(
                {
                    "index": entry.index,
                    "start_page": entry.start_page,
                    "end_page": entry.end_page or "",
                    "old_filer": old_filer,
                    "new_filer": entry.filer,
                    "mapped_by": entry.mapped_by,
                    "match_value": signals[0] if signals else f"score={score}",
                }
            )
    return entries, overrides


def _write_csv(path: Path, rows: List[BookmarkEntry], leaf_only: bool) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "index",
                "depth",
                "title",
                "date",
                "filer",
                "substance",
                "start_page",
                "end_page",
                "page_count",
                "has_children",
                "score",
                "mapped_by",
            ]
        )
        for row in rows:
            if leaf_only and row.has_children:
                continue
            if row.start_page and row.end_page:
                page_count = row.end_page - row.start_page + 1
            else:
                page_count = ""
            writer.writerow(
                [
                    row.index,
                    row.depth,
                    row.title,
                    row.date or "",
                    row.filer,
                    row.substance,
                    row.start_page or "",
                    row.end_page or "",
                    page_count,
                    "yes" if row.has_children else "no",
                    row.score,
                    row.mapped_by,
                ]
            )


def _write_summary(path: Path, rows: List[BookmarkEntry], leaf_only: bool) -> None:
    counts = Counter(row.filer for row in rows if not leaf_only or not row.has_children)
    total = sum(counts.values())
    lines = [
        "# 28B Bookmark Filer Summary",
        "",
        f"Total bookmarks: {total}",
        "",
        "## Counts by filer",
        "",
    ]
    for filer in FILER_PRIORITY:
        if filer in counts:
            lines.append(f"- {filer}: {counts[filer]}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _apply_parent_child_mapping(entries: List[BookmarkEntry]) -> List[BookmarkEntry]:
    parent_by_idx: List[int | None] = []
    stack: List[Tuple[int, int]] = []
    for idx, entry in enumerate(entries):
        depth = entry.depth
        while stack and stack[-1][0] >= depth:
            stack.pop()
        parent_idx = stack[-1][1] if stack else None
        parent_by_idx.append(parent_idx)
        stack.append((depth, idx))

    children_map: Dict[int, List[int]] = defaultdict(list)
    for idx, parent_idx in enumerate(parent_by_idx):
        if parent_idx is not None:
            children_map[parent_idx].append(idx)

    min_parent_score = 2
    for idx, entry in enumerate(entries):
        if entry.filer != "unknown":
            continue
        parent_idx = parent_by_idx[idx]
        if parent_idx is None:
            continue
        parent = entries[parent_idx]
        if parent.filer != "unknown" and parent.score >= min_parent_score:
            entry.filer = parent.filer
            entry.mapped_by = "parent"
            entry.score = max(entry.score, parent.score - 1)

    for idx in range(len(entries) - 1, -1, -1):
        entry = entries[idx]
        if entry.filer != "unknown" and entry.score >= 2:
            continue
        child_idxs = children_map.get(idx, [])
        if not child_idxs:
            continue
        counts = Counter(
            entries[child_idx].filer
            for child_idx in child_idxs
            if entries[child_idx].filer != "unknown"
        )
        if not counts:
            continue
        top_filer, top_count = counts.most_common(1)[0]
        total_known = sum(counts.values())
        if top_count >= 2 and (top_count / total_known) >= 0.6:
            entry.filer = top_filer
            entry.mapped_by = "children"
            entry.score = max(entry.score, 2)

    for idx, entry in enumerate(entries):
        if entry.filer != "unknown":
            continue
        parent_idx = parent_by_idx[idx]
        if parent_idx is None:
            continue
        parent = entries[parent_idx]
        if parent.filer != "unknown" and parent.score >= min_parent_score:
            entry.filer = parent.filer
            entry.mapped_by = "parent"
            entry.score = max(entry.score, parent.score - 1)

    return entries


def _write_overrides(path: Path, overrides: List[dict]) -> None:
    if not overrides:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "index",
                "start_page",
                "end_page",
                "old_filer",
                "new_filer",
                "mapped_by",
                "match_value",
            ]
        )
        for row in overrides:
            writer.writerow(
                [
                    row.get("index", ""),
                    row.get("start_page", ""),
                    row.get("end_page", ""),
                    row.get("old_filer", ""),
                    row.get("new_filer", ""),
                    row.get("mapped_by", ""),
                    row.get("match_value", ""),
                ]
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse PDF bookmarks into filer/date/page ranges.")
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path("CASE DOCS/28B.pdf"),
        help="Path to the PDF with bookmarks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory to write outputs.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional page-level JSON to resolve unknown filers by context.",
    )
    parser.add_argument(
        "--docket-filer-map",
        type=Path,
        default=Path("reports/28b_docket_filer_map.csv"),
        help="Optional docket filemark-to-filer CSV for context matching.",
    )
    parser.add_argument(
        "--context-pages",
        type=int,
        default=2,
        help="Number of pages to sample for context matching.",
    )
    parser.add_argument(
        "--min-context-score",
        type=int,
        default=3,
        help="Minimum score for text-based filer resolution.",
    )
    parser.add_argument(
        "--manual-overrides",
        type=Path,
        default=None,
        help="Optional CSV with manual filer labels (e.g., 28b_bookmark_unknown.csv).",
    )
    parser.add_argument(
        "--manual-score",
        type=int,
        default=6,
        help="Score to assign to manual overrides.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    pdf_path = args.pdf.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reader = PyPDF2.PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    try:
        outlines = reader.outline
    except Exception:
        outlines = reader.outlines

    nodes = _parse_outline(reader, outlines)
    flat: List[dict] = []
    _flatten(nodes, flat)
    _compute_ranges(flat, total_pages)
    entries = _build_entries(flat)
    entries = _apply_parent_child_mapping(entries)
    overrides: List[dict] = []

    if args.json:
        json_path = args.json.expanduser().resolve()
        if json_path.exists():
            pages = _load_page_text(json_path)
            docket_map = _load_docket_filer_map(args.docket_filer_map.expanduser().resolve())
            docket_entries = _extract_docket_entries(pages)
            entries, overrides = _resolve_unknowns_by_context(
                entries,
                pages,
                docket_map,
                docket_entries,
                args.context_pages,
                args.min_context_score,
            )
            entries = _apply_parent_child_mapping(entries)

    manual_overrides = _load_manual_overrides(args.manual_overrides.expanduser().resolve()) if args.manual_overrides else {}
    manual_applied: List[dict] = []
    if manual_overrides:
        entries, manual_applied = _apply_manual_overrides(entries, manual_overrides, args.manual_score)
        entries = _apply_parent_child_mapping(entries)

    _write_csv(output_dir / "28b_bookmark_index.csv", entries, leaf_only=False)
    _write_csv(output_dir / "28b_bookmark_documents.csv", entries, leaf_only=True)
    _write_summary(output_dir / "28b_bookmark_filer_summary.md", entries, leaf_only=False)
    _write_summary(output_dir / "28b_bookmark_filer_summary_leaf.md", entries, leaf_only=True)
    if overrides:
        _write_overrides(output_dir / "28b_bookmark_context_overrides.csv", overrides)
    if manual_applied:
        _write_overrides(output_dir / "28b_bookmark_manual_overrides.csv", manual_applied)

    print(f"Wrote bookmark reports to {output_dir}")


if __name__ == "__main__":
    main()
