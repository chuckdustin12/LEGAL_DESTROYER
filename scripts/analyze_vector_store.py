"""Generate legal-task reports from a vector store."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


ISSUES = {
    "recusal": [
        "recusal",
        "motion to recuse",
        "notice of recusal",
        "recuse judge",
        "disqualify judge",
    ],
    "emergency relief denied": [
        "emergency relief denied",
        "motion for emergency relief denied",
        "petition for writ of mandamus denied",
        "mandamus denied",
    ],
    "temporary orders": [
        "temporary orders",
        "temporary order",
        "temporary injunction",
        "ex parte order",
    ],
    "notice of hearing": [
        "notice of hearing",
        "hearing notice",
        "notice of trial setting",
        "hearing set",
    ],
}

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

RE_CASE_V = re.compile(
    r"\b[A-Z][A-Za-z0-9.&'\- ]{1,50}\s+v\.?\s+[A-Z][A-Za-z0-9.&'\- ]{1,50}\b"
)
RE_IN_RE = re.compile(r"\bIn re\s+[A-Z][A-Za-z0-9.&'\- ]{1,50}\b")
RE_DOCKET = re.compile(r"\b\d{2}-\d{2}-\d{5}(?:-CV)?\b")
RE_TRIAL_NO = re.compile(r"\b\d{3}-\d{6}-\d{2}\b")
RE_RULE = re.compile(
    r"\b(?:Tex\.?\s+R\.?\s+(?:Civ|App)\.?\s+P\.?|TRCP|Rule)\s*\d+(?:\.\d+)?\b",
    re.IGNORECASE,
)
RE_USC = re.compile(r"\b\d+\s*U\.S\.C\.?\s*§+\s*\d+[A-Za-z0-9.-]*\b")
RE_TEX_CODE = re.compile(
    r"\bTex\.?\s+(?:Fam|Gov|Penal|Civ\.?\s+Prac\.?\s+&\s+Rem\.?|Code\s+Crim\.?\s+"
    r"Proc\.?)\s+Code\b[^\n]{0,60}",
    re.IGNORECASE,
)


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _normalize_ascii(text: str) -> str:
    cleaned = text.translate(NON_ASCII_MAP)
    return cleaned.encode("ascii", "ignore").decode("ascii")


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _snippet(text: str, max_len: int = 400) -> str:
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


def _extract_dates(text: str) -> List[Tuple[str, datetime | None, Tuple[int, int]]]:
    results: List[Tuple[str, datetime | None, Tuple[int, int]]] = []
    for pattern in DATE_PATTERNS:
        for match in pattern.finditer(text):
            date_text = match.group(0)
            parsed = _parse_date(date_text)
            results.append((date_text, parsed, (match.start(), match.end())))
    return results


def _load_store(store_dir: Path) -> Tuple[np.ndarray, List[dict]]:
    embeddings = np.load(store_dir / "embeddings.npy")
    metadata_path = store_dir / "metadata.jsonl"
    records = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]
    return embeddings, records


def _issue_search(
    embeddings: np.ndarray,
    records: List[dict],
    model: SentenceTransformer,
    issues: Dict[str, List[str]],
    top_k: int,
) -> Dict[str, List[Tuple[float, dict]]]:
    results: Dict[str, List[Tuple[float, dict]]] = {}
    for issue, queries in issues.items():
        query_embeddings = model.encode(queries, normalize_embeddings=True)
        scores = embeddings @ query_embeddings.T
        best_scores = scores.max(axis=1)
        ranked = np.argsort(-best_scores)
        seen = set()
        picks: List[Tuple[float, dict]] = []
        for idx in ranked:
            text = records[idx]["text"]
            if text in seen:
                continue
            seen.add(text)
            picks.append((float(best_scores[idx]), records[idx]))
            if len(picks) >= top_k:
                break
        results[issue] = picks
    return results


def _issue_tags_for_text(text: str, issues: Dict[str, List[str]]) -> List[str]:
    lowered = text.lower()
    tags = []
    for issue, queries in issues.items():
        if any(query.lower() in lowered for query in queries):
            tags.append(issue)
    return tags


def _write_issue_matrix(
    output_path: Path,
    issue_hits: Dict[str, List[Tuple[float, dict]]],
    store_dir: Path,
    top_k: int,
) -> None:
    lines = [
        "# 28B Issue Evidence Matrix",
        "",
        f"Generated: {_timestamp()}",
        f"Store: {store_dir}",
        f"Top hits per issue: {top_k}",
        "",
    ]
    for issue, hits in issue_hits.items():
        lines.append(f"## Issue: {issue}")
        if not hits:
            lines.append("- No matches found.")
            lines.append("")
            continue
        for score, record in hits:
            snippet = _snippet(record["text"], 500)
            lines.append(
                f"- Score {score:.4f} | vector_id {record['vector_id']} | "
                f"chunk {record['chunk_index']} | id {record['id']}"
            )
            lines.append(f"  Quote: \"{snippet}\"")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_timeline(
    output_path: Path,
    records: List[dict],
    issue_hits: Dict[str, List[Tuple[float, dict]]],
    max_events_per_issue: int,
    scope: str,
    max_dates_per_chunk: int,
) -> None:
    events: List[dict] = []
    seen = set()
    if scope == "issues":
        for issue, hits in issue_hits.items():
            for _, record in hits[:max_events_per_issue]:
                text = record["text"]
                dates = _extract_dates(text)
                if max_dates_per_chunk > 0:
                    dates = dates[:max_dates_per_chunk]
                for date_text, parsed, span in dates:
                    if parsed is None:
                        continue
                    key = (issue, date_text, record["vector_id"])
                    if key in seen:
                        continue
                    seen.add(key)
                    start = max(0, span[0] - 80)
                    end = min(len(text), span[1] + 120)
                    snippet = _snippet(text[start:end], 240)
                    events.append(
                        {
                            "issue_tags": [issue],
                            "date_text": date_text,
                            "parsed": parsed,
                            "vector_id": record["vector_id"],
                            "chunk_index": record["chunk_index"],
                            "snippet": snippet,
                        }
                    )
    else:
        for record in records:
            text = record["text"]
            tags = _issue_tags_for_text(text, ISSUES)
            dates = _extract_dates(text)
            if max_dates_per_chunk > 0:
                dates = dates[:max_dates_per_chunk]
            for date_text, parsed, span in dates:
                if parsed is None:
                    continue
                key = (date_text, record["vector_id"])
                if key in seen:
                    continue
                seen.add(key)
                start = max(0, span[0] - 80)
                end = min(len(text), span[1] + 120)
                snippet = _snippet(text[start:end], 240)
                events.append(
                    {
                        "issue_tags": tags,
                        "date_text": date_text,
                        "parsed": parsed,
                        "vector_id": record["vector_id"],
                        "chunk_index": record["chunk_index"],
                        "snippet": snippet,
                    }
                )
    events.sort(
        key=lambda item: (
            item["parsed"] is None,
            item["parsed"] or datetime.max,
            item["date_text"],
        )
    )
    scope_label = "entire document" if scope == "all" else "top issue hits"
    lines = [
        f"# 28B Timeline ({scope_label})",
        "",
        f"Generated: {_timestamp()}",
        "Note: Timeline includes only parsed dates between 1800 and 2100.",
        "",
    ]
    if not events:
        lines.append("No parsed dates found in the selected scope.")
    for event in events:
        if event["parsed"]:
            date_label = event["parsed"].strftime("%Y-%m-%d")
        else:
            date_label = "unknown"
        tags = event["issue_tags"]
        issue_label = "none" if not tags else ", ".join(tags)
        lines.append(
            f"- {date_label} | \"{_normalize_ascii(event['date_text'])}\" | "
            f"issues: {issue_label} | vector_id {event['vector_id']} | "
            f"chunk {event['chunk_index']}"
        )
        lines.append(f"  Snippet: {event['snippet']}")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _collect_citations(records: List[dict]) -> Dict[str, Dict[str, List[int]]]:
    buckets = {
        "docket_numbers": {},
        "trial_numbers": {},
        "case_captions": {},
        "rules": {},
        "statutes": {},
    }
    for record in records:
        text = record["text"]
        vector_id = record["vector_id"]

        def _add(bucket: Dict[str, List[int]], match: str) -> None:
            normalized = _normalize_ascii(_normalize_ws(match))
            if not normalized:
                return
            bucket.setdefault(normalized, [])
            if vector_id not in bucket[normalized]:
                bucket[normalized].append(vector_id)

        for match in RE_DOCKET.findall(text):
            _add(buckets["docket_numbers"], match)
        for match in RE_TRIAL_NO.findall(text):
            _add(buckets["trial_numbers"], match)
        for match in RE_CASE_V.findall(text):
            _add(buckets["case_captions"], match)
        for match in RE_IN_RE.findall(text):
            _add(buckets["case_captions"], match)
        for match in RE_RULE.findall(text):
            _add(buckets["rules"], match)
        for match in RE_USC.findall(text):
            _add(buckets["statutes"], match)
        for match in RE_TEX_CODE.findall(text):
            _add(buckets["statutes"], match)
    return buckets


def _write_citations(output_path: Path, citations: Dict[str, Dict[str, List[int]]]) -> None:
    lines = [
        "# 28B Citation Spotting",
        "",
        f"Generated: {_timestamp()}",
        "",
    ]

    def _write_section(title: str, bucket: Dict[str, List[int]], max_items: int = 100) -> None:
        lines.append(f"## {title}")
        if not bucket:
            lines.append("- None found.")
            lines.append("")
            return
        items = sorted(bucket.items(), key=lambda item: (-len(item[1]), item[0]))
        for citation, ids in items[:max_items]:
            sample_ids = ", ".join(str(i) for i in ids[:5])
            lines.append(f"- {citation} (count: {len(ids)}; vector_ids: {sample_ids})")
        lines.append("")

    _write_section("Docket Numbers", citations["docket_numbers"])
    _write_section("Trial Court Numbers", citations["trial_numbers"])
    _write_section("Case Captions", citations["case_captions"])
    _write_section("Rule References", citations["rules"])
    _write_section("Statutes and Codes", citations["statutes"])

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _find_duplicate_clusters(
    embeddings: np.ndarray, threshold: float
) -> List[List[int]]:
    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims, 0.0)
    pairs = np.argwhere(sims >= threshold)

    parent = list(range(len(embeddings)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in pairs:
        if i >= j:
            continue
        union(int(i), int(j))

    clusters: Dict[int, List[int]] = {}
    for idx in range(len(embeddings)):
        root = find(idx)
        clusters.setdefault(root, []).append(idx)

    result = [members for members in clusters.values() if len(members) > 1]
    result.sort(key=len, reverse=True)
    return result


def _write_consistency(
    output_path: Path,
    records: List[dict],
    embeddings: np.ndarray,
    threshold: float,
    max_clusters: int,
    max_members: int,
) -> None:
    clusters = _find_duplicate_clusters(embeddings, threshold)
    lines = [
        "# 28B Consistency Check (Near-Duplicate Clusters)",
        "",
        f"Generated: {_timestamp()}",
        f"Similarity threshold: {threshold}",
        "",
    ]
    if not clusters:
        lines.append("No near-duplicate clusters found at this threshold.")
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return

    for idx, cluster in enumerate(clusters[:max_clusters], start=1):
        lines.append(f"## Cluster {idx} (size {len(cluster)})")
        for member in cluster[:max_members]:
            record = records[member]
            lines.append(
                f"- vector_id {record['vector_id']} | chunk {record['chunk_index']} | "
                f"{_snippet(record['text'], 200)}"
            )
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_scaffolds(
    output_path: Path,
    issue_hits: Dict[str, List[Tuple[float, dict]]],
) -> None:
    lines = [
        "# 28B Draft Scaffolds (Issue-Based)",
        "",
        f"Generated: {_timestamp()}",
        "",
        "Use these as placeholders for drafting. Attach supporting excerpts and citations as needed.",
        "",
    ]
    for issue, hits in issue_hits.items():
        lines.append(f"## {issue.title()}")
        lines.append("### Goal")
        lines.append("- [Insert requested relief or objective]")
        lines.append("")
        lines.append("### Key Facts / Excerpts")
        if hits:
            for score, record in hits[:3]:
                lines.append(
                    f"- Score {score:.4f} | vector_id {record['vector_id']} | chunk {record['chunk_index']}"
                )
                lines.append(f"  Quote: \"{_snippet(record['text'], 400)}\"")
        else:
            lines.append("- [No excerpts found in top hits]")
        lines.append("")
        lines.append("### Legal Standard")
        lines.append("- [Insert controlling standard and supporting authorities]")
        lines.append("")
        lines.append("### Argument")
        lines.append("- [Point-by-point argument tied to excerpts]")
        lines.append("")
        lines.append("### Requested Relief")
        lines.append("- [Insert requested order or relief]")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a vector store for legal-task outputs.")
    parser.add_argument(
        "--store",
        type=Path,
        required=True,
        help="Vector store directory (contains embeddings.npy + metadata.jsonl).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory to write report files.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top hits per issue.")
    parser.add_argument(
        "--label",
        type=str,
        default="28b",
        help="Label prefix for report filenames.",
    )
    parser.add_argument(
        "--timeline-events",
        type=int,
        default=5,
        help="Max records per issue to scan for timeline dates.",
    )
    parser.add_argument(
        "--timeline-scope",
        choices=["issues", "all"],
        default="all",
        help="Scope for timeline generation.",
    )
    parser.add_argument(
        "--timeline-max-dates-per-chunk",
        type=int,
        default=0,
        help="Limit dates per chunk (0 = no limit).",
    )
    parser.add_argument(
        "--dup-threshold",
        type=float,
        default=0.985,
        help="Cosine similarity threshold for near-duplicate clusters.",
    )
    parser.add_argument(
        "--dup-clusters",
        type=int,
        default=20,
        help="Max duplicate clusters to write.",
    )
    parser.add_argument(
        "--dup-members",
        type=int,
        default=5,
        help="Max members per duplicate cluster to show.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    store_dir = args.store.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings, records = _load_store(store_dir)
    model = SentenceTransformer(args.model)

    issue_hits = _issue_search(embeddings, records, model, ISSUES, args.top_k)

    _write_issue_matrix(
        output_dir / f"{args.label}_issue_evidence_matrix.md",
        issue_hits,
        store_dir,
        args.top_k,
    )
    _write_timeline(
        output_dir / f"{args.label}_timeline.md",
        records,
        issue_hits,
        args.timeline_events,
        args.timeline_scope,
        args.timeline_max_dates_per_chunk,
    )
    citations = _collect_citations(records)
    _write_citations(output_dir / f"{args.label}_citations.md", citations)
    _write_consistency(
        output_dir / f"{args.label}_consistency_clusters.md",
        records,
        embeddings,
        args.dup_threshold,
        args.dup_clusters,
        args.dup_members,
    )
    _write_scaffolds(output_dir / f"{args.label}_draft_scaffolds.md", issue_hits)

    print(f"Wrote reports to {output_dir}")


if __name__ == "__main__":
    main()
