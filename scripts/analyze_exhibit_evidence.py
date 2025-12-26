"""Analyze OCRed exhibits for evidentiary markers and potential contradictions."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


NEGATION_TERMS = [
    "no ",
    "not ",
    "never",
    "denied",
    "without",
    "lack",
    "failed",
    "refused",
    "cannot",
    "void",
]

AFFIRM_TERMS = [
    "granted",
    "ordered",
    "finds",
    "concludes",
    "determines",
    "approved",
    "sustained",
    "affirmed",
]

SHIFT_KEYWORDS = {
    "protective order / family violence": [
        r"protective order",
        r"family violence",
        r"domestic violence",
        r"\bassault\b",
    ],
    "agreement / settlement": [
        r"\bagreement\b",
        r"agreed order",
        r"\bsettlement\b",
        r"mediated settlement",
        r"rule 11 agreement",
        r"\bstipulation\b",
    ],
    "child abuse / neglect": [
        r"child abuse",
        r"child neglect",
        r"abuse of a child",
        r"child endangerment",
        r"injury to a child",
        r"sexual abuse",
        r"physical abuse",
    ],
}

EVIDENCE_MARKERS = {
    "affidavit_or_sworn": [
        r"\baffidavit\b",
        r"\bsworn\b",
        r"under penalty of perjury",
        r"\bdeclaration\b",
        r"\bverified\b",
        r"subscribed and sworn",
        r"\bjurat\b",
    ],
    "notary_or_seal": [
        r"\bnotary\b",
        r"\bnotarized\b",
        r"\bseal\b",
        r"commission expires",
    ],
    "signature_block": [
        r"/s/",
        r"\bsignature\b",
        r"\bsigned\b",
        r"sign here",
    ],
    "authentication": [
        r"certified copy",
        r"true and correct copy",
        r"business records",
        r"custodian of records",
        r"authenticated",
    ],
}

SCREENSHOT_MARKERS = [
    r"text message",
    r"\bsms\b",
    r"call log",
    r"screenshot",
    r"\bphoto\b",
    r"\bimage\b",
    r"\bfacebook\b",
    r"\binstagram\b",
    r"\bgmail\b",
    r"\bemail\b",
    r"\bappclose\b",
    r"\bmessage\b",
]

FILING_INCLUDE = re.compile(
    r"(petition|application for protective order|statement of inability)",
    re.IGNORECASE,
)
FILING_EXCLUDE = re.compile(
    r"(respondent|response|objection|temporary orders|ex parte|exhibit)",
    re.IGNORECASE,
)


@dataclass
class ExhibitSummary:
    label: str
    source: str
    pages: int
    char_count: int
    topic_counts: Dict[str, int]
    marker_counts: Dict[str, int]
    screenshot_hits: int


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _snippet(text: str, max_len: int = 260) -> str:
    cleaned = _normalize_ws(text)
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3].rstrip() + "..."


def _snippet_around_match(text: str, patterns: List[re.Pattern], max_len: int = 260) -> str:
    cleaned = _normalize_ws(text)
    for pattern in patterns:
        match = pattern.search(cleaned)
        if match:
            start = max(0, match.start() - max_len // 2)
            end = min(len(cleaned), start + max_len)
            snippet = cleaned[start:end].strip()
            if start > 0:
                snippet = "..." + snippet
            if end < len(cleaned):
                snippet = snippet + "..."
            return snippet
    return _snippet(cleaned, max_len)


def _load_exhibit_text(exhibit_dir: Path) -> Tuple[str, int, str]:
    json_path = exhibit_dir / f"{exhibit_dir.name}.json"
    text_path = exhibit_dir / f"{exhibit_dir.name}.txt"
    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        pages = [page.get("text", "") for page in payload.get("pages", [])]
        return "\n".join(pages), len(pages), payload.get("source", exhibit_dir.name)
    if text_path.exists():
        text = text_path.read_text(encoding="utf-8")
        return text, text.count("\n\n") + 1, exhibit_dir.name
    return "", 0, exhibit_dir.name


def _count_matches(text: str, patterns: List[re.Pattern]) -> int:
    total = 0
    for pattern in patterns:
        total += len(pattern.findall(text))
    return total


def _compile_patterns(patterns: Dict[str, List[str]]) -> Dict[str, List[re.Pattern]]:
    return {key: [re.compile(p, re.IGNORECASE) for p in values] for key, values in patterns.items()}


def _collect_exhibits(root: Path) -> List[ExhibitSummary]:
    compiled_topics = _compile_patterns(SHIFT_KEYWORDS)
    compiled_markers = _compile_patterns(EVIDENCE_MARKERS)
    screenshot_patterns = [re.compile(p, re.IGNORECASE) for p in SCREENSHOT_MARKERS]

    summaries: List[ExhibitSummary] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if "exhibit" not in path.name.lower():
            continue
        text, pages, source = _load_exhibit_text(path)
        text_lower = text.lower()
        topic_counts = {
            topic: _count_matches(text_lower, patterns)
            for topic, patterns in compiled_topics.items()
        }
        marker_counts = {
            marker: _count_matches(text_lower, patterns)
            for marker, patterns in compiled_markers.items()
        }
        screenshot_hits = _count_matches(text_lower, screenshot_patterns)
        summaries.append(
            ExhibitSummary(
                label=path.name,
                source=source,
                pages=pages,
                char_count=len(text),
                topic_counts=topic_counts,
                marker_counts=marker_counts,
                screenshot_hits=screenshot_hits,
            )
        )
    return summaries


def _polarity_sign(text: str, min_hits: int) -> int:
    text_lower = text.lower()
    pos = sum(text_lower.count(term) for term in AFFIRM_TERMS)
    neg = sum(text_lower.count(term) for term in NEGATION_TERMS)
    score = pos - neg
    if score >= min_hits:
        return 1
    if score <= -min_hits:
        return -1
    return 0


def _load_store(store_dir: Path) -> Tuple[np.ndarray, List[dict]]:
    embeddings = np.load(store_dir / "embeddings.npy")
    records = [
        json.loads(line)
        for line in (store_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return embeddings, records


def _find_contradictions(
    embeddings: np.ndarray,
    records: List[dict],
    *,
    similarity_threshold: float,
    min_polarity_hits: int,
    top_k: int,
) -> List[dict]:
    exhibit_indices = [
        idx for idx, record in enumerate(records) if "exhibit" in str(record.get("source_pdf", "")).lower()
    ]
    filing_indices = [
        idx
        for idx, record in enumerate(records)
        if FILING_INCLUDE.search(str(record.get("source_pdf", "")))
        and not FILING_EXCLUDE.search(str(record.get("source_pdf", "")))
    ]
    if not exhibit_indices or not filing_indices:
        return []

    exhibit_vectors = embeddings[exhibit_indices]
    filing_vectors = embeddings[filing_indices]
    sims = exhibit_vectors @ filing_vectors.T

    contradictions: List[dict] = []
    for i, ex_idx in enumerate(exhibit_indices):
        ex_record = records[ex_idx]
        ex_text = ex_record.get("text", "")
        ex_sign = _polarity_sign(ex_text, min_polarity_hits)
        if ex_sign == 0:
            continue
        scores = sims[i]
        ranked = np.argsort(-scores)[:top_k]
        for pos in ranked:
            fil_idx = filing_indices[int(pos)]
            fil_record = records[fil_idx]
            fil_text = fil_record.get("text", "")
            fil_sign = _polarity_sign(fil_text, min_polarity_hits)
            if fil_sign == 0 or ex_sign * fil_sign >= 0:
                continue
            score = float(scores[int(pos)])
            if score < similarity_threshold:
                continue
            contradictions.append(
                {
                    "similarity": round(score, 4),
                    "exhibit_source": ex_record.get("source_pdf", ""),
                    "filing_source": fil_record.get("source_pdf", ""),
                    "exhibit_chunk": ex_record.get("chunk_index"),
                    "filing_chunk": fil_record.get("chunk_index"),
                    "exhibit_snippet": _snippet(ex_text, 260),
                    "filing_snippet": _snippet(fil_text, 260),
                    "exhibit_polarity": ex_sign,
                    "filing_polarity": fil_sign,
                }
            )
    contradictions.sort(key=lambda item: item["similarity"], reverse=True)
    return contradictions


def _find_overlap(
    embeddings: np.ndarray,
    records: List[dict],
    *,
    similarity_threshold: float,
    top_k: int,
) -> List[dict]:
    exhibit_indices = [
        idx for idx, record in enumerate(records) if "exhibit" in str(record.get("source_pdf", "")).lower()
    ]
    filing_indices = [
        idx
        for idx, record in enumerate(records)
        if FILING_INCLUDE.search(str(record.get("source_pdf", "")))
        and not FILING_EXCLUDE.search(str(record.get("source_pdf", "")))
    ]
    if not exhibit_indices or not filing_indices:
        return []

    exhibit_vectors = embeddings[exhibit_indices]
    filing_vectors = embeddings[filing_indices]
    sims = exhibit_vectors @ filing_vectors.T

    overlaps: List[dict] = []
    for i, ex_idx in enumerate(exhibit_indices):
        scores = sims[i]
        ranked = np.argsort(-scores)[:top_k]
        for pos in ranked:
            score = float(scores[int(pos)])
            if score < similarity_threshold:
                continue
            fil_idx = filing_indices[int(pos)]
            ex_record = records[ex_idx]
            fil_record = records[fil_idx]
            overlaps.append(
                {
                    "similarity": round(score, 4),
                    "exhibit_source": ex_record.get("source_pdf", ""),
                    "filing_source": fil_record.get("source_pdf", ""),
                    "exhibit_chunk": ex_record.get("chunk_index"),
                    "filing_chunk": fil_record.get("chunk_index"),
                    "exhibit_snippet": _snippet(ex_record.get("text", ""), 220),
                    "filing_snippet": _snippet(fil_record.get("text", ""), 220),
                }
            )
    overlaps.sort(key=lambda item: item["similarity"], reverse=True)
    return overlaps


def _write_contradictions_csv(path: Path, rows: List[dict]) -> None:
    header = [
        "similarity",
        "exhibit_source",
        "filing_source",
        "exhibit_chunk",
        "filing_chunk",
        "exhibit_polarity",
        "filing_polarity",
        "exhibit_snippet",
        "filing_snippet",
    ]
    lines = [",".join(header)]
    for row in rows:
        line = [
            str(row.get("similarity", "")),
            str(row.get("exhibit_source", "")),
            str(row.get("filing_source", "")),
            str(row.get("exhibit_chunk", "")),
            str(row.get("filing_chunk", "")),
            str(row.get("exhibit_polarity", "")),
            str(row.get("filing_polarity", "")),
            str(row.get("exhibit_snippet", "")).replace(",", " "),
            str(row.get("filing_snippet", "")).replace(",", " "),
        ]
        lines.append(",".join(line))
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_report(
    output_path: Path,
    exhibits: List[ExhibitSummary],
    overlaps: List[dict],
    contradictions: List[dict],
) -> None:
    lines = [
        "# Exhibit OCR Evidence Review",
        "",
        f"Generated: {_timestamp()}",
        "",
        "This report flags evidentiary markers in OCRed exhibits and highlights potential contradictions",
        "based on semantic similarity with opposite polarity. It does not determine admissibility.",
        "",
        "## Exhibit Summary",
        "",
        "| Exhibit | Pages | Chars | Protective Order | Agreement | Child Abuse | Affidavit/Sworn | Notary/Seal | Signature | Authentication | Screenshot Indicators |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for exhibit in exhibits:
        lines.append(
            "| {label} | {pages} | {chars} | {protective} | {agreement} | {abuse} | {affidavit} | {notary} | {signature} | {auth} | {screens} |".format(
                label=Path(exhibit.source).name,
                pages=exhibit.pages,
                chars=exhibit.char_count,
                protective=exhibit.topic_counts.get("protective order / family violence", 0),
                agreement=exhibit.topic_counts.get("agreement / settlement", 0),
                abuse=exhibit.topic_counts.get("child abuse / neglect", 0),
                affidavit=exhibit.marker_counts.get("affidavit_or_sworn", 0),
                notary=exhibit.marker_counts.get("notary_or_seal", 0),
                signature=exhibit.marker_counts.get("signature_block", 0),
                auth=exhibit.marker_counts.get("authentication", 0),
                screens=exhibit.screenshot_hits,
            )
        )

    lines.extend(
        [
            "",
            "## Exhibit Overlap With Filings (High Similarity)",
            "",
        ]
    )
    if not overlaps:
        lines.append("No high-similarity overlaps found at the current threshold.")
    else:
        for row in overlaps[:25]:
            lines.append(
                f"- Similarity {row['similarity']:.4f} | {Path(row['exhibit_source']).name} vs {Path(row['filing_source']).name}"
            )
            lines.append(f"  Exhibit: \"{row['exhibit_snippet']}\"")
            lines.append(f"  Filing: \"{row['filing_snippet']}\"")

    lines.extend(
        [
            "",
            "## Potential Contradictions (Semantic Similarity + Opposite Polarity)",
            "",
        ]
    )
    if not contradictions:
        lines.append("No contradiction pairs found at the current threshold.")
    else:
        for row in contradictions[:25]:
            lines.append(
                f"- Similarity {row['similarity']:.4f} | {Path(row['exhibit_source']).name} vs {Path(row['filing_source']).name}"
            )
            lines.append(f"  Exhibit: \"{row['exhibit_snippet']}\"")
            lines.append(f"  Filing: \"{row['filing_snippet']}\"")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze OCRed exhibits for evidentiary markers.")
    parser.add_argument(
        "--exhibit-root",
        type=Path,
        default=Path("extracted_text_full/inconsistencies"),
        help="Root directory containing OCRed exhibit outputs.",
    )
    parser.add_argument(
        "--store",
        type=Path,
        default=Path("vector_store_inconsistencies"),
        help="Vector store used for contradiction checks.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/exhibit_evidence_review.md"),
        help="Markdown report output.",
    )
    parser.add_argument(
        "--contradictions-csv",
        type=Path,
        default=Path("reports/exhibit_contradiction_pairs.csv"),
        help="CSV output for contradiction pairs.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.78,
        help="Cosine similarity threshold for contradiction pairs.",
    )
    parser.add_argument(
        "--min-polarity-hits",
        type=int,
        default=2,
        help="Minimum polarity hits to treat a chunk as affirmative/negative.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="Top similar filing chunks to consider per exhibit chunk.",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.82,
        help="Cosine similarity threshold for overlap pairs.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    exhibit_root = args.exhibit_root.expanduser().resolve()
    store_dir = args.store.expanduser().resolve()

    exhibits = _collect_exhibits(exhibit_root)
    if not exhibits:
        raise ValueError(f"No exhibit OCR outputs found under {exhibit_root}")

    embeddings, records = _load_store(store_dir)
    overlaps = _find_overlap(
        embeddings,
        records,
        similarity_threshold=args.overlap_threshold,
        top_k=args.top_k,
    )
    contradictions = _find_contradictions(
        embeddings,
        records,
        similarity_threshold=args.similarity_threshold,
        min_polarity_hits=args.min_polarity_hits,
        top_k=args.top_k,
    )

    _write_contradictions_csv(args.contradictions_csv.expanduser().resolve(), contradictions)
    _write_report(args.report.expanduser().resolve(), exhibits, overlaps, contradictions)

    print(f"Wrote report to {args.report}")


if __name__ == "__main__":
    main()
