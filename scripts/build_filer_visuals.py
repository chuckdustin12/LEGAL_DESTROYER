"""Split pages by filer and regenerate advanced visuals per filer."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from scripts import build_advanced_semantic_visuals as vis


@dataclass
class PageRecord:
    page_number: int
    text: str


FILER_RULES = {
    "charles_dustin_myers": [
        (re.compile(r"/s/\s*Charles\s+Dustin\s+Myers", re.IGNORECASE), 6),
        (re.compile(r"Charles\s+Dustin\s+Myers", re.IGNORECASE), 3),
        (re.compile(r"Charles\s+D\s+Myers", re.IGNORECASE), 3),
        (re.compile(r"chuckdustin12@gmail\.com", re.IGNORECASE), 4),
        (re.compile(r"CSD-legal", re.IGNORECASE), 4),
        (re.compile(r"pro\s+se", re.IGNORECASE), 2),
    ],
    "cooper_carter": [
        (re.compile(r"/s/\s*Cooper\s+L\.?\s+Carter", re.IGNORECASE), 6),
        (re.compile(r"Cooper\s+L\.?\s+Carter", re.IGNORECASE), 4),
        (re.compile(r"Cooper\s+Carter", re.IGNORECASE), 3),
        (re.compile(r"majadmin\.com", re.IGNORECASE), 3),
        (re.compile(r"Max\s+Altman\s*&\s*Johnson", re.IGNORECASE), 2),
    ],
    "morgan_michelle_myers": [
        (re.compile(r"/s/\s*Morgan\s+Michelle\s+Myers", re.IGNORECASE), 6),
        (re.compile(r"Morgan\s+Michelle\s+Myers", re.IGNORECASE), 3),
        (re.compile(r"Morgan\s+Myers", re.IGNORECASE), 2),
    ],
    "court": [
        (re.compile(r"Court\s+of\s+Appeals", re.IGNORECASE), 4),
        (re.compile(r"Supreme\s+Court\s+of\s+Texas", re.IGNORECASE), 4),
        (re.compile(r"Per\s+Curiam", re.IGNORECASE), 4),
        (re.compile(r"MEMORANDUM\s+OPINION", re.IGNORECASE), 4),
        (re.compile(r"\bOPINION\b", re.IGNORECASE), 2),
        (re.compile(r"\bPanel:\b", re.IGNORECASE), 3),
        (re.compile(r"\bJudgment\b", re.IGNORECASE), 2),
        (re.compile(r"\bORDER\b", re.IGNORECASE), 1),
    ],
    "clerk": [
        (re.compile(r"District\s+Clerk", re.IGNORECASE), 4),
        (re.compile(r"Clerk's\s+Office", re.IGNORECASE), 4),
        (re.compile(r"ALL\s+TRANSACTIONS\s+FOR\s+A\s+CASE", re.IGNORECASE), 6),
        (re.compile(r"FILE\s+COPY", re.IGNORECASE), 3),
        (re.compile(r"Certified\s+Copy", re.IGNORECASE), 3),
        (re.compile(r"Payment\s+received", re.IGNORECASE), 2),
    ],
    "oag": [
        (re.compile(r"Office\s+of\s+the\s+Attorney\s+General", re.IGNORECASE), 4),
        (re.compile(r"oag\.texas\.gov", re.IGNORECASE), 4),
        (re.compile(r"OAG", re.IGNORECASE), 2),
    ],
}

FILER_PRIORITY = [
    "charles_dustin_myers",
    "cooper_carter",
    "morgan_michelle_myers",
    "court",
    "clerk",
    "oag",
    "unknown",
]

DOC_START_PATTERNS = [
    re.compile(r"(?i)\bpage[: ]+1\b"),
    re.compile(r"(?i)\bpage\s+1\s+of\b"),
    re.compile(r"(?i)\bIN\s+THE\s+.*\s+COURT\b"),
    re.compile(r"(?i)\bCAUSE\s+NO\.?\b"),
    re.compile(r"(?i)\bCAUSE\s+NUMBER\b"),
]

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


def _normalize_ascii(text: str) -> str:
    return vis._normalize_ascii(text)


def _normalize_for_match(text: str) -> str:
    cleaned = _normalize_ascii(text).lower()
    cleaned = re.sub(r"[^a-z0-9\s]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _iter_pages(json_path: Path) -> Iterable[PageRecord]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    for page in payload.get("pages", []):
        yield PageRecord(page_number=page["page_number"], text=page.get("text", ""))


def _is_doc_start(text: str) -> bool:
    for pattern in DOC_START_PATTERNS:
        if pattern.search(text):
            return True
    return False


def _score_filer(text: str) -> Tuple[str, int, List[str]]:
    scores: Dict[str, int] = defaultdict(int)
    signals: Dict[str, List[str]] = defaultdict(list)
    for filer, rules in FILER_RULES.items():
        for pattern, weight in rules:
            if pattern.search(text):
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


def _smooth_labels(labels: List[str], window: int = 2) -> List[str]:
    smoothed = labels[:]
    for idx, label in enumerate(labels):
        if label != "unknown":
            continue
        left = max(0, idx - window)
        right = min(len(labels), idx + window + 1)
        neighbors = [labels[i] for i in range(left, right) if labels[i] != "unknown"]
        if not neighbors:
            continue
        counts = Counter(neighbors)
        if counts:
            smoothed[idx] = counts.most_common(1)[0][0]
    return smoothed


def _load_docket_filer_map(path: Path | None) -> Dict[str, str]:
    if not path:
        return {}
    if not path.exists():
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


def _extract_docket_entries(pages: List[PageRecord]) -> List[dict]:
    entries: List[dict] = []
    for page in pages:
        if DOCKET_HEADER not in page.text:
            continue
        for line in page.text.splitlines():
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


def _apply_docket_overrides(
    docs: List[dict],
    pages: List[PageRecord],
    docket_map: Dict[str, str],
    docket_entries: List[dict],
) -> Tuple[List[dict], List[dict]]:
    page_by_number = {page.page_number: page for page in pages}
    entries = []
    for entry in docket_entries:
        filer = docket_map.get(entry["filemark"])
        if not filer:
            continue
        key = _build_match_key(entry["description"])
        if not key:
            continue
        entries.append(
            {
                "filemark": entry["filemark"],
                "filer": filer,
                "key": key,
                "key_len": len(key),
            }
        )
    entries.sort(key=lambda item: item["key_len"], reverse=True)

    overrides: List[dict] = []
    for doc in docs:
        first_page = page_by_number.get(doc["start_page"])
        if not first_page:
            continue
        raw_text = first_page.text
        normalized = _normalize_for_match(raw_text)[:6000]
        matched_filer = None
        match_type = None
        match_value = None

        filemark = _detect_filemark(raw_text)
        if filemark and filemark in docket_map:
            matched_filer = docket_map[filemark]
            match_type = "filemark"
            match_value = filemark
        else:
            for entry in entries:
                if entry["key"] in normalized:
                    matched_filer = entry["filer"]
                    match_type = "description"
                    match_value = entry["filemark"]
                    break

        if matched_filer and matched_filer != doc["filer"]:
            overrides.append(
                {
                    "doc_id": doc["doc_id"],
                    "start_page": doc["start_page"],
                    "old_filer": doc["filer"],
                    "new_filer": matched_filer,
                    "match_type": match_type or "",
                    "match_value": match_value or "",
                }
            )
            doc["filer"] = matched_filer
    return docs, overrides


def _group_documents(pages: List[PageRecord], labels: List[str]) -> List[dict]:
    docs: List[dict] = []
    current = {
        "doc_id": 1,
        "filer": labels[0],
        "start_page": pages[0].page_number,
        "end_page": pages[0].page_number,
        "pages": [pages[0].page_number],
    }
    for page, label in zip(pages[1:], labels[1:]):
        if _is_doc_start(page.text) or label != current["filer"]:
            docs.append(current)
            current = {
                "doc_id": current["doc_id"] + 1,
                "filer": label,
                "start_page": page.page_number,
                "end_page": page.page_number,
                "pages": [page.page_number],
            }
        else:
            current["end_page"] = page.page_number
            current["pages"].append(page.page_number)
    docs.append(current)
    return docs


def _write_docket_overrides(output_path: Path, overrides: List[dict]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["doc_id", "start_page", "old_filer", "new_filer", "match_type", "match_value"]
        )
        for item in overrides:
            writer.writerow(
                [
                    item["doc_id"],
                    item["start_page"],
                    item["old_filer"],
                    item["new_filer"],
                    item["match_type"],
                    item["match_value"],
                ]
            )


def _write_page_map(output_path: Path, pages: List[PageRecord], labels: List[str]) -> None:
    lines = ["page_number,filer"]
    for page, label in zip(pages, labels):
        lines.append(f"{page.page_number},{label}")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_doc_map(output_path: Path, docs: List[dict]) -> None:
    lines = ["doc_id,filer,start_page,end_page,page_count"]
    for doc in docs:
        lines.append(
            f"{doc['doc_id']},{doc['filer']},{doc['start_page']},{doc['end_page']},{len(doc['pages'])}"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_summary(output_path: Path, labels: List[str]) -> None:
    counts = Counter(labels)
    total = len(labels)
    lines = [
        "# Filer Split Summary",
        "",
        f"Total pages: {total}",
        "",
    ]
    for filer in FILER_PRIORITY:
        if filer in counts:
            lines.append(f"- {filer}: {counts[filer]}")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _render_for_filer(
    output_dir: Path,
    pages: List[PageRecord],
    embeddings: np.ndarray,
    parties: List[str],
    baseline_embeddings: np.ndarray | None,
    min_year: int,
    max_year: int,
    bin_size: int,
    top_issues: int,
    contradiction_nodes: int,
    contradiction_threshold: float,
    role_sample: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked_issues = vis._issue_ranking(pages)[:top_issues]
    drift_images = vis._semantic_drift(
        output_dir,
        pages,
        embeddings,
        parties,
        ranked_issues,
        min_year,
        max_year,
    )
    contradiction_img, contradiction_edges = vis._contradiction_map(
        output_dir,
        pages,
        embeddings,
        contradiction_nodes,
        contradiction_threshold,
    )
    vis._authority_leakage(output_dir / "authority_leakage.png", pages, bin_size)
    vis._procedural_gravity(output_dir / "procedural_gravity_wells.png", pages, embeddings)
    vis._selective_attention(output_dir / "selective_attention.png", pages, embeddings, bin_size)
    role_blind, role_labeled = vis._role_blind_plots(output_dir, pages, embeddings, role_sample)
    vis._counterfactual_overlay(
        output_dir / "counterfactual_anomaly_overlay.png",
        embeddings,
        baseline_embeddings,
        bin_size,
    )
    vis._issue_cannibalization(output_dir / "issue_cannibalization.png", pages)

    images = []
    for name in drift_images:
        images.append((name, f"Semantic drift timeline ({name.replace('semantic_drift_', '').replace('.png', '')})"))
    if contradiction_img:
        images.append((contradiction_img, "Contradiction map (similar + opposite polarity)"))
    images.extend(
        [
            ("authority_leakage.png", "Authority leakage (authority vs statute terms)"),
            ("procedural_gravity_wells.png", "Procedural gravity wells (dominant vs tagged issues)"),
            ("selective_attention.png", "Selective attention (brief/order similarity)"),
            (role_blind, "Role-blind cluster projection"),
            (role_labeled, "Role-labeled comparison"),
            ("counterfactual_anomaly_overlay.png", "Counterfactual anomaly overlay (baseline similarity)"),
            ("issue_cannibalization.png", "Issue cannibalization (dominant vs secondary)"),
        ]
    )
    vis._write_index(output_dir / "index.html", images)

    if contradiction_edges:
        (output_dir / "contradiction_edges.csv").write_text(
            (output_dir / contradiction_edges).read_text(encoding="utf-8"),
            encoding="utf-8",
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split pages by filer and build visuals.")
    parser.add_argument(
        "--json",
        type=Path,
        required=True,
        help="Path to the extracted JSON (page-level) file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/visuals_by_filer"),
        help="Root directory for filer visuals.",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=Path("reports"),
        help="Directory to write filer summaries.",
    )
    parser.add_argument("--min-year", type=int, default=1800, help="Minimum year for dates.")
    parser.add_argument("--max-year", type=int, default=2100, help="Maximum year for dates.")
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size.")
    parser.add_argument("--bin-size", type=int, default=100, help="Page bin size.")
    parser.add_argument("--top-issues", type=int, default=3, help="Top issues for drift.")
    parser.add_argument("--contradiction-nodes", type=int, default=220, help="Nodes for map.")
    parser.add_argument(
        "--contradiction-threshold",
        type=float,
        default=0.78,
        help="Similarity threshold for contradictions.",
    )
    parser.add_argument(
        "--role-sample",
        type=int,
        default=0,
        help="Sample size for role plots (0 = all pages).",
    )
    parser.add_argument("--min-pages", type=int, default=30, help="Minimum pages per filer.")
    parser.add_argument(
        "--docket-filer-map",
        type=Path,
        default=None,
        help="CSV map of filemark-to-filer from color-coded docket images.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    json_path = args.json.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    summary_dir = args.summary_dir.expanduser().resolve()
    summary_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    pages = list(_iter_pages(json_path))
    if not pages:
        raise ValueError("No pages found in JSON input.")

    labels = []
    for page in pages:
        filer, score, _signals = _score_filer(_normalize_ascii(page.text))
        labels.append(filer)

    labels = _smooth_labels(labels, window=2)
    docs = _group_documents(pages, labels)

    docket_map = _load_docket_filer_map(args.docket_filer_map)
    if docket_map:
        docket_entries = _extract_docket_entries(pages)
        docs, overrides = _apply_docket_overrides(docs, pages, docket_map, docket_entries)
        _write_docket_overrides(summary_dir / "28b_filer_docket_overrides.csv", overrides)
        page_index = {page.page_number: idx for idx, page in enumerate(pages)}
        for doc in docs:
            for page_num in doc["pages"]:
                idx = page_index.get(page_num)
                if idx is not None:
                    labels[idx] = doc["filer"]

    _write_page_map(summary_dir / "28b_filer_pages.csv", pages, labels)
    _write_doc_map(summary_dir / "28b_filer_docs.csv", docs)
    _write_summary(summary_dir / "28b_filer_summary.md", labels)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = vis._page_embeddings(model, pages, args.batch_size)

    header = vis._extract_case_header(pages[0].text)
    parties = [value for key, value in header.items() if key.startswith("party_")]
    if len(parties) < 2:
        parties = ["relator", "respondent"]

    baseline_embeddings = vis._load_baseline_embeddings(Path("vector_store_research"))

    filer_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        filer_indices[label].append(idx)

    for filer in FILER_PRIORITY:
        idxs = filer_indices.get(filer, [])
        if len(idxs) < args.min_pages:
            continue
        filer_pages = [pages[idx] for idx in idxs]
        filer_embeddings = embeddings[idxs]
        filer_dir = output_root / filer
        _render_for_filer(
            filer_dir,
            filer_pages,
            filer_embeddings,
            parties,
            baseline_embeddings,
            args.min_year,
            args.max_year,
            args.bin_size,
            args.top_issues,
            args.contradiction_nodes,
            args.contradiction_threshold,
            args.role_sample,
        )

    print(f"Wrote filer visuals to {output_root}")


if __name__ == "__main__":
    main()
