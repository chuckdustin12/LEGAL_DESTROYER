"""Build inconsistency visuals and vector search reports for petitioner docs."""

from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer

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

TOPIC_QUERIES = {
    "protective order / family violence": [
        "protective order",
        "temporary protective order",
        "family violence",
        "assault",
        "threatened violence",
    ],
    "custody / conservatorship": [
        "custody",
        "conservatorship",
        "possession and access",
        "best interest of the child",
    ],
    "child support / income": [
        "child support",
        "income",
        "support obligation",
        "arrears",
    ],
    "property / financial claims": [
        "community property",
        "separate property",
        "bank account",
        "debt",
        "fraud",
    ],
    "jurisdiction / venue": [
        "jurisdiction",
        "subject matter jurisdiction",
        "venue",
    ],
    "summary judgment / evidence": [
        "summary judgment",
        "no-evidence summary judgment",
        "evidence",
    ],
}

SHIFT_TOPICS = {
    "protective order / family violence": [
        "protective order",
        "family violence",
        "domestic violence",
        "assault",
        "threatened violence",
    ],
    "agreement / settlement": [
        "agreement",
        "agreed order",
        "settlement",
        "mediated settlement agreement",
        "rule 11 agreement",
        "stipulation",
    ],
    "child abuse / neglect": [
        "child abuse",
        "abuse of a child",
        "child neglect",
        "sexual abuse",
        "physical abuse",
        "child endangerment",
        "injury to a child",
    ],
}

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

DATE_RE = re.compile(r"(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})")


@dataclass
class DocInfo:
    label: str
    short_label: str
    source_pdf: str
    date: datetime | None
    indices: List[int]


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _normalize_ascii(text: str) -> str:
    cleaned = text.translate(NON_ASCII_MAP)
    return cleaned.encode("ascii", "ignore").decode("ascii")


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _snippet(text: str, max_len: int = 280) -> str:
    cleaned = _normalize_ascii(_normalize_ws(text))
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3].rstrip() + "..."


def _snippet_around_match(text: str, patterns: List[re.Pattern], max_len: int = 320) -> str:
    cleaned = _normalize_ascii(_normalize_ws(text))
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


def _load_store(store_dir: Path) -> Tuple[np.ndarray, List[dict]]:
    embeddings = np.load(store_dir / "embeddings.npy")
    records = [
        json.loads(line)
        for line in (store_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return embeddings, records


def _clean_label(name: str) -> str:
    cleaned = re.sub(r"[_]+", " ", name).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"-[a-f0-9]{6,}(?:-\d+)?$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[-_]\d+$", "", cleaned)
    return cleaned.strip()


def _shorten(label: str, width: int = 28) -> str:
    if len(label) <= width:
        return label
    return label[: width - 3] + "..."


def _parse_date(text: str) -> datetime | None:
    for match in DATE_RE.finditer(text):
        month = int(match.group(1))
        day = int(match.group(2))
        year = int(match.group(3))
        if year < 100:
            year += 2000
        try:
            return datetime(year, month, day)
        except ValueError:
            continue
    return None


def _doc_info(records: List[dict]) -> List[DocInfo]:
    groups: Dict[str, List[int]] = {}
    labels: Dict[str, str] = {}
    dates: Dict[str, datetime | None] = {}
    for idx, record in enumerate(records):
        source_pdf = record.get("source_pdf") or "unknown"
        stem = Path(source_pdf).stem
        label = _clean_label(stem)
        groups.setdefault(label, []).append(idx)
        labels[label] = source_pdf
        if label not in dates:
            dates[label] = _parse_date(source_pdf) or _parse_date(label)

    docs: List[DocInfo] = []
    for label, indices in groups.items():
        docs.append(
            DocInfo(
                label=label,
                short_label=_shorten(label),
                source_pdf=labels.get(label, ""),
                date=dates.get(label),
                indices=indices,
            )
        )
    docs.sort(key=lambda doc: (doc.date or datetime.max, doc.label.lower()))
    return docs


def _filter_docs(
    docs: List[DocInfo], include_pattern: str | None, exclude_pattern: str | None
) -> List[DocInfo]:
    include_re = re.compile(include_pattern, re.IGNORECASE) if include_pattern else None
    exclude_re = re.compile(exclude_pattern, re.IGNORECASE) if exclude_pattern else None
    filtered: List[DocInfo] = []
    for doc in docs:
        haystack = f"{doc.label} {doc.source_pdf}"
        if include_re and not include_re.search(haystack):
            continue
        if exclude_re and exclude_re.search(haystack):
            continue
        filtered.append(doc)
    return filtered


def _subset_by_docs(
    embeddings: np.ndarray, records: List[dict], docs: List[DocInfo]
) -> Tuple[np.ndarray, List[dict], List[DocInfo]]:
    selected_indices = sorted({idx for doc in docs for idx in doc.indices})
    index_map = {old: new for new, old in enumerate(selected_indices)}
    sub_embeddings = embeddings[selected_indices]
    sub_records = [records[i] for i in selected_indices]
    final_docs: List[DocInfo] = []
    for doc in docs:
        mapped = [index_map[i] for i in doc.indices if i in index_map]
        if not mapped:
            continue
        doc.indices = mapped
        final_docs.append(doc)
    return sub_embeddings, sub_records, final_docs


def _unit_vector(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0:
        return vec
    return vec / norm


def _doc_embeddings(embeddings: np.ndarray, docs: List[DocInfo]) -> np.ndarray:
    vectors = []
    for doc in docs:
        doc_vec = embeddings[doc.indices].mean(axis=0)
        vectors.append(_unit_vector(doc_vec))
    return np.vstack(vectors)


def _polarity_counts(text: str) -> Tuple[int, int]:
    text_lower = text.lower()
    neg = sum(text_lower.count(term) for term in NEGATION_TERMS)
    pos = sum(text_lower.count(term) for term in AFFIRM_TERMS)
    return pos, neg


def _polarity_sign(pos: int, neg: int, min_hits: int) -> int:
    score = pos - neg
    if score >= min_hits:
        return 1
    if score <= -min_hits:
        return -1
    return 0


def _pca_2d(vectors: np.ndarray) -> np.ndarray:
    centered = vectors - vectors.mean(axis=0)
    _, _, v = np.linalg.svd(centered, full_matrices=False)
    return centered @ v[:2].T


def _wrap_labels(labels: List[str], width: int = 24) -> List[str]:
    wrapped = []
    for label in labels:
        wrapped.append("\n".join(textwrap.wrap(label, width=width)) or label)
    return wrapped


def _plot_doc_similarity(
    output_path: Path, docs: List[DocInfo], doc_vectors: np.ndarray
) -> None:
    sim = doc_vectors @ doc_vectors.T
    labels = _wrap_labels([doc.short_label for doc in docs], width=20)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title("Document Similarity (Petitioner Filings)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_polarity_balance(
    output_path: Path, docs: List[DocInfo], records: List[dict]
) -> None:
    neg_counts = []
    pos_counts = []
    labels = []
    for doc in docs:
        pos = 0
        neg = 0
        for idx in doc.indices:
            text = _normalize_ascii(records[idx].get("text", ""))
            pos_chunk, neg_chunk = _polarity_counts(text)
            pos += pos_chunk
            neg += neg_chunk
        labels.append(doc.short_label)
        pos_counts.append(pos)
        neg_counts.append(neg)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.2, pos_counts, width=0.4, label="affirmation")
    ax.bar(x + 0.2, neg_counts, width=0.4, label="negation")
    ax.set_xticks(x)
    ax.set_xticklabels(_wrap_labels(labels, width=18), rotation=45, ha="right")
    ax.set_ylabel("Term hits")
    ax.set_title("Polarity Balance by Document")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _topic_trends(
    output_path: Path,
    docs: List[DocInfo],
    doc_vectors: np.ndarray,
    model: SentenceTransformer,
) -> None:
    topic_names = list(TOPIC_QUERIES.keys())
    query_lists = list(TOPIC_QUERIES.values())
    query_embeddings = [
        model.encode(queries, normalize_embeddings=True) for queries in query_lists
    ]

    scores = []
    for q_embeds in query_embeddings:
        topic_scores = doc_vectors @ q_embeds.T
        scores.append(topic_scores.max(axis=1))

    x = np.arange(len(docs))
    fig, ax = plt.subplots(figsize=(11, 5))
    for idx, topic in enumerate(topic_names):
        ax.plot(x, scores[idx], marker="o", label=topic)
    labels = [doc.short_label for doc in docs]
    ax.set_xticks(x)
    ax.set_xticklabels(_wrap_labels(labels, width=18), rotation=45, ha="right")
    ax.set_ylabel("Similarity")
    ax.set_title("Topic Emphasis Across Petitioner Filings")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _best_topic_hit(
    doc_vectors: np.ndarray,
    doc_indices: List[int],
    query_embeddings: np.ndarray,
) -> Tuple[float, int]:
    if not doc_indices:
        return 0.0, -1
    sims = doc_vectors @ query_embeddings.T
    max_scores = sims.max(axis=1)
    best_pos = int(np.argmax(max_scores))
    return float(max_scores[best_pos]), doc_indices[best_pos]


def _shift_topic_scores(
    embeddings: np.ndarray,
    records: List[dict],
    docs: List[DocInfo],
    model: SentenceTransformer,
) -> Tuple[Dict[str, List[int]], Dict[str, Dict[int, Tuple[int, int]]], Dict[str, Dict[int, Tuple[float, int]]]]:
    counts: Dict[str, List[int]] = {}
    keyword_hits: Dict[str, Dict[int, Tuple[int, int]]] = {}
    semantic_hits: Dict[str, Dict[int, Tuple[float, int]]] = {}
    query_embeddings = {
        topic: model.encode(queries, normalize_embeddings=True)
        for topic, queries in SHIFT_TOPICS.items()
    }
    keyword_patterns = {
        topic: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        for topic, patterns in SHIFT_KEYWORDS.items()
    }

    for topic in SHIFT_TOPICS:
        counts[topic] = []
        keyword_hits[topic] = {}
        semantic_hits[topic] = {}

    for doc_idx, doc in enumerate(docs):
        doc_vectors = embeddings[doc.indices]
        for topic, q_embed in query_embeddings.items():
            best_score, best_record_idx = _best_topic_hit(
                doc_vectors, doc.indices, q_embed
            )
            semantic_hits[topic][doc_idx] = (best_score, best_record_idx)

        for topic, patterns in keyword_patterns.items():
            total_hits = 0
            best_hits = 0
            best_record = -1
            for record_idx in doc.indices:
                text = records[record_idx].get("text", "")
                hits = 0
                for pattern in patterns:
                    hits += len(pattern.findall(text))
                total_hits += hits
                if hits > best_hits:
                    best_hits = hits
                    best_record = record_idx
            counts[topic].append(total_hits)
            keyword_hits[topic][doc_idx] = (best_hits, best_record)

    return counts, keyword_hits, semantic_hits


def _plot_shift_timeline(
    output_path: Path,
    docs: List[DocInfo],
    topic_counts: Dict[str, List[int]],
) -> None:
    labels = [
        f"{doc.date.strftime('%Y-%m-%d') if doc.date else 'unknown'}\n{doc.short_label}"
        for doc in docs
    ]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 5))
    for topic, values in topic_counts.items():
        ax.plot(x, values, marker="o", label=topic)
    ax.set_xticks(x)
    ax.set_xticklabels(_wrap_labels(labels, width=22), rotation=45, ha="right")
    ax.set_ylabel("Keyword hits")
    ax.set_title("Narrative Shift Timeline (Petitioner Filings)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, ncol=1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _write_shift_report(
    output_path: Path,
    docs: List[DocInfo],
    records: List[dict],
    topic_counts: Dict[str, List[int]],
    keyword_hits: Dict[str, Dict[int, Tuple[int, int]]],
    semantic_hits: Dict[str, Dict[int, Tuple[float, int]]],
) -> None:
    keyword_patterns = {
        topic: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        for topic, patterns in SHIFT_KEYWORDS.items()
    }
    lines = [
        "# Narrative Shift: Protective Order -> Agreement -> Child Abuse",
        "",
        f"Generated: {_timestamp()}",
        "",
        "Counts reflect keyword hits per document; excerpts use keyword matches with semantic fallback.",
        "",
        "## Topic Intensity By Document",
        "",
    ]
    for idx, doc in enumerate(docs):
        date_str = doc.date.strftime("%Y-%m-%d") if doc.date else "unknown date"
        lines.append(f"### {doc.label} ({date_str})")
        best_topic = None
        best_score = -1
        for topic, values in topic_counts.items():
            score = values[idx]
            if score > best_score:
                best_score = score
                best_topic = topic
            lines.append(f"- {topic}: {score}")
        if best_score <= 0:
            lines.append("- dominant_topic: none")
        else:
            lines.append(f"- dominant_topic: {best_topic} ({best_score})")
        lines.append("")

    lines.append("## Transition Summary")
    lines.append("")
    for topic, values in topic_counts.items():
        max_hits = max(values) if values else 0
        first_idx = next((i for i, value in enumerate(values) if value > 0), None)
        if max_hits <= 0 or first_idx is None:
            lines.append(f"- {topic}: no keyword hits found.")
            continue
        peak_idx = values.index(max_hits)
        peak_doc = docs[peak_idx]
        peak_date = peak_doc.date.strftime("%Y-%m-%d") if peak_doc.date else "unknown date"
        first_doc = docs[first_idx]
        first_date = first_doc.date.strftime("%Y-%m-%d") if first_doc.date else "unknown date"
        lines.append(
            f"- {topic}: first appears in {first_doc.label} ({first_date}); "
            f"peaks in {peak_doc.label} ({peak_date}) with {max_hits} hits."
        )
    lines.append("")

    lines.append("## Supporting Excerpts (Top Match Per Topic)")
    lines.append("")
    for topic in SHIFT_TOPICS:
        lines.append(f"### {topic}")
        has_keyword_hits = any(
            hits > 0 for hits, _ in keyword_hits[topic].values()
        )
        wrote_any = False
        for idx, doc in enumerate(docs):
            best_keyword_hits, record_idx = keyword_hits[topic][idx]
            if best_keyword_hits > 0 and record_idx >= 0:
                record = records[record_idx]
                lines.append(
                    f"- {doc.label} | keyword hits {best_keyword_hits} | chunk {record.get('chunk_index')}"
                )
                lines.append(
                    f"  Quote: \"{_snippet_around_match(record.get('text', ''), keyword_patterns[topic], 320)}\""
                )
                wrote_any = True
            elif not has_keyword_hits:
                score, record_idx = semantic_hits[topic][idx]
                if record_idx < 0:
                    continue
                record = records[record_idx]
                lines.append(
                    f"- {doc.label} | semantic score {score:.4f} | chunk {record.get('chunk_index')}"
                )
                lines.append(f"  Quote: \"{_snippet(record.get('text', ''), 320)}\"")
                wrote_any = True
        if not wrote_any:
            lines.append("- No keyword hits found for this topic.")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _build_contradictions(
    embeddings: np.ndarray,
    records: List[dict],
    docs: List[DocInfo],
    *,
    min_polarity_hits: int,
    max_candidates: int,
    similarity_threshold: float,
    max_edges: int,
) -> Tuple[List[dict], np.ndarray, List[int], List[str], List[int], List[int]]:
    doc_lookup = {}
    for doc in docs:
        for idx in doc.indices:
            doc_lookup[idx] = doc.label

    candidates = []
    strengths = []
    for idx, record in enumerate(records):
        text = _normalize_ascii(record.get("text", ""))
        pos, neg = _polarity_counts(text)
        sign = _polarity_sign(pos, neg, min_polarity_hits)
        if sign == 0:
            continue
        strength = abs(pos - neg)
        candidates.append(idx)
        strengths.append(strength)

    if not candidates:
        return [], np.zeros((0, 2)), [], [], [], []

    if len(candidates) > max_candidates:
        ranked = sorted(range(len(candidates)), key=lambda i: strengths[i], reverse=True)
        keep = set(ranked[:max_candidates])
        candidates = [candidates[i] for i in range(len(candidates)) if i in keep]
        strengths = [strengths[i] for i in range(len(strengths)) if i in keep]

    cand_vectors = embeddings[candidates]
    cand_vectors = np.asarray([_unit_vector(vec) for vec in cand_vectors])
    coords = _pca_2d(cand_vectors) if len(candidates) > 1 else np.zeros((len(candidates), 2))

    signs = []
    doc_labels = []
    for idx in candidates:
        text = _normalize_ascii(records[idx].get("text", ""))
        pos, neg = _polarity_counts(text)
        signs.append(_polarity_sign(pos, neg, min_polarity_hits))
        doc_labels.append(doc_lookup.get(idx, "unknown"))

    sim = cand_vectors @ cand_vectors.T
    edges = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            if doc_labels[i] == doc_labels[j]:
                continue
            if signs[i] == 0 or signs[j] == 0:
                continue
            if signs[i] * signs[j] > 0:
                continue
            score = float(sim[i, j])
            if score < similarity_threshold:
                continue
            edges.append((i, j, score))

    edges.sort(key=lambda item: item[2], reverse=True)
    edges = edges[:max_edges]

    edge_rows = []
    for i, j, score in edges:
        rec_a = records[candidates[i]]
        rec_b = records[candidates[j]]
        edge_rows.append(
            {
                "doc_a": doc_labels[i],
                "doc_b": doc_labels[j],
                "score": round(score, 4),
                "chunk_a": rec_a.get("chunk_index"),
                "chunk_b": rec_b.get("chunk_index"),
                "source_a": rec_a.get("source_pdf"),
                "source_b": rec_b.get("source_pdf"),
                "snippet_a": _snippet(rec_a.get("text", ""), 240),
                "snippet_b": _snippet(rec_b.get("text", ""), 240),
            }
        )

    return edge_rows, coords, candidates, doc_labels, signs, strengths


def _plot_contradiction_map(
    output_path: Path,
    coords: np.ndarray,
    doc_labels: List[str],
    edges: List[dict],
    edge_indices: List[Tuple[int, int]],
) -> None:
    if coords.size == 0:
        return
    unique_docs = sorted(set(doc_labels))
    cmap = plt.cm.get_cmap("tab20", max(len(unique_docs), 3))
    color_map = {doc: cmap(i) for i, doc in enumerate(unique_docs)}

    fig, ax = plt.subplots(figsize=(10, 8))
    for idx, doc in enumerate(unique_docs):
        points = [i for i, label in enumerate(doc_labels) if label == doc]
        ax.scatter(
            coords[points, 0],
            coords[points, 1],
            s=18,
            color=color_map[doc],
            label=_shorten(doc, 20),
            alpha=0.85,
        )

    for i, j in edge_indices:
        ax.plot(
            [coords[i, 0], coords[j, 0]],
            [coords[i, 1], coords[j, 1]],
            color="#999999",
            alpha=0.35,
            linewidth=0.8,
        )

    ax.set_title("Contradiction Map (Similar + Opposite Polarity)")
    ax.axis("off")
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _write_edges(path: Path, edges: List[dict]) -> None:
    header = [
        "doc_a",
        "doc_b",
        "score",
        "chunk_a",
        "chunk_b",
        "source_a",
        "source_b",
        "snippet_a",
        "snippet_b",
    ]
    lines = [",".join(header)]
    for edge in edges:
        row = [
            str(edge.get("doc_a", "")),
            str(edge.get("doc_b", "")),
            str(edge.get("score", "")),
            str(edge.get("chunk_a", "")),
            str(edge.get("chunk_b", "")),
            str(edge.get("source_a", "")),
            str(edge.get("source_b", "")),
            str(edge.get("snippet_a", "")).replace(",", " "),
            str(edge.get("snippet_b", "")).replace(",", " "),
        ]
        lines.append(",".join(row))
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_index(output_path: Path, images: List[Tuple[str, str]]) -> None:
    lines = [
        "<!doctype html>",
        "<html lang=\"en\">",
        "<head>",
        "<meta charset=\"utf-8\" />",
        "<title>Inconsistency Visuals</title>",
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
        "<h1>Petitioner Inconsistency Visuals</h1>",
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


def _write_report(
    output_path: Path,
    docs: List[DocInfo],
    records: List[dict],
    edges: List[dict],
    topic_hits: Dict[str, List[Tuple[float, int]]],
) -> None:
    lines = [
        "# Petitioner Inconsistency Vector Search",
        "",
        f"Generated: {_timestamp()}",
        "",
        "## Documents Scanned",
        "",
    ]
    for doc in docs:
        date_str = doc.date.strftime("%Y-%m-%d") if doc.date else "unknown date"
        lines.append(f"- {doc.label} ({date_str})")

    lines.append("")
    lines.append("## Topic Searches")
    lines.append("")
    for topic, hits in topic_hits.items():
        lines.append(f"### {topic}")
        if not hits:
            lines.append("- No hits found.")
            lines.append("")
            continue
        for score, idx in hits:
            record = records[idx]
            doc_label = Path(record.get("source_pdf", "")).stem
            lines.append(
                f"- Score {score:.4f} | {doc_label} | chunk {record.get('chunk_index')}"
            )
            lines.append(f"  Quote: \"{_snippet(record.get('text', ''), 320)}\"")
        lines.append("")

    lines.append("## Top Contradiction Pairs")
    lines.append("")
    if not edges:
        lines.append("No contradiction pairs found at the current threshold.")
    else:
        for edge in edges[:25]:
            lines.append(
                f"- Similarity {edge['score']:.4f} | {edge['doc_a']} vs {edge['doc_b']}"
            )
            lines.append(f"  A: \"{edge['snippet_a']}\"")
            lines.append(f"  B: \"{edge['snippet_b']}\"")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build inconsistency visuals from a vector store.")
    parser.add_argument(
        "--store",
        type=Path,
        default=Path("vector_store_inconsistencies"),
        help="Merged vector store directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/visuals_inconsistencies"),
        help="Output directory for images and HTML.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/inconsistencies_vector_search.md"),
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--shift-report",
        type=Path,
        default=Path("reports/inconsistencies_narrative_shift.md"),
        help="Markdown report for narrative shift output.",
    )
    parser.add_argument(
        "--edges",
        type=Path,
        default=Path("reports/inconsistencies_contradiction_edges.csv"),
        help="CSV output for contradiction edges.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--min-polarity-hits",
        type=int,
        default=2,
        help="Minimum polarity hits to treat a chunk as affirmative/negative.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=1200,
        help="Max chunks considered for contradiction mapping.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.78,
        help="Cosine similarity threshold for contradiction edges.",
    )
    parser.add_argument("--max-edges", type=int, default=200, help="Max contradiction edges.")
    parser.add_argument("--top-hits", type=int, default=6, help="Top hits per topic.")
    parser.add_argument(
        "--include",
        type=str,
        default=None,
        help="Regex to include documents by label/path.",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="Regex to exclude documents by label/path.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    store_dir = args.store.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings, records = _load_store(store_dir)
    docs = _doc_info(records)
    docs = _filter_docs(docs, args.include, args.exclude)
    embeddings, records, docs = _subset_by_docs(embeddings, records, docs)
    if not docs:
        raise ValueError("No documents matched the include/exclude filters.")
    doc_vectors = _doc_embeddings(embeddings, docs)

    model = SentenceTransformer(args.model)

    similarity_img = output_dir / "document_similarity_heatmap.png"
    _plot_doc_similarity(similarity_img, docs, doc_vectors)

    polarity_img = output_dir / "polarity_balance.png"
    _plot_polarity_balance(polarity_img, docs, records)

    topic_img = output_dir / "topic_emphasis.png"
    _topic_trends(topic_img, docs, doc_vectors, model)

    shift_counts, shift_keyword_hits, shift_semantic_hits = _shift_topic_scores(
        embeddings, records, docs, model
    )
    shift_img = output_dir / "narrative_shift_timeline.png"
    _plot_shift_timeline(shift_img, docs, shift_counts)
    _write_shift_report(
        args.shift_report.expanduser().resolve(),
        docs,
        records,
        shift_counts,
        shift_keyword_hits,
        shift_semantic_hits,
    )

    edges, coords, candidates, doc_labels, signs, strengths = _build_contradictions(
        embeddings,
        records,
        docs,
        min_polarity_hits=args.min_polarity_hits,
        max_candidates=args.max_candidates,
        similarity_threshold=args.similarity_threshold,
        max_edges=args.max_edges,
    )

    contradiction_img = output_dir / "contradiction_map.png"
    if coords.size:
        edge_indices = []
        if edges:
            cand_lookup = {}
            for pos, idx in enumerate(candidates):
                record = records[idx]
                key = (record.get("chunk_index"), record.get("source_pdf"))
                cand_lookup[key] = pos
            for edge in edges:
                chunk_a = edge.get("chunk_a")
                chunk_b = edge.get("chunk_b")
                source_a = edge.get("source_a", "")
                source_b = edge.get("source_b", "")
                idx_a = cand_lookup.get((chunk_a, source_a))
                idx_b = cand_lookup.get((chunk_b, source_b))
                if idx_a is None or idx_b is None:
                    continue
                edge_indices.append((idx_a, idx_b))
        _plot_contradiction_map(contradiction_img, coords, doc_labels, edges, edge_indices)

    _write_edges(args.edges.expanduser().resolve(), edges)

    topic_hits: Dict[str, List[Tuple[float, int]]] = {}
    for topic, queries in TOPIC_QUERIES.items():
        query_embeddings = model.encode(queries, normalize_embeddings=True)
        scores = embeddings @ query_embeddings.T
        best_scores = scores.max(axis=1)
        ranked = np.argsort(-best_scores)
        seen = set()
        picks: List[Tuple[float, int]] = []
        for idx in ranked:
            text = records[idx]["text"]
            if text in seen:
                continue
            seen.add(text)
            picks.append((float(best_scores[idx]), idx))
            if len(picks) >= args.top_hits:
                break
        topic_hits[topic] = picks

    _write_report(args.report.expanduser().resolve(), docs, records, edges, topic_hits)

    images = [
        ("document_similarity_heatmap.png", "Document similarity heatmap"),
        ("polarity_balance.png", "Polarity balance by document"),
        ("topic_emphasis.png", "Topic emphasis across documents"),
        ("narrative_shift_timeline.png", "Narrative shift timeline"),
    ]
    if contradiction_img.exists():
        images.append(("contradiction_map.png", "Contradiction map (similar + opposite polarity)"))

    index_path = output_dir / "index.html"
    _write_index(index_path, images)

    print(f"Wrote visuals to {output_dir}")


if __name__ == "__main__":
    main()
