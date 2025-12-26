"""Create advanced semantic visuals for the 28B record."""

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
from sentence_transformers import SentenceTransformer

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

AUTHORITY_TERMS = [
    "authority",
    "inherent",
    "discretion",
    "emergency",
    "equitable",
    "sua sponte",
    "powers",
    "protect",
    "best interest",
    "public policy",
]

STATUTORY_TERMS = [
    "statute",
    "code",
    "section",
    "rule",
    "tex.",
    "texas",
    "family code",
    "gov't code",
    "civ. prac",
    "app. p.",
    "u.s.c.",
    "constitution",
]

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

DOC_TYPE_RULES = {
    "order": ["order", "judgment", "opinion", "decree", "signed"],
    "finding": ["findings of fact", "conclusions of law"],
    "sworn": ["affidavit", "declaration", "sworn", "verified", "under oath"],
    "pleading": ["petition", "motion", "response", "brief", "answer", "counterpetition"],
}

BRIEF_TERMS = ["brief", "memorandum", "response", "petition", "motion", "counterpetition"]
ORDER_TERMS = ["order", "judgment", "opinion", "decree", "signed"]


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
    parsed = []
    for pattern in DATE_PATTERNS:
        for match in pattern.finditer(text):
            value = _parse_date(match.group(0), min_year, max_year)
            if value:
                parsed.append(value)
    return parsed


def _iter_pages(json_path: Path) -> Iterable[PageRecord]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    for page in payload.get("pages", []):
        yield PageRecord(page_number=page["page_number"], text=page.get("text", ""))


def _extract_case_header(text: str) -> Dict[str, str]:
    normalized = _normalize_ascii(_normalize_ws(text))
    info = {}
    match = re.search(r"Cause Number:\s*([A-Z0-9-]+)", normalized)
    if match:
        info["cause_number"] = match.group(1)
    match = re.search(r"([A-Z][A-Z ]+?)\s+v\s+([A-Z][A-Z ]+)", normalized)
    if match:
        info["party_a"] = match.group(1).strip()
        info["party_b"] = match.group(2).strip()
    return info


def _doc_type(text_lower: str) -> str | None:
    for label in ("order", "finding", "sworn", "pleading"):
        if any(term in text_lower for term in DOC_TYPE_RULES[label]):
            return label
    return None


def _issue_tags(text_lower: str) -> List[str]:
    tags = []
    for issue, keywords in ISSUE_CATEGORIES.items():
        if any(keyword in text_lower for keyword in keywords):
            tags.append(issue)
    return tags


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    if np.all(a == 0) or np.all(b == 0):
        return 0.0
    return float(1.0 - np.dot(a, b))


def _pca_2d(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] == 0:
        return np.zeros((0, 2))
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def _bin_series(values: List[float], bin_size: int) -> Tuple[np.ndarray, np.ndarray]:
    count = len(values)
    bin_count = math.ceil(count / bin_size)
    binned = np.zeros(bin_count, dtype=float)
    for idx, value in enumerate(values):
        binned[idx // bin_size] += value
    x = np.arange(bin_count) * bin_size + 1
    return x, binned


def _page_embeddings(
    model: SentenceTransformer, pages: List[PageRecord], batch_size: int
) -> np.ndarray:
    texts = [_normalize_ascii(page.text) for page in pages]
    nonempty_idx = [idx for idx, text in enumerate(texts) if text.strip()]
    embeddings = np.zeros((len(texts), model.get_sentence_embedding_dimension()), dtype=np.float32)
    if nonempty_idx:
        chunk = [texts[idx] for idx in nonempty_idx]
        vectors = model.encode(
            chunk,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        embeddings[nonempty_idx] = vectors
    return embeddings


def _semantic_drift(
    output_dir: Path,
    pages: List[PageRecord],
    embeddings: np.ndarray,
    parties: List[str],
    issues: List[str],
    min_year: int,
    max_year: int,
) -> List[str]:
    outputs = []
    for issue in issues:
        keywords = ISSUE_CATEGORIES[issue]
        plt.figure(figsize=(10, 5))
        plotted = False
        for party in parties:
            points = []
            for idx, page in enumerate(pages):
                text_norm = _normalize_ascii(page.text)
                text_lower = text_norm.lower()
                if party.lower() not in text_lower:
                    continue
                if not any(keyword in text_lower for keyword in keywords):
                    continue
                dates = _extract_dates(text_norm, min_year, max_year)
                if not dates:
                    continue
                points.append((min(dates), embeddings[idx]))
            if len(points) < 3:
                continue
            points.sort(key=lambda item: item[0])
            base = points[0][1]
            series = [(date, _cosine_distance(base, vector)) for date, vector in points]
            dates = [item[0] for item in series]
            distances = [item[1] for item in series]
            plt.plot(dates, distances, marker="o", label=party)
            plotted = True
        if plotted:
            plt.title(f"Semantic Drift Timeline - {issue}")
            plt.ylabel("Cosine Distance from First Mention")
            plt.xlabel("Date")
            plt.legend()
            plt.tight_layout()
            path = output_dir / f"semantic_drift_{issue.replace('/', '_')}.png"
            plt.savefig(path, dpi=160)
            outputs.append(path.name)
        plt.close()
    return outputs


def _polarity_score(text_lower: str) -> int:
    neg = sum(text_lower.count(term) for term in NEGATION_TERMS)
    pos = sum(text_lower.count(term) for term in AFFIRM_TERMS)
    return pos - neg


def _contradiction_map(
    output_dir: Path,
    pages: List[PageRecord],
    embeddings: np.ndarray,
    max_nodes: int,
    similarity_threshold: float,
) -> Tuple[str, str]:
    candidates = []
    for idx, page in enumerate(pages):
        text_norm = _normalize_ascii(page.text)
        text_lower = text_norm.lower()
        doc_type = _doc_type(text_lower)
        if doc_type is None:
            continue
        polarity = _polarity_score(text_lower)
        candidates.append((idx, doc_type, polarity))

    if not candidates:
        return "", ""

    candidates = candidates[:max_nodes]
    indices = [item[0] for item in candidates]
    doc_types = [item[1] for item in candidates]
    polarities = [item[2] for item in candidates]
    vectors = embeddings[indices]
    coords = _pca_2d(vectors)

    similarity = vectors @ vectors.T
    edges = []
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            if similarity[i, j] < similarity_threshold:
                continue
            if polarities[i] == 0 or polarities[j] == 0:
                continue
            if polarities[i] * polarities[j] > 0:
                continue
            edges.append((i, j, similarity[i, j]))

    edges.sort(key=lambda item: item[2], reverse=True)
    edges = edges[:200]

    plt.figure(figsize=(10, 8))
    for i, j, _ in edges:
        plt.plot(
            [coords[i, 0], coords[j, 0]],
            [coords[i, 1], coords[j, 1]],
            color="#999999",
            alpha=0.35,
            linewidth=0.8,
        )

    colors = {
        "order": "#204a87",
        "finding": "#4e9a06",
        "sworn": "#ce5c00",
        "pleading": "#5c3566",
    }
    for doc_type in set(doc_types):
        idxs = [i for i, value in enumerate(doc_types) if value == doc_type]
        plt.scatter(
            coords[idxs, 0],
            coords[idxs, 1],
            s=18,
            c=colors.get(doc_type, "#555555"),
            label=doc_type,
            alpha=0.9,
        )

    plt.title("Contradiction Map (High Similarity, Opposite Polarity)")
    plt.legend(loc="best", fontsize=8)
    plt.axis("off")
    plt.tight_layout()
    path = output_dir / "contradiction_map.png"
    plt.savefig(path, dpi=160)
    plt.close()

    edge_path = output_dir / "contradiction_edges.csv"
    with edge_path.open("w", encoding="utf-8") as handle:
        handle.write("node_a,node_b,page_a,page_b,doc_type_a,doc_type_b,similarity\n")
        for i, j, sim in edges:
            handle.write(
                f"{i},{j},{pages[indices[i]].page_number},{pages[indices[j]].page_number},"
                f"{doc_types[i]},{doc_types[j]},{sim:.4f}\n"
            )

    return path.name, edge_path.name


def _authority_leakage(
    output_path: Path, pages: List[PageRecord], bin_size: int
) -> None:
    authority = []
    statutory = []
    for page in pages:
        text_lower = _normalize_ascii(page.text).lower()
        authority.append(sum(text_lower.count(term) for term in AUTHORITY_TERMS))
        statutory.append(sum(text_lower.count(term) for term in STATUTORY_TERMS))

    x, authority_binned = _bin_series(authority, bin_size)
    _, statutory_binned = _bin_series(statutory, bin_size)
    ratio = np.divide(
        authority_binned + 1,
        statutory_binned + 1,
        out=np.zeros_like(authority_binned),
        where=True,
    )

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(x, authority_binned, color="#3465a4", label="Authority terms")
    ax1.plot(x, statutory_binned, color="#4e9a06", label="Statutory terms")
    ax1.set_xlabel(f"Page Range (bin size = {bin_size})")
    ax1.set_ylabel("Term Count")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(x, ratio, color="#cc0000", linestyle="--", label="Authority/Statute ratio")
    ax2.set_ylabel("Ratio")
    ax2.legend(loc="upper right")

    plt.title("Authority Leakage (Authority vs Statutory Language)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _issue_centroids(
    pages: List[PageRecord], embeddings: np.ndarray
) -> Dict[str, np.ndarray]:
    centroids = {}
    for issue, keywords in ISSUE_CATEGORIES.items():
        idxs = []
        for i, page in enumerate(pages):
            text_lower = _normalize_ascii(page.text).lower()
            if any(keyword in text_lower for keyword in keywords):
                idxs.append(i)
        if not idxs:
            continue
        centroid = embeddings[idxs].mean(axis=0)
        if np.linalg.norm(centroid) > 0:
            centroid = centroid / np.linalg.norm(centroid)
        centroids[issue] = centroid
    return centroids


def _procedural_gravity(
    output_path: Path,
    pages: List[PageRecord],
    embeddings: np.ndarray,
) -> None:
    centroids = _issue_centroids(pages, embeddings)
    issues = list(centroids.keys())
    index = {issue: idx for idx, issue in enumerate(issues)}
    flows = np.zeros((len(issues), len(issues)), dtype=int)

    for i, page in enumerate(pages):
        text_lower = _normalize_ascii(page.text).lower()
        tags = [issue for issue, keywords in ISSUE_CATEGORIES.items() if any(k in text_lower for k in keywords)]
        if len(tags) < 2:
            continue
        vector = embeddings[i]
        sims = {issue: float(np.dot(vector, centroids[issue])) for issue in tags if issue in centroids}
        if not sims:
            continue
        dominant = max(sims, key=sims.get)
        for tag in tags:
            if tag not in centroids:
                continue
            flows[index[dominant], index[tag]] += 1

    plt.figure(figsize=(8, 7))
    plt.imshow(flows, cmap="viridis")
    plt.title("Procedural Gravity Wells (Dominant vs Tagged Issues)")
    plt.xticks(np.arange(len(issues)), issues, rotation=45, ha="right")
    plt.yticks(np.arange(len(issues)), issues)
    plt.colorbar(label="Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _selective_attention(
    output_path: Path,
    pages: List[PageRecord],
    embeddings: np.ndarray,
    bin_size: int,
) -> None:
    brief_idxs = []
    order_idxs = []
    for idx, page in enumerate(pages):
        text_lower = _normalize_ascii(page.text).lower()
        if any(term in text_lower for term in BRIEF_TERMS):
            brief_idxs.append(idx)
        if any(term in text_lower for term in ORDER_TERMS):
            order_idxs.append(idx)

    if not brief_idxs or not order_idxs:
        return

    brief_centroid = embeddings[brief_idxs].mean(axis=0)
    order_centroid = embeddings[order_idxs].mean(axis=0)
    if np.linalg.norm(brief_centroid) > 0:
        brief_centroid = brief_centroid / np.linalg.norm(brief_centroid)
    if np.linalg.norm(order_centroid) > 0:
        order_centroid = order_centroid / np.linalg.norm(order_centroid)

    attention = []
    for vector in embeddings:
        score = max(float(np.dot(vector, brief_centroid)), float(np.dot(vector, order_centroid)))
        attention.append(score)

    x, attention_binned = _bin_series(attention, bin_size)
    silence = 1.0 - np.clip(attention_binned, 0, 1)

    plt.figure(figsize=(12, 6))
    plt.plot(x, attention_binned, color="#204a87", label="Engagement score")
    plt.plot(x, silence, color="#cc0000", linestyle="--", label="Silence score")
    plt.title("Selective Attention (Brief/Order Similarity Across Record)")
    plt.xlabel(f"Page Range (bin size = {bin_size})")
    plt.ylabel("Score")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _role_blind_plots(
    output_dir: Path,
    pages: List[PageRecord],
    embeddings: np.ndarray,
    sample_size: int,
) -> Tuple[str, str]:
    if sample_size <= 0 or sample_size >= len(pages):
        idxs = list(range(len(pages)))
    else:
        idxs = np.linspace(0, len(pages) - 1, sample_size, dtype=int).tolist()

    vectors = embeddings[idxs]
    coords = _pca_2d(vectors)

    roles = []
    for idx in idxs:
        text_lower = _normalize_ascii(pages[idx].text).lower()
        if any(term in text_lower for term in ORDER_TERMS):
            role = "judge"
        elif any(term in text_lower for term in BRIEF_TERMS):
            role = "party"
        elif "exhibit" in text_lower or "appendix" in text_lower:
            role = "record"
        else:
            role = "other"
        roles.append(role)

    plt.figure(figsize=(10, 7))
    plt.scatter(coords[:, 0], coords[:, 1], s=12, c="#888888", alpha=0.6)
    plt.title("Role-Blind Clustering (PCA Projection)")
    plt.axis("off")
    path_blind = output_dir / "role_blind_clusters.png"
    plt.tight_layout()
    plt.savefig(path_blind, dpi=160)
    plt.close()

    colors = {
        "judge": "#204a87",
        "party": "#4e9a06",
        "record": "#ce5c00",
        "other": "#555555",
    }
    plt.figure(figsize=(10, 7))
    for role in sorted(set(roles)):
        role_idxs = [i for i, value in enumerate(roles) if value == role]
        plt.scatter(
            coords[role_idxs, 0],
            coords[role_idxs, 1],
            s=14,
            c=colors.get(role, "#555555"),
            label=role,
            alpha=0.75,
        )
    plt.title("Role-Labeled Comparison (Same Projection)")
    plt.legend(loc="best", fontsize=8)
    plt.axis("off")
    path_labeled = output_dir / "role_labeled_clusters.png"
    plt.tight_layout()
    plt.savefig(path_labeled, dpi=160)
    plt.close()

    return path_blind.name, path_labeled.name


def _counterfactual_overlay(
    output_path: Path,
    embeddings: np.ndarray,
    baseline_embeddings: np.ndarray | None,
    bin_size: int,
) -> None:
    if baseline_embeddings is None or baseline_embeddings.size == 0:
        return
    baseline = baseline_embeddings.mean(axis=0)
    if np.linalg.norm(baseline) > 0:
        baseline = baseline / np.linalg.norm(baseline)
    similarity = embeddings @ baseline
    anomaly = 1.0 - np.clip(similarity, 0, 1)
    x, anomaly_binned = _bin_series(anomaly.tolist(), bin_size)

    plt.figure(figsize=(12, 6))
    plt.plot(x, anomaly_binned, color="#cc0000", linewidth=2)
    plt.title("Counterfactual Similarity Overlay (Anomaly Zones)")
    plt.xlabel(f"Page Range (bin size = {bin_size})")
    plt.ylabel("Anomaly (1 - baseline similarity)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _issue_cannibalization(
    output_path: Path,
    pages: List[PageRecord],
) -> None:
    issues = list(ISSUE_CATEGORIES.keys())
    index = {issue: idx for idx, issue in enumerate(issues)}
    matrix = np.zeros((len(issues), len(issues)), dtype=int)

    for page in pages:
        text_lower = _normalize_ascii(page.text).lower()
        scores = {issue: sum(text_lower.count(term) for term in keywords) for issue, keywords in ISSUE_CATEGORIES.items()}
        active = {issue: score for issue, score in scores.items() if score > 0}
        if len(active) < 2:
            continue
        dominant = max(active, key=active.get)
        for issue in active:
            if issue == dominant:
                continue
            matrix[index[dominant], index[issue]] += 1

    plt.figure(figsize=(8, 7))
    plt.imshow(matrix, cmap="plasma")
    plt.title("Issue Cannibalization (Dominant vs Secondary Issues)")
    plt.xticks(np.arange(len(issues)), issues, rotation=45, ha="right")
    plt.yticks(np.arange(len(issues)), issues)
    plt.colorbar(label="Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _authority_terms(series: List[int], bin_size: int) -> Tuple[np.ndarray, np.ndarray]:
    return _bin_series(series, bin_size)


def _issue_ranking(pages: List[PageRecord]) -> List[str]:
    totals = {}
    for issue, keywords in ISSUE_CATEGORIES.items():
        totals[issue] = sum(
            sum(_normalize_ascii(page.text).lower().count(term) for term in keywords)
            for page in pages
        )
    return sorted(totals, key=totals.get, reverse=True)


def _write_index(output_path: Path, images: List[Tuple[str, str]]) -> None:
    lines = [
        "<!doctype html>",
        "<html lang=\"en\">",
        "<head>",
        "<meta charset=\"utf-8\" />",
        "<title>28B Advanced Semantic Visuals</title>",
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
        "<h1>28B Advanced Semantic Visuals</h1>",
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


def _load_baseline_embeddings(path: Path | None) -> np.ndarray | None:
    if not path:
        return None
    if not path.exists():
        return None
    embeddings_path = path / "embeddings.npy"
    if not embeddings_path.exists():
        return None
    return np.load(embeddings_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build advanced semantic visuals.")
    parser.add_argument(
        "--json",
        type=Path,
        required=True,
        help="Path to the extracted JSON (page-level) file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/visuals_advanced"),
        help="Directory to write images and HTML.",
    )
    parser.add_argument(
        "--baseline-store",
        type=Path,
        default=Path("vector_store_research"),
        help="Vector store for baseline procedural language.",
    )
    parser.add_argument("--min-year", type=int, default=1800, help="Minimum year for dates.")
    parser.add_argument("--max-year", type=int, default=2100, help="Maximum year for dates.")
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size.")
    parser.add_argument("--bin-size", type=int, default=100, help="Page bin size.")
    parser.add_argument("--top-issues", type=int, default=3, help="Top issues for drift plots.")
    parser.add_argument("--contradiction-nodes", type=int, default=220, help="Max nodes for map.")
    parser.add_argument(
        "--contradiction-threshold",
        type=float,
        default=0.78,
        help="Cosine similarity threshold for contradictions.",
    )
    parser.add_argument(
        "--role-sample",
        type=int,
        default=0,
        help="Sample size for role plots (0 = all pages).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    json_path = args.json.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pages = list(_iter_pages(json_path))
    if not pages:
        raise ValueError("No pages found in JSON input.")

    header = _extract_case_header(pages[0].text)
    parties = [value for key, value in header.items() if key.startswith("party_")]
    if len(parties) < 2:
        parties = ["relator", "respondent"]

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = _page_embeddings(model, pages, args.batch_size)

    ranked_issues = _issue_ranking(pages)[: args.top_issues]
    drift_images = _semantic_drift(
        output_dir,
        pages,
        embeddings,
        parties,
        ranked_issues,
        args.min_year,
        args.max_year,
    )

    contradiction_img, contradiction_edges = _contradiction_map(
        output_dir,
        pages,
        embeddings,
        args.contradiction_nodes,
        args.contradiction_threshold,
    )

    authority_img = "authority_leakage.png"
    _authority_leakage(output_dir / authority_img, pages, args.bin_size)

    gravity_img = "procedural_gravity_wells.png"
    _procedural_gravity(output_dir / gravity_img, pages, embeddings)

    attention_img = "selective_attention.png"
    _selective_attention(output_dir / attention_img, pages, embeddings, args.bin_size)

    role_blind, role_labeled = _role_blind_plots(
        output_dir, pages, embeddings, args.role_sample
    )

    baseline_embeddings = _load_baseline_embeddings(args.baseline_store)
    anomaly_img = "counterfactual_anomaly_overlay.png"
    _counterfactual_overlay(output_dir / anomaly_img, embeddings, baseline_embeddings, args.bin_size)

    cannibal_img = "issue_cannibalization.png"
    _issue_cannibalization(output_dir / cannibal_img, pages)

    images = []
    for name in drift_images:
        images.append((name, f"Semantic drift timeline ({name.replace('semantic_drift_', '').replace('.png', '')})"))
    if contradiction_img:
        images.append((contradiction_img, "Contradiction map (similar + opposite polarity)"))
    images.extend(
        [
            (authority_img, "Authority leakage (authority vs statute terms)"),
            (gravity_img, "Procedural gravity wells (dominant vs tagged issues)"),
            (attention_img, "Selective attention (brief/order similarity)"),
            (role_blind, "Role-blind cluster projection"),
            (role_labeled, "Role-labeled comparison"),
            (anomaly_img, "Counterfactual anomaly overlay (baseline similarity)"),
            (cannibal_img, "Issue cannibalization (dominant vs secondary)"),
        ]
    )

    index_path = output_dir / "index.html"
    _write_index(index_path, images)

    if contradiction_edges:
        print(f"Saved contradiction edges to {contradiction_edges}")
    print(f"Wrote advanced visuals to {output_dir}")


if __name__ == "__main__":
    main()
