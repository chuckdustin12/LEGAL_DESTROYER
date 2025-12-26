"""Build expanded visuals for the 28B record."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
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
    "recusal": ["recusal", "recuse", "disqualify", "disqualification"],
    "emergency relief": [
        "emergency relief",
        "emergency motion",
        "temporary restraining order",
        "ex parte",
        "tro",
    ],
    "temporary orders": ["temporary orders", "temporary order", "temporary injunction"],
    "notice/hearing": [
        "notice of hearing",
        "notice of trial",
        "hearing set",
        "trial setting",
    ],
    "jurisdiction/void": ["jurisdiction", "void order", "lack of jurisdiction"],
    "mandamus/appeal": ["mandamus", "writ", "appeal", "appellate"],
    "sanctions/fees": ["sanctions", "attorney's fees", "attorneys fees", "fees", "contempt"],
    "custody/child": ["custody", "conservatorship", "possession", "child support"],
    "property/financial": ["property", "bank", "account", "transfer", "fraud", "asset"],
}

CLAIM_CATEGORIES = {
    "protective order": ["protective order", "order of protection", "ex parte order"],
    "indigency/affidavit": [
        "affidavit of inability",
        "statement of inability",
        "indigent",
        "pauper",
    ],
    "paypal/transfer": ["paypal", "branthoover", "1,576", "1576"],
    "recusal process": ["recusal", "order of referral", "associate judge", "201.006"],
    "temporary orders": ["temporary orders", "temporary order"],
    "family violence": ["family violence", "domestic violence"],
    "child abuse": ["child abuse", "child neglect", "injury to a child"],
    "agreement/settlement": ["agreement", "agreed order", "settlement", "rule 11"],
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

AFFIDAVIT_PATTERN = re.compile(
    r"\baffidavit\b|\bsworn\b|under penalty of perjury|declaration|verified|jurat",
    re.IGNORECASE,
)
FILING_PATTERN = re.compile(
    r"\bpetition\b|\bapplication\b|\bmotion\b|\bnotice\b|\bresponse\b|\banswer\b|"
    r"\bcounterpetition\b|\bstatement of inability\b",
    re.IGNORECASE,
)

DOCKET_HEADER = "ALL TRANSACTIONS FOR A CASE"
DOCKET_ENTRY = re.compile(r"(\d{1,4})\s+(\d{2}/\d{2}/\d{4})\s+")

FILER_ORDER = [
    "charles_dustin_myers",
    "cooper_carter",
    "morgan_michelle_myers",
    "court",
    "clerk",
    "oag",
    "unknown",
]

FILER_LABELS = {
    "charles_dustin_myers": "Charles",
    "cooper_carter": "Counsel",
    "morgan_michelle_myers": "Petitioner",
    "court": "Court",
    "clerk": "Clerk",
    "oag": "OAG",
    "unknown": "Unknown",
}

FILER_RULES = {
    "charles_dustin_myers": [
        (re.compile(r"/s/\s*Charles\s+Dustin\s+Myers", re.IGNORECASE), 6),
        (re.compile(r"Charles\s+Dustin\s+Myers", re.IGNORECASE), 3),
        (re.compile(r"chuckdustin12@gmail\.com", re.IGNORECASE), 4),
        (re.compile(r"\bpro\s+se\b", re.IGNORECASE), 2),
    ],
    "cooper_carter": [
        (re.compile(r"/s/\s*Cooper\s+L\.?\s+Carter", re.IGNORECASE), 6),
        (re.compile(r"Cooper\s+L\.?\s+Carter", re.IGNORECASE), 4),
        (re.compile(r"majadmin\.com", re.IGNORECASE), 3),
    ],
    "morgan_michelle_myers": [
        (re.compile(r"/s/\s*Morgan\s+Michelle\s+Myers", re.IGNORECASE), 6),
        (re.compile(r"Morgan\s+Michelle\s+Myers", re.IGNORECASE), 3),
    ],
    "court": [
        (re.compile(r"Court\s+of\s+Appeals", re.IGNORECASE), 4),
        (re.compile(r"Supreme\s+Court\s+of\s+Texas", re.IGNORECASE), 4),
        (re.compile(r"\bOPINION\b", re.IGNORECASE), 2),
        (re.compile(r"\bORDER\b", re.IGNORECASE), 1),
    ],
    "clerk": [
        (re.compile(r"District\s+Clerk", re.IGNORECASE), 4),
        (re.compile(r"Clerk's\s+Office", re.IGNORECASE), 4),
        (re.compile(r"ALL\s+TRANSACTIONS\s+FOR\s+A\s+CASE", re.IGNORECASE), 6),
        (re.compile(r"Certified\s+Copy", re.IGNORECASE), 3),
    ],
    "oag": [
        (re.compile(r"Office\s+of\s+the\s+Attorney\s+General", re.IGNORECASE), 4),
        (re.compile(r"oag\.texas\.gov", re.IGNORECASE), 4),
    ],
}


@dataclass
class PageRecord:
    page_number: int
    text: str


@dataclass
class DocketEntry:
    filemark: str
    date: datetime
    description: str


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _normalize_ascii(text: str) -> str:
    cleaned = text.translate(NON_ASCII_MAP)
    return cleaned.encode("ascii", "ignore").decode("ascii")


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _iter_pages(json_path: Path) -> Iterable[PageRecord]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    for page in payload.get("pages", []):
        yield PageRecord(page_number=page["page_number"], text=page.get("text", ""))


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


def _extract_dates_with_context(
    text: str, min_year: int, max_year: int
) -> List[Tuple[datetime, str, str]]:
    normalized = _normalize_ascii(text)
    matches: List[Tuple[datetime, str, str]] = []
    for pattern in DATE_PATTERNS:
        for match in pattern.finditer(normalized):
            parsed = _parse_date(match.group(0), min_year, max_year)
            if not parsed:
                continue
            start = max(0, match.start() - 80)
            end = min(len(normalized), match.end() + 120)
            context = normalized[start:end]
            matches.append((parsed, match.group(0), context))
    return matches


def _month_key(date_value: datetime) -> str:
    return date_value.strftime("%Y-%m")


def _classify_context(context: str) -> str:
    lowered = context.lower()
    order_terms = ["order", "judgment", "signed", "decree", "ruling", "report"]
    hearing_terms = ["hearing", "setting", "trial", "conference", "appearance"]
    filing_terms = [
        "filed",
        "filing",
        "petition",
        "application",
        "motion",
        "notice",
        "response",
        "answer",
        "counterpetition",
        "objection",
    ]
    if any(term in lowered for term in order_terms):
        return "orders"
    if any(term in lowered for term in hearing_terms):
        return "hearings"
    if any(term in lowered for term in filing_terms):
        return "filings"
    return "events"


def _timeline_counts(
    pages: List[PageRecord], min_year: int, max_year: int
) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
    total_counts: Dict[str, int] = {}
    type_counts: Dict[str, Dict[str, int]] = {
        "filings": {},
        "orders": {},
        "hearings": {},
        "events": {},
    }
    for page in pages:
        for parsed, _raw, context in _extract_dates_with_context(
            page.text, min_year, max_year
        ):
            month = _month_key(parsed)
            total_counts[month] = total_counts.get(month, 0) + 1
            bucket = _classify_context(context)
            type_counts[bucket][month] = type_counts[bucket].get(month, 0) + 1
    return total_counts, type_counts


def _parse_docket_entries(json_path: Path) -> List[DocketEntry]:
    entries: List[DocketEntry] = []
    for page in _iter_pages(json_path):
        if DOCKET_HEADER not in page.text:
            continue
        normalized = _normalize_ascii(page.text)
        normalized = re.sub(
            r"(\d+\.\d{2})(\d{1,4})\s+(\d{2}/\d{2}/\d{4})",
            r"\1 \2 \3",
            normalized,
        )
        marker = normalized.find("Filemark")
        if marker != -1:
            normalized = normalized[marker:]
        matches = list(DOCKET_ENTRY.finditer(normalized))
        for idx, match in enumerate(matches):
            filemark_raw = match.group(1)
            try:
                filemark = str(int(filemark_raw))
            except ValueError:
                continue
            if filemark == "0":
                continue
            date_text = match.group(2)
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(normalized)
            raw_desc = normalized[start:end]
            if not raw_desc.strip():
                continue
            if "Date Filed" in raw_desc or "Page" in raw_desc:
                continue
            try:
                date_value = datetime.strptime(date_text, "%m/%d/%Y")
            except ValueError:
                continue
            description = _normalize_ws(raw_desc)
            entries.append(
                DocketEntry(
                    filemark=filemark,
                    date=date_value,
                    description=description,
                )
            )
    return entries


def _month_series(month_counts: Dict[str, int]) -> Tuple[List[str], List[int]]:
    if not month_counts:
        return [], []
    months = sorted(month_counts.keys())
    return months, [month_counts.get(month, 0) for month in months]


def _plot_line_series(
    output_path: Path, months: List[str], values: List[int], title: str, ylabel: str
) -> None:
    if not months:
        return
    x = np.arange(len(months))
    plt.figure(figsize=(12, 5))
    plt.plot(x, values, color="#3465a4", linewidth=2)
    step = max(1, len(months) // 12)
    plt.xticks(x[::step], months[::step], rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_stack_series(
    output_path: Path, months: List[str], series: Dict[str, List[int]], title: str
) -> None:
    if not months or not series:
        return
    x = np.arange(len(months))
    labels = list(series.keys())
    data = np.vstack([series[label] for label in labels])
    plt.figure(figsize=(12, 5))
    plt.stackplot(x, data, labels=labels, alpha=0.85)
    step = max(1, len(months) // 12)
    plt.xticks(x[::step], months[::step], rotation=45, ha="right")
    plt.ylabel("Mentions")
    plt.title(title)
    plt.legend(loc="upper left", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _bin_series(values: List[int], bin_size: int) -> Tuple[np.ndarray, np.ndarray]:
    count = len(values)
    bin_count = math.ceil(count / bin_size)
    binned = np.zeros(bin_count, dtype=int)
    for idx, value in enumerate(values):
        binned[idx // bin_size] += value
    x = np.arange(bin_count) * bin_size + 1
    return x, binned


def _issue_counts_by_page(
    pages: List[PageRecord], categories: Dict[str, List[str]]
) -> Dict[str, List[int]]:
    results: Dict[str, List[int]] = {key: [0] * len(pages) for key in categories}
    for idx, page in enumerate(pages):
        text_lower = _normalize_ascii(page.text).lower()
        for label, keywords in categories.items():
            results[label][idx] = sum(text_lower.count(keyword) for keyword in keywords)
    return results


def _plot_heatmap(
    output_path: Path,
    counts_by_page: Dict[str, List[int]],
    bin_size: int,
    title: str,
) -> None:
    labels = list(counts_by_page.keys())
    if not labels:
        return
    total_pages = len(next(iter(counts_by_page.values())))
    bins = math.ceil(total_pages / bin_size)
    matrix = np.zeros((len(labels), bins), dtype=int)
    for row, label in enumerate(labels):
        _, binned = _bin_series(counts_by_page[label], bin_size)
        matrix[row, :] = binned

    x_labels = []
    for idx in range(bins):
        start = idx * bin_size + 1
        end = min((idx + 1) * bin_size, total_pages)
        x_labels.append(f"{start}-{end}")

    plt.figure(figsize=(13, 6))
    im = plt.imshow(matrix, aspect="auto", cmap="viridis")
    plt.title(title)
    plt.ylabel("Category")
    plt.xlabel(f"Page Range (bin size = {bin_size})")
    plt.yticks(np.arange(len(labels)), labels)
    step = max(1, len(x_labels) // 12)
    plt.xticks(
        np.arange(0, len(x_labels), step),
        [x_labels[i] for i in range(0, len(x_labels), step)],
        rotation=45,
        ha="right",
    )
    plt.colorbar(im, label="Keyword hits")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _load_docket_filer_map(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not path.exists():
        return mapping
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            filemark = row.get("filemark", "").strip()
            filer = row.get("filer", "").strip()
            if filemark and filer:
                mapping[filemark] = filer
    return mapping


def _series_by_filer(
    entries: List[DocketEntry], docket_map: Dict[str, str]
) -> Tuple[List[str], Dict[str, List[int]]]:
    counts: Dict[str, Dict[str, int]] = {}
    all_months: set[str] = set()
    for entry in entries:
        filer = docket_map.get(entry.filemark, "unknown")
        month = _month_key(entry.date)
        counts.setdefault(filer, {})
        counts[filer][month] = counts[filer].get(month, 0) + 1
        all_months.add(month)
    months = sorted(all_months)
    series: Dict[str, List[int]] = {}
    for filer in FILER_ORDER:
        if filer not in counts:
            continue
        series[filer] = [counts[filer].get(month, 0) for month in months]
    return months, series


def _plot_filer_stack(
    output_path: Path, months: List[str], series: Dict[str, List[int]]
) -> None:
    if not months or not series:
        return
    filers = [f for f in FILER_ORDER if f in series]
    data = np.vstack([series[filer] for filer in filers])
    x = np.arange(len(months))
    labels = [FILER_LABELS.get(filer, filer) for filer in filers]
    plt.figure(figsize=(12, 6))
    plt.stackplot(x, data, labels=labels, alpha=0.85)
    step = max(1, len(months) // 12)
    plt.xticks(x[::step], months[::step], rotation=45, ha="right")
    plt.ylabel("Filings")
    plt.title("Filings by Month (Stacked by Filer)")
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_filer_cumulative(
    output_path: Path, months: List[str], series: Dict[str, List[int]]
) -> None:
    if not months or not series:
        return
    x = np.arange(len(months))
    plt.figure(figsize=(12, 6))
    for filer in FILER_ORDER:
        if filer not in series:
            continue
        cumulative = np.cumsum(series[filer])
        label = FILER_LABELS.get(filer, filer)
        plt.plot(x, cumulative, label=label, linewidth=2)
    step = max(1, len(months) // 12)
    plt.xticks(x[::step], months[::step], rotation=45, ha="right")
    plt.ylabel("Cumulative Filings")
    plt.title("Cumulative Filings by Month")
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _score_filer(text: str) -> str:
    scores: Dict[str, int] = {}
    for filer, rules in FILER_RULES.items():
        for pattern, weight in rules:
            if pattern.search(text):
                scores[filer] = scores.get(filer, 0) + weight
    if not scores:
        return "unknown"
    max_score = max(scores.values())
    winners = [filer for filer, score in scores.items() if score == max_score]
    for filer in FILER_ORDER:
        if filer in winners:
            return filer
    return "unknown"


def _load_page_filer_labels(
    page_map_path: Path, pages: List[PageRecord]
) -> List[str]:
    if page_map_path.exists():
        labels = []
        with page_map_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                labels.append(row.get("filer", "unknown"))
        if len(labels) == len(pages):
            return labels
    labels = []
    for page in pages:
        labels.append(_score_filer(_normalize_ascii(page.text)))
    return labels


def _plot_page_share(output_path: Path, labels: List[str]) -> None:
    counts: Dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    keys = [f for f in FILER_ORDER if f in counts]
    values = [counts[f] for f in keys]
    plt.figure(figsize=(8, 6))
    plt.bar([FILER_LABELS.get(f, f) for f in keys], values, color="#3465a4")
    plt.title("Page Share by Filer")
    plt.ylabel("Pages")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _load_vector_store(store_dir: Path) -> Tuple[np.ndarray, List[dict]]:
    embeddings = np.load(store_dir / "embeddings.npy")
    records = [
        json.loads(line)
        for line in (store_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return embeddings, records


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


def _find_contradictions(
    embeddings: np.ndarray,
    records: List[dict],
    *,
    similarity_threshold: float,
    min_polarity_hits: int,
    top_k: int,
    max_edges: int,
) -> List[dict]:
    affidavit_indices = [
        idx for idx, record in enumerate(records) if AFFIDAVIT_PATTERN.search(record.get("text", ""))
    ]
    filing_indices = [
        idx for idx, record in enumerate(records) if FILING_PATTERN.search(record.get("text", ""))
    ]
    if not affidavit_indices or not filing_indices:
        return []

    affidavit_vectors = embeddings[affidavit_indices]
    filing_vectors = embeddings[filing_indices]
    sims = affidavit_vectors @ filing_vectors.T

    polarity_cache: Dict[int, int] = {}
    edges: List[dict] = []
    for i, aff_idx in enumerate(affidavit_indices):
        aff_record = records[aff_idx]
        aff_text = aff_record.get("text", "")
        if aff_idx not in polarity_cache:
            polarity_cache[aff_idx] = _polarity_sign(aff_text, min_polarity_hits)
        aff_sign = polarity_cache[aff_idx]
        if aff_sign == 0:
            continue
        scores = sims[i]
        ranked = np.argsort(-scores)[:top_k]
        for pos in ranked:
            fil_idx = filing_indices[int(pos)]
            if records[fil_idx].get("vector_id", fil_idx) >= records[aff_idx].get(
                "vector_id", aff_idx
            ):
                continue
            fil_text = records[fil_idx].get("text", "")
            if fil_idx not in polarity_cache:
                polarity_cache[fil_idx] = _polarity_sign(fil_text, min_polarity_hits)
            fil_sign = polarity_cache[fil_idx]
            if fil_sign == 0 or aff_sign * fil_sign >= 0:
                continue
            score = float(scores[int(pos)])
            if score < similarity_threshold:
                continue
            edges.append(
                {
                    "similarity": round(score, 4),
                    "affidavit_idx": aff_idx,
                    "filing_idx": fil_idx,
                    "affidavit_vector_id": records[aff_idx].get("vector_id"),
                    "filing_vector_id": records[fil_idx].get("vector_id"),
                    "affidavit_source": records[aff_idx].get("source_pdf", ""),
                    "filing_source": records[fil_idx].get("source_pdf", ""),
                }
            )
    edges.sort(key=lambda item: item["similarity"], reverse=True)
    return edges[:max_edges]


def _pca_2d(vectors: np.ndarray) -> np.ndarray:
    if vectors.shape[0] == 1:
        return np.zeros((1, 2))
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    u, s, _vt = np.linalg.svd(centered, full_matrices=False)
    coords = u[:, :2] * s[:2]
    return coords


def _plot_contradiction_network(
    output_path: Path,
    edges: List[dict],
    embeddings: np.ndarray,
    records: List[dict],
) -> None:
    if not edges:
        return
    node_ids = sorted({edge["affidavit_idx"] for edge in edges} | {edge["filing_idx"] for edge in edges})
    vectors = embeddings[node_ids]
    coords = _pca_2d(vectors)
    node_pos = {node_id: coords[idx] for idx, node_id in enumerate(node_ids)}

    aff_nodes = {edge["affidavit_idx"] for edge in edges}
    fil_nodes = {edge["filing_idx"] for edge in edges}

    plt.figure(figsize=(10, 8))
    for edge in edges:
        a = node_pos[edge["affidavit_idx"]]
        b = node_pos[edge["filing_idx"]]
        alpha = min(0.8, 0.2 + (edge["similarity"] - 0.7))
        plt.plot([a[0], b[0]], [a[1], b[1]], color="#555555", alpha=alpha, linewidth=1)

    aff_coords = np.array([node_pos[node_id] for node_id in node_ids if node_id in aff_nodes])
    fil_coords = np.array([node_pos[node_id] for node_id in node_ids if node_id in fil_nodes])
    if aff_coords.size:
        plt.scatter(
            aff_coords[:, 0],
            aff_coords[:, 1],
            color="#c0392b",
            label="Affidavit chunks",
            s=50,
            edgecolors="white",
            linewidth=0.6,
        )
    if fil_coords.size:
        plt.scatter(
            fil_coords[:, 0],
            fil_coords[:, 1],
            color="#2980b9",
            label="Earlier filings",
            s=40,
            edgecolors="white",
            linewidth=0.6,
        )

    if len(node_ids) <= 40:
        for node_id in node_ids:
            vector_id = records[node_id].get("vector_id", node_id)
            x, y = node_pos[node_id]
            plt.text(x, y, f"{vector_id}", fontsize=7, ha="center", va="center")

    plt.title("Affidavit vs Earlier Filings - Contradiction Network (heuristic)")
    plt.axis("off")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _write_edges_csv(output_path: Path, edges: List[dict]) -> None:
    if not edges:
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "similarity",
                "affidavit_vector_id",
                "filing_vector_id",
                "affidavit_source",
                "filing_source",
            ],
        )
        writer.writeheader()
        for edge in edges:
            writer.writerow(
                {
                    "similarity": edge["similarity"],
                    "affidavit_vector_id": edge["affidavit_vector_id"],
                    "filing_vector_id": edge["filing_vector_id"],
                    "affidavit_source": edge["affidavit_source"],
                    "filing_source": edge["filing_source"],
                }
            )


def _compile_patterns(patterns: Dict[str, List[str]]) -> Dict[str, List[re.Pattern]]:
    return {key: [re.compile(p, re.IGNORECASE) for p in values] for key, values in patterns.items()}


def _evidence_counts_by_page(pages: List[PageRecord]) -> Dict[str, List[int]]:
    compiled = _compile_patterns(EVIDENCE_MARKERS)
    screenshot_patterns = [re.compile(p, re.IGNORECASE) for p in SCREENSHOT_MARKERS]
    counts_by_page: Dict[str, List[int]] = {key: [0] * len(pages) for key in compiled}
    counts_by_page["screenshot"] = [0] * len(pages)

    for idx, page in enumerate(pages):
        text_lower = _normalize_ascii(page.text).lower()
        for key, patterns in compiled.items():
            counts_by_page[key][idx] = sum(len(pattern.findall(text_lower)) for pattern in patterns)
        counts_by_page["screenshot"][idx] = sum(
            len(pattern.findall(text_lower)) for pattern in screenshot_patterns
        )
    return counts_by_page


def _plot_evidence_totals(output_path: Path, counts_by_page: Dict[str, List[int]]) -> None:
    labels = list(counts_by_page.keys())
    totals = [sum(counts_by_page[label]) for label in labels]
    plt.figure(figsize=(10, 6))
    plt.bar(labels, totals, color="#4e9a06")
    plt.title("Evidence Marker Totals (Full Record)")
    plt.ylabel("Total hits")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_evidence_density(
    output_path: Path, counts_by_page: Dict[str, List[int]], bin_size: int
) -> None:
    plt.figure(figsize=(12, 6))
    for label, counts in counts_by_page.items():
        x, binned = _bin_series(counts, bin_size)
        plt.plot(x, binned, label=label, linewidth=2)
    plt.title("Evidence Marker Density by Page Range")
    plt.xlabel(f"Page Range (bin size = {bin_size})")
    plt.ylabel("Hits")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_evidence_compare(
    output_path: Path, counts_by_page: Dict[str, List[int]], bin_size: int
) -> None:
    x, affidavit = _bin_series(counts_by_page.get("affidavit_or_sworn", []), bin_size)
    _, auth = _bin_series(counts_by_page.get("authentication", []), bin_size)
    _, screenshots = _bin_series(counts_by_page.get("screenshot", []), bin_size)
    plt.figure(figsize=(12, 6))
    plt.plot(x, affidavit, label="affidavit_or_sworn", color="#c0392b", linewidth=2)
    plt.plot(x, auth, label="authentication", color="#2980b9", linewidth=2)
    plt.plot(x, screenshots, label="screenshot", color="#8e44ad", linewidth=2)
    plt.title("Affidavit vs Authentication vs Screenshot Density")
    plt.xlabel(f"Page Range (bin size = {bin_size})")
    plt.ylabel("Hits")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _write_index(output_path: Path, sections: List[Tuple[str, List[Tuple[str, str]]]]) -> None:
    lines = [
        "<!doctype html>",
        "<html lang=\"en\">",
        "<head>",
        "<meta charset=\"utf-8\" />",
        "<title>28B Expanded Visuals</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 24px; background: #f5f5f5; }",
        "h1 { margin-bottom: 8px; }",
        "h2 { margin-top: 28px; }",
        ".grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; }",
        ".card { background: #fff; padding: 16px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }",
        ".card img { width: 100%; height: auto; border-radius: 6px; }",
        ".caption { margin-top: 8px; font-size: 14px; color: #444; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>28B Expanded Visuals</h1>",
        f"<p>Generated: {_timestamp()}</p>",
    ]
    for title, images in sections:
        lines.append(f"<h2>{title}</h2>")
        lines.append("<div class=\"grid\">")
        for filename, caption in images:
            lines.extend(
                [
                    "<div class=\"card\">",
                    f"<img src=\"{filename}\" alt=\"{caption}\" />",
                    f"<div class=\"caption\">{caption}</div>",
                    "</div>",
                ]
            )
        lines.append("</div>")
    lines.extend(["</body>", "</html>"])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build expanded visuals for the 28B record.")
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("extracted_text_full/28b_merged/28B_merged.json"),
        help="Path to the extracted JSON (page-level) file.",
    )
    parser.add_argument(
        "--store",
        type=Path,
        default=Path("vector_store_28b"),
        help="Vector store directory for contradiction network.",
    )
    parser.add_argument(
        "--docket-filer-map",
        type=Path,
        default=Path("reports/28b_docket_filer_map.csv"),
        help="CSV map of filemark-to-filer.",
    )
    parser.add_argument(
        "--page-filer-map",
        type=Path,
        default=Path("reports/28b_filer_pages.csv"),
        help="CSV page-to-filer map (optional).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/visuals_expanded"),
        help="Directory to write images and HTML.",
    )
    parser.add_argument("--min-year", type=int, default=1800, help="Minimum year for dates.")
    parser.add_argument("--max-year", type=int, default=2100, help="Maximum year for dates.")
    parser.add_argument("--bin-size", type=int, default=100, help="Page bin size for heatmaps.")
    parser.add_argument("--similarity-threshold", type=float, default=0.78)
    parser.add_argument("--min-polarity-hits", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--max-edges", type=int, default=140)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    json_path = args.json.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pages = list(_iter_pages(json_path))
    if not pages:
        raise ValueError("No pages found in JSON input.")

    total_counts, type_counts = _timeline_counts(pages, args.min_year, args.max_year)
    months, totals = _month_series(total_counts)

    timeline_total_path = output_dir / "timeline_event_mentions.png"
    _plot_line_series(
        timeline_total_path, months, totals, "Event Date Mentions by Month", "Mentions"
    )

    timeline_type_months = months
    type_series = {}
    for label in ["filings", "orders", "hearings", "events"]:
        values = [type_counts.get(label, {}).get(month, 0) for month in timeline_type_months]
        type_series[label] = values
    timeline_type_path = output_dir / "timeline_event_types.png"
    _plot_stack_series(
        timeline_type_path,
        timeline_type_months,
        type_series,
        "Event Types Over Time (Date Mentions)",
    )

    docket_entries = _parse_docket_entries(json_path)
    docket_month_counts: Dict[str, int] = {}
    for entry in docket_entries:
        month = _month_key(entry.date)
        docket_month_counts[month] = docket_month_counts.get(month, 0) + 1
    docket_months, docket_values = _month_series(docket_month_counts)
    docket_timeline_path = output_dir / "timeline_docket_filings.png"
    _plot_line_series(
        docket_timeline_path,
        docket_months,
        docket_values,
        "Docket Filings by Month",
        "Filings",
    )

    issue_counts = _issue_counts_by_page(pages, ISSUE_CATEGORIES)
    issue_heatmap_path = output_dir / "issue_heatmap.png"
    _plot_heatmap(issue_heatmap_path, issue_counts, args.bin_size, "Issue Heatmap Across 28B")

    claim_counts = _issue_counts_by_page(pages, CLAIM_CATEGORIES)
    claim_heatmap_path = output_dir / "claim_heatmap.png"
    _plot_heatmap(claim_heatmap_path, claim_counts, args.bin_size, "Claim Heatmap Across 28B")

    docket_map = _load_docket_filer_map(args.docket_filer_map.expanduser().resolve())
    filer_months, filer_series = _series_by_filer(docket_entries, docket_map)
    filer_stack_path = output_dir / "filer_filings_by_month.png"
    _plot_filer_stack(filer_stack_path, filer_months, filer_series)
    filer_cumulative_path = output_dir / "filer_filings_cumulative.png"
    _plot_filer_cumulative(filer_cumulative_path, filer_months, filer_series)

    page_labels = _load_page_filer_labels(args.page_filer_map.expanduser().resolve(), pages)
    page_share_path = output_dir / "filer_page_share.png"
    _plot_page_share(page_share_path, page_labels)

    embeddings, records = _load_vector_store(args.store.expanduser().resolve())
    contradiction_edges = _find_contradictions(
        embeddings,
        records,
        similarity_threshold=args.similarity_threshold,
        min_polarity_hits=args.min_polarity_hits,
        top_k=args.top_k,
        max_edges=args.max_edges,
    )
    contradiction_path = output_dir / "affidavit_contradiction_network.png"
    _plot_contradiction_network(contradiction_path, contradiction_edges, embeddings, records)
    _write_edges_csv(output_dir / "affidavit_contradiction_edges.csv", contradiction_edges)

    evidence_counts = _evidence_counts_by_page(pages)
    evidence_totals_path = output_dir / "evidence_marker_totals.png"
    evidence_density_path = output_dir / "evidence_marker_density.png"
    evidence_compare_path = output_dir / "evidence_marker_comparison.png"
    _plot_evidence_totals(evidence_totals_path, evidence_counts)
    _plot_evidence_density(evidence_density_path, evidence_counts, args.bin_size)
    _plot_evidence_compare(evidence_compare_path, evidence_counts, args.bin_size)

    sections = [
        (
            "Timeline Visuals",
            [
                (timeline_total_path.name, "Event date mentions by month"),
                (timeline_type_path.name, "Event type breakdown (filings, orders, hearings)"),
                (docket_timeline_path.name, "Docket filings by month"),
            ],
        ),
        (
            "Issue and Claim Heatmaps",
            [
                (issue_heatmap_path.name, "Issue keyword heatmap (page bins)"),
                (claim_heatmap_path.name, "Claim keyword heatmap (page bins)"),
            ],
        ),
        (
            "Filer-Based Trends",
            [
                (filer_stack_path.name, "Filings by month (stacked by filer)"),
                (filer_cumulative_path.name, "Cumulative filings by filer"),
                (page_share_path.name, "Page share by filer (full record)"),
            ],
        ),
        (
            "Contradiction Network",
            [
                (
                    contradiction_path.name,
                    "Affidavit vs earlier filings (semantic + polarity heuristic)",
                )
            ],
        ),
        (
            "Evidence Marker Dashboard",
            [
                (evidence_totals_path.name, "Evidence marker totals"),
                (evidence_density_path.name, "Evidence marker density by page range"),
                (evidence_compare_path.name, "Affidavit vs authentication vs screenshot density"),
            ],
        ),
    ]
    _write_index(output_dir / "index.html", sections)

    print(f"Wrote expanded visuals to {output_dir}")


if __name__ == "__main__":
    main()
