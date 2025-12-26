"""Map draft allegations to 28B record excerpts for quick cite checking."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parent.parent

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


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _normalize_ascii(text: str) -> str:
    return text.translate(NON_ASCII_MAP).encode("ascii", "ignore").decode("ascii")


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _snippet_around_match(text: str, patterns: List[re.Pattern], max_len: int = 320) -> str:
    cleaned = _normalize_ws(_normalize_ascii(text))
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
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3].rstrip() + "..."


def _load_records(store_dir: Path) -> List[dict]:
    metadata_path = store_dir / "metadata.jsonl"
    return [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]


def _score_record(text: str, patterns: Iterable[re.Pattern]) -> int:
    return sum(1 for pattern in patterns if pattern.search(text))


def _claim_hits(
    records: List[dict],
    patterns: List[re.Pattern],
    max_hits: int,
) -> List[dict]:
    scored: List[tuple[int, dict]] = []
    for record in records:
        text = record.get("text", "")
        score = _score_record(text, patterns)
        if score <= 0:
            continue
        scored.append((score, record))
    scored.sort(key=lambda item: (-item[0], item[1].get("vector_id", 0)))

    hits: List[dict] = []
    seen = set()
    for score, record in scored:
        key = record.get("id")
        if key in seen:
            continue
        seen.add(key)
        hits.append(
            {
                "score": score,
                "vector_id": record.get("vector_id"),
                "chunk_index": record.get("chunk_index"),
                "source_pdf": record.get("source_pdf", ""),
                "snippet": _snippet_around_match(record.get("text", ""), patterns),
            }
        )
        if len(hits) >= max_hits:
            break
    return hits


CLAIMS = [
    {
        "id": "M1",
        "section": "I. Morgan - Funding/PayPal",
        "claim": (
            "Transfer of $1,576 in marital funds via Daniel Branthoover's PayPal "
            "on or around 12/15/2023."
        ),
        "queries": [
            r"1,576",
            r"\b1576\b",
            r"PayPal",
            r"Branthoover",
            r"12/15/2023",
            r"December 15, 2023",
        ],
    },
    {
        "id": "M2",
        "section": "I. Morgan - Oklahoma Trip/Phone",
        "claim": (
            "Travel to Yukon, Oklahoma to draft initial documents and acquire a secondary "
            "phone registered to 817-940-0852."
        ),
        "queries": [
            r"Yukon, Oklahoma",
            r"Oklahoma",
            r"817-940-0852",
            r"drafting the initial documents",
            r"assistance in drafting",
            r"secondary phone",
        ],
    },
    {
        "id": "M3",
        "section": "I. Morgan - Affidavit/Indigency",
        "claim": (
            "Affidavit/statement of inability to pay court costs and allegations that it was false."
        ),
        "queries": [
            r"affidavit of inability",
            r"statement of inability",
            r"indigenc",
            r"pauper",
            r"false",
        ],
    },
    {
        "id": "M4",
        "section": "I. Morgan - Protective Order Representations",
        "claim": (
            "Representation of an active protective order with a family-violence finding."
        ),
        "queries": [
            r"active protective order",
            r"order for emergency protection",
            r"finding that my spouse committed family violence",
        ],
    },
    {
        "id": "M5",
        "section": "I. Morgan - Protective Order Status",
        "claim": (
            "Protective order application filed in late 2023 and later nonsuited; "
            "no final protective order/family-violence finding."
        ),
        "queries": [
            r"application for protective order",
            r"nonsuit",
            r"no protective order",
            r"no finding of family violence",
            r"nonsuited",
        ],
    },
    {
        "id": "M6",
        "section": "I. Morgan - Vacatur/Kick-Out",
        "claim": "Court-ordered vacatur/removal from the residence on January 16, 2024.",
        "queries": [
            r"January 16, 2024",
            r"01/16/2024",
            r"ordered to vacate",
            r"kick-out",
            r"vacate the residence",
            r"ordered to vacate his",
        ],
    },
    {
        "id": "M7",
        "section": "II. Court - No Hearing/Evidence",
        "claim": "Removal/orders entered without hearing, evidence, or transcript.",
        "queries": [
            r"without any factual basis",
            r"without holding an evidentiary hearing",
            r"no hearing took place",
            r"no transcript",
            r"no reporter",
        ],
    },
    {
        "id": "C1",
        "section": "II. Court - Referral Defect",
        "claim": "Associate judge acted without an order of referral or standing order (Tex. Fam. Code 201.006).",
        "queries": [
            r"201\.006",
            r"order of referral",
            r"standing order of referral",
            r"associate judge",
        ],
    },
    {
        "id": "C2",
        "section": "II. Court - Protective Order Findings",
        "claim": "Missing required findings for protective order (Tex. Fam. Code 85.001).",
        "queries": [
            r"85\.001",
            r"find whether family violence has occurred",
            r"no such findings",
        ],
    },
    {
        "id": "K1",
        "section": "III. Counsel - Cooper L. Carter Entry",
        "claim": (
            "Carter retained around Jan 22, 2024; signature first appears on associate judge report."
        ),
        "queries": [
            r"Cooper L\. Carter",
            r"January 22, 2024",
            r"signature first appears",
            r"associate judge",
        ],
    },
    {
        "id": "K2",
        "section": "III. Counsel - Temporary Orders",
        "claim": "Carter prepared/handed temporary orders and proposed denial order.",
        "queries": [
            r"Cooper L\. Carter handed",
            r"temporary orders",
            r"proposed order",
            r"associate judge",
        ],
    },
    {
        "id": "S1",
        "section": "IV. Court Staff - Recusal Handling",
        "claim": "Court coordinator involvement in recusal handling/splitting files.",
        "queries": [
            r"court coordinator",
            r"recusal",
            r"split",
            r"order of referral",
        ],
    },
    {
        "id": "R1",
        "section": "V. Regional Presiding Judge - Assignment",
        "claim": "Assignment of retired judge John H. Cayce, Jr. and eligibility objections.",
        "queries": [
            r"John H\. Cayce",
            r"retired judge",
            r"assignment",
            r"74\.055",
            r"74\.054",
        ],
    },
]


def _write_report(
    output_path: Path,
    records: List[dict],
    max_hits: int,
) -> None:
    lines = [
        "# 28B Record Integration - Lawful Violations Draft",
        "",
        f"Generated: {_timestamp()}",
        "",
        "This map surfaces record excerpts that mention the draft's factual claims.",
        "It does not determine legal conclusions or the truth of any allegation.",
        "",
    ]

    for entry in CLAIMS:
        patterns = [re.compile(q, re.IGNORECASE) for q in entry["queries"]]
        hits = _claim_hits(records, patterns, max_hits)
        lines.append(f"## {entry['id']} | {entry['section']}")
        lines.append("")
        lines.append(f"Draft claim: {entry['claim']}")
        lines.append("")
        if not hits:
            lines.append("Record hits: none found with the current query set.")
            lines.append("")
            continue
        lines.append("Record hits:")
        for hit in hits:
            source_pdf = _normalize_ascii(str(hit["source_pdf"]))
            lines.append(
                f"- vector_id {hit['vector_id']} | chunk {hit['chunk_index']} | "
                f"score {hit['score']} | source {source_pdf}"
            )
            lines.append(f"  Quote: \"{hit['snippet']}\"")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map draft allegations to 28B record excerpts."
    )
    parser.add_argument(
        "--store",
        type=Path,
        default=Path("vector_store_28b"),
        help="Vector store directory (metadata.jsonl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/28b_lawful_violations_record_map.md"),
        help="Output markdown report path.",
    )
    parser.add_argument(
        "--max-hits",
        type=int,
        default=3,
        help="Max excerpts per claim.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    store_dir = args.store.expanduser().resolve()
    records = _load_records(store_dir)
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_report(output_path, records, args.max_hits)
    print(f"Wrote record map to {output_path}")


if __name__ == "__main__":
    main()
