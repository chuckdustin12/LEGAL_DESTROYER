import argparse
import json
import re
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Iterable


REPORTER_PATTERN = re.compile(
    r"\b\d{1,4}\s+(S\.W\.3d|S\.W\.2d|S\.W\.|U\.S\.|F\.3d|F\.2d|F\. Supp\. 2d|F\. Supp\.|S\. Ct\.|L\. Ed\. 2d|L\. Ed\.)\s+\d+\b"
)
CASE_NAME_PATTERN = re.compile(r"\b[A-Z][A-Za-z.&'-]{2,}\s+v\.?\s+[A-Z][A-Za-z.&'-]{2,}\b")
IN_RE_PATTERN = re.compile(r"\bIn re\s+[A-Z][A-Za-z.&'-]{2,}\b")
EX_PARTE_PATTERN = re.compile(r"\bEx parte\s+[A-Z][A-Za-z.&'-]{2,}\b")


@dataclass(frozen=True)
class CaseMention:
    case_name: str
    reporter: str
    source_txt: str
    line_no: int
    context: str


def iter_text_files(root: Path) -> list[Path]:
    return sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".txt"],
        key=lambda p: str(p).lower(),
    )


def strip_headers(content: str) -> str:
    lines = content.splitlines()
    filtered = []
    for line in lines:
        if line.startswith(("FILE:", "PAGES:", "OCR_MODE:", "DPI:")):
            continue
        if line.startswith("===") or line.startswith("--- EXTRACTED TEXT ---"):
            continue
        if line.startswith("--- OCR TEXT ---"):
            continue
        filtered.append(line)
    return "\n".join(filtered)


def normalize_case_name(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip(" ,.;:)("))


def normalize_reporter(text: str) -> str:
    return text.strip(" ,.;:)(")


def extract_mentions(lines: Iterable[str], source_txt: str) -> list[CaseMention]:
    mentions: list[CaseMention] = []
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        reporters = [normalize_reporter(m.group(0)) for m in REPORTER_PATTERN.finditer(line)]
        case_names = [normalize_case_name(m.group(0)) for m in CASE_NAME_PATTERN.finditer(line)]
        case_names += [normalize_case_name(m.group(0)) for m in IN_RE_PATTERN.finditer(line)]
        case_names += [normalize_case_name(m.group(0)) for m in EX_PARTE_PATTERN.finditer(line)]

        if not reporters and not case_names:
            continue

        if case_names and reporters:
            for case_name in case_names:
                for reporter in reporters:
                    mentions.append(
                        CaseMention(
                            case_name=case_name,
                            reporter=reporter,
                            source_txt=source_txt,
                            line_no=line_no,
                            context=line.strip(),
                        )
                    )
        else:
            for case_name in case_names:
                mentions.append(
                    CaseMention(
                        case_name=case_name,
                        reporter="",
                        source_txt=source_txt,
                        line_no=line_no,
                        context=line.strip(),
                    )
                )
            for reporter in reporters:
                mentions.append(
                    CaseMention(
                        case_name="",
                        reporter=reporter,
                        source_txt=source_txt,
                        line_no=line_no,
                        context=line.strip(),
                    )
                )
    return mentions


def build_payload(mention: CaseMention) -> str:
    case_name = mention.case_name or "N/A"
    reporter = mention.reporter or "N/A"
    return "\n".join(
        [
            f"CASE_NAME: {case_name}",
            f"REPORTER: {reporter}",
            f"SOURCE_FILE: {mention.source_txt}",
            f"LINE_NUMBER: {mention.line_no}",
            "CONTEXT:",
            mention.context,
        ]
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_text(path: Path, payloads: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for payload in payloads:
            f.write(payload)
            f.write("\n\n" + ("-" * 80) + "\n\n")


def load_lines(path: Path) -> list[str]:
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    return strip_headers(content).splitlines()


def collect_mentions(roots: list[Path]) -> list[CaseMention]:
    mentions: list[CaseMention] = []
    for root in roots:
        for txt_path in iter_text_files(root):
            lines = load_lines(txt_path)
            if not lines:
                continue
            mentions.extend(extract_mentions(lines, str(txt_path)))
    return mentions


def build_records(mentions: list[CaseMention]) -> tuple[list[dict], list[str]]:
    records: list[dict] = []
    payloads: list[str] = []
    for mention in mentions:
        payload = build_payload(mention)
        record_id = sha1(
            f"{mention.source_txt}:{mention.line_no}:{mention.case_name}:{mention.reporter}:{mention.context}".encode(
                "utf-8"
            )
        ).hexdigest()
        records.append(
            {
                "id": record_id,
                "case_name": mention.case_name,
                "reporter": mention.reporter,
                "source_txt": mention.source_txt,
                "line_no": mention.line_no,
                "context": mention.context,
                "text": payload,
            }
        )
        payloads.append(payload)
    return records, payloads


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract case mentions from text files for semantic search."
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Input root folder containing .txt files. Repeat for multiple roots.",
    )
    parser.add_argument(
        "--output",
        default="reports/case_mentions",
        help="Output folder for extracted case mentions.",
    )
    args = parser.parse_args()

    roots = [Path(p) for p in args.input] if args.input else [Path("extracted_text_full")]
    roots = [root for root in roots if root.exists()]
    if not roots:
        print("No valid input roots found.")
        return 1

    mentions = collect_mentions(roots)
    records, payloads = build_records(mentions)

    output_root = Path(args.output)
    write_jsonl(output_root / "case_mentions.jsonl", records)
    write_text(output_root / "case_mentions.txt", payloads)

    summary = {
        "input_roots": [str(root) for root in roots],
        "mention_count": len(records),
    }
    summary_path = output_root / "case_mentions_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Extracted {len(records)} case mentions into {output_root}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
