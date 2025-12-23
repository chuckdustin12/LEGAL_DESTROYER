import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


USC_PATTERN = re.compile(
    r"\b(?P<title>\d+)\s+U\.?S\.?C\.?\s*(?:§+)?\s*(?P<section>[\w\.\-]+)"
)
TEX_CODE_PATTERN = re.compile(
    r"\bTex\.?\s*(?P<code>Gov't|Govt|Fam\.|Family|Penal|Civ\.|Admin\.|Const\.)\s*Code\.?\s*(?:§+)?\s*(?P<section>[\w\.\-]+)"
)
TRCP_PATTERN = re.compile(r"\bTex\.?\s*R\.?\s*Civ\.?\s*P\.?\s*(?P<section>\d+[A-Za-z]?)")
TRCP_SHORT_PATTERN = re.compile(r"\bTRCP\s*(?P<section>\d+[A-Za-z]?)")
TAC_PATTERN = re.compile(r"\bTex\.?\s*Admin\.?\s*Code\s*(?:§+)?\s*(?P<section>[\w\.\-]+)")

REPORTER_PATTERN = re.compile(
    r"\b\d{1,4}\s+(S\.W\.3d|S\.W\.2d|S\.W\.|U\.S\.|F\.3d|F\.2d|F\. Supp\. 2d|F\. Supp\.|S\. Ct\.|L\. Ed\. 2d|L\. Ed\.)\s+\d+\b"
)
CASE_NAME_PATTERN = re.compile(r"\b[A-Z][A-Za-z.&'-]{2,}\s+v\.?\s+[A-Z][A-Za-z.&'-]{2,}\b")
IN_RE_PATTERN = re.compile(r"\bIn re\s+[A-Z][A-Za-z.&'-]{2,}\b")
EX_PARTE_PATTERN = re.compile(r"\bEx parte\s+[A-Z][A-Za-z.&'-]{2,}\b")


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            filtered = {field: row.get(field, "") for field in fieldnames}
            writer.writerow(filtered)


def normalize_us_code(match: re.Match) -> str:
    title = match.group("title")
    section = match.group("section").strip(" ,.;:)(")
    return f"{title} U.S.C. {section}"


def normalize_tex_code(match: re.Match) -> str:
    code = match.group("code")
    section = match.group("section").strip(" ,.;:)(")
    code = code.replace("Govt", "Gov't")
    code = code.replace("Family", "Fam.")
    return f"Tex. {code} Code {section}"


def normalize_trcp(match: re.Match) -> str:
    section = match.group("section").strip(" ,.;:)(")
    return f"Tex. R. Civ. P. {section}"


def normalize_tac(match: re.Match) -> str:
    section = match.group("section").strip(" ,.;:)(")
    return f"Tex. Admin. Code {section}"


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


def load_text_lines(path: Path) -> list[str]:
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    content = strip_headers(content)
    return [line.strip() for line in content.splitlines() if line.strip()]


def ascii_safe(value: str) -> str:
    return value.encode("ascii", errors="ignore").decode("ascii")


def add_citation(stats: dict, key: str, source_pdf: str, date_val: str, line: str, max_examples: int) -> None:
    if key not in stats:
        stats[key] = {"count": 0, "docs": set(), "examples": []}
    entry = stats[key]
    entry["count"] += 1
    entry["docs"].add(source_pdf)
    if len(entry["examples"]) < max_examples:
        snippet = ascii_safe(line)
        snippet = snippet[:200] + "..." if len(snippet) > 200 else snippet
        entry["examples"].append(f"{date_val} | {ascii_safe(source_pdf)} | {snippet}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a citation pack with snippets.")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--case-text", default="extracted_text")
    parser.add_argument("--research-text", default="extracted_research")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()

    reports_dir = Path(args.reports)
    doc_index = read_csv(reports_dir / "doc_index.csv")
    meta_map = {row.get("source_txt", ""): row for row in doc_index}

    statute_stats = {}
    caselaw_stats = {}
    case_name_stats = {}

    for root in [Path(args.case_text), Path(args.research_text)]:
        for txt_path in sorted(root.rglob("*.txt")):
            meta = meta_map.get(str(txt_path), {})
            source_pdf = meta.get("source_pdf", "")
            date_val = meta.get("date", "undated")
            lines = load_text_lines(txt_path)
            if not lines:
                continue
            for line in lines:
                for match in USC_PATTERN.finditer(line):
                    add_citation(
                        statute_stats,
                        normalize_us_code(match),
                        source_pdf,
                        date_val,
                        line,
                        args.examples,
                    )
                for match in TEX_CODE_PATTERN.finditer(line):
                    add_citation(
                        statute_stats,
                        normalize_tex_code(match),
                        source_pdf,
                        date_val,
                        line,
                        args.examples,
                    )
                for match in TRCP_PATTERN.finditer(line):
                    add_citation(
                        statute_stats,
                        normalize_trcp(match),
                        source_pdf,
                        date_val,
                        line,
                        args.examples,
                    )
                for match in TRCP_SHORT_PATTERN.finditer(line):
                    add_citation(
                        statute_stats,
                        normalize_trcp(match),
                        source_pdf,
                        date_val,
                        line,
                        args.examples,
                    )
                for match in TAC_PATTERN.finditer(line):
                    add_citation(
                        statute_stats,
                        normalize_tac(match),
                        source_pdf,
                        date_val,
                        line,
                        args.examples,
                    )

                for match in REPORTER_PATTERN.finditer(line):
                    add_citation(
                        caselaw_stats,
                        match.group(0).strip(" ,.;:)("),
                        source_pdf,
                        date_val,
                        line,
                        args.examples,
                    )
                for match in CASE_NAME_PATTERN.finditer(line):
                    add_citation(
                        case_name_stats,
                        match.group(0).strip(" ,.;:)("),
                        source_pdf,
                        date_val,
                        line,
                        args.examples,
                    )
                for match in IN_RE_PATTERN.finditer(line):
                    add_citation(
                        case_name_stats,
                        match.group(0).strip(" ,.;:)("),
                        source_pdf,
                        date_val,
                        line,
                        args.examples,
                    )
                for match in EX_PARTE_PATTERN.finditer(line):
                    add_citation(
                        case_name_stats,
                        match.group(0).strip(" ,.;:)("),
                        source_pdf,
                        date_val,
                        line,
                        args.examples,
                    )

    def sorted_stats(stats: dict) -> list[tuple[str, dict]]:
        return sorted(stats.items(), key=lambda x: (-len(x[1]["docs"]), -x[1]["count"]))

    statute_sorted = sorted_stats(statute_stats)[: args.top]
    caselaw_sorted = sorted_stats(caselaw_stats)[: args.top]
    case_name_sorted = sorted_stats(case_name_stats)[: args.top]

    write_csv(
        reports_dir / "citation_pack_statutes.csv",
        [
            {
                "citation": citation,
                "doc_count": len(meta["docs"]),
                "total_hits": meta["count"],
                "sample_source": meta["examples"][0] if meta["examples"] else "",
            }
            for citation, meta in statute_sorted
        ],
        ["citation", "doc_count", "total_hits", "sample_source"],
    )

    write_csv(
        reports_dir / "citation_pack_caselaw.csv",
        [
            {
                "citation": citation,
                "doc_count": len(meta["docs"]),
                "total_hits": meta["count"],
                "sample_source": meta["examples"][0] if meta["examples"] else "",
            }
            for citation, meta in caselaw_sorted
        ],
        ["citation", "doc_count", "total_hits", "sample_source"],
    )

    write_csv(
        reports_dir / "citation_pack_case_names.csv",
        [
            {
                "citation": citation,
                "doc_count": len(meta["docs"]),
                "total_hits": meta["count"],
                "sample_source": meta["examples"][0] if meta["examples"] else "",
            }
            for citation, meta in case_name_sorted
        ],
        ["citation", "doc_count", "total_hits", "sample_source"],
    )

    out_md = reports_dir / "citation_pack.md"
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Citation Pack (Document-Derived)\n\n")
        f.write("This pack lists citations and example contexts. It is not legal advice.\n\n")

        f.write("## Statutes (Top by doc count)\n\n")
        for citation, meta in statute_sorted:
            f.write(
                f"- {citation} | docs: {len(meta['docs'])} | hits: {meta['count']}\n"
            )
            for example in meta["examples"]:
                f.write(f"  - {example}\n")
        f.write("\n")

        f.write("## Caselaw Reporter Citations (Top by doc count)\n\n")
        for citation, meta in caselaw_sorted:
            f.write(
                f"- {citation} | docs: {len(meta['docs'])} | hits: {meta['count']}\n"
            )
            for example in meta["examples"]:
                f.write(f"  - {example}\n")
        f.write("\n")

        f.write("## Case Name Citations (Top by doc count)\n\n")
        for citation, meta in case_name_sorted:
            f.write(
                f"- {citation} | docs: {len(meta['docs'])} | hits: {meta['count']}\n"
            )
            for example in meta["examples"]:
                f.write(f"  - {example}\n")
        f.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
