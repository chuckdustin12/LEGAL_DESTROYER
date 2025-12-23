import argparse
import csv
import itertools
import json
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path


DATE_PATTERNS = [
    re.compile(
        r"(?P<month>\d{1,2})[./_-](?P<day>\d{1,2})[./_-](?P<year>\d{2,4})"
    ),
    re.compile(
        r"(?P<year>\d{4})[./_-](?P<month>\d{1,2})[./_-](?P<day>\d{1,2})"
    ),
]


ISSUES = {
    "due_process": [
        r"\bdue process\b",
        r"opportunity to be heard",
        r"notice and hearing",
        r"without a hearing",
    ],
    "notice": [r"\bnotice\b", r"\bnotified\b", r"\bnotification\b"],
    "hearing": [r"\bhearing\b", r"show cause", r"trial setting"],
    "recusal": [
        r"\brecusal\b",
        r"\brecuse\b",
        r"disqualif",
        r"regional presiding",
        r"order of assignment",
    ],
    "associate_judge": [r"associate judge", r"order of referral", r"\bde novo\b"],
    "mandamus": [r"\bmandamus\b", r"writ of mandamus", r"original proceeding"],
    "appeal": [r"\bappeal\b", r"appellate", r"supreme court", r"court of appeals"],
    "jurisdiction": [
        r"\bjurisdiction\b",
        r"\bvenue\b",
        r"\b1332\b",
        r"diversity",
        r"federal question",
    ],
    "removal_remand": [r"\bremoval\b", r"\bremand\b", r"\b1441\b", r"\b1446\b"],
    "federal_court": [
        r"\bfederal\b",
        r"u\.s\. district",
        r"district court",
        r"united states",
    ],
    "civil_rights_1983": [r"\b1983\b", r"42 u\.s\.c\.", r"civil rights"],
    "rico": [r"\brico\b", r"racketeering", r"enterprise"],
    "protective_order": [r"protective order", r"family violence"],
    "temporary_orders": [r"temporary order", r"temporary orders", r"temp order"],
    "custody": [
        r"\bcustody\b",
        r"conservatorship",
        r"\bpossession\b",
        r"\baccess\b",
        r"parenting",
    ],
    "property_financial": [
        r"\bproperty\b",
        r"\bassets\b",
        r"\bfunds\b",
        r"\bbank\b",
        r"\bfinancial\b",
        r"\bbusiness\b",
        r"\bresidence\b",
        r"\bhome\b",
    ],
    "fraud_perjury": [r"\bfraud\b", r"\bperjury\b", r"misrepresent", r"forg", r"fabricat"],
    "discovery": [r"\bdiscovery\b", r"interrogator", r"request for production", r"deposition"],
    "sanctions_contempt": [r"\bsanction", r"\bcontempt\b"],
    "injunction": [r"\binjunction\b", r"\binjunctive\b", r"\btro\b"],
    "child_support_oag": [r"\boag\b", r"child support", r"arrears"],
    "law_enforcement": [r"police", r"sheriff", r"law enforcement", r"warrant"],
    "ex_parte": [r"ex parte", r"without notice"],
    "lockout_eviction": [r"evict", r"lockout", r"\bvacate\b", r"removed from the home"],
}


def iter_text_files(root: Path) -> list[Path]:
    return sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".txt"],
        key=lambda p: str(p).lower(),
    )


def normalize_year(year: int) -> int:
    if year < 100:
        return 2000 + year
    return year


def parse_date_from_name(name: str) -> date | None:
    candidates = []
    for pattern in DATE_PATTERNS:
        for match in pattern.finditer(name):
            try:
                year = normalize_year(int(match.group("year")))
                month = int(match.group("month"))
                day = int(match.group("day"))
                if year < 1990 or year > 2100:
                    continue
                candidates.append((match.start(), date(year, month, day)))
            except Exception:
                continue
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


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


def load_text_body(txt_path: Path) -> str:
    try:
        content = txt_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    return strip_headers(content)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            filtered = {field: row.get(field, "") for field in fieldnames}
            writer.writerow(filtered)


def ascii_safe(value: str) -> str:
    return value.encode("ascii", errors="ignore").decode("ascii")


def bar(value: float, scale: int = 40) -> str:
    blocks = int(round(value * scale))
    return "#" * max(blocks, 0)


def build_issue_radars(
    case_text_root: Path,
    case_pdf_root: Path,
    research_text_root: Path,
    research_pdf_root: Path,
    out_dir: Path,
    skip_patterns: list[re.Pattern],
) -> None:
    issue_patterns = {
        issue: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        for issue, patterns in ISSUES.items()
    }

    doc_rows = []
    issue_doc_rows = []
    issue_month_rows = []
    cooccurrence = Counter()
    corpus_doc_counts = Counter()
    issue_totals = defaultdict(lambda: Counter())
    issue_doc_counts = defaultdict(lambda: Counter())

    corpora = [
        ("case", case_text_root, case_pdf_root),
        ("research", research_text_root, research_pdf_root),
    ]

    for corpus, text_root, pdf_root in corpora:
        for txt_path in iter_text_files(text_root):
            rel = txt_path.relative_to(text_root)
            source_pdf = pdf_root / rel.with_suffix(".pdf")
            if skip_patterns:
                path_str = f"{source_pdf} {txt_path}"
                if any(pat.search(path_str) for pat in skip_patterns):
                    continue
            date_val = parse_date_from_name(str(rel))
            month_val = date_val.strftime("%Y-%m") if date_val else ""
            corpus_doc_counts[corpus] += 1

            body = load_text_body(txt_path)
            if not body:
                continue

            issue_hits = {}
            for issue, patterns in issue_patterns.items():
                count = sum(len(pattern.findall(body)) for pattern in patterns)
                if count:
                    issue_hits[issue] = count
                    issue_totals[corpus][issue] += count
                    issue_doc_counts[corpus][issue] += 1

            if not issue_hits:
                continue

            doc_row = {
                "corpus": corpus,
                "source_pdf": str(source_pdf),
                "source_txt": str(txt_path),
                "date": date_val.isoformat() if date_val else "",
                "issue_list": ";".join(sorted(issue_hits.keys())),
            }
            doc_rows.append(doc_row)

            for issue, count in issue_hits.items():
                issue_doc_rows.append(
                    {
                        "corpus": corpus,
                        "issue": issue,
                        "source_pdf": str(source_pdf),
                        "source_txt": str(txt_path),
                        "date": date_val.isoformat() if date_val else "",
                        "hit_count": count,
                    }
                )
                if month_val:
                    issue_month_rows.append(
                        {
                            "corpus": corpus,
                            "month": month_val,
                            "issue": issue,
                            "doc_count": 1,
                            "hit_count": count,
                        }
                    )

            issues_present = sorted(issue_hits.keys())
            for a, b in itertools.combinations(issues_present, 2):
                cooccurrence[(a, b)] += 1

    out_dir.mkdir(parents=True, exist_ok=True)

    radar_rows = []
    for corpus in ("case", "research"):
        total_docs = corpus_doc_counts[corpus] or 1
        for issue in ISSUES.keys():
            docs_with = issue_doc_counts[corpus].get(issue, 0)
            total_hits = issue_totals[corpus].get(issue, 0)
            radar_rows.append(
                {
                    "corpus": corpus,
                    "issue": issue,
                    "docs_with_hits": docs_with,
                    "total_hits": total_hits,
                    "pct_docs": round(docs_with / total_docs, 4),
                }
            )

    overall_rows = []
    for issue in ISSUES.keys():
        docs_with = issue_doc_counts["case"].get(issue, 0) + issue_doc_counts[
            "research"
        ].get(issue, 0)
        total_hits = issue_totals["case"].get(issue, 0) + issue_totals["research"].get(
            issue, 0
        )
        total_docs = corpus_doc_counts["case"] + corpus_doc_counts["research"]
        overall_rows.append(
            {
                "issue": issue,
                "docs_with_hits": docs_with,
                "total_hits": total_hits,
                "pct_docs": round(docs_with / (total_docs or 1), 4),
            }
        )

    radar_fields = ["corpus", "issue", "docs_with_hits", "total_hits", "pct_docs"]
    write_csv(out_dir / "issue_radar_case.csv", [r for r in radar_rows if r["corpus"] == "case"], radar_fields)
    write_csv(out_dir / "issue_radar_research.csv", [r for r in radar_rows if r["corpus"] == "research"], radar_fields)
    write_csv(out_dir / "issue_radar_overall.csv", overall_rows, ["issue", "docs_with_hits", "total_hits", "pct_docs"])

    write_csv(
        out_dir / "issue_doc_map.csv",
        issue_doc_rows,
        ["corpus", "issue", "source_pdf", "source_txt", "date", "hit_count"],
    )

    month_agg = defaultdict(lambda: Counter())
    for row in issue_month_rows:
        key = (row["corpus"], row["month"], row["issue"])
        month_agg[key]["doc_count"] += 1
        month_agg[key]["hit_count"] += row["hit_count"]
    month_rows = [
        {
            "corpus": corpus,
            "month": month,
            "issue": issue,
            "doc_count": counts["doc_count"],
            "hit_count": counts["hit_count"],
        }
        for (corpus, month, issue), counts in month_agg.items()
    ]
    write_csv(
        out_dir / "issue_month_map.csv",
        month_rows,
        ["corpus", "month", "issue", "doc_count", "hit_count"],
    )

    co_rows = [
        {"issue_a": a, "issue_b": b, "doc_count": count}
        for (a, b), count in cooccurrence.items()
    ]
    co_rows.sort(key=lambda r: (-r["doc_count"], r["issue_a"], r["issue_b"]))
    write_csv(out_dir / "issue_cooccurrence.csv", co_rows, ["issue_a", "issue_b", "doc_count"])

    radar_md = out_dir / "issue_radar.md"
    with radar_md.open("w", encoding="utf-8") as f:
        f.write("# Issue Radar (Document-Derived)\n\n")
        f.write("This is a keyword-based index for navigation, not legal advice.\n\n")
        f.write(f"Case docs scanned: {corpus_doc_counts['case']}\n")
        f.write(f"Research docs scanned: {corpus_doc_counts['research']}\n\n")

        for corpus in ("case", "research"):
            f.write(f"## {corpus.title()} Issue Radar (by documents with hits)\n\n")
            rows = [r for r in radar_rows if r["corpus"] == corpus]
            rows.sort(key=lambda r: (-r["docs_with_hits"], r["issue"]))
            total_docs = corpus_doc_counts[corpus] or 1
            for row in rows:
                pct = row["docs_with_hits"] / total_docs
                label = f"{row['issue']}: {row['docs_with_hits']} docs ({pct:.1%}), {row['total_hits']} hits"
                f.write(f"- {label} | {bar(pct)}\n")
            f.write("\n")

        f.write("Outputs:\n")
        f.write("- reports/issue_radar_case.csv\n")
        f.write("- reports/issue_radar_research.csv\n")
        f.write("- reports/issue_radar_overall.csv\n")
        f.write("- reports/issue_doc_map.csv\n")
        f.write("- reports/issue_month_map.csv\n")
        f.write("- reports/issue_cooccurrence.csv\n")

    issue_argument_md = out_dir / "issue_argument_map.md"
    with issue_argument_md.open("w", encoding="utf-8") as f:
        f.write("# Document-Derived Analysis and Argument Map\n\n")
        f.write("This is a document-derived indexing map, not legal advice.\n")
        f.write("It highlights where issues are discussed in the record and research.\n\n")

        for issue in ISSUES.keys():
            f.write(f"## {issue}\n\n")
            case_hits = [
                row
                for row in issue_doc_rows
                if row["corpus"] == "case" and row["issue"] == issue
            ]
            research_hits = [
                row
                for row in issue_doc_rows
                if row["corpus"] == "research" and row["issue"] == issue
            ]
            case_hits.sort(key=lambda r: (-r["hit_count"], r["source_pdf"]))
            research_hits.sort(key=lambda r: (-r["hit_count"], r["source_pdf"]))

            if case_hits:
                f.write("Case docs (top 5 by keyword hits):\n")
                for row in case_hits[:5]:
                    source_pdf = ascii_safe(row["source_pdf"])
                    date_val = row["date"] or "undated"
                    f.write(f"- {date_val}: {source_pdf} (hits: {row['hit_count']})\n")
                f.write("\n")
            else:
                f.write("Case docs: no keyword hits found.\n\n")

            if research_hits:
                f.write("Research docs (top 5 by keyword hits):\n")
                for row in research_hits[:5]:
                    source_pdf = ascii_safe(row["source_pdf"])
                    date_val = row["date"] or "undated"
                    f.write(f"- {date_val}: {source_pdf} (hits: {row['hit_count']})\n")
                f.write("\n")
            else:
                f.write("Research docs: no keyword hits found.\n\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build issue radars and mappings from extracted text."
    )
    parser.add_argument("--case-text", default="extracted_text")
    parser.add_argument("--case-pdf", default="CASE DOCS")
    parser.add_argument("--research-text", default="extracted_research")
    parser.add_argument("--research-pdf", default="RESEARCH")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument(
        "--skip-pattern",
        action="append",
        default=[],
        help="Regex pattern to skip files (case-insensitive). Can be repeated.",
    )
    args = parser.parse_args()

    skip_patterns = [re.compile(pat, re.IGNORECASE) for pat in args.skip_pattern]

    build_issue_radars(
        case_text_root=Path(args.case_text),
        case_pdf_root=Path(args.case_pdf),
        research_text_root=Path(args.research_text),
        research_pdf_root=Path(args.research_pdf),
        out_dir=Path(args.out_dir),
        skip_patterns=skip_patterns,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
