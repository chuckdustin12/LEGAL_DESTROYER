import argparse
import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_QUERIES = [
    "due process without a hearing notice and opportunity to be heard",
    "recusal motion regional presiding judge order of assignment disqualification",
    "associate judge order of referral de novo hearing",
    "mandamus petition original proceeding court of appeals supreme court",
    "notice of removal remand federal district court 28 U.S.C. 1332",
    "federal civil rights 42 U.S.C. 1983 deprivation of property due process",
    "civil RICO enterprise pattern of racketeering predicate acts",
    "protective order temporary orders ex parte",
    "fraud perjury forged documents misrepresentation",
    "child support OAG arrears",
    "lockout eviction removed from home property",
    "discovery requests nonresponse sanctions",
    "jurisdiction venue federal question diversity",
    "law enforcement police involvement property removal",
    "Judge Munford recusal",
    "Judge Kaitcer recusal",
    "Judge Newell mandamus",
    "Justice Gabriel recusal",
]


def ascii_safe(value: str) -> str:
    return value.encode("ascii", errors="ignore").decode("ascii")


def load_metadata(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def clean_snippet(text: str, max_len: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    text = ascii_safe(text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def query_embeddings(
    embeddings: np.ndarray,
    metadata: list[dict],
    queries: list[str],
    model_name: str,
    top_k: int,
    dedupe: bool,
    output_path: Path,
) -> None:
    model = SentenceTransformer(model_name)
    query_vectors = model.encode(
        queries, convert_to_numpy=True, normalize_embeddings=True
    )
    if embeddings.size == 0:
        raise RuntimeError("Embeddings matrix is empty.")

    with output_path.open("w", encoding="utf-8") as f:
        f.write("# Evidence Packets (Semantic Search)\n\n")
        f.write("Document-derived matches from vector_store_all. Not legal advice.\n\n")

        for idx, query in enumerate(queries):
            qvec = query_vectors[idx]
            scores = embeddings @ qvec
            order = np.argsort(-scores)

            f.write(f"## Query: {ascii_safe(query)}\n\n")
            seen = set()
            count = 0
            for rank in order:
                record = metadata[int(rank)]
                source_pdf = record.get("source_pdf", "")
                source_txt = record.get("source_txt", "")
                key = source_pdf if dedupe else f"{source_pdf}:{record.get('chunk_index')}"
                if dedupe and key in seen:
                    continue
                seen.add(key)

                snippet = clean_snippet(record.get("text", ""), 360)
                score = float(scores[int(rank)])
                page = record.get("page")
                f.write(
                    f"- score {score:.3f} | page {page} | {ascii_safe(source_pdf)}\n"
                )
                f.write(f"  source_txt: {ascii_safe(source_txt)}\n")
                f.write(f"  snippet: {snippet}\n")
                count += 1
                if count >= top_k:
                    break
            f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Query vector store for evidence packets.")
    parser.add_argument("--store", default="vector_store_all")
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name or path.",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--dedupe", action="store_true", help="Deduplicate by source PDF.")
    parser.add_argument(
        "--queries",
        default="",
        help="Optional JSON array of queries to override the defaults.",
    )
    parser.add_argument("--output", default="reports/evidence_packets.md")
    args = parser.parse_args()

    queries = DEFAULT_QUERIES
    if args.queries:
        try:
            queries = json.loads(args.queries)
        except Exception as exc:
            raise SystemExit(f"Failed to parse --queries JSON: {exc}")

    store_dir = Path(args.store)
    embeddings_path = store_dir / "embeddings.npy"
    metadata_path = store_dir / "metadata.jsonl"
    embeddings = np.load(embeddings_path)
    metadata = load_metadata(metadata_path)
    if embeddings.shape[0] != len(metadata):
        raise SystemExit("Embeddings/metadata size mismatch.")

    query_embeddings(
        embeddings=embeddings,
        metadata=metadata,
        queries=queries,
        model_name=args.model,
        top_k=args.top_k,
        dedupe=args.dedupe,
        output_path=Path(args.output),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
