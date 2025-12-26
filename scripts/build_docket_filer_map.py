"""Parse color-coded docket images to map filemarks to filers."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import pytesseract


COLOR_BANDS = {
    "red": [(0, 15), (170, 179)],
    "yellow": [(16, 35)],
    "green": [(36, 85)],
    "light_blue": [(80, 115)],
    "purple": [(120, 160)],
}

ROLE_BY_COLOR = {
    "green": "petitioner",
    "purple": "counsel",
    "red": "courts",
    "yellow": "me",
    "light_blue": "oag",
}

DEFAULT_FILER_BY_ROLE = {
    "petitioner": "morgan_michelle_myers",
    "counsel": "cooper_carter",
    "courts": "court",
    "me": "charles_dustin_myers",
    "oag": "oag",
}

DOCKET_LINE = re.compile(r"^\s*(\d{1,4})\s+\d{2}/\d{2}/\d{4}\b")


@dataclass
class OcrHit:
    filemark_raw: str
    filemark: str | None
    corrected: bool
    color: str
    role: str
    filer: str
    source_image: str


def _ensure_tesseract() -> None:
    current = os.environ.get("TESSDATA_PREFIX")
    if current and (Path(current) / "eng.traineddata").exists():
        return
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tessdata",
        r"C:\Program Files (x86)\Tesseract-OCR\tessdata",
    ]
    for candidate in candidates:
        if (Path(candidate) / "eng.traineddata").exists():
            os.environ["TESSDATA_PREFIX"] = candidate
            return


def _iter_images(images_dir: Path) -> Iterable[Path]:
    for path in sorted(images_dir.iterdir()):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            yield path


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1]


def _correct_filemark(raw: str, valid_set: set[str]) -> Tuple[str | None, bool]:
    if not raw:
        return None, False
    if raw in valid_set:
        return raw, False
    candidates = []
    for candidate in valid_set:
        if abs(len(candidate) - len(raw)) > 1:
            continue
        distance = _edit_distance(raw, candidate)
        if distance <= 1:
            diff = abs(int(candidate) - int(raw))
            candidates.append((distance, diff, len(candidate), candidate))
    if candidates:
        candidates.sort()
        return candidates[0][3], True
    return raw, False


def _extract_filemarks(json_path: Path) -> set[str]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    filemarks: set[str] = set()
    for page in payload.get("pages", []):
        text = page.get("text", "")
        for line in text.splitlines():
            match = DOCKET_LINE.match(line)
            if match:
                filemarks.add(match.group(1))
    return filemarks


def _mask_for_color(hsv: np.ndarray, bands: List[Tuple[int, int]], s_thr: int, v_thr: int) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in bands:
        lower = np.array([lo, s_thr, v_thr])
        upper = np.array([hi, 255, 255])
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
    return mask


def _extract_numbers(block: np.ndarray) -> List[str]:
    scale = 6
    block_big = cv2.resize(block, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(block_big, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)

    row_counts = (bw > 0).sum(axis=1)
    threshold = max(5, int(0.03 * bw.shape[1]))
    segments: List[Tuple[int, int]] = []
    start = None
    for idx, count in enumerate(row_counts):
        if count > threshold and start is None:
            start = idx
        elif count <= threshold and start is not None:
            end = idx
            if end - start > 10:
                segments.append((start, end))
            start = None
    if start is not None:
        segments.append((start, len(row_counts) - 1))

    numbers: List[str] = []
    if not segments:
        segments = [(0, bw.shape[0])]

    for y0, y1 in segments:
        line = bw[y0:y1, :]
        inv = 255 - line
        inv = cv2.copyMakeBorder(inv, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=255)
        text = pytesseract.image_to_string(
            inv, config="--psm 7 -c tessedit_char_whitelist=0123456789"
        )
        text = "".join(ch for ch in text if ch.isdigit())
        if text:
            numbers.append(text)

    if not numbers:
        text = pytesseract.image_to_string(
            255 - bw, config="--psm 6 -c tessedit_char_whitelist=0123456789"
        )
        for line in text.splitlines():
            digits = "".join(ch for ch in line if ch.isdigit())
            if digits:
                numbers.append(digits)

    return numbers


def _parse_image(image_path: Path) -> List[Tuple[str, List[str]]]:
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    height, width = hsv.shape[:2]
    margin_limit = min(160, max(60, int(width * 0.2)))
    results: List[Tuple[str, List[str]]] = []
    s_thr = 40
    v_thr = 40
    for color, bands in COLOR_BANDS.items():
        mask = _mask_for_color(hsv, bands, s_thr, v_thr)
        mask[:, margin_limit:] = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: List[Tuple[int, int, int, int]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 10 or h < 10:
                continue
            area = w * h
            if area < 80 or area > 8000:
                continue
            boxes.append((x, y, w, h))
        boxes.sort(key=lambda b: (b[1], b[0]))
        for x, y, w, h in boxes:
            crop = img[y : y + h, x : x + w]
            numbers = _extract_numbers(crop)
            if numbers:
                results.append((color, numbers))
    return results


def _write_ocr_hits(output_path: Path, hits: List[OcrHit]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["filemark_raw", "filemark", "corrected", "color", "role", "filer", "source_image"]
        )
        for hit in hits:
            writer.writerow(
                [
                    hit.filemark_raw,
                    hit.filemark or "",
                    "yes" if hit.corrected else "no",
                    hit.color,
                    hit.role,
                    hit.filer,
                    hit.source_image,
                ]
            )


def _write_aggregated_map(output_path: Path, hits: List[OcrHit]) -> None:
    by_filemark: Dict[str, Counter[str]] = defaultdict(Counter)
    meta: Dict[str, Dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for hit in hits:
        if not hit.filemark:
            continue
        by_filemark[hit.filemark][hit.filer] += 1
        meta[hit.filemark]["color"][hit.color] += 1
        meta[hit.filemark]["role"][hit.role] += 1

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filemark", "filer", "role", "color", "count"])
        for filemark in sorted(by_filemark, key=lambda x: int(x)):
            filer_counts = by_filemark[filemark]
            filer, count = filer_counts.most_common(1)[0]
            role = meta[filemark]["role"].most_common(1)[0][0]
            color = meta[filemark]["color"].most_common(1)[0][0]
            writer.writerow([filemark, filer, role, color, count])


def _write_conflicts(output_path: Path, hits: List[OcrHit]) -> None:
    by_filemark: Dict[str, Counter[str]] = defaultdict(Counter)
    for hit in hits:
        if hit.filemark:
            by_filemark[hit.filemark][hit.filer] += 1
    lines = [
        "# Docket Filer Conflicts",
        "",
    ]
    conflicts = 0
    for filemark in sorted(by_filemark, key=lambda x: int(x)):
        if len(by_filemark[filemark]) <= 1:
            continue
        conflicts += 1
        entries = ", ".join(
            f"{filer} ({count})" for filer, count in by_filemark[filemark].most_common()
        )
        lines.append(f"- {filemark}: {entries}")
    if conflicts == 0:
        lines.append("- None.")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a docket filemark-to-filer map.")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("CASE DOCKET"),
        help="Directory containing color-coded docket images.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("extracted_text_full/28b_merged/28B_merged.json"),
        help="Page-level JSON to extract valid filemarks for correction.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory to write mapping outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    images_dir = args.images_dir.expanduser().resolve()
    json_path = args.json.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _ensure_tesseract()
    valid_set = _extract_filemarks(json_path) if json_path.exists() else set()

    hits: List[OcrHit] = []
    for image_path in _iter_images(images_dir):
        parsed = _parse_image(image_path)
        for color, numbers in parsed:
            role = ROLE_BY_COLOR.get(color, "unknown")
            filer = DEFAULT_FILER_BY_ROLE.get(role, "unknown")
            for raw in numbers:
                corrected, corrected_flag = _correct_filemark(raw, valid_set) if valid_set else (raw, False)
                hits.append(
                    OcrHit(
                        filemark_raw=raw,
                        filemark=corrected,
                        corrected=corrected_flag,
                        color=color,
                        role=role,
                        filer=filer,
                        source_image=image_path.name,
                    )
                )

    _write_ocr_hits(output_dir / "28b_docket_filer_ocr.csv", hits)
    _write_aggregated_map(output_dir / "28b_docket_filer_map.csv", hits)
    _write_conflicts(output_dir / "28b_docket_filer_conflicts.md", hits)

    print(f"Wrote docket filer mapping to {output_dir}")


if __name__ == "__main__":
    main()
