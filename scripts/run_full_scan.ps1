param(
    [string]$CaseInput = "CASE DOCS",
    [string]$CaseOutput = "extracted_text_full",
    [string]$ResearchInput = "RESEARCH",
    [string]$ResearchOutput = "extracted_research_full"
)

$ErrorActionPreference = "Stop"

Write-Host "Starting full OCR scan for case docs..."
$caseLog = Join-Path $CaseOutput "extraction_log_full.csv"
python scripts\extract_case_docs.py --input $CaseInput --output $CaseOutput --log $caseLog --ocr-mode always --resume --log-skips

Write-Host "Starting full OCR scan for research docs..."
$researchLog = Join-Path $ResearchOutput "extraction_log_full.csv"
python scripts\extract_case_docs.py --input $ResearchInput --output $ResearchOutput --log $researchLog --ocr-mode always --resume --log-skips

Write-Host "Extracting case mentions from all text outputs..."
python scripts\extract_case_mentions.py --input $CaseOutput --input $ResearchOutput --output reports\case_mentions

Write-Host "Full OCR scan complete."
