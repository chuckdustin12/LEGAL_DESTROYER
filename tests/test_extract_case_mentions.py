import unittest

from scripts.extract_case_mentions import (
    CASE_NAME_PATTERN,
    EX_PARTE_PATTERN,
    IN_RE_PATTERN,
    REPORTER_PATTERN,
    build_payload,
    extract_mentions,
    strip_headers,
)


class TestCaseExtraction(unittest.TestCase):
    def test_strip_headers_removes_metadata(self) -> None:
        content = "\n".join(
            [
                "FILE: example.pdf",
                "PAGES: 2",
                "OCR_MODE: always",
                "DPI: 300",
                "=== PAGE 1 ===",
                "--- EXTRACTED TEXT ---",
                "Smith v. Jones",
                "--- OCR TEXT ---",
                "123 S.W.3d 456",
            ]
        )
        cleaned = strip_headers(content)
        self.assertNotIn("FILE:", cleaned)
        self.assertNotIn("OCR_MODE:", cleaned)
        self.assertIn("Smith v. Jones", cleaned)
        self.assertIn("123 S.W.3d 456", cleaned)

    def test_extract_mentions_pairs_case_and_reporter(self) -> None:
        line = "See Smith v. Jones, 123 S.W.3d 456 (Tex. 2005)."
        mentions = extract_mentions([line], "source.txt")
        self.assertEqual(len(mentions), 1)
        mention = mentions[0]
        self.assertEqual(mention.case_name, "Smith v. Jones")
        self.assertEqual(mention.reporter, "123 S.W.3d 456")
        payload = build_payload(mention)
        self.assertIn("CASE_NAME: Smith v. Jones", payload)
        self.assertIn("REPORTER: 123 S.W.3d 456", payload)

    def test_patterns_match_alternative_case_formats(self) -> None:
        self.assertIsNotNone(IN_RE_PATTERN.search("In re Estate"))
        self.assertIsNotNone(EX_PARTE_PATTERN.search("Ex parte Garcia"))
        self.assertIsNotNone(CASE_NAME_PATTERN.search("Brown v. Board"))
        self.assertIsNotNone(REPORTER_PATTERN.search("410 U.S. 113"))


if __name__ == "__main__":
    unittest.main()
