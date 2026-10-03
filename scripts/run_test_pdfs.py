"""
Run every PDF in /testings through the full pipeline and report the results.

    python scripts/run_test_pdfs.py                 # all languages
    python scripts/run_test_pdfs.py --only te,hi    # just these
    python scripts/run_test_pdfs.py --english-model # use the English-only model

This checks the whole chain -- PDF -> text extraction -> chunking -> QA -- and
separates the two things that can go wrong:

    extraction  did the text survive being written into, and read back out of,
                a PDF? (Indic scripts are where this usually breaks)
    answers     did the model find the right span in that text?

Needs the multilingual model (~2.2 GB on first run). On Windows run `chcp 65001`
first, or the non-Latin output shows as "?".
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import ENGLISH_MODEL, MULTILINGUAL_MODEL  # noqa: E402
from app.pdf_extractor import extract_text  # noqa: E402
from app.qa_engine import QAEngine  # noqa: E402

TESTINGS_DIR = ROOT / "testings"


def ocr_languages_for(code: str):
    """
    EasyOCR codes to use for a language, or None if it has no OCR model.

    EasyOCR covers fewer languages than the QA model does: Gujarati, Malayalam,
    Odia and Punjabi can be *understood* but not *read off an image*.
    """
    try:
        from easyocr.config import all_lang_list
    except ImportError:
        return None
    if code not in all_lang_list:
        return None
    # Each Indic script loads on its own or together with English.
    return (code,) if code == "en" else (code, "en")


def main() -> int:
    parser = argparse.ArgumentParser(description="Test every PDF in /testings.")
    parser.add_argument("--only", help="comma-separated language codes, e.g. te,hi")
    parser.add_argument("--english-model", action="store_true",
                        help="use the English-only model (expect failures elsewhere)")
    parser.add_argument("--ocr", action="store_true",
                        help="read the pages as images instead of using the PDF's "
                             "text layer; recovers Indic conjuncts (slow)")
    args = parser.parse_args()

    cases = json.loads((TESTINGS_DIR / "questions.json").read_text(encoding="utf-8"))
    if args.only:
        wanted = {code.strip().lower() for code in args.only.split(",")}
        cases = [c for c in cases if c["code"] in wanted]

    model = ENGLISH_MODEL if args.english_model else MULTILINGUAL_MODEL
    print(f"Loading {model} ...\n")
    engine = QAEngine(model)

    rows, total_ok, total_q = [], 0, 0
    for case in cases:
        path = TESTINGS_DIR / case["pdf"]
        if not path.exists():
            print(f"  missing: {case['pdf']} (run scripts/make_test_pdfs.py)")
            continue

        if args.ocr:
            from app.ocr import OCRError
            from app.pdf_extractor import extract_with_ocr
            languages = ocr_languages_for(case["code"])
            if languages is None:
                print(f"  skip: EasyOCR has no model for {case['language']}")
                continue
            doc = None
            # Try the language with English first, then on its own: some
            # EasyOCR models refuse to pair, and a few are simply broken.
            for attempt in (languages, languages[:1]):
                try:
                    doc = extract_with_ocr(path, attempt, first_page=1, max_pages=1)
                    break
                except OCRError as exc:
                    last_error = exc
            if doc is None:
                print(f"  skip: OCR failed for {case['language']}: "
                      f"{str(last_error)[:90]}")
                continue
        else:
            doc = extract_text(path)
        # Did the text survive the PDF round trip? Check that each expected
        # answer is still present in the extracted text before blaming the model.
        intact = sum(1 for item in case["questions"] if item["expected"] in doc.text)

        ok = 0
        for item in case["questions"]:
            result = engine.answer(item["q"], pages=doc.pages)
            hit = item["expected"].lower() in (result.answer or "").lower()
            ok += hit
            print(f"  [{'PASS' if hit else 'FAIL'}] {case['language']:<11} {item['q']}")
            print(f"         -> {result.answer or '(no answer)'}   "
                  f"(score {result.score:.2f}, expected: {item['expected']})")

        total_ok += ok
        total_q += len(case["questions"])
        rows.append((case["language"], doc.word_count, intact,
                     len(case["questions"]), ok))

    print(f"\n{'Language':<12} {'Words':>6} {'Text OK':>9} {'Answers':>9}")
    print("-" * 40)
    for language, words, intact, asked, ok in rows:
        print(f"{language:<12} {words:>6} {intact:>5}/{asked:<3} {ok:>5}/{asked:<3}")
    print("-" * 40)
    print(f"{'TOTAL':<12} {'':>6} {'':>9} {total_ok:>5}/{total_q:<3}")
    print("\n'Text OK' counts expected answers still present in the extracted text: "
          "a low number there is a PDF extraction problem, not a model problem.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
