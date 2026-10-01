"""
Indian-language demo: how well does the multilingual model answer questions in
Hindi, Bengali, Tamil, Telugu, Kannada, Malayalam, Marathi, Gujarati, Punjabi,
Urdu and Odia?

    python scripts/run_indic_examples.py                # all languages
    python scripts/run_indic_examples.py --language Telugu

The passages are typed text (samples/indic_examples.json), NOT PDFs, so this
measures the *model* only -- separately from PDF text extraction, which is the
weak link for Indic scripts (see the README).

Needs the multilingual model (~2.2 GB on first run) and outputs UTF-8. On
Windows, run `chcp 65001` first if the Telugu/Tamil text shows as "?".
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import MULTILINGUAL_MODEL, SAMPLES_DIR  # noqa: E402
from app.qa_engine import QAEngine  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Indian-language QA examples.")
    parser.add_argument("--language", help="only run languages whose name contains this text")
    args = parser.parse_args()

    groups = json.loads((SAMPLES_DIR / "indic_examples.json").read_text(encoding="utf-8"))
    if args.language:
        groups = [g for g in groups if args.language.lower() in g["language"].lower()]
        if not groups:
            print(f"No language matching {args.language!r}.")
            return 1

    print(f"Loading {MULTILINGUAL_MODEL} ...")
    engine = QAEngine(MULTILINGUAL_MODEL)

    passed = total = 0
    for group in groups:
        print(f"\n=== {group['language']} ===")
        for item in group["questions"]:
            result = engine.answer(item["question"], text=group["context"])
            ok = item["expected"].lower() in result.answer.lower()
            passed += ok
            total += 1
            print(f"  [{'PASS' if ok else 'FAIL'}] Q: {item['question']}")
            print(f"         A: {result.answer or '(no answer)'}   "
                  f"(score {result.score:.2f}, expected: {item['expected']})")

    print(f"\n{passed}/{total} questions answered correctly.")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
