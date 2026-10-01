"""
End-to-end demo: run the sample questions against the sample PDFs and check
the answers against samples/expected_answers.json.

    python scripts/run_examples.py                 # English examples only
    python scripts/run_examples.py --multilingual  # + French examples (downloads ~2.2 GB model)

An example passes if the expected text appears in the predicted answer
(case-insensitive), or -- for expected = null -- if no answer is returned.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make `app` importable

from app.config import ENGLISH_MODEL, MULTILINGUAL_MODEL, SAMPLES_DIR  # noqa: E402
from app.pdf_extractor import extract_text  # noqa: E402
from app.qa_engine import QAEngine  # noqa: E402


def is_correct(expected, answer: str) -> bool:
    if expected is None:
        return answer == ""
    return expected.lower() in answer.lower()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--multilingual", action="store_true",
                        help="also run the French examples with the multilingual model")
    args = parser.parse_args()

    examples = json.loads((SAMPLES_DIR / "expected_answers.json").read_text(encoding="utf-8"))
    if not args.multilingual:
        examples = [e for e in examples if not e.get("multilingual")]

    engines = {}
    passed = 0
    for ex in examples:
        model = MULTILINGUAL_MODEL if ex.get("multilingual") else ENGLISH_MODEL
        if model not in engines:
            print(f"\nLoading {model} ...")
            engines[model] = QAEngine(model)

        doc = extract_text(SAMPLES_DIR / ex["pdf"])
        t0 = time.time()
        result = engines[model].answer(ex["question"], pages=doc.pages)
        elapsed = time.time() - t0

        ok = is_correct(ex["expected"], result.answer)
        passed += ok
        source = result.best.chunk.page_label if result.best else "-"
        print(f"\n[{'PASS' if ok else 'FAIL'}] {ex['pdf']}  ({result.mode}, "
              f"{result.num_chunks} chunk(s), {elapsed:.1f}s)")
        print(f"  Q: {ex['question']}")
        print(f"  A: {result.answer or '(no answer found)'}"
              f"   score={result.score:.3f}   source={source}")
        print(f"  expected: {ex['expected'] if ex['expected'] is not None else '(no answer)'}")

    print(f"\n{passed}/{len(examples)} examples passed.")
    return 0 if passed == len(examples) else 1


if __name__ == "__main__":
    sys.exit(main())
