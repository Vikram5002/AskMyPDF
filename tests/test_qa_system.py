"""
Tests for the PDF QA system (standard-library unittest, no extra deps).

    python -m unittest discover -s tests -v

The end-to-end tests load the English model (~500 MB, downloaded on first
run). Set SKIP_MODEL_TESTS=1 to run only the fast tests, or
RUN_MULTILINGUAL_TESTS=1 to also test the multilingual model (~2.2 GB).
"""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.chunker import TextTooShortError, chunk_text  # noqa: E402
from app.config import SAMPLES_DIR  # noqa: E402
from app.pdf_extractor import EmptyPDFError, PDFReadError, extract_text  # noqa: E402
from scripts.make_sample_pdfs import build_pdf  # noqa: E402


class TestChunker(unittest.TestCase):

    def test_short_text_is_single_chunk(self):
        chunks = chunk_text("one two three four five six", chunk_size=400, overlap=50)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "one two three four five six")

    def test_long_text_overlaps(self):
        words = [f"w{i}" for i in range(1000)]
        chunks = chunk_text(" ".join(words), chunk_size=400, overlap=50)
        # windows start at 0, 350, 700 -> 3 chunks
        self.assertEqual([c.start_word for c in chunks], [0, 350, 700])
        self.assertEqual(chunks[-1].end_word, 1000)
        # the last 50 words of chunk 0 are the first 50 words of chunk 1
        self.assertEqual(chunks[0].text.split()[-50:], chunks[1].text.split()[:50])

    def test_every_word_is_covered(self):
        words = [f"w{i}" for i in range(937)]
        chunks = chunk_text(" ".join(words), chunk_size=100, overlap=20)
        covered = set()
        for c in chunks:
            covered.update(range(c.start_word, c.end_word))
        self.assertEqual(covered, set(range(937)))

    def test_page_tracking(self):
        pages = [" ".join(["a"] * 300), " ".join(["b"] * 300)]
        chunks = chunk_text(pages=pages, chunk_size=400, overlap=50)
        self.assertEqual(chunks[0].pages, [1, 2])
        self.assertEqual(chunks[-1].page_label, "page 2")

    def test_too_short_raises(self):
        with self.assertRaises(TextTooShortError):
            chunk_text("page 1")

    def test_invalid_overlap_raises(self):
        with self.assertRaises(ValueError):
            chunk_text("a b c d e f", chunk_size=10, overlap=10)


class TestPDFExtractor(unittest.TestCase):

    def test_extracts_sample(self):
        doc = extract_text(SAMPLES_DIR / "sample_long.pdf")
        self.assertEqual(doc.method, "pdfplumber")
        self.assertEqual(doc.num_pages, 3)
        self.assertIn("Dartmouth College", doc.text)

    def test_accepts_bytes(self):
        data = (SAMPLES_DIR / "sample_short.pdf").read_bytes()
        self.assertIn("Eiffel", extract_text(data).text)

    def test_accented_text(self):
        self.assertIn("mètres", extract_text(SAMPLES_DIR / "sample_french.pdf").text)

    def test_falls_back_to_pypdf2(self):
        with mock.patch("app.pdf_extractor._extract_with_pdfplumber",
                        side_effect=RuntimeError("simulated failure")):
            doc = extract_text(SAMPLES_DIR / "sample_short.pdf")
        self.assertEqual(doc.method, "PyPDF2")
        self.assertIn("Eiffel", doc.text)
        self.assertTrue(doc.warnings)

    def test_blank_pdf_raises_empty(self):
        with self.assertRaises(EmptyPDFError):
            extract_text(build_pdf([[]]))   # one page, no text

    def test_scan_is_reported_as_empty_even_if_one_library_crashes(self):
        # Regression: a scanned PDF that also trips one parser must still raise
        # EmptyPDFError, because that is the error the UI turns into an OCR offer.
        # PDFReadError would be fatal and the user could never reach OCR.
        with mock.patch("app.pdf_extractor._extract_with_pdfplumber",
                        side_effect=RuntimeError("simulated parser crash")), \
             mock.patch("app.pdf_extractor._extract_with_pypdf2", return_value=["", "", ""]):
            with self.assertRaises(EmptyPDFError):
                extract_text(SAMPLES_DIR / "sample_short.pdf")

    def test_garbage_raises_read_error(self):
        with self.assertRaises(PDFReadError):
            extract_text(b"this is not a pdf at all")

    def test_zero_bytes_raises_read_error(self):
        with self.assertRaises(PDFReadError):
            extract_text(b"")

    def test_missing_file_raises_read_error(self):
        with self.assertRaises(PDFReadError):
            extract_text(ROOT / "does_not_exist.pdf")


class TestScannedDetection(unittest.TestCase):
    """A scanned PDF yields almost no text; the app must spot that and offer OCR."""

    def test_text_pdf_is_not_flagged(self):
        self.assertFalse(extract_text(SAMPLES_DIR / "sample_long.pdf").looks_scanned)

    def test_partly_scanned_pdf_is_flagged(self):
        # 5 typed pages + 30 scanned ones averages 57 words/page, which looks
        # fine, but the 30 image pages would be silently ignored without OCR.
        from app.pdf_extractor import ExtractionResult
        mixed = ExtractionResult(pages=["word " * 400] * 5 + [""] * 30,
                                 method="pdfplumber")
        self.assertTrue(mixed.looks_scanned)

    def test_watermark_only_pdf_is_flagged(self):
        from app.pdf_extractor import ExtractionResult
        # 39 pages, 10 words in total -- a real scanned book behaves like this.
        scan = ExtractionResult(pages=["tk for more books... www.telugubooks.tk"] + [""] * 38,
                                method="pdfplumber")
        self.assertTrue(scan.looks_scanned)

    def test_ocr_result_is_never_flagged(self):
        from app.pdf_extractor import ExtractionResult
        self.assertFalse(ExtractionResult(pages=["short ocr text"], method="OCR").looks_scanned)


class TestChunkSizeByScript(unittest.TestCase):
    """Non-Latin scripts need smaller chunks to fit the 512-token window."""

    def test_english_gets_the_default(self):
        from app.chunker import suggest_chunk_size
        from app.config import CHUNK_SIZE_WORDS
        size, overlap = suggest_chunk_size("The Eiffel Tower is a wrought-iron lattice tower.")
        self.assertEqual(size, CHUNK_SIZE_WORDS)
        self.assertLess(overlap, size)

    def test_telugu_gets_a_smaller_chunk(self):
        from app.chunker import suggest_chunk_size
        from app.config import CHUNK_SIZE_WORDS_NON_LATIN
        size, _ = suggest_chunk_size("తాజ్ మహల్ ఆగ్రా నగరంలో ఉంది. నిర్మాణం 1653లో పూర్తయింది.")
        self.assertEqual(size, CHUNK_SIZE_WORDS_NON_LATIN)

    def test_hindi_gets_a_smaller_chunk(self):
        from app.chunker import suggest_chunk_size
        from app.config import CHUNK_SIZE_WORDS_NON_LATIN
        size, _ = suggest_chunk_size("ताजमहल आगरा शहर में स्थित है। इसका निर्माण 1653 में पूरा हुआ।")
        self.assertEqual(size, CHUNK_SIZE_WORDS_NON_LATIN)

    def test_empty_text_falls_back_to_default(self):
        from app.chunker import suggest_chunk_size
        from app.config import CHUNK_SIZE_WORDS
        self.assertEqual(suggest_chunk_size("12345 ...")[0], CHUNK_SIZE_WORDS)


class TestRealWorldExtraction(unittest.TestCase):
    """Bugs found on real LaTeX/two-column PDFs rather than the generated samples."""

    def test_word_spacing_is_not_lost(self):
        # pdfplumber's default x_tolerance glues tight LaTeX spacing into
        # "WeusedtheAdamoptimizer"; app.config.X_TOLERANCE fixes it.
        from app.config import X_TOLERANCE
        self.assertLessEqual(X_TOLERANCE, 2)

    def test_real_hyphens_survive(self):
        from app.pdf_extractor import clean_text
        # "self" appears on its own, so the hyphen is real and must be kept.
        out = clean_text("self-\nattention is used; self and attention matter")
        self.assertIn("self-attention", out)
        self.assertNotIn("selfattention", out)

    def test_broken_words_are_rejoined(self):
        from app.pdf_extractor import clean_text
        # "infor" is not a word elsewhere in the text, so the halves join up.
        self.assertIn("information", clean_text("the infor-\nmation was useful"))

    def test_two_column_page_is_detected(self):
        import pdfplumber
        from app.config import X_TOLERANCE
        from app.pdf_extractor import _column_boundary
        # The bundled samples are single-column: they must NOT be split.
        with pdfplumber.open(SAMPLES_DIR / "sample_long.pdf") as pdf:
            words = pdf.pages[0].extract_words(x_tolerance=X_TOLERANCE)
            self.assertIsNone(_column_boundary(words))

    def test_chunk_size_shrinks_for_token_dense_text(self):
        # A document that tokenises into many tokens per word must get a
        # smaller chunk, or every chunk overflows the 512-token window.
        from app.chunker import suggest_chunk_size
        from app.config import CHUNK_SIZE_WORDS

        class FakeTokenizer:
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [0] * (len(text.split()) * 4)}   # 4 tokens/word

        size, overlap = suggest_chunk_size(" ".join(["word"] * 500), tokenizer=FakeTokenizer())
        self.assertLess(size, CHUNK_SIZE_WORDS)
        self.assertLess(overlap, size)


class TestModelLoadFailure(unittest.TestCase):

    def test_memory_failure_gives_a_readable_message(self):
        # Windows reports exhaustion as "The paging file is too small", and
        # transformers then says "Could not load model with any of the
        # following classes", which tells the user nothing useful.
        from app.qa_engine import ModelLoadError, load_local_pipeline
        # NOTE: patch transformers.pipelines.pipeline, not transformers.pipeline:
        # transformers is a _LazyModule and ignores patches on the facade.
        with mock.patch("transformers.pipelines.pipeline",
                        side_effect=OSError("The paging file is too small (os error 1455)")):
            with self.assertRaises(ModelLoadError) as caught:
                load_local_pipeline("deepset/roberta-base-squad2")
        self.assertIn("memory", str(caught.exception).lower())


class TestSpacelessScripts(unittest.TestCase):

    def test_chinese_text_is_chunked_not_rejected(self):
        from app.chunker import chunk_text
        chinese = "北京是中国的首都。故宫位于北京市中心，建于1420年。" * 20
        chunks = chunk_text(chinese, chunk_size=100, overlap=20)
        self.assertGreater(len(chunks), 1)
        # Characters must not be glued back together with spaces.
        self.assertNotIn(" ", chunks[0].text)

    def test_truly_tiny_text_still_raises(self):
        from app.chunker import TextTooShortError, chunk_text
        with self.assertRaises(TextTooShortError):
            chunk_text("北京")


class TestEngineArgumentHandling(unittest.TestCase):
    """QAEngine.answer's chunk_size/overlap defaults, without loading a model."""

    def setUp(self):
        from app.qa_engine import QAEngine
        self.engine = QAEngine.__new__(QAEngine)        # skip model loading
        self.engine.model_name, self.engine.backend = "test", "local"
        self.calls = []
        self.engine._ask = lambda q, c: (self.calls.append(c),
                                         {"answer": "", "score": 0.0, "start": 0, "end": 0})[1]

    def test_explicit_zero_chunk_size_still_raises(self):
        # 0 must not be silently replaced by the suggested size.
        with self.assertRaises(ValueError):
            self.engine.answer("q?", text="a b c d e f g h", chunk_size=0)

    def test_small_chunk_size_does_not_clash_with_suggested_overlap(self):
        # chunk_size=30 with the default overlap of 50 would be invalid;
        # the engine must shrink the overlap instead of raising.
        result = self.engine.answer("q?", text=" ".join(f"w{i}" for i in range(100)),
                                    chunk_size=30)
        self.assertGreater(result.num_chunks, 1)

    def test_ocr_choices_are_valid_combinations(self):
        # Each entry must be one Indic language, optionally plus English.
        from app.ocr import OCR_CHOICES
        for label, langs in OCR_CHOICES.items():
            indic = [code for code in langs if code != "en"]
            self.assertLessEqual(len(indic), 1, f"{label} combines two Indic scripts")


@unittest.skipUnless(os.environ.get("RUN_OCR_TESTS"), "set RUN_OCR_TESTS=1 (slow, needs easyocr)")
class TestOCR(unittest.TestCase):

    def test_ocr_reads_a_rendered_page(self):
        from app.pdf_extractor import extract_with_ocr
        doc = extract_with_ocr(SAMPLES_DIR / "sample_short.pdf", languages=("en",), max_pages=1)
        self.assertEqual(doc.method, "OCR")
        self.assertIn("eiffel", doc.text.lower())

    def test_bad_language_combination_raises(self):
        from app.ocr import OCRError, ocr_pdf
        with self.assertRaises(OCRError):   # two Indic scripts can't share a model
            ocr_pdf(SAMPLES_DIR / "sample_short.pdf", languages=("te", "hi"), pages=[1])

    def test_out_of_range_pages_raise(self):
        from app.ocr import OCRError, ocr_pdf
        with self.assertRaises(OCRError):
            ocr_pdf(SAMPLES_DIR / "sample_short.pdf", languages=("en",), pages=[99])

    def test_page_cache_avoids_repeat_work(self):
        import time
        from app.ocr import ocr_pdf
        cache = {}
        t0 = time.time()
        first = ocr_pdf(SAMPLES_DIR / "sample_short.pdf", languages=("en",),
                        pages=[1], cache=cache)
        cold = time.time() - t0
        t0 = time.time()
        second = ocr_pdf(SAMPLES_DIR / "sample_short.pdf", languages=("en",),
                         pages=[1], cache=cache)
        warm = time.time() - t0
        self.assertEqual(first, second)
        self.assertLess(warm, cold / 5)     # cached run must be far faster

    def test_keeps_real_page_numbers(self):
        # Reading page 1 only of the 3-page sample: page numbering must not shift.
        from app.pdf_extractor import extract_with_ocr
        doc = extract_with_ocr(SAMPLES_DIR / "sample_long.pdf", languages=("en",),
                               first_page=1, max_pages=1)
        self.assertEqual(doc.total_pages, 3)
        self.assertTrue(doc.pages[0].strip())


class TestDotEnv(unittest.TestCase):
    """The .env loader that keeps HF_TOKEN out of the source code."""

    def _write_env(self, text: str) -> Path:
        import tempfile
        path = Path(tempfile.mkdtemp()) / ".env"
        path.write_text(text, encoding="utf-8")
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        return path

    def test_reads_keys_comments_and_quotes(self):
        from app import load_dotenv
        env = self._write_env('# a comment\n\nHF_TOKEN="hf_from_file"\nexport OTHER=plain\n')
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HF_TOKEN", None)
            os.environ.pop("OTHER", None)
            load_dotenv(env)
            self.assertEqual(os.environ["HF_TOKEN"], "hf_from_file")
            self.assertEqual(os.environ["OTHER"], "plain")

    def test_real_environment_wins(self):
        from app import load_dotenv
        env = self._write_env("HF_TOKEN=hf_from_file\n")
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_from_shell"}):
            load_dotenv(env)
            self.assertEqual(os.environ["HF_TOKEN"], "hf_from_shell")

    def test_missing_file_is_fine(self):
        from app import load_dotenv
        load_dotenv(ROOT / "no_such.env")   # must not raise

    def test_byte_order_mark_and_inline_comment(self):
        # Windows Notepad writes a BOM; without utf-8-sig the key becomes
        # "﻿HF_TOKEN" and the token is silently missed.
        from app import load_dotenv
        env = self._write_env_bytes('HF_TOKEN=hf_abc  # my token\n')
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HF_TOKEN", None)
            load_dotenv(env)
            self.assertEqual(os.environ.get("HF_TOKEN"), "hf_abc")
            self.assertFalse([k for k in os.environ if k.startswith("﻿")])

    def _write_env_bytes(self, text: str) -> Path:
        import tempfile
        path = Path(tempfile.mkdtemp()) / ".env"
        path.write_text(text, encoding="utf-8-sig")
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        return path


class TestInferenceAPIWithoutToken(unittest.TestCase):

    def test_missing_token_gives_clear_error(self):
        from app.hf_inference_api import InferenceAPIError, make_api_qa_function
        with mock.patch.dict(os.environ, {"HF_TOKEN": ""}), \
                mock.patch("app.hf_inference_api.get_token", return_value=None):
            with self.assertRaises(InferenceAPIError):
                make_api_qa_function("deepset/roberta-base-squad2")


@unittest.skipIf(os.environ.get("SKIP_MODEL_TESTS"), "SKIP_MODEL_TESTS is set")
class TestEndToEnd(unittest.TestCase):
    """Runs the real English model on the sample PDFs."""

    @classmethod
    def setUpClass(cls):
        from app.qa_engine import QAEngine
        cls.engine = QAEngine()

    def test_expected_answers(self):
        examples = json.loads((SAMPLES_DIR / "expected_answers.json").read_text(encoding="utf-8"))
        for ex in (e for e in examples if not e.get("multilingual")):
            with self.subTest(question=ex["question"]):
                doc = extract_text(SAMPLES_DIR / ex["pdf"])
                result = self.engine.answer(ex["question"], pages=doc.pages)
                if ex["expected"] is None:
                    self.assertFalse(result.found, f"expected no answer, got {result.answer!r}")
                else:
                    self.assertIn(ex["expected"].lower(), result.answer.lower())
                    self.assertLessEqual(result.score, 1.0)
                    # the answer span really is inside the source chunk
                    best = result.best
                    self.assertEqual(best.chunk.text[best.start:best.end].strip(), result.answer)

    def test_modes(self):
        short = extract_text(SAMPLES_DIR / "sample_short.pdf")
        long = extract_text(SAMPLES_DIR / "sample_long.pdf")
        self.assertEqual(self.engine.answer("How tall is it?", pages=short.pages).mode, "direct")
        self.assertEqual(self.engine.answer("Who wrote ELIZA?", pages=long.pages).mode, "chunked")

    def test_empty_question_raises(self):
        with self.assertRaises(ValueError):
            self.engine.answer("   ", text="some text that is long enough here")


@unittest.skipUnless(os.environ.get("RUN_MULTILINGUAL_TESTS"),
                     "set RUN_MULTILINGUAL_TESTS=1 (downloads ~2.2 GB model)")
class TestMultilingual(unittest.TestCase):

    def test_french_examples(self):
        from app.config import MULTILINGUAL_MODEL
        from app.qa_engine import QAEngine
        engine = QAEngine(MULTILINGUAL_MODEL)
        examples = json.loads((SAMPLES_DIR / "expected_answers.json").read_text(encoding="utf-8"))
        for ex in (e for e in examples if e.get("multilingual")):
            with self.subTest(question=ex["question"]):
                doc = extract_text(SAMPLES_DIR / ex["pdf"])
                result = engine.answer(ex["question"], pages=doc.pages)
                self.assertIn(ex["expected"].lower(), result.answer.lower())


if __name__ == "__main__":
    unittest.main()
