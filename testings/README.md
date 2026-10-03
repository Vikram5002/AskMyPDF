# /testings — multilingual test PDFs

Synthetic PDFs for checking AskMyPDF in many languages, plus the questions to ask.

| File | What it is |
|---|---|
| `test_<code>_<language>.pdf` | One short passage about the Taj Mahal, 26 languages |
| `ALL_QUESTIONS.pdf` | Every question with its expected answer — open this while testing |
| `questions.json` | The same questions, for the automated runner |
| `test_corpus.json` | The source text. Edit this, then regenerate |

Every passage states the **same facts** (Agra; built by Shah Jahan for Mumtaz Mahal;
1632–1653; white marble), so results are directly comparable across languages.

## Using them

**By hand:** `streamlit run app.py` → upload one of the PDFs → set **Model** to
*Multilingual (XLM-RoBERTa-large)* → ask a question from `ALL_QUESTIONS.pdf`.

**Automatically:**

```bash
python scripts/run_test_pdfs.py                  # all languages
python scripts/run_test_pdfs.py --only te,hi,ta  # a few
python scripts/run_test_pdfs.py --ocr            # read pages as images instead
python scripts/make_test_pdfs.py                 # regenerate the PDFs
```

The runner separates the two things that can fail:
**Text OK** = the answer survived the PDF round trip; **Answers** = the model found it.

## Results (multilingual model, this machine)

**45 of 55 questions correct** using the PDF text layer.

| Group | Result |
|---|---|
| English, French, Spanish, German, Portuguese, Russian, Turkish | 100% |
| Chinese, Japanese, Korean | 100% |
| Arabic, Urdu, Sindhi | 100% |
| Hindi, Tamil, Punjabi | 100% |
| Bengali, Telugu, Kannada, Malayalam, Marathi, Gujarati, Odia, Assamese, Nepali, Sanskrit | ~50% |

The ~50% group fails on **one specific question** — the one whose answer contains a
conjunct consonant, such as Telugu `ఆగ్రా` or Kannada `ಆಗ್ರಾ`. The PDF stores those
as a single ligature glyph with no mapping back to Unicode, so extraction returns
`ఆ␣␣` and the answer is simply not in the text any more. The date question in the
same language works fine. **This is a PDF limitation, not a model limitation.**

**The workaround: read the page as an image instead.** Running the same files with
`--ocr` recovers the conjuncts:

| Language | Via text layer | Via OCR |
|---|---|---|
| Telugu | 1/2 | **2/2** |
| Kannada | 1/2 | **2/2** |
| Bengali | 1/2 | **2/2** |

In the app, tick **"Use OCR even if the PDF has text"** under *OCR (scanned PDFs)*.
Use it whenever the *Preview extracted text* panel shows missing or broken letters.

OCR has no model at all for **Gujarati, Malayalam, Odia and Punjabi**, and Sanskrit
and Tamil are broken in EasyOCR 1.7.2 — those languages can be understood but a
scanned page in them cannot be read.

## Notes

- Right-to-left scripts (Arabic, Urdu, Sindhi) only work because the extractor
  repairs them: PDFs store those lines backwards and in "presentation form"
  characters. See `_repair_rtl` in [../app/pdf_extractor.py](../app/pdf_extractor.py).
- Chinese and Japanese report ~1 "word" because those scripts have no spaces; the
  chunker switches to counting characters for them.
- The passages were written for testing and are deliberately short and simple. For
  the languages the author cannot read fluently (Sanskrit, Sindhi, Assamese, Odia),
  treat the wording as approximate — the facts and the expected answers are what matter.
- The model supports ~100 languages; these 26 cover all the Indian languages it
  supports plus the major world ones. Add more by editing `test_corpus.json`.
