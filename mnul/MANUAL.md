# AskMyPDF — User Manual

Ask a question about a PDF in plain language and get the exact answer back, with
the page it came from.

![The app answering a question](../docs/screenshot.png)

---

## Contents

1. [What it does](#1-what-it-does)
2. [First-time setup](#2-first-time-setup)
3. [Starting the app](#3-starting-the-app)
4. [The screen, explained](#4-the-screen-explained)
5. [Walkthrough A — an English PDF](#5-walkthrough-a--an-english-pdf)
6. [Walkthrough B — a non-English PDF](#6-walkthrough-b--a-non-english-pdf)
7. [Walkthrough C — a scanned PDF](#7-walkthrough-c--a-scanned-pdf)
8. [Reading the results](#8-reading-the-results)
9. [Settings reference](#9-settings-reference)
10. [Asking good questions](#10-asking-good-questions)
11. [What it cannot do](#11-what-it-cannot-do)
12. [Troubleshooting](#12-troubleshooting)
13. [Command reference](#13-command-reference)

---

## 1. What it does

You give it a PDF and a question. It finds the sentence containing the answer and
shows you the exact words, how confident it is, and which page they came from.

**It copies text out of your document — it never writes new text.** That means it
cannot invent a fact, but it also cannot summarise, add up numbers, or answer in
its own words. If the document doesn't contain the answer, it says so.

A real example, from the bundled sample:

> **Q:** Which computer defeated Garry Kasparov in 1997?
> **A:** Deep Blue — 76.7% confident, chunk 3 of 4, pages 2–3

It works in about 100 languages, reads scanned books through OCR, and runs entirely
on your machine — nothing is uploaded unless you deliberately choose the hosted option.

---

## 2. First-time setup

You need **Python 3.9 or newer**. Open a terminal in the project folder:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux the second line is `source .venv/bin/activate`.

That's all. The language model itself (about 500 MB) downloads automatically the
first time you ask a question, and is reused from then on — after that the app
works offline.

> **If you only want to read scanned PDFs later**, nothing extra is needed now;
> OCR is already included in `requirements.txt`.

---

## 3. Starting the app

```powershell
.venv\Scripts\activate
streamlit run app.py
```

Your browser opens at **http://localhost:8501**. If it doesn't, type that address
yourself.

To stop the app, press **Ctrl + C** in the terminal.

---

## 4. The screen, explained

### Sidebar (left)

| Control | What it's for |
|---|---|
| **Model** | *English* is faster; *Multilingual* understands ~100 languages. Pick Multilingual for any non-English PDF |
| **Run the model** | *Locally* (default, private, no internet needed) or *Hugging Face Inference API* (runs on Hugging Face's servers, needs a token) |
| **Chunking settings** | Leave *Auto* ticked unless you are experimenting |
| **OCR (scanned PDFs)** | Language, page range, and the option to force OCR |
| **⚡ Running on…** | Whether it is using your GPU or the CPU |
| **Model info** | Live details about the model, fetched from Hugging Face |

### Main area

1. **Upload a PDF** — or pick one of the bundled samples from the dropdown.
2. **Four figures** — pages, word count, which library read the file, and whether
   the document was short enough to read in one go ("direct") or had to be split
   ("chunked").
3. **Preview extracted text** — open this to see what the app actually read. Worth
   a glance whenever an answer looks wrong.
4. **Your question** and the **Get answer** button.
5. **The answer**, with confidence, source chunk, page and the surrounding text
   with the answer highlighted.

---

## 5. Walkthrough A — an English PDF

1. Start the app.
2. Under *…or try a sample PDF*, choose **sample_long.pdf**.
3. In **Example questions for this sample**, choose
   *"Which computer defeated Garry Kasparov in 1997?"* — or type your own.
4. Click **Get answer**.

You should see **Deep Blue**, about 77% confidence, from pages 2–3, with the
sentence shown underneath and the answer highlighted in yellow.

Ask another question without reloading; each answer is added to **Question history**,
which you can download as a CSV for your report.

---

## 6. Walkthrough B — a non-English PDF

1. Upload your PDF (or pick one from the `testings` folder, e.g. `test_hi_hindi.pdf`).
2. If it isn't in the Latin alphabet, a yellow warning appears with a
   **Switch to the multilingual model** button — click it. The larger model takes
   a few seconds to load the first time (and about 2.2 GB on first download).
3. Ask your question **in the same language as the document**, or in English —
   both work.
4. Click **Get answer**.

**The answer comes back in the document's language.** Asking "What fruit did Narada
bring?" about a Telugu book returns మామిడిపండు, not "a mango". That is what
"extractive" means: it hands you the words that are in the document.

> **Important for Indian languages.** Open *Preview extracted text* first. If letters
> look missing or words are broken (`ఆగ్రా` showing as `ఆ`), the PDF stores its
> letters in a way that cannot be read back properly. Fix it by ticking
> **"Use OCR even if the PDF has text"** in the sidebar — see Walkthrough C.
> Hindi, Tamil and Punjabi PDFs usually read correctly without this.

---

## 7. Walkthrough C — a scanned PDF

A scanned book is a *photograph* of each page. There is no text inside it to search,
so the app reads the pages as images instead — this is OCR.

The app spots this by itself:

> ⚠️ **This PDF looks scanned.** Only 10 words were found across 39 pages…

1. In the sidebar under **OCR (scanned PDFs)**, set:
   - **OCR language** — e.g. *Telugu + English*
   - **Start at page** — skip covers and contents. If the chapter you want starts
     on page 5, start there
   - **How many pages** — start with 5 or 10
2. Tick **🔍 Read this PDF with OCR**.
3. Wait. **OCR takes about 20 seconds per page**, so 6 pages is roughly two minutes.
   A progress bar shows where it is.
4. Ask your question as usual.

Pages you have already read are remembered, so widening the range later only reads
the new pages.

**For a PDF that has text but reads badly** (the Indian-language case in
Walkthrough B), tick **"Use OCR even if the PDF has text"** instead. Example: a
Telugu document where the answer came out as `ఆ నగరంలో` returns the correct
`ఆగ్రా నగరంలో` once OCR is on.

---

## 8. Reading the results

| What you see | What it means |
|---|---|
| **Answer** | The exact words from your document |
| **Confidence** | How sure the model is, 0–100%. Above ~50% is usually reliable |
| **Source: Chunk 3 of 4** | Long documents are read in pieces; this says which piece held the answer |
| **Location: pages 2–3** | Where to look in the original PDF |
| **Source context** | The passage, with the answer highlighted — *always worth checking* |
| ⚠️ **Low confidence** | Below 10%. Treat the answer as a guess |
| **Answers from each chunk** | What every other part of the document proposed |

### When it finds nothing

> ❌ **No confident answer.** Every chunk preferred "no answer here"…

Two possible reasons:

1. **The document genuinely doesn't say.** This is correct behaviour — the model
   declining rather than making something up.
2. **It is there but the model missed it.** Look at **Closest guesses** underneath;
   the right answer is often in that list, just below the model's confidence bar.

If you expected an answer, try: rephrasing the question, checking the extracted-text
preview, or switching the model.

---

## 9. Settings reference

**Model** — *English (RoBERTa-base)* is smaller and faster, English only.
*Multilingual (XLM-RoBERTa-large)* covers ~100 languages, is about four times larger
and slower, and needs roughly 2.5 GB of free memory.

**Run the model** — *Locally* keeps your document on your machine. *Hugging Face
Inference API* sends each question and passage to Hugging Face's servers; it needs a
free token in a `.env` file and is rate-limited, but needs no download.

**Chunking settings** — the model can only read about 512 word-pieces at a time, so
long documents are cut into overlapping chunks. *Auto* measures the right size for
your document; only untick it if you want to experiment.

**OCR** — language (one Indian script at a time, plus English), the page range, and
the option to use OCR even when the PDF already contains text.

---

## 10. Asking good questions

| Works well | Works badly |
|---|---|
| "When was the tower completed?" | "Summarise this document" |
| "Who built the Taj Mahal?" | "How many times is X mentioned?" |
| "What BLEU score was achieved?" | "Is this a good paper?" |
| "Which hardware was used for training?" | "What will happen next year?" |

Ask for **a fact that is written in the document**, in the way the document would
phrase it. Short, specific questions beat long ones.

---

## 11. What it cannot do

- **Summarise, count, calculate or compare.** It only points at existing text.
- **Join facts from distant pages.** It reads one chunk at a time, so a question
  needing page 2 *and* page 40 together will fail.
- **Read handwriting**, or scanned pages in Gujarati, Malayalam, Odia, Punjabi or
  Tamil (no OCR model available for those).
- **Read tables reliably.** Numbers in a table arrive as a flat row of digits, so
  "how many layers does the model have?" often fails on research papers.
- **Guarantee accuracy on dense technical PDFs.** Expect roughly 4 correct out of 10
  on a research paper, against near-perfect on ordinary prose.

---

## 12. Troubleshooting

| Problem | Fix |
|---|---|
| Browser doesn't open | Go to http://localhost:8501 manually |
| "This document isn't in the Latin alphabet…" | Click **Switch to the multilingual model** |
| Letters missing, words broken in the preview | Tick **Use OCR even if the PDF has text** |
| "This PDF looks scanned" | Tick **Read this PDF with OCR** and set a page range |
| "No confident answer" but you know it's there | Check **Closest guesses**; rephrase; check the preview |
| OCR is very slow | Normal: ~20 s per page. Read fewer pages, starting at the one you need |
| "Could not load the model… too little free memory" | Close other applications, or switch to the English model |
| "The token is valid but lacks inference permission (403)" | Create a **Read** token at huggingface.co/settings/tokens and put it in `.env` |
| "Rate limit reached" | Switch **Run the model** back to *Locally* |
| First question takes a minute | The model is downloading. Only happens once |
| `KeyError: Unknown task question-answering` | `pip install "transformers<5"` |

---

## 13. Command reference

Run these with the virtual environment activated.

```powershell
streamlit run app.py                     # start the app

python scripts/run_examples.py           # 7 English sample questions
python scripts/run_indic_examples.py     # 25 questions in 11 Indian languages
python scripts/run_test_pdfs.py          # the 26 language PDFs in /testings
python scripts/run_test_pdfs.py --ocr    # the same, read as images

python -m unittest discover -s tests -v  # the test suite (48 tests)
python -m app.hub_info                   # live model details from Hugging Face
```

**Sharing a demo as a link** — the app accepts URL parameters, so you can open it
with a question already answered:

```
http://localhost:8501/?sample=sample_long.pdf&q=Who+coined+the+term+artificial+intelligence
```

---

*For how the system works internally — extraction, chunking, scoring — see the
project [README](../README.md). Test results by language are in
[testings/README.md](../testings/README.md).*
