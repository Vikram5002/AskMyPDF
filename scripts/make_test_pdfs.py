"""
Build the multilingual test PDFs in /testings.

    python scripts/make_test_pdfs.py
    python scripts/make_test_pdfs.py --only te,hi,ta

One PDF per language (testings/test_<code>_<name>.pdf) plus:
    testings/ALL_QUESTIONS.pdf   - every question, for reading while you test
    testings/questions.json      - the same questions, for automated checking

WHY NOT THE OTHER GENERATOR?
    scripts/make_sample_pdfs.py writes PDFs by hand with the built-in Helvetica
    font, which only covers Latin characters -- Telugu or Chinese would come out
    blank. Here we write an HTML page and let Microsoft Edge print it to PDF,
    because the browser picks a font that can show each script and embeds it.

    That is also how most real-world PDFs are produced, which makes these files
    a realistic test of text extraction, not just of the QA model.
"""

import argparse
import html
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTINGS_DIR = ROOT / "testings"
CORPUS = TESTINGS_DIR / "test_corpus.json"

EDGE_LOCATIONS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]
# Chrome works just as well; it takes the same switches.
CHROME_LOCATIONS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]

PAGE_CSS = """
  @page { size: A4; margin: 20mm; }
  body { font-family: "Nirmala UI", "Segoe UI", "Microsoft YaHei",
         "Yu Gothic", "Malgun Gothic", sans-serif; font-size: 13pt;
         line-height: 1.9; color: #000; }
  h1 { font-size: 20pt; margin-bottom: 4pt; }
  h2 { font-size: 15pt; margin-top: 18pt; }
  .meta { color: #555; font-size: 10pt; margin-bottom: 18pt; }
  .q { margin: 3pt 0 3pt 14pt; }
  .expected { color: #666; }
"""


def find_browser() -> str:
    """Locate a Chromium browser that can print HTML to PDF."""
    for candidate in EDGE_LOCATIONS + CHROME_LOCATIONS:
        if Path(candidate).exists():
            return candidate
    for name in ("msedge", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    raise SystemExit(
        "Could not find Microsoft Edge or Google Chrome, which this script uses "
        "to render non-Latin text into PDFs. Install either one, or generate the "
        "PDFs on another machine."
    )


def html_to_pdf(browser: str, page_html: str, out_path: Path) -> None:
    """
    Print one HTML page to a PDF file using headless Chromium.

    --user-data-dir matters: without it, a browser window the user already has
    open takes over the request, and the new process exits successfully having
    written nothing. A throwaway profile forces a separate instance.
    """
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "page.html"
        source.write_text(page_html, encoding="utf-8")
        profile = Path(tmp) / "profile"
        subprocess.run(
            [browser, "--headless", "--disable-gpu", "--no-pdf-header-footer",
             f"--user-data-dir={profile}", f"--print-to-pdf={out_path}",
             source.as_uri()],
            check=True, capture_output=True, timeout=180,
        )
        if not out_path.exists():
            raise RuntimeError(
                f"{Path(browser).name} reported success but wrote no PDF to "
                f"{out_path}. Close any open {Path(browser).stem} windows and retry."
            )


def document_html(entry: dict) -> str:
    """The test document for one language: a title and a short passage."""
    return f"""<!doctype html><html lang="{entry['code']}" dir="{entry['dir']}">
<meta charset="utf-8"><title>{html.escape(entry['title'])}</title>
<style>{PAGE_CSS}</style>
<body>
  <h1>{html.escape(entry['title'])}</h1>
  <div class="meta">{html.escape(entry['name'])} ({entry['code']}) &middot;
      AskMyPDF test document</div>
  <p>{html.escape(entry['context'])}</p>
</body></html>"""


def questions_html(entries: list) -> str:
    """One page listing every question, grouped by language."""
    blocks = []
    for entry in entries:
        rows = "\n".join(
            f'<div class="q" dir="{entry["dir"]}">{i}. {html.escape(item["q"])}'
            f' <span class="expected">&rarr; {html.escape(item["expected"])}</span></div>'
            for i, item in enumerate(entry["questions"], start=1))
        blocks.append(
            f'<h2>{html.escape(entry["name"])} ({entry["code"]}) '
            f'&mdash; test_{entry["code"]}_{entry["name"].lower()}.pdf</h2>\n{rows}')
    return f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>AskMyPDF test questions</title><style>{PAGE_CSS}</style>
<body>
  <h1>AskMyPDF &mdash; test questions</h1>
  <div class="meta">Open the PDF named under each heading in the app, ask the
  questions listed, and compare with the expected answer in grey.
  Every passage states the same facts, so answers are comparable across languages.</div>
  {"".join(blocks)}
</body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the /testings PDFs.")
    parser.add_argument("--only", help="comma-separated language codes, e.g. te,hi,ta")
    args = parser.parse_args()

    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    entries = corpus["languages"]
    if args.only:
        wanted = {code.strip().lower() for code in args.only.split(",")}
        entries = [e for e in entries if e["code"] in wanted]
        if not entries:
            print(f"No languages matching {args.only!r}.")
            return 1

    browser = find_browser()
    TESTINGS_DIR.mkdir(exist_ok=True)
    print(f"Rendering with {Path(browser).name}\n")

    for entry in entries:
        out = TESTINGS_DIR / f"test_{entry['code']}_{entry['name'].lower()}.pdf"
        html_to_pdf(browser, document_html(entry), out)
        print(f"  {out.name:<34} {entry['name']:<12} "
              f"{len(entry['context'].split()):>4} words, "
              f"{len(entry['questions'])} questions")

    all_entries = corpus["languages"]
    html_to_pdf(browser, questions_html(all_entries), TESTINGS_DIR / "ALL_QUESTIONS.pdf")
    (TESTINGS_DIR / "questions.json").write_text(
        json.dumps([{"pdf": f"test_{e['code']}_{e['name'].lower()}.pdf",
                     "language": e["name"], "code": e["code"],
                     "questions": e["questions"]} for e in all_entries],
                   ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  ALL_QUESTIONS.pdf               every question, for reading")
    print(f"  questions.json                  the same, for scripts/run_test_pdfs.py")
    print(f"\nWrote {len(entries)} language PDFs to {TESTINGS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
