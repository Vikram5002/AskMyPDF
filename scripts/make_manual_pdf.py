"""
Turn mnul/MANUAL.md into a printable mnul/MANUAL.pdf.

    python scripts/make_manual_pdf.py

DEV TOOL -- not needed to use AskMyPDF, and deliberately not in
requirements.txt. It needs:

    pip install markdown playwright

Playwright drives the Edge or Chrome already on the machine (no extra browser
download) and prints the page. A plain `msedge --print-to-pdf` is used
elsewhere in this project, but it quietly does nothing when a browser window
is already open, so the manual uses Playwright instead.
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SOURCE = ROOT / "mnul" / "MANUAL.md"
OUTPUT = ROOT / "mnul" / "MANUAL.pdf"

CSS = """
  @page { size: A4; margin: 18mm 16mm; }
  body { font-family: "Segoe UI", system-ui, sans-serif; font-size: 10.5pt;
         line-height: 1.55; color: #1a1a1a; }
  h1 { font-size: 22pt; border-bottom: 2px solid #ff4b4b; padding-bottom: 6pt; }
  h2 { font-size: 15pt; margin-top: 22pt; border-bottom: 1px solid #ddd;
       padding-bottom: 3pt; page-break-after: avoid; }
  h3 { font-size: 12pt; margin-top: 16pt; page-break-after: avoid; }
  table { border-collapse: collapse; width: 100%; margin: 10pt 0;
          page-break-inside: avoid; }
  th, td { border: 1px solid #ccc; padding: 5pt 7pt; text-align: left;
           vertical-align: top; }
  th { background: #f4f4f4; }
  code { background: #f4f4f4; padding: 1pt 4pt; border-radius: 3px;
         font-family: Consolas, monospace; font-size: 9.5pt; }
  pre { background: #f7f7f7; border: 1px solid #e2e2e2; border-radius: 4px;
        padding: 8pt; overflow-x: auto; page-break-inside: avoid; }
  pre code { background: none; padding: 0; }
  blockquote { border-left: 3px solid #ff4b4b; margin-left: 0; padding-left: 12pt;
               color: #444; }
  img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
  ol, ul { padding-left: 20pt; }
"""


def main() -> int:
    try:
        import markdown
    except ImportError:
        print("The markdown package is missing. Install it with:  pip install markdown")
        return 1
    if not SOURCE.exists():
        print(f"{SOURCE} not found.")
        return 1

    body = markdown.markdown(
        SOURCE.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "toc", "sane_lists"],
    )
    # Image paths are relative to mnul/; the HTML is rendered from a temp folder,
    # so point them at the real files instead.
    body = body.replace('src="../docs/', f'src="{(ROOT / "docs").as_uri()}/')

    document = (f'<!doctype html><html lang="en"><meta charset="utf-8">'
                f"<title>AskMyPDF Manual</title><style>{CSS}</style>"
                f"<body>{body}</body></html>")

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright is missing. Install it with:  pip install playwright")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "manual.html"
        source.write_text(document, encoding="utf-8")
        with sync_playwright() as playwright:
            for channel in ("msedge", "chrome", None):
                try:
                    browser = (playwright.chromium.launch(channel=channel) if channel
                               else playwright.chromium.launch())
                    break
                except Exception:
                    browser = None
            if browser is None:
                print("No Edge or Chrome found. Run:  playwright install chromium")
                return 1
            page = browser.new_page()
            page.goto(source.as_uri(), wait_until="networkidle")
            page.pdf(path=str(OUTPUT), format="A4", print_background=True,
                     margin={"top": "18mm", "bottom": "18mm",
                             "left": "16mm", "right": "16mm"})
            browser.close()

    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
