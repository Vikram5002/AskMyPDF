"""
Render report/REPORT.md into a two-page report/REPORT.pdf.

    python scripts/make_report_pdf.py

DEV TOOL -- not needed to use AskMyPDF. It needs `pip install markdown` and
Microsoft Edge or Google Chrome, and runs on any Python (it does not depend
on the project's virtual environment).

The layout is deliberately tight: a project report has a page budget, so the
font, margins and table padding here are smaller than in the user manual.
Put <div class="page-break"></div> in the Markdown to force the page break.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "report" / "REPORT.md"
OUTPUT = ROOT / "report" / "REPORT.pdf"

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]

CSS = """
  @page { size: A4; margin: 13mm 14mm; }
  body { font-family: "Segoe UI", system-ui, sans-serif; font-size: 9.3pt;
         line-height: 1.38; color: #1a1a1a; }
  h1 { font-size: 17pt; margin: 0 0 4pt; border-bottom: 2px solid #c00000;
       padding-bottom: 4pt; }
  h1 + p { margin-top: 4pt; font-size: 9pt; color: #333; }
  h3 { font-size: 11.5pt; margin: 11pt 0 4pt; color: #c00000;
       page-break-after: avoid; }
  p { margin: 0 0 6pt; text-align: justify; }
  table { border-collapse: collapse; width: 100%; margin: 5pt 0 7pt;
          font-size: 8.8pt; page-break-inside: avoid; }
  th, td { border: 1px solid #ccc; padding: 2.5pt 4pt; text-align: left;
           vertical-align: top; }
  th { background: #f2f2f2; }
  code { font-family: Consolas, monospace; font-size: 8.6pt;
         background: #f4f4f4; padding: 0 2pt; }
  ol { margin: 4pt 0; padding-left: 16pt; }
  li { margin-bottom: 2pt; }
  .page-break { page-break-before: always; }
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

    browser = next((b for b in BROWSERS if Path(b).exists()), None)
    if browser is None:
        print("Microsoft Edge or Google Chrome is needed to print the PDF.")
        return 1

    body = markdown.markdown(SOURCE.read_text(encoding="utf-8"),
                             extensions=["tables", "sane_lists"])
    page = (f'<!doctype html><html lang="en"><meta charset="utf-8">'
            f"<title>AskMyPDF Report</title><style>{CSS}</style><body>{body}</body></html>")

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "report.html"
        source.write_text(page, encoding="utf-8")
        # A throwaway profile: without it an open browser window takes over the
        # request and the new process exits successfully having written nothing.
        subprocess.run(
            [browser, "--headless", "--disable-gpu", "--no-pdf-header-footer",
             f"--user-data-dir={Path(tmp) / 'profile'}",
             f"--print-to-pdf={OUTPUT}", source.as_uri()],
            check=True, capture_output=True, timeout=180,
        )

    if not OUTPUT.exists():
        print("The browser reported success but wrote no PDF. Close open windows and retry.")
        return 1
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
