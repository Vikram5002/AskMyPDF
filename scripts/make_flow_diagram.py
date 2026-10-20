"""
Render flow/flow.html into flow/project_flow.png (and .pdf).

    python scripts/make_flow_diagram.py

DEV TOOL -- not needed to use AskMyPDF, and deliberately not in
requirements.txt. It needs Playwright driving the Edge or Chrome already on
the machine:

    pip install playwright

Edit flow/flow.html and re-run this to regenerate the picture.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "flow" / "flow.html"
PNG = ROOT / "flow" / "project_flow.png"
PDF = ROOT / "flow" / "project_flow.pdf"


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright is missing. Install it with:  pip install playwright")
        return 1
    if not SOURCE.exists():
        print(f"{SOURCE} not found.")
        return 1

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

        # device_scale_factor=2 renders at twice the pixel size, so the text
        # stays sharp when the picture is zoomed or printed.
        page = browser.new_page(viewport={"width": 1640, "height": 1200},
                                device_scale_factor=2)
        page.goto(SOURCE.as_uri(), wait_until="networkidle")
        page.screenshot(path=str(PNG), full_page=True)
        page.pdf(path=str(PDF), width="1640px", height="2150px",
                 print_background=True, margin={"top": "0", "bottom": "0",
                                                "left": "0", "right": "0"})
        browser.close()

    for path in (PNG, PDF):
        print(f"Wrote {path} ({path.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
