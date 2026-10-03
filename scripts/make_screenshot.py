"""
Take the README screenshot of the running app (docs/screenshot.png).

    streamlit run app.py          # in one terminal
    python scripts/make_screenshot.py

DEV TOOL -- not needed to use AskMyPDF, and deliberately not in
requirements.txt. It needs Playwright driving your installed Edge or Chrome:

    pip install playwright

A plain `--screenshot` from headless Chrome captures only Streamlit's grey
loading skeleton: the page arrives empty and fills in over a websocket.
Playwright can wait for the answer to actually appear before taking the shot.

The URL uses the app's own query parameters (?sample=...&q=...), so the page
loads with the question already answered.
"""

import argparse
import sys
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "screenshot.png"

SAMPLE = "sample_long.pdf"
QUESTION = "Which computer defeated Garry Kasparov in 1997?"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--port", type=int, default=8501, help="port the app runs on")
    parser.add_argument("--sample", default=SAMPLE)
    parser.add_argument("--question", default=QUESTION)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright is missing. Install it with:  pip install playwright")
        return 1

    url = (f"http://localhost:{args.port}/?sample={quote(args.sample)}"
           f"&q={quote(args.question)}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        # channel="msedge" uses the Edge already on the machine, so Playwright
        # doesn't have to download its own copy of Chromium.
        for channel in ("msedge", "chrome", None):
            try:
                browser = playwright.chromium.launch(channel=channel) if channel \
                    else playwright.chromium.launch()
                break
            except Exception:
                browser = None
        if browser is None:
            print("No Edge or Chrome found, and no bundled Chromium. "
                  "Run:  playwright install chromium")
            return 1

        page = browser.new_page(viewport={"width": 1500, "height": 1000},
                                device_scale_factor=2)   # crisp on high-DPI screens
        page.goto(url, wait_until="networkidle", timeout=120_000)
        # Wait for the real result. "Confidence" is the metric shown only once
        # an answer exists -- waiting for "Answer" would match the "Get answer"
        # button and capture the model-loading spinner instead.
        page.wait_for_selector("text=Confidence", timeout=300_000)
        page.wait_for_timeout(2000)                       # let the layout settle
        page.screenshot(path=str(args.out), full_page=True)
        browser.close()

    print(f"Wrote {args.out} ({args.out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
