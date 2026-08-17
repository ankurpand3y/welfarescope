"""Build the submission PDF.

Embeds the figures as base64 data URIs so the HTML is self contained, then prints
it with headless Chrome. Chrome is used rather than reportlab because it is the
only route on this machine with modern CSS support.

Run from the experiment root:  python report/build_report.py
"""

import base64
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "report_template.html")
HTML_OUT = os.path.join(HERE, "report.html")
PDF_OUT = os.path.join(HERE, "WelfareScope_Pandey_DigitalMinds2026.pdf")

FIGURES = {
    "FIG1": "fig1_design.png",
    "FIG2": "fig2_layer_sweep.png",
    "FIG3": "fig3_sign_flip.png",
    "FIG4": "fig4_heldout.png",
    "FIG5": "fig5_cosine_groups.png",
    "FIG6": "fig6_magnitude.png",
    "FIG7": "fig7_leace.png",
}

BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    os.path.expanduser(
        r"~\AppData\Local\ms-playwright\chromium-1217\chrome-win64\chrome.exe"),
]


def embed():
    with open(TEMPLATE, encoding="utf-8") as f:
        html = f.read()

    for key, name in FIGURES.items():
        path = os.path.join(HERE, "figures", name)
        if not os.path.exists(path):
            sys.exit(f"missing figure: {path}\nrun make_figures.py first")
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        token = "{{" + key + "}}"
        if token not in html:
            sys.exit(f"template has no placeholder for {key}")
        html = html.replace(token, f"data:image/png;base64,{b64}")

    leftover = [k for k in FIGURES if "{{" + k + "}}" in html]
    assert not leftover, leftover

    with open(HTML_OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {os.path.basename(HTML_OUT)} ({len(html)/1e6:.1f} MB with figures)")
    return html


def to_pdf():
    """Print via Playwright.

    Chrome's `--print-to-pdf-no-header` CLI flag is ignored in Chrome 151 (both
    --headless=new and --headless=old still stamp a date, URL and page number
    into the margins). Playwright drives the same engine through the DevTools
    protocol, where display_header_footer=False is actually honoured.
    """
    from playwright.sync_api import sync_playwright

    if os.path.exists(PDF_OUT):
        os.remove(PDF_OUT)

    exe = next((b for b in BROWSERS if os.path.exists(b)), None)
    if exe is None:
        sys.exit("no Chrome or Edge found")
    print(f"printing with {os.path.basename(exe)} via playwright")

    with sync_playwright() as pw:
        # Use the already installed system browser rather than making playwright
        # download its own pinned revision.
        browser = pw.chromium.launch(executable_path=exe)
        page = browser.new_page()
        page.goto("file:///" + HTML_OUT.replace("\\", "/"), wait_until="load")
        page.emulate_media(media="print")
        page.pdf(
            path=PDF_OUT,
            format="A4",
            print_background=True,      # keeps the callout box fills
            display_header_footer=False,
            prefer_css_page_size=True,  # honour the @page rule
        )
        browser.close()

    size = os.path.getsize(PDF_OUT)
    print(f"wrote {os.path.basename(PDF_OUT)} ({size/1e6:.2f} MB)")

    import fitz
    d = fitz.open(PDF_OUT)
    stamped = any("file:///" in p.get_text() for p in d)
    print(f"page count: {len(d)}   header/footer stamp: {stamped}")
    if stamped:
        sys.exit("print header/footer leaked into the PDF")


if __name__ == "__main__":
    embed()
    to_pdf()
