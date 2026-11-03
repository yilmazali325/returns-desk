import pathlib, sys
from playwright.sync_api import sync_playwright

src = pathlib.Path(sys.argv[1]); out = pathlib.Path(sys.argv[2]); shots = src.parent
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width": 1920, "height": 1080})
    pg.goto(src.as_uri()); pg.wait_for_load_state("networkidle"); pg.evaluate("document.fonts.ready")
    # report any slide whose content overflows its 1080px frame
    over = pg.evaluate("""[...document.querySelectorAll('section')].map((s, i) => {
        const r = s.getBoundingClientRect(); let max = 0;
        s.querySelectorAll('*').forEach(e => { const b = e.getBoundingClientRect(); if (b.height) max = Math.max(max, b.bottom - r.top); });
        return [i + 1, Math.round(max)]; }).filter(x => x[1] > 1080)""")
    print("overflow:", over or "none")
    for i, el in enumerate(pg.query_selector_all("section")):
        el.screenshot(path=str(shots / f"s{i+1:02d}.png"))
    pg.pdf(path=str(out), width="1920px", height="1080px", print_background=True)
    b.close()
print("ok")
