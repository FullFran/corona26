from collections import Counter
from contextlib import contextmanager
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse
from urllib.request import urlopen


ROOT = Path(__file__).parents[1]
DOCS = ROOT / "docs"
HISTORIC_IDS = {"top", "pipeline", "phase-a", "phase-b", "explorer", "phase-c", "prediction", "next"}


class SiteParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids = []
        self.links = []
        self.assets = []
        self.main_count = 0
        self.h1_count = 0
        self.nav_labels = []
        self.tabs = []
        self.panels = []
        self.tablist_label = None
        self.rss_controls = []
        self.localized_rss_caption = False
        self.hidden = []
        self.images_without_alt = []
        self.headings = []
        self.tables = []
        self._table = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        element_id = attrs.get("id")
        if element_id:
            self.ids.append(element_id)
        if "hidden" in attrs:
            self.hidden.append(element_id or tag)
        if tag == "main":
            self.main_count += 1
        if tag == "nav":
            self.nav_labels.append(attrs.get("aria-label"))
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(tag[1])
            self.headings.append(level)
            if tag == "h1":
                self.h1_count += 1
        if tag == "a":
            href = attrs.get("href")
            if href:
                self.links.append(href)
            if "data-panel" in attrs:
                self.tabs.append((element_id, attrs["data-panel"], href))
            if "sim__btn" in attrs.get("class", "").split():
                self.rss_controls.append((href, attrs.get("hx-get"), attrs.get("aria-pressed")))
        if tag == "ul" and "tabs-nav__list" in attrs.get("class", "").split():
            self.tablist_label = (attrs.get("aria-label"), attrs.get("data-es"))
        if tag == "div" and "data-tab-panel" in attrs:
            self.panels.append(element_id)
        if tag == "img":
            if not attrs.get("alt"):
                self.images_without_alt.append(attrs.get("src"))
            if attrs.get("src"):
                self.assets.append(attrs["src"])
        if tag == "script" and attrs.get("src"):
            self.assets.append(attrs["src"])
        if tag == "table":
            self._table = {"caption": False, "bad_th": 0}
            self.tables.append(self._table)
        elif tag == "caption" and self._table is not None:
            self._table["caption"] = True
        elif tag == "figcaption" and "inversiones de polaridad" in attrs.get("data-es", ""):
            self.localized_rss_caption = True
        elif tag == "th" and self._table is not None and attrs.get("scope") not in {"row", "col"}:
            self._table["bad_th"] += 1

    def handle_endtag(self, tag):
        if tag == "table":
            self._table = None


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


@contextmanager
def served_docs():
    handler = lambda *args, **kwargs: QuietHandler(*args, directory=DOCS, **kwargs)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()


def load_site():
    with served_docs() as base_url:
        with urlopen(f"{base_url}/index.html") as response:
            html = response.read().decode("utf-8")
    parser = SiteParser()
    parser.feed(html)
    return html, parser


def test_document_landmarks_ids_links_and_assets():
    _, site = load_site()
    assert site.main_count == 1
    assert site.h1_count == 1
    assert site.nav_labels and all(site.nav_labels)
    assert not [item for item, count in Counter(site.ids).items() if count > 1]
    assert HISTORIC_IDS | {"field-guide"} <= set(site.ids)
    assert not site.hidden, "Progressive enhancement requires all base panels to be visible"

    unresolved = [href for href in site.links if href.startswith("#") and href[1:] not in site.ids]
    assert not unresolved
    local_links = [href for href in site.links if not urlparse(href).scheme and not href.startswith("#")]
    missing = [href for href in local_links + site.assets if not urlparse(href).scheme and not (DOCS / href).is_file()]
    assert not missing


def test_five_progressively_enhanced_tabs_match_panels():
    html, site = load_site()
    assert len(site.tabs) == 5
    assert len(site.panels) == 5
    assert site.tabs[0] == ("tab-guide", "panel-guide", "#field-guide")
    assert {panel for _, panel, _ in site.tabs} == set(site.panels)
    assert all(tab_id and panel and href for tab_id, panel, href in site.tabs)
    assert site.tablist_label == ("Page sections", "Secciones de la página")
    assert "nav.setAttribute('role', 'tablist')" in html
    assert "panel.setAttribute('tabindex', '0')" in html
    assert "el.setAttribute('aria-label'" in html


def test_hero_reports_post_eclipse_validation_status():
    html, _ = load_site()
    assert "Totality observed" in html
    assert "Totalidad observada" in html
    assert "Validation underway" in html
    assert "Validación en curso" in html
    assert 'id="countdown"' not in html
    assert "Totality in" not in html


def test_validation_is_provisional_and_not_an_official_claim():
    html, _ = load_site()
    panel = html[html.index('id="panel-validation"') : html.index("</main>")]
    assert "General shape partially supported" in panel
    assert "La forma general está parcialmente apoyada" in panel
    assert "provisional · qualitative only" in panel
    assert "qualitative_only" in panel
    assert "the official score remains pending" in panel
    assert "it is not an official score" in panel

    strong_claims = [
        "partially confirmed",
        "parcialmente confirmada",
        "prediction validated",
        "predicción validada",
        "acierto demostrado",
    ]
    assert not [claim for claim in strong_claims if claim in panel.lower()]


def test_validation_records_esa_evidence_candidate_state_and_sources():
    html, site = load_site()
    required = [
        "01:03:50",
        "01:04:40",
        "Six frames",
        "3&ndash;4 broad",
        "~2&ndash;2.5",
        "official_candidate: none_available",
        "12 Aug 2026 21:34 CEST",
        "CustomRendered",
        "No suitable public originals yet",
    ]
    assert not [text for text in required if text not in html]

    sources = {
        "https://www.youtube.com/watch?v=j2TjlTbHwsw",
        "https://commons.wikimedia.org/wiki/File:Eclipse_solar_total_12_de_agosto_de_2026_desde_Logro%C3%B1o.jpg",
        "https://trioeclipses.es/directo",
        "https://www.iac.es/es/divulgacion/noticias",
        "validation-protocol.md",
    }
    assert sources <= set(site.links)


def test_rss_selection_follows_successful_swap_and_has_no_js_fallback():
    html, site = load_site()
    assert len(site.rss_controls) == 5
    assert all(pressed in {"true", "false"} for _, _, pressed in site.rss_controls)
    assert all(href.startswith("assets/rss/ss_") and href.endswith(".png") for href, _, _ in site.rss_controls)
    assert all(hx_get.startswith("fragments/rss-") and hx_get.endswith(".html") for _, hx_get, _ in site.rss_controls)
    assert all((DOCS / href).is_file() for href, _, _ in site.rss_controls)
    assert all((DOCS / hx_get).is_file() for _, hx_get, _ in site.rss_controls)
    assert "var btn = e.target.closest('.sim__btn');" not in html
    assert "addEventListener('htmx:afterSwap'" in html
    assert "event.detail.requestConfig && event.detail.requestConfig.elt" in html
    assert "event.detail.target.id !== 'rss-view'" in html
    assert site.localized_rss_caption


def test_accessible_media_tables_and_heading_order():
    _, site = load_site()
    assert not site.images_without_alt
    assert site.tables and all(table["caption"] for table in site.tables)
    assert all(table["bad_th"] == 0 for table in site.tables)
    assert all(current <= previous + 1 for previous, current in zip(site.headings, site.headings[1:]))


def test_guide_contains_bilingual_safety_and_local_circumstances():
    html, _ = load_site()
    required = [
        "data-es=",
        "ISO 12312-2",
        "TODA la fotosfera",
        "in front of the objective",
        "19:36:24",
        "20:31:42",
        "20:32:00",
        "20:32:24",
        "21:17:06",
        "37&ndash;42 seconds",
        "7.4&deg; altitude",
        "283.2&deg;",
        "A cloud is <strong>never</strong>",
        "a solar filter, even when the Sun looks dim",
    ]
    assert not [text for text in required if text not in html]
