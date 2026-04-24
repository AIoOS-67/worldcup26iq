"""
WorldCup26AI — Streamlit app for the 2026 FIFA World Cup model.

Self-contained for Streamlit Cloud deploy. Reads parquet files from `./data/`
or the script directory.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from i18n import t, set_language_from_sidebar, LANGUAGES, translate_reason, team_name, TEAMS  # noqa: E402
from ask_model import (  # noqa: E402
    ask_claude,
    ask_gemini,
    parse_routing,
    build_data_context,
    EXAMPLE_QUESTIONS,
)

# Lazy markdown lib for chat bubble content
try:
    import markdown as _md
    def _md_to_html(text: str) -> str:
        return _md.markdown(text or "", extensions=["fenced_code", "nl2br"])
except ImportError:
    def _md_to_html(text: str) -> str:
        import html as _html
        escaped = _html.escape(text or "")
        return escaped.replace("\n\n", "</p><p>").replace("\n", "<br>")


def _render_merch_card(rec: dict) -> str:
    """Inline shoppable card — smaller than Team Explorer version, designed to
    slot inside a chat bubble below Claude's text.

    If rec carries a 'product' dict (populated whenever Claude picked a
    specific Fanatics SKU — player jersey, authentic kit, hat, etc.), the
    card uses that SKU's real image and deep link. Otherwise falls back to
    the team's TheSportsDB crest + generic jersey and the wrapped search URL.

    If rec carries a 'pricing' dict, shows SALE badge / strikethrough / promo.
    """
    media = globals()["load_team_media"]().get(rec["team"], {})
    product = rec.get("product") or {}
    # Prefer the real Fanatics SKU link (already publisher-tagged, deep-linked
    # to that specific product page — highest conversion). Fall back to the
    # team-level wrapped search URL.
    shop_url = product.get("link") or globals()["merch_link"](rec["team"])
    localized = globals()["team_name"](rec["team"])
    pitch = rec.get("pitch", "")
    pricing = rec.get("pricing") or {}

    # Crest always comes from TheSportsDB (matches app-wide style + has the
    # national badge). Right-side image prefers the specific Fanatics product
    # photo — so a "Messi jersey" card shows the Messi shirt, not a generic
    # Argentina kit.
    badge_html = (
        f'<img src="{media["badge"]}" alt="{rec["team"]} crest" '
        f'style="height:56px;width:56px;object-fit:contain;flex-shrink:0;">'
        if media.get("badge") else ""
    )
    right_img_src = product.get("image_url") or media.get("jersey")
    jersey_html = (
        f'<img src="{right_img_src}" alt="{product.get("name") or rec["team"]}" '
        f'style="height:88px;width:auto;flex-shrink:0;'
        f'background:rgba(255,255,255,0.04);border-radius:4px;padding:4px;">'
        if right_img_src else ""
    )

    # Pricing strip — only rendered when pricing was looked up this turn
    price_html = ""
    sale_pill = ""
    disclaimer = ""
    if pricing:
        currency = "$" if pricing.get("currency") == "USD" else ""
        list_p = pricing.get("list_price")
        sale_p = pricing.get("sale_price")
        if sale_p and list_p and sale_p < list_p:
            sale_pill = (
                f'<span style="background:#ef4444;color:#fff;font-size:0.7rem;'
                f'font-weight:700;padding:2px 8px;border-radius:10px;'
                f'margin-left:8px;letter-spacing:0.5px;">'
                f'SALE −{pricing.get("discount_pct", 0)}%</span>'
            )
            countdown = pricing.get("sale_ends_hours")
            countdown_html = (
                f'<span style="color:#94a3c5;font-size:0.72rem;margin-left:8px;">'
                f'ends in {countdown}h</span>' if countdown else ""
            )
            price_html = (
                f'<div style="margin-top:4px;font-size:0.85rem;">'
                f'<span style="color:#94a3c5;text-decoration:line-through;">'
                f'{currency}{list_p}</span> '
                f'<span style="color:#3dd68c;font-weight:700;">'
                f'{currency}{sale_p}</span>'
                f'{countdown_html}</div>'
            )
        elif list_p:
            price_html = (
                f'<div style="margin-top:4px;font-size:0.85rem;color:#cfd7e8;">'
                f'{currency}{list_p}</div>'
            )
        promo = pricing.get("promo_code")
        if promo:
            price_html += (
                f'<div style="margin-top:4px;font-size:0.78rem;color:#cfd7e8;">'
                f'Code <code style="background:rgba(247,201,72,0.15);'
                f'color:#f7c948;padding:1px 6px;border-radius:3px;'
                f'font-weight:600;">{promo}</code> '
                f'= extra −{pricing.get("promo_discount_pct", 0)}%</div>'
            )
        is_real = bool(pricing.get("_source"))
        disclaimer_text = (
            f'Live from {pricing["_source"]}' if is_real
            else 'Fallback estimate — no current inventory in Fanatics feed for this team.'
        )
        disclaimer = (
            f'<div style="margin-top:6px;font-size:0.68rem;color:#5b6c8a;'
            f'font-style:italic;">{disclaimer_text}</div>'
        )

    border_color = "rgba(239,68,68,0.5)" if sale_pill else "rgba(247,201,72,0.3)"
    bg_color = "rgba(239,68,68,0.06)" if sale_pill else "rgba(247,201,72,0.08)"

    # Age-group pill ("👨 Men's" / "👦 Youth" / "👩 Women's" / ...) — helps
    # users spot kids-sized SKUs without having to click through. Only shown
    # when the real Fanatics product provides an age_group.
    _AGE_EMOJI = {
        "Men's":   "👨",
        "Women's": "👩",
        "Adult":   "👥",
        "Youth":   "👦",
        "Boys'":   "👦",
        "Girls'":  "👧",
        "Toddler": "🧒",
        "Infant":  "👶",
    }
    age_group = (product or {}).get("age_group") or ""
    age_pill = ""
    if age_group:
        emoji = _AGE_EMOJI.get(age_group, "")
        age_pill = (
            f'<span style="background:rgba(148,163,197,0.15);'
            f'color:#cfd7e8;font-size:0.7rem;font-weight:600;'
            f'padding:2px 8px;border-radius:10px;margin-left:8px;'
            f'letter-spacing:0.3px;">{emoji} {age_group}</span>'
        )

    return (
        f'<a href="{shop_url}" target="_blank" rel="noopener sponsored" '
        f'style="display:flex;gap:12px;align-items:center;margin-top:10px;'
        f'padding:10px 12px;background:{bg_color};'
        f'border:1px solid {border_color};border-radius:8px;'
        f'text-decoration:none;color:inherit;">'
        f'{badge_html}{jersey_html}'
        f'<div style="flex:1;min-width:0;">'
        f'<div style="font-weight:600;color:#f7c948;font-size:0.95rem;">'
        f'🛒 {localized}{sale_pill}{age_pill}</div>'
        f'<div style="font-size:0.82rem;color:#cfd7e8;margin-top:2px;">{pitch}</div>'
        f'{price_html}{disclaimer}'
        f'</div></a>'
    )


def _render_msg(role: str, content) -> str:
    """Return HTML for a single WeChat-style message.

    `content` is a str for most roles. For claude it may also be a dict
    shaped like {'text': str, 'merch': [{'team','pitch'}, ...]} — in that
    case we render the text as markdown and append merch cards below.
    """
    if role == "user":
        avatar, name, cls = "🧑", "You", "user"
    elif role == "claude":
        avatar, name, cls = "🤖", "Claude · Anthropic", "claude"
    elif role == "gemini":
        avatar, name, cls = "💎", "Gemini · Google", "gemini"
    elif role == "thinking_claude":
        avatar, name, cls = "🤖", "Claude · Anthropic", "claude"
        inner = '<div class="wc-thinking">typing…</div>'
        return (f'<div class="wc-row claude"><div class="wc-avatar">{avatar}</div>'
                f'<div class="wc-bwrap"><div class="wc-name">{name}</div>'
                f'<div class="wc-bubble claude">{inner}</div></div></div>')
    elif role == "thinking_gemini":
        avatar, name, cls = "💎", "Gemini · Google", "gemini"
        inner = '<div class="wc-thinking">typing…</div>'
        return (f'<div class="wc-row gemini"><div class="wc-avatar">{avatar}</div>'
                f'<div class="wc-bwrap"><div class="wc-name">{name}</div>'
                f'<div class="wc-bubble gemini">{inner}</div></div></div>')
    else:
        avatar, name, cls = "⚠️", "System", "error"

    if isinstance(content, dict):
        text = content.get("text", "")
        merch = content.get("merch", []) or []
    else:
        text = content
        merch = []
    html_content = _md_to_html(text)
    merch_html = "".join(_render_merch_card(m) for m in merch)
    return (
        f'<div class="wc-row {cls}"><div class="wc-avatar">{avatar}</div>'
        f'<div class="wc-bwrap"><div class="wc-name">{name}</div>'
        f'<div class="wc-bubble {cls}">{html_content}{merch_html}</div></div></div>'
    )


def team_with_flag(english: str) -> str:
    """Localized team name prefixed with its flag emoji."""
    return f"{flag(english)} {team_name(english)}"


HERE = Path(__file__).resolve().parent


def _p(name: str) -> Path:
    here = HERE / name
    sub = HERE / "data" / name
    return here if here.exists() else sub


# ---------- flag map ----------
FLAGS = {
    "Algeria": "🇩🇿", "Argentina": "🇦🇷", "Australia": "🇦🇺", "Austria": "🇦🇹",
    "Belgium": "🇧🇪", "Bosnia and Herzegovina": "🇧🇦", "Brazil": "🇧🇷",
    "Canada": "🇨🇦", "Cape Verde": "🇨🇻", "Colombia": "🇨🇴", "Croatia": "🇭🇷",
    "Curaçao": "🇨🇼", "Curacao": "🇨🇼", "Czech Republic": "🇨🇿", "DR Congo": "🇨🇩",
    "Ecuador": "🇪🇨", "Egypt": "🇪🇬",
    # UK subdivision flag emoji (🏴󠁧󠁢󠁥󠁮󠁧󠁿 et al.) don't render on Windows/Chrome —
    # fall back to 🇬🇧 so tables/plotly axes/buttons stay visually consistent.
    "England": "🇬🇧",
    "France": "🇫🇷", "Germany": "🇩🇪", "Ghana": "🇬🇭", "Haiti": "🇭🇹",
    "Iran": "🇮🇷", "Iraq": "🇮🇶", "Ivory Coast": "🇨🇮", "Japan": "🇯🇵",
    "Jordan": "🇯🇴", "Mexico": "🇲🇽", "Morocco": "🇲🇦", "Netherlands": "🇳🇱",
    "New Zealand": "🇳🇿", "Norway": "🇳🇴", "Panama": "🇵🇦", "Paraguay": "🇵🇾",
    "Portugal": "🇵🇹", "Qatar": "🇶🇦", "Saudi Arabia": "🇸🇦",
    "Scotland": "🇬🇧",
    "Senegal": "🇸🇳", "South Africa": "🇿🇦", "South Korea": "🇰🇷", "Spain": "🇪🇸",
    "Sweden": "🇸🇪", "Switzerland": "🇨🇭", "Tunisia": "🇹🇳", "Turkey": "🇹🇷",
    "United States": "🇺🇸", "Uruguay": "🇺🇾", "Uzbekistan": "🇺🇿",
    # Extras that show up in historical Polymarket markets
    "Republic of Ireland": "🇮🇪", "Northern Ireland": "🇬🇧", "Wales": "🇬🇧",
    "Chile": "🇨🇱", "Peru": "🇵🇪", "Bolivia": "🇧🇴", "Venezuela": "🇻🇪",
    "Poland": "🇵🇱", "Ukraine": "🇺🇦", "Serbia": "🇷🇸", "Greece": "🇬🇷",
    "Denmark": "🇩🇰", "Hungary": "🇭🇺", "Romania": "🇷🇴", "Slovakia": "🇸🇰",
    "Slovenia": "🇸🇮", "Albania": "🇦🇱", "Israel": "🇮🇱", "Italy": "🇮🇹",
    "Cameroon": "🇨🇲", "Nigeria": "🇳🇬", "Kenya": "🇰🇪",
}


def flag(team: str) -> str:
    return FLAGS.get(team, "🏳️")


def with_flag(team: str) -> str:
    return f"{flag(team)} {team}"


# ---------- FlagCDN: real flag images for table contexts ----------
# Used by dataframes via st.column_config.ImageColumn. Includes ISO 3166-2
# subdivision codes (gb-eng, gb-sct, gb-wls) so England, Scotland, and Wales
# render their proper flags — not the Union Jack fallback used in emoji mode.
ISO = {
    "Algeria": "dz", "Argentina": "ar", "Australia": "au", "Austria": "at",
    "Belgium": "be", "Bosnia and Herzegovina": "ba", "Brazil": "br",
    "Canada": "ca", "Cape Verde": "cv", "Colombia": "co", "Croatia": "hr",
    "Curaçao": "cw", "Curacao": "cw", "Czech Republic": "cz", "DR Congo": "cd",
    "Ecuador": "ec", "Egypt": "eg", "England": "gb-eng",
    "France": "fr", "Germany": "de", "Ghana": "gh", "Haiti": "ht",
    "Iran": "ir", "Iraq": "iq", "Ivory Coast": "ci", "Japan": "jp",
    "Jordan": "jo", "Mexico": "mx", "Morocco": "ma", "Netherlands": "nl",
    "New Zealand": "nz", "Norway": "no", "Panama": "pa", "Paraguay": "py",
    "Portugal": "pt", "Qatar": "qa", "Saudi Arabia": "sa", "Scotland": "gb-sct",
    "Senegal": "sn", "South Africa": "za", "South Korea": "kr", "Spain": "es",
    "Sweden": "se", "Switzerland": "ch", "Tunisia": "tn", "Turkey": "tr",
    "United States": "us", "Uruguay": "uy", "Uzbekistan": "uz",
    # Extras that show up in historical Polymarket markets
    "Republic of Ireland": "ie", "Northern Ireland": "gb-nir", "Wales": "gb-wls",
    "Chile": "cl", "Peru": "pe", "Bolivia": "bo", "Venezuela": "ve",
    "Poland": "pl", "Ukraine": "ua", "Serbia": "rs", "Greece": "gr",
    "Denmark": "dk", "Hungary": "hu", "Romania": "ro", "Slovakia": "sk",
    "Slovenia": "si", "Albania": "al", "Israel": "il", "Italy": "it",
    "Cameroon": "cm", "Nigeria": "ng", "Kenya": "ke",
}


def flag_url(team: str, size: str = "w40") -> str | None:
    """FlagCDN PNG URL for a team. None if team not mapped — caller should fall
    back to emoji. Sizes: w20, w40, w80, w160, w320, w640."""
    code = ISO.get(team)
    return f"https://flagcdn.com/{size}/{code}.png" if code else None


def flag_img(team: str, h: int = 16, size: str = "w40") -> str:
    """Inline <img> tag for a team's flag (HTML contexts only). Falls back to
    the emoji flag() if team isn't in ISO map. Use this instead of flag() in
    any st.markdown(..., unsafe_allow_html=True) block so flags render on
    browsers that don't support regional-indicator emoji (e.g. some Chrome
    installs on Windows)."""
    url = flag_url(team, size)
    if not url:
        return flag(team)
    return (
        f'<img src="{url}" alt="{team}" '
        f'style="height:{h}px;vertical-align:-2px;border-radius:2px;'
        f'box-shadow:0 0 0 1px rgba(255,255,255,0.08);">'
    )


def team_with_flag_img(team: str, h: int = 16) -> str:
    """<img> flag + localized team name (HTML contexts)."""
    return f"{flag_img(team, h)} {team_name(team)}"


# Fanatics affiliate tracking — live via Impact.com, approved Apr 23 2026.
# Every click through merch_link() carries this publisher tag, so a purchase
# on the landing page attributes commission (1%–8%, 2%–8% on Online Sale)
# back to FoodyePay Technology's Impact account.
FANATICS_PUBLISHER_ID = "7225697"
FANATICS_AD_ID        = "586570"
FANATICS_PROGRAM_ID   = "9663"  # Fanatics (Global)
FANATICS_CLICK_BASE = (
    f"https://fanatics.93n6tx.net/c/{FANATICS_PUBLISHER_ID}"
    f"/{FANATICS_AD_ID}/{FANATICS_PROGRAM_ID}"
)


def merch_link(team: str, kind: str = "jersey") -> str:
    """Return an Impact-wrapped Fanatics affiliate link for this team.

    Prefers a deep link straight to a specific SKU (from the Fanatics product
    feed) — the feed's `link` column is already publisher-tagged and points at
    an exact product page, which converts 3-5x better than a search landing.

    Falls back to a wrapped search URL when the team has no inventory in the
    feed (~11 of the 48 WC26 teams, mostly lower-profile sides)."""
    product = find_team_product(team, prefer_keyword=kind)
    if product and product.get("link"):
        return product["link"]
    # Fallback — wrapped Fanatics search
    import urllib.parse
    q = urllib.parse.quote_plus(f"{team} {kind}")
    destination = f"https://www.fanatics.com/search?query={q}"
    return f"{FANATICS_CLICK_BASE}?u={urllib.parse.quote(destination, safe='')}"


# ---------- global config + CSS ----------
def _page_icon():
    """Prefer the gold-ball logo as favicon; fall back to ⚽ emoji if PIL
    can't open it (e.g. missing file)."""
    try:
        from PIL import Image
        p = HERE / "static" / "logo.png"
        if p.exists():
            return Image.open(p)
    except Exception:
        pass
    return "⚽"


st.set_page_config(
    page_title="WorldCup26AI",
    page_icon=_page_icon(),
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- PWA manifest injection (makes the app installable on mobile home screens) ----------
# st.markdown strips <script>, so we use components.html which runs in an iframe
# and reaches into window.parent.document to modify the main page's <head>.
import streamlit.components.v1 as _components
_components.html(
    """
    <script>
      (function() {
        try {
          const doc = window.parent.document;
          const head = doc.head;
          const origin = window.parent.location.origin;
          // Streamlit Cloud serves static files at /app/static/...
          const baseStatic = origin + '/app/static';

          function ensure(selector, create) {
            if (!doc.querySelector(selector)) head.appendChild(create());
          }

          ensure('link[rel="manifest"]', () => {
            const l = doc.createElement('link');
            l.rel = 'manifest'; l.href = baseStatic + '/manifest.webmanifest';
            return l;
          });
          ensure('meta[name="theme-color"]', () => {
            const m = doc.createElement('meta');
            m.name = 'theme-color'; m.content = '#f7c948'; return m;
          });
          [
            ['apple-mobile-web-app-capable',          'yes'],
            ['apple-mobile-web-app-status-bar-style', 'black-translucent'],
            ['apple-mobile-web-app-title',            'WC26AI'],
            ['mobile-web-app-capable',                'yes'],
          ].forEach(([name, content]) => {
            ensure(`meta[name="${name}"]`, () => {
              const m = doc.createElement('meta');
              m.name = name; m.content = content; return m;
            });
          });
          ensure('link[rel="apple-touch-icon"]', () => {
            const l = doc.createElement('link');
            l.rel = 'apple-touch-icon'; l.href = baseStatic + '/icon-192.png';
            return l;
          });
        } catch (e) { /* Cross-origin or sandbox — PWA disabled, page still works */ }
      })();
    </script>
    """,
    height=0,
)

CUSTOM_CSS = """
<style>
  /* hide Streamlit chrome */
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }

  /* --- WeChat-style chat bubbles --- */
  .wc-row { display: flex; margin: 10px 0; gap: 10px; align-items: flex-start; }
  .wc-row.user { flex-direction: row-reverse; }
  .wc-avatar {
    width: 40px; height: 40px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; background: #1b2742; flex-shrink: 0;
    border: 1px solid #23315c;
  }
  .wc-row.user .wc-avatar { background: #2a3f22; border-color: #4a7a2a; }
  .wc-row.claude .wc-avatar { background: #3a2e15; border-color: #f7c94880; }
  .wc-row.gemini .wc-avatar { background: #291f4a; border-color: #a78bfa80; }
  .wc-bwrap { max-width: 72%; display: flex; flex-direction: column; }
  .wc-row.user .wc-bwrap { align-items: flex-end; }
  .wc-name {
    font-size: 0.72rem; color: #94a3c5; margin-bottom: 4px; padding: 0 4px;
    letter-spacing: 0.02em;
  }
  .wc-bubble {
    padding: 10px 14px; border-radius: 10px; line-height: 1.55;
    font-size: 0.95rem; word-wrap: break-word; overflow-wrap: anywhere;
  }
  .wc-bubble.user {
    background: #95ec69 !important; color: #111 !important;
    border-top-right-radius: 3px;
  }
  .wc-bubble.claude {
    background: #1e2638 !important; color: #f5f7fb !important;
    border: 1px solid #f7c94850; border-top-left-radius: 3px;
  }
  .wc-bubble.gemini {
    background: #1f1a36 !important; color: #f5f7fb !important;
    border: 1px solid #a78bfa50; border-top-left-radius: 3px;
  }
  .wc-bubble.error {
    background: #2a1a1e !important; color: #fca5a5 !important;
    border: 1px solid #7a2a2a;
  }
  .wc-bubble p { margin: 0 0 6px 0; }
  .wc-bubble p:last-child { margin-bottom: 0 !important; }
  .wc-bubble ul, .wc-bubble ol { margin: 4px 0 4px 20px; padding: 0; }
  .wc-bubble li { margin: 2px 0; }
  .wc-bubble.user strong { color: #000; }
  .wc-bubble.claude strong { color: #f7c948; }
  .wc-bubble.gemini strong { color: #c4b5fd; }
  .wc-bubble code {
    background: rgba(148,163,197,0.15); padding: 1px 5px; border-radius: 4px;
    font-size: 0.88em;
  }
  .wc-thinking {
    color: #94a3c5; font-style: italic; padding: 8px 14px;
  }
  .wc-hint {
    background: #121c2e; border: 1px dashed #23315c; border-radius: 8px;
    padding: 8px 12px; color: #94a3c5; font-size: 0.85rem; margin: 8px 0 16px 0;
  }
  .wc-hint code { color: #f7c948; background: transparent; padding: 0; }

  /* Groups Overview cards */
  .grp-grid {
    display: grid !important;
    grid-template-columns: repeat(auto-fit, minmax(310px, 1fr)) !important;
    gap: 14px !important;
    margin-top: 8px !important;
  }
  .grp-card {
    background: #121c2e !important;
    border: 1px solid #23315c !important;
    border-radius: 12px !important;
    padding: 14px 16px !important;
  }
  .grp-card .letter {
    display: inline-block; width: 32px; height: 32px; line-height: 32px;
    text-align: center; border-radius: 8px;
    background: #f7c948; color: #0b1220 !important;
    font-weight: 800 !important; font-size: 1rem;
    margin-right: 10px;
  }
  .grp-card h3 {
    display: inline-block; color: #f7c948 !important;
    margin: 0 !important; font-size: 1.1rem !important; vertical-align: middle;
  }
  .grp-card .dates {
    color: #94a3c5 !important; font-size: 0.78rem;
    margin: 6px 0 10px 0;
  }
  .grp-card table {
    width: 100%; border-collapse: collapse; margin: 4px 0 8px 0;
  }
  .grp-card td {
    padding: 5px 0 !important; font-size: 0.92rem;
    border-bottom: 1px solid #1b2742;
  }
  .grp-card td.team { color: #e8edf7; font-weight: 500; }
  .grp-card td.prob { text-align: right; color: #f7c948; font-weight: 700; font-variant-numeric: tabular-nums; }
  .grp-card tr.lock td.prob { color: #4ade80; }
  .grp-card tr.out td.prob { color: #94a3c5; }
  .grp-card tr.out td.team { color: #94a3c5; }
  .grp-card .verdict {
    margin-top: 8px; padding: 8px 10px;
    background: rgba(247, 201, 72, 0.08);
    border-left: 3px solid #f7c948;
    color: #e8edf7 !important; font-size: 0.88rem;
    border-radius: 4px;
  }
  .grp-card .verdict strong { color: #f7c948 !important; }

  /* Schedule table */
  .sch-day {
    color: #f7c948 !important; font-size: 1.05rem !important;
    font-weight: 700 !important;
    margin: 18px 0 4px 0 !important;
    padding-bottom: 4px; border-bottom: 1px solid #23315c;
  }
  .sch-row {
    display: grid !important;
    grid-template-columns: 60px 1fr 70px 2fr !important;
    gap: 12px !important;
    padding: 10px 8px !important; align-items: center !important;
    border-bottom: 1px solid #1b2742;
    font-size: 0.93rem;
  }
  .sch-row:hover { background: rgba(247,201,72,0.04); }
  .sch-row .md { color: #94a3c5 !important; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.06em; }
  .sch-row .match { color: #e8edf7 !important; font-weight: 500; }
  .sch-row .match .vs { color: #94a3c5; padding: 0 6px; }
  .sch-row .group-chip {
    display: inline-block; padding: 3px 9px; border-radius: 999px;
    background: rgba(247,201,72,0.14); color: #f7c948 !important;
    font-weight: 700; font-size: 0.78rem; text-align: center;
  }
  .sch-row .venue { color: #94a3c5 !important; font-size: 0.85rem; }
  .sch-row .venue .stadium { color: #cbd5e8 !important; font-weight: 500; }

  /* Squad list with headshots */
  .squad-list { display: flex; flex-direction: column; margin: 8px 0; }
  .s-row {
    display: grid !important;
    grid-template-columns: 52px 1.6fr 2fr 0.8fr !important;
    gap: 12px !important;
    align-items: center !important;
    padding: 8px 10px !important;
    border-bottom: 1px solid #1b2742;
    font-size: 0.92rem;
  }
  .s-row:hover { background: rgba(247,201,72,0.04); }
  .s-avatar {
    width: 48px; height: 48px; border-radius: 50%; object-fit: cover;
    background: #1b2742; border: 2px solid #23315c;
  }
  .s-avatar.placeholder {
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; color: #94a3c5;
  }
  .s-name { color: #e8edf7 !important; font-weight: 600 !important; }
  .s-pos { color: #94a3c5 !important; font-size: 0.78rem !important; font-weight: 400 !important; margin-top: 2px; }
  .s-club { color: #cbd5e8 !important; font-weight: 500; }
  .s-league { color: #94a3c5 !important; font-size: 0.78rem !important; font-weight: 400 !important; margin-top: 2px; }
  .s-value { color: #f7c948 !important; font-weight: 700 !important; text-align: right !important; font-variant-numeric: tabular-nums; }

  /* What-If group picker cards */
  .gpick-title {
    font-weight: 700; color: #f7c948;
    letter-spacing: 0.05em;
    padding: 6px 0 4px 2px;
    font-size: 0.95rem;
    border-bottom: 1px solid #1b2742;
    margin-bottom: 6px;
  }

  /* What-If bracket view */
  .bracket {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 10px;
    margin: 14px 0 8px 0;
  }
  .bracket-col {
    display: flex; flex-direction: column; gap: 4px;
    min-width: 0;
  }
  .bracket-col h4 {
    color: #f7c948 !important; font-size: 0.88rem !important;
    letter-spacing: 0.06em; text-transform: uppercase;
    margin: 0 0 6px 0 !important; padding: 0 !important;
    text-align: center;
  }
  .bracket-cell {
    position: relative;
    padding: 6px 8px;
    border-radius: 6px;
    background: #121c2e;
    border: 1px solid #1b2742;
    font-size: 0.78rem;
    color: #e8edf7;
    display: flex; align-items: center; justify-content: space-between;
    gap: 6px;
    overflow: hidden;
  }
  .bracket-cell .fill {
    position: absolute; left: 0; top: 0; bottom: 0;
    background: linear-gradient(90deg, rgba(247,201,72,.25) 0%, rgba(247,201,72,.08) 100%);
    z-index: 0;
  }
  .bracket-cell .label, .bracket-cell .pct {
    position: relative; z-index: 1;
  }
  .bracket-cell .label {
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    flex: 1 1 auto; font-weight: 500;
  }
  .bracket-cell .pct {
    color: #f7c948; font-weight: 700; font-variant-numeric: tabular-nums;
    font-size: 0.75rem;
  }
  .bracket-cell.top { border-color: #f7c94860; background: #1a2340; }

  /* Roadmap "Coming May 31" card */
  .roadmap {
    background: linear-gradient(135deg, #1a1034 0%, #2a1640 50%, #1a2a52 100%) !important;
    border: 1px solid #a78bfa60 !important;
    border-radius: 14px !important;
    padding: 24px 28px !important;
    margin: 20px 0 !important;
    position: relative;
    overflow: hidden;
  }
  .roadmap::before {
    content: ""; position: absolute; top: -30px; right: -30px;
    width: 120px; height: 120px; border-radius: 50%;
    background: radial-gradient(circle, #f7c94830 0%, transparent 70%);
  }
  .roadmap h2 {
    color: #f7c948 !important; font-size: 1.4rem !important;
    margin: 0 0 6px 0 !important; padding: 0 !important;
  }
  .roadmap p.date {
    color: #c4b5fd !important; margin: 0 0 14px 0 !important;
    font-size: 0.95rem !important;
  }
  .roadmap ul { list-style: none !important; padding: 0 !important; margin: 0 !important; }
  .roadmap li {
    color: #e8edf7 !important; padding: 6px 0 6px 24px !important;
    position: relative; font-size: 0.92rem !important;
  }
  .roadmap li::before {
    content: "→"; position: absolute; left: 0; color: #f7c948; font-weight: 700;
  }
  .roadmap li strong { color: #f7c948 !important; }
  .roadmap .cta {
    margin-top: 14px !important; padding: 10px 14px !important;
    background: rgba(247, 201, 72, 0.08) !important;
    border-left: 3px solid #f7c948 !important;
    color: #f5f7fb !important; font-style: italic; border-radius: 4px !important;
  }

  /* Hero card */
  div.hero {
    background: linear-gradient(135deg, #0f1b34 0%, #1a2752 100%) !important;
    padding: 28px 32px !important;
    border-radius: 16px !important;
    border: 1px solid #23315c !important;
    margin-bottom: 18px !important;
  }
  div.hero h1, div.hero h1 span {
    font-size: 2.2rem !important;
    margin: 0 0 8px 0 !important;
    color: #f7c948 !important;
    padding: 0 !important;
  }
  div.hero p.subtitle {
    color: #bcc7e0 !important;
    margin: 0 !important;
    font-size: 1.0rem !important;
    padding: 0 !important;
  }
  div.hero p.subtitle b { color: #f7c948 !important; font-weight: 700 !important; }
  /* Kill Streamlit heading anchor link icon inside hero */
  div.hero a[aria-label="Link to heading"] { display: none !important; }

  /* KPI cards */
  div.kpi-grid {
    display: grid !important;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)) !important;
    gap: 14px !important;
    margin: 10px 0 18px 0 !important;
  }
  div.kpi {
    background: #121c2e !important;
    border: 1px solid #23315c !important;
    border-radius: 12px !important;
    padding: 16px 18px !important;
  }
  div.kpi .label { color:#94a3c5 !important; font-size:0.78rem !important; text-transform:uppercase !important; letter-spacing:0.06em !important; }
  div.kpi .value { color:#f5f7fb !important; font-size:1.4rem !important; font-weight:700 !important; margin-top:6px !important; }
  div.kpi .delta { font-size:0.85rem !important; margin-top:4px !important; }
  div.kpi .delta.up { color:#22c55e !important; }
  div.kpi .delta.down { color:#ef4444 !important; }
  div.kpi .delta.muted { color:#94a3c5 !important; }

  /* Section heading */
  div.section-title {
    font-size: 1.35rem !important;
    color: #f7c948 !important;
    margin: 14px 0 8px 0 !important;
    letter-spacing: 0.02em !important;
    font-weight: 700 !important;
  }
  p.section-caption {
    color: #94a3c5 !important;
    margin: 0 0 14px 0 !important;
  }

  /* Mispricing row cards */
  div.mis-row {
    display: grid !important;
    grid-template-columns: 60px 1.2fr 1fr 0.7fr 0.7fr 2.5fr !important;
    gap: 10px !important;
    padding: 10px 12px !important;
    border-bottom: 1px solid #1b2742 !important;
    align-items: center !important;
  }
  div.mis-row.header { color: #94a3c5 !important; font-size: 0.78rem !important; text-transform: uppercase !important; letter-spacing: 0.06em !important; border-bottom-color: #23315c !important; }
  div.mis-row .rank { color: #94a3c5 !important; font-weight: 600 !important; }
  div.mis-row .team { font-weight: 600 !important; font-size: 1.02rem !important; }
  div.mis-row .dir-under { color: #22c55e !important; font-weight: 700 !important; }
  div.mis-row .dir-over { color: #ef4444 !important; font-weight: 700 !important; }
  div.mis-row .reason { color: #cbd5e8 !important; font-size: 0.88rem !important; }

  /* reduce sidebar padding */
  section[data-testid="stSidebar"] { background:#0b1220 !important; border-right:1px solid #1b2742 !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------- cached loaders ----------
@st.cache_data
def load_probs() -> pd.DataFrame:
    return pd.read_parquet(_p("sim_team_probs.parquet"))


@st.cache_data
def load_edges() -> pd.DataFrame:
    return pd.read_parquet(_p("edge_table.parquet"))


@st.cache_data
def load_leaderboard() -> pd.DataFrame:
    return pd.read_parquet(_p("mispricing_leaderboard.parquet"))


@st.cache_data
def load_backtest_summary() -> pd.DataFrame:
    return pd.read_parquet(_p("backtest_summary.parquet"))


@st.cache_data
def load_backtest_reliability() -> pd.DataFrame:
    return pd.read_parquet(_p("backtest_reliability.parquet"))


@st.cache_data
def load_backtest_predictions() -> pd.DataFrame:
    return pd.read_parquet(_p("backtest_predictions.parquet"))


@st.cache_data
def load_squads() -> pd.DataFrame:
    p = _p("squads.parquet")
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_player_photos() -> dict:
    """Return {player_name: photo_url} map (skips rows without photo)."""
    p = _p("player_photos.parquet")
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    df = df[df["photo_url"].notna()]
    return dict(zip(df["player"], df["photo_url"]))


@st.cache_data
def load_squad_metrics() -> pd.DataFrame:
    p = _p("team_squad_metrics.parquet")
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_recent_matches() -> pd.DataFrame:
    p = _p("team_recent_matches.parquet")
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_groups() -> pd.DataFrame:
    p = _p("wc2026_groups.parquet")
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_team_media() -> dict:
    """Return {team_name: {badge, jersey, logo, banner, desc_en}} map populated
    by scripts/fetch_team_media.py (TheSportsDB). Empty dict if not present."""
    p = _p("team_media.json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_data
def logo_data_url() -> str:
    """Return static/logo.png as a base64 data URL so we can embed it inline
    in markdown/HTML without depending on Streamlit's static-file URL (which
    differs between local dev and Streamlit Cloud hosting)."""
    import base64
    p = HERE / "static" / "logo.png"
    if not p.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def find_team_product(team: str, prefer_keyword: str = "jersey") -> dict | None:
    """Thin re-export of ask_model._pick_team_product so the same ranking
    logic is used by merch_link() (this module) and the Claude pricing tool
    (ask_model). Single source of truth for the Fanatics feed lookup avoids
    the cross-module import race we hit earlier."""
    from ask_model import _pick_team_product
    return _pick_team_product(team, prefer_keyword=prefer_keyword)


@st.cache_data
def load_fixtures() -> pd.DataFrame:
    p = _p("wc2026_fixtures.parquet")
    if not p.exists():
        return pd.DataFrame()
    f = pd.read_parquet(p)
    f["date"] = pd.to_datetime(f["date"])
    return f


@st.cache_data
def load_schedule() -> pd.DataFrame:
    p = _p("wc2026_schedule.parquet")
    if not p.exists():
        return pd.DataFrame()
    s = pd.read_parquet(p)
    s["date"] = pd.to_datetime(s["date"])
    return s


def _team_group(team: str, groups_df: pd.DataFrame) -> str | None:
    """Return 'A' / 'B' / ... for a team, or None if unknown."""
    if groups_df.empty:
        return None
    hit = groups_df[groups_df["team"] == team]
    return hit.iloc[0]["group"] if len(hit) else None


def _last_n_for_team(matches: pd.DataFrame, team: str, n: int = 10):
    """Return list of result chars ('W'/'D'/'L') for team's last n internationals."""
    if matches.empty:
        return {"results": [], "W": 0, "D": 0, "L": 0, "avg_gd": 0.0, "n": 0}
    m = matches[(matches["home_team"] == team) | (matches["away_team"] == team)]
    m = m.sort_values("date").tail(n)
    results, gds = [], []
    for _, r in m.iterrows():
        if r["home_team"] == team:
            gd = int(r["home_goals"]) - int(r["away_goals"])
        else:
            gd = int(r["away_goals"]) - int(r["home_goals"])
        results.append("W" if gd > 0 else "D" if gd == 0 else "L")
        gds.append(gd)
    return {
        "results": results,
        "W": results.count("W"),
        "D": results.count("D"),
        "L": results.count("L"),
        "avg_gd": float(sum(gds) / len(gds)) if gds else 0.0,
        "n": len(results),
    }


@st.cache_data
def load_dc_params():
    p = pd.read_parquet(_p("dc_params.parquet"))
    s = pd.read_parquet(_p("dc_scalars.parquet")).iloc[0]
    teams = p["team"].tolist()
    return {
        "teams": teams,
        "team_index": {t: i for i, t in enumerate(teams)},
        "attack": p["attack"].to_numpy(),
        "defense": p["defense"].to_numpy(),
        "home_adv": float(s["home_adv"]),
        "rho": float(s["rho"]),
    }


@st.cache_data
def load_fixtures() -> pd.DataFrame:
    return pd.read_parquet(_p("wc2026_fixtures.parquet"))


# ---------- What-If simulator primitives ----------
MAX_GOALS = 7
N_STATES = (MAX_GOALS + 1) ** 2


def _poisson_pmf(lam: float, n: int) -> np.ndarray:
    p = np.empty(n + 1)
    p[0] = np.exp(-lam)
    for k in range(n):
        p[k + 1] = p[k] * lam / (k + 1)
    return p


def _score_cdf(lam_h: float, lam_a: float, rho: float) -> np.ndarray:
    ph = _poisson_pmf(lam_h, MAX_GOALS)
    pa = _poisson_pmf(lam_a, MAX_GOALS)
    mat = np.outer(ph, pa)
    mat[0, 0] *= max(1 - lam_h * lam_a * rho, 1e-12)
    mat[0, 1] *= max(1 + lam_h * rho, 1e-12)
    mat[1, 0] *= max(1 + lam_a * rho, 1e-12)
    mat[1, 1] *= max(1 - rho, 1e-12)
    mat = mat / mat.sum()
    return np.cumsum(mat.ravel())


def _lambdas(dc, home, away, neutral):
    i, j = dc["team_index"][home], dc["team_index"][away]
    gamma = 0.0 if neutral else dc["home_adv"]
    lam_h = float(np.exp(dc["attack"][i] + dc["defense"][j] + gamma))
    lam_a = float(np.exp(dc["attack"][j] + dc["defense"][i]))
    return lam_h, lam_a


def _sample_from_cdf(cdf, u):
    idx = int(np.searchsorted(cdf, u, side="right"))
    if idx >= N_STATES:
        idx = N_STATES - 1
    return idx // (MAX_GOALS + 1), idx % (MAX_GOALS + 1)


def _knockout_winner(dc, home, away, rng):
    lam_h, lam_a = _lambdas(dc, home, away, True)
    hg, ag = _sample_from_cdf(_score_cdf(lam_h, lam_a, dc["rho"]), rng.random())
    if hg != ag:
        return home if hg > ag else away
    hg2, ag2 = _sample_from_cdf(_score_cdf(lam_h / 3.0, lam_a / 3.0, dc["rho"]), rng.random())
    if hg2 != ag2:
        return home if hg2 > ag2 else away
    return home if rng.random() < 0.5 else away


def infer_groups(fixtures: pd.DataFrame) -> dict:
    """Official FIFA 2026 group draw (A-L) with team memberships.

    Reads wc2026_groups.parquet (canonical source), falling back to
    fixture-derived connected components only if that file is missing."""
    groups_df = load_groups()
    if not groups_df.empty:
        return {
            f"Group {letter}": sorted(g["team"].tolist())
            for letter, g in groups_df.groupby("group")
        }

    # Fallback (should not run on deployed app — parquet ships with repo)
    from collections import defaultdict
    adj = defaultdict(set)
    for _, r in fixtures.iterrows():
        adj[r["home_team"]].add(r["away_team"])
        adj[r["away_team"]].add(r["home_team"])
    groups, seen = [], set()
    for team in adj:
        if team in seen:
            continue
        stack, comp = [team], set()
        while stack:
            t = stack.pop()
            if t in comp:
                continue
            comp.add(t)
            for nb in adj[t]:
                if nb not in comp:
                    stack.append(nb)
        seen |= comp
        groups.append(sorted(comp))
    groups.sort()
    return {f"G{chr(ord('A') + i)}": g for i, g in enumerate(groups)}


@st.cache_data
def prepare_group_cdfs(_fixtures: pd.DataFrame, _dc_teams: tuple, _dc_attack_hash: float):
    """Precompute CDF per group fixture (indep of which sim)."""
    dc = load_dc_params()
    teams_in_groups = {t for t in _dc_teams}
    entries = []
    for _, r in _fixtures.iterrows():
        h, a = r["home_team"], r["away_team"]
        if h not in dc["team_index"] or a not in dc["team_index"]:
            continue
        lam_h, lam_a = _lambdas(dc, h, a, bool(r["neutral"]))
        entries.append((h, a, _score_cdf(lam_h, lam_a, dc["rho"])))
    return entries


def _group_from_team(groups: dict, team: str) -> str | None:
    for g, ts in groups.items():
        if team in ts:
            return g
    return None


def run_whatif(groups: dict, fixtures: pd.DataFrame, locks: dict,
               n_sims: int, seed: int = 42) -> pd.DataFrame:
    """locks: {group_key: {'1st': team, '2nd': team}} - force these positions."""
    dc = load_dc_params()
    rng = np.random.default_rng(seed)

    cdfs = prepare_group_cdfs(fixtures, tuple(dc["teams"]), float(dc["attack"].sum()))
    # Build list of (home, away, cdf, group_key) in fixture order
    fx_rows = []
    for h, a, cdf in cdfs:
        gkey = _group_from_team(groups, h)
        fx_rows.append((h, a, cdf, gkey))

    STAGES = ["group", "R32", "R16", "QF", "SF", "F", "W"]
    stage_rank = {s: i for i, s in enumerate(STAGES)}
    team_to_group = {t: g for g, ts in groups.items() for t in ts}
    all_teams = sorted(team_to_group)
    counts = {t: [0] * len(STAGES) for t in all_teams}
    third_pos_ct = {t: 0 for t in all_teams}   # times team finished 3rd in group
    third_adv_ct = {t: 0 for t in all_teams}   # times team advanced as one of the top 8 thirds
    third_pts_sum = {t: 0 for t in all_teams}  # sum pts when 3rd (for averaging)
    third_gd_sum = {t: 0 for t in all_teams}   # sum goal diff when 3rd

    for _sim in range(n_sims):
        tables = {g: {t: {"pts": 0, "gf": 0, "ga": 0} for t in ts} for g, ts in groups.items()}
        for h, a, cdf, gkey in fx_rows:
            hg, ag = _sample_from_cdf(cdf, rng.random())
            tables[gkey][h]["gf"] += hg; tables[gkey][h]["ga"] += ag
            tables[gkey][a]["gf"] += ag; tables[gkey][a]["ga"] += hg
            if hg > ag:
                tables[gkey][h]["pts"] += 3
            elif hg < ag:
                tables[gkey][a]["pts"] += 3
            else:
                tables[gkey][h]["pts"] += 1
                tables[gkey][a]["pts"] += 1

        first_place, second_place, third_place = [], [], []
        for gkey, tbl in tables.items():
            ordered = sorted(
                tbl.items(),
                key=lambda kv: (kv[1]["pts"], kv[1]["gf"] - kv[1]["ga"], kv[1]["gf"], rng.random()),
                reverse=True,
            )
            # Apply locks for this group
            lk = locks.get(gkey, {})
            ordered_teams = [t for t, _ in ordered]
            final = [None, None]
            remaining = list(ordered_teams)
            # Pull locked teams out of remaining first (so they don't double-slot)
            for slot in ("1st", "2nd", "3rd"):
                if slot in lk and lk[slot] in remaining:
                    remaining.remove(lk[slot])
            if "1st" in lk:
                final[0] = lk["1st"]
            if "2nd" in lk:
                final[1] = lk["2nd"]
            # Fill unlocked 1st/2nd slots with natural top of remaining
            for pos in range(2):
                if final[pos] is None:
                    final[pos] = remaining.pop(0)
            # 3rd: locked team wins, else natural top of remaining
            if "3rd" in lk:
                third = lk["3rd"]
            else:
                third = remaining[0] if remaining else None
            first_place.append((gkey, final[0], tables[gkey][final[0]]))
            second_place.append((gkey, final[1], tables[gkey][final[1]]))
            if third:
                third_place.append((gkey, third, tables[gkey][third]))
                t_stats = tables[gkey][third]
                third_pos_ct[third] += 1
                third_pts_sum[third] += t_stats["pts"]
                third_gd_sum[third] += t_stats["gf"] - t_stats["ga"]

        third_sorted = sorted(
            third_place,
            key=lambda t: (t[2]["pts"], t[2]["gf"] - t[2]["ga"], t[2]["gf"], rng.random()),
            reverse=True,
        )
        # Any team the user locked as 3rd gets guaranteed advancement;
        # remaining slots filled by natural top-ranked 3rds.
        locked_thirds = [lk["3rd"] for lk in locks.values() if "3rd" in lk]
        natural_order = [t for (_, t, _) in third_sorted]
        third_adv = list(locked_thirds)
        for t in natural_order:
            if len(third_adv) >= 8:
                break
            if t not in third_adv:
                third_adv.append(t)
        for t in third_adv:
            third_adv_ct[t] += 1

        r32 = [t for (_, t, _) in first_place] + [t for (_, t, _) in second_place] + third_adv
        reached = {t: "group" for t in all_teams}
        for t in r32:
            reached[t] = "R32"

        def advance(teams, stage_out):
            winners = []
            n = len(teams)
            for k in range(n // 2):
                w = _knockout_winner(dc, teams[k], teams[n - 1 - k], rng)
                winners.append(w)
                reached[w] = stage_out
            return winners

        r16 = advance(r32, "R16")
        qf = advance(r16, "QF")
        sf = advance(qf, "SF")
        final = advance(sf, "F")
        advance(final, "W")

        for team, stage in reached.items():
            r = stage_rank[stage]
            for s_idx in range(r + 1):
                counts[team][s_idx] += 1

    rows = []
    for team in all_teams:
        row = {"team": team, "group": team_to_group[team]}
        for k, s in enumerate(STAGES):
            row[f"p_{s}"] = counts[team][k] / n_sims
        pos_ct = third_pos_ct[team]
        row["p_third_pos"] = pos_ct / n_sims
        row["p_third_adv"] = third_adv_ct[team] / n_sims
        row["third_avg_pts"] = (third_pts_sum[team] / pos_ct) if pos_ct else float("nan")
        row["third_avg_gd"] = (third_gd_sum[team] / pos_ct) if pos_ct else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("p_W", ascending=False).reset_index(drop=True)


def render_bracket(res: pd.DataFrame) -> str:
    """Return HTML for a 6-column bracket showing top teams reaching each round,
    with fill-bar proportional to P(reach stage)."""
    rounds = [
        ("R32", "p_R32", 16),
        ("R16", "p_R16", 12),
        ("QF",  "p_QF",  8),
        ("SF",  "p_SF",  6),
        ("F",   "p_F",   4),
        ("W",   "p_W",   3),
    ]
    html = ['<div class="bracket">']
    for label, col, top_n in rounds:
        html.append('<div class="bracket-col">')
        html.append(f"<h4>{label}</h4>")
        top = res.nlargest(top_n, col)
        # Max of this column (for relative fill)
        max_p = max(top[col].max(), 1e-6)
        for rank_i, (_, r) in enumerate(top.iterrows()):
            p = float(r[col])
            if p < 0.001:
                continue
            pct_fill = min(100, 100 * p / max_p)
            cls = "bracket-cell top" if rank_i < 2 else "bracket-cell"
            tw = team_with_flag_img(r["team"], h=14)
            html.append(
                f'<div class="{cls}">'
                f'<div class="fill" style="width:{pct_fill:.0f}%"></div>'
                f'<span class="label">{tw}</span>'
                f'<span class="pct">{p*100:.0f}%</span>'
                f"</div>"
            )
        html.append("</div>")
    html.append("</div>")
    return "\n".join(html)


# ---------- sidebar ----------
set_language_from_sidebar()
_logo = logo_data_url()
if _logo:
    st.sidebar.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin:4px 0 8px;">'
        f'<img src="{_logo}" alt="WorldCup26AI" '
        f'style="width:44px;height:44px;object-fit:contain;flex-shrink:0;">'
        f'<span style="font-size:1.25rem;font-weight:700;color:#f7c948;'
        f'letter-spacing:-0.01em;">{t("app_title")}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(f"# ⚽ {t('app_title')}")
st.sidebar.caption(t("app_tagline"))
PAGE_KEYS = [
    ("hero",     t("nav_hero")),
    ("champ",    t("nav_champ")),
    ("misp",     t("nav_misp")),
    ("groups",   t("nav_groups")),
    ("schedule", t("nav_schedule")),
    ("whatif",   t("nav_whatif")),
    ("ask",      t("nav_ask")),
    ("explorer", t("nav_explorer")),
    ("stage",    t("nav_stage")),
    ("calib",    t("nav_calib")),
    ("method",   t("nav_method")),
]
page_labels = [lbl for _, lbl in PAGE_KEYS]
page_label = st.sidebar.radio(t("navigate"), page_labels, index=0)
page_id = dict(zip(page_labels, [k for k, _ in PAGE_KEYS]))[page_label]
st.sidebar.markdown("---")
st.sidebar.caption(t("sidebar_foot"))


# ---------- HERO / landing ----------
if page_id == "hero":
    lb = load_leaderboard()
    probs = load_probs()
    summary = load_backtest_summary()
    all_dc = summary[(summary["year"] == "ALL") & (summary["model"] == "DixonColes")].iloc[0]
    all_base = summary[(summary["year"] == "ALL") & (summary["model"] == "Uniform(1/3)")].iloc[0]
    skill = 1 - all_dc["brier"] / all_base["brier"]

    best_under = lb[lb["direction"] == "UNDER"].iloc[0]
    best_over = lb[lb["direction"] == "OVER"].iloc[0]

    headline_text = t("hero_wrong",
                       over_pp=f"{best_over['edge']*100:+.0f}",
                       over_flag=flag_img(best_over["team"], h=18),
                       over_team=team_name(best_over["team"]),
                       under_pp=f"{best_under['edge']*100:+.0f}",
                       under_flag=flag_img(best_under["team"], h=18),
                       under_team=team_name(best_under["team"]))
    _logo_url = logo_data_url()
    _logo_tag = (
        f'<img src="{_logo_url}" alt="WorldCup26AI" '
        f'style="height:56px;width:56px;vertical-align:middle;margin-right:14px;">'
        if _logo_url else '⚽ '
    )
    st.markdown(
        f"""
        <div class="hero">
          <h1>{_logo_tag}{t('app_title')}</h1>
          <p class="subtitle">{headline_text}</p>
          <p class="subtitle" style="margin-top:8px;">{t('hero_intro')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="kpi-grid">
          <div class="kpi">
            <div class="label">{t('kpi_biggest_under')}</div>
            <div class="value">{team_with_flag_img(best_under['team'], h=18)}</div>
            <div class="delta up">{best_under['edge']*100:+.1f} pp · {t('kpi_mkt_model', mkt=f"{best_under['market_p_W']*100:.1f}", model=f"{best_under['p_W']*100:.1f}")}</div>
          </div>
          <div class="kpi">
            <div class="label">{t('kpi_biggest_over')}</div>
            <div class="value">{team_with_flag_img(best_over['team'], h=18)}</div>
            <div class="delta down">{best_over['edge']*100:+.1f} pp · {t('kpi_mkt_model', mkt=f"{best_over['market_p_W']*100:.1f}", model=f"{best_over['p_W']*100:.1f}")}</div>
          </div>
          <div class="kpi">
            <div class="label">{t('kpi_brier')}</div>
            <div class="value">{skill*100:+.1f}%</div>
            <div class="delta muted">{t('kpi_brier_sub')}</div>
          </div>
          <div class="kpi">
            <div class="label">{t('kpi_coverage')}</div>
            <div class="value">{int(lb['team'].nunique())} / 48</div>
            <div class="delta muted">{t('kpi_coverage_sub')}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top 5 model champions
    top = probs.sort_values("p_W", ascending=False).head(5).reset_index(drop=True)
    st.markdown(f'<div class="section-title">{t("section_top5_fav")}</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    for i, row in top.iterrows():
        with cols[i]:
            st.markdown(
                f"""
                <div class="kpi">
                  <div class="label">#{i+1}</div>
                  <div class="value">{team_with_flag_img(row['team'], h=18)}</div>
                  <div class="delta muted">{row['p_W']*100:.1f}% {t('champion_suffix')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Mispricing top 5
    st.markdown(f'<div class="section-title">{t("section_top5_mis")}</div>', unsafe_allow_html=True)
    top_lb = lb.head(5)
    rows_html = (
        f'<div class="mis-row header"><div>#</div>'
        f'<div>{t("col_team")}</div><div>{t("col_direction")}</div>'
        f'<div>{t("col_model")}</div><div>{t("col_market")}</div>'
        f'<div>{t("col_reason")}</div></div>'
    )
    for _, r in top_lb.iterrows():
        dir_class = "dir-under" if r["direction"] == "UNDER" else "dir-over"
        dir_label = t("dir_under") if r["direction"] == "UNDER" else t("dir_over")
        edge_pp = f"{r['edge']*100:+.1f}pp"
        rows_html += (
            f'<div class="mis-row">'
            f'<div class="rank">#{int(r["rank"])}</div>'
            f'<div class="team">{team_with_flag_img(r["team"], h=16)}</div>'
            f'<div class="{dir_class}">{dir_label} {edge_pp}</div>'
            f'<div>{r["p_W"]*100:.1f}%</div>'
            f'<div>{r["market_p_W"]*100:.1f}%</div>'
            f'<div class="reason">{translate_reason(r["reason"])}</div>'
            f"</div>"
        )
    st.markdown(rows_html, unsafe_allow_html=True)

    # Small reliability + quick-explain row
    rel = load_backtest_reliability()
    c_left, c_right = st.columns([1.2, 1.8])
    with c_left:
        st.markdown(
            f'<div class="section-title" style="font-size:1.1rem;">{t("section_credibility")}</div>',
            unsafe_allow_html=True,
        )
        st.caption(t("credibility_text", skill=skill * 100))
        mini = go.Figure()
        mini.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                       line=dict(dash="dash", color="#94a3c5", width=1))
        mini.add_trace(go.Scatter(
            x=rel["avg_predicted"], y=rel["empirical"],
            mode="markers",
            marker=dict(size=np.sqrt(rel["count"]) * 4 + 5, color="#f7c948",
                        line=dict(color="#0b1220", width=1)),
            showlegend=False,
        ))
        mini.update_layout(
            height=260, margin=dict(l=30, r=20, t=10, b=30),
            paper_bgcolor="#0b1220", plot_bgcolor="#121c2e",
            font=dict(color="#e8edf7", size=10),
            xaxis=dict(title="Predicted", range=[0, 1], gridcolor="#1b2742", tickformat=".0%"),
            yaxis=dict(title="Actual", range=[0, 1], gridcolor="#1b2742", tickformat=".0%"),
        )
        st.plotly_chart(mini, use_container_width=True)
    with c_right:
        st.markdown(
            f'<div class="section-title" style="font-size:1.1rem;">{t("section_different")}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"- {t('diff_1')}\n\n- {t('diff_2')}\n\n- {t('diff_3')}\n\n- {t('diff_4')}"
        )
        st.caption(t("explore_sidebar"))

    # --- Coming May 31 roadmap card ---
    import re as _re
    def _inline_md(s: str) -> str:
        s = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = _re.sub(r"«(.+?)»", r'<em>«\1»</em>', s)
        return s
    st.markdown(
        f"""
        <div class="roadmap">
          <h2>{t('roadmap_title')}</h2>
          <p class="date">{_inline_md(t('roadmap_date'))}</p>
          <ul>
            <li>{_inline_md(t('roadmap_b1'))}</li>
            <li>{_inline_md(t('roadmap_b2'))}</li>
            <li>{_inline_md(t('roadmap_b3'))}</li>
            <li>{_inline_md(t('roadmap_b4'))}</li>
          </ul>
          <div class="cta">{t('roadmap_cta')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------- Champion ----------
elif page_id == "champ":
    probs = load_probs()
    st.markdown(f'<div class="section-title">{t("champ_title")}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="section-caption">{t("champ_caption", n=len(probs))}</p>',
        unsafe_allow_html=True,
    )

    top = probs.sort_values("p_W", ascending=False).head(20).copy()
    top["label"] = top["team"].apply(team_with_flag)

    fig = px.bar(
        top.sort_values("p_W"),
        x="p_W", y="label", orientation="h",
        labels={"p_W": "P(Win)", "label": ""},
        title=t("champ_chart"),
    )
    fig.update_traces(marker_color="#f7c948")
    fig.update_layout(
        height=620,
        margin=dict(l=140, r=40, t=60, b=40),
        paper_bgcolor="#0b1220",
        plot_bgcolor="#121c2e",
        font=dict(color="#e8edf7"),
        xaxis=dict(gridcolor="#1b2742"),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander(t("full_table")):
        show = probs.copy()
        show["flag"] = show["team"].apply(lambda tm: flag_url(tm, "w80") or "")
        show["team_disp"] = show["team"].apply(team_name)
        st.dataframe(
            show[["flag", "team_disp", "p_R32", "p_R16", "p_QF", "p_SF", "p_F", "p_W"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "flag": st.column_config.ImageColumn("Flag", width="small"),
                "team_disp": st.column_config.TextColumn(t("whatif_col_team")),
                **{c: st.column_config.NumberColumn(c, format="percent")
                   for c in ["p_R32", "p_R16", "p_QF", "p_SF", "p_F", "p_W"]},
            },
        )


# ---------- Mispricing Leaderboard ----------
elif page_id == "misp":
    lb = load_leaderboard()
    edges = load_edges()
    st.markdown(f'<div class="section-title">{t("misp_title")}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="section-caption">{t("misp_caption", n=len(lb))}</p>',
        unsafe_allow_html=True,
    )

    under = lb[lb["direction"] == "UNDER"]
    over = lb[lb["direction"] == "OVER"]
    best_under = under.iloc[0] if len(under) else None
    best_over = over.iloc[0] if len(over) else None

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(t("markets_analysed"), len(lb))
    if best_under is not None:
        k2.metric(f"{t('kpi_biggest_under')}: {team_with_flag(best_under['team'])}",
                  f"{best_under['edge']*100:+.1f} pp",
                  t("kpi_mkt_model", mkt=f"{best_under['market_p_W']*100:.1f}", model=f"{best_under['p_W']*100:.1f}"))
    if best_over is not None:
        k3.metric(f"{t('kpi_biggest_over')}: {team_with_flag(best_over['team'])}",
                  f"{best_over['edge']*100:+.1f} pp",
                  t("kpi_mkt_model", mkt=f"{best_over['market_p_W']*100:.1f}", model=f"{best_over['p_W']*100:.1f}"))
    k4.metric(t("total_liquidity"), f"${lb['liquidity'].sum()/1e6:.1f}M")

    st.markdown(f'<div class="section-title" style="font-size:1.1rem;">{t("misp_rank_head")}</div>', unsafe_allow_html=True)

    # Custom rows
    rows_html = (
        f'<div class="mis-row header"><div>{t("col_rank")}</div><div>{t("col_team")}</div>'
        f'<div>{t("col_direction")}</div><div>{t("col_model")}</div>'
        f'<div>{t("col_market")}</div><div>{t("col_reason")}</div></div>'
    )
    for _, r in lb.iterrows():
        dir_class = "dir-under" if r["direction"] == "UNDER" else "dir-over"
        dir_word = t("dir_under") if r["direction"] == "UNDER" else t("dir_over")
        dir_label = f"{dir_word} {r['edge']*100:+.1f}pp"
        rows_html += (
            f'<div class="mis-row">'
            f'<div class="rank">#{int(r["rank"])}</div>'
            f'<div class="team">{team_with_flag_img(r["team"], h=16)}</div>'
            f'<div class="{dir_class}">{dir_label}</div>'
            f'<div>{r["p_W"]*100:.1f}%</div>'
            f'<div>{r["market_p_W"]*100:.1f}%</div>'
            f'<div class="reason">{translate_reason(r["reason"])}</div>'
            f"</div>"
        )
    st.markdown(rows_html, unsafe_allow_html=True)

    st.markdown(f'<div class="section-title" style="margin-top:24px;">{t("scatter_title")}</div>', unsafe_allow_html=True)
    scat = edges[edges["market_covered"]].copy()
    fig = px.scatter(
        scat, x="market_p_W", y="p_W", text="team", size="liquidity",
        color="edge", color_continuous_scale="RdYlGn",
        labels={"market_p_W": "Polymarket implied P(Win)", "p_W": "Model P(Win)"},
        hover_data={"edge": ":.1%", "liquidity": ":,.0f"},
    )
    mx = max(scat["market_p_W"].max(), scat["p_W"].max())
    fig.add_shape(type="line", x0=0, y0=0, x1=mx, y1=mx, line=dict(dash="dash", color="#94a3c5"))
    fig.update_traces(textposition="top center")
    fig.update_layout(
        height=620,
        paper_bgcolor="#0b1220",
        plot_bgcolor="#121c2e",
        font=dict(color="#e8edf7"),
        xaxis=dict(gridcolor="#1b2742"),
        yaxis=dict(gridcolor="#1b2742"),
        coloraxis_colorbar=dict(title="Edge"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------- What-If ----------
elif page_id == "whatif":
    st.markdown(f'<div class="section-title">{t("whatif_title")}</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="section-caption">{t("whatif_caption")}</p>', unsafe_allow_html=True)

    fx = load_fixtures()
    groups = infer_groups(fx)
    dc = load_dc_params()
    baseline_probs = load_probs().set_index("team")["p_W"].to_dict()

    # Auto-init session state
    if "wc26_locks" not in st.session_state:
        st.session_state["wc26_locks"] = {}

    st.markdown(f'<div class="section-title" style="font-size:1.1rem;">{t("whatif_lock")}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="section-caption" style="font-size:0.85rem;">{t("whatif_pick_caption")}</p>',
        unsafe_allow_html=True,
    )

    # Cycle: unset → 🏆 (1st) → 🥈 (2nd) → 🥉 (3rd) → unset. Selecting a slot
    # displaces any other team already in that slot for the same group.
    # 3rd-placed teams advance only if among top 8 across all 12 groups.
    def _cycle_lock(gkey: str, team: str):
        locks = st.session_state["wc26_locks"].setdefault(gkey, {})
        if locks.get("1st") == team:
            locks.pop("1st", None)
            locks["2nd"] = team
        elif locks.get("2nd") == team:
            locks.pop("2nd", None)
            locks["3rd"] = team
        elif locks.get("3rd") == team:
            locks.pop("3rd", None)
        else:
            locks["1st"] = team
        if not locks:
            st.session_state["wc26_locks"].pop(gkey, None)

    group_keys = list(groups.keys())
    for i in range(0, len(group_keys), 4):
        cols = st.columns(4)
        for j, gkey in enumerate(group_keys[i:i + 4]):
            with cols[j]:
                st.markdown(f'<div class="gpick-title">{gkey}</div>',
                            unsafe_allow_html=True)
                locks = st.session_state["wc26_locks"].get(gkey, {})
                for tm in groups[gkey]:
                    if locks.get("1st") == tm:
                        badge, btn_type = "🏆", "primary"
                    elif locks.get("2nd") == tm:
                        badge, btn_type = "🥈", "primary"
                    elif locks.get("3rd") == tm:
                        badge, btn_type = "🥉", "primary"
                    else:
                        badge, btn_type = "　", "secondary"
                    label = f"{badge} {flag(tm)} {team_name(tm)}"
                    st.button(
                        label,
                        key=f"gpick_{gkey}_{tm}",
                        use_container_width=True,
                        type=btn_type,
                        on_click=_cycle_lock,
                        args=(gkey, tm),
                    )

    new_locks = {g: dict(v) for g, v in st.session_state["wc26_locks"].items() if v}

    st.markdown("---")
    c1, c2, c3 = st.columns([1, 1, 3])
    n_sims = c1.selectbox(t("whatif_sims"), [500, 1000, 2000, 5000], index=1)
    run = c2.button(t("whatif_run"), type="primary", use_container_width=True)
    if c3.button(t("whatif_reset")):
        st.session_state["wc26_locks"] = {}
        st.rerun()

    if run:
        with st.spinner(t("whatif_spinner", n=n_sims)):
            res = run_whatif(groups, fx, new_locks, n_sims=int(n_sims))

        # Compare to baseline
        res["baseline_p_W"] = res["team"].map(baseline_probs).fillna(0.0)
        res["delta"] = res["p_W"] - res["baseline_p_W"]

        n_locked = sum(len(v) for v in new_locks.values())
        st.markdown(
            f'<div class="section-title" style="margin-top:12px;">{t("whatif_results", n_locked=n_locked, n_sims=n_sims)}</div>',
            unsafe_allow_html=True,
        )

        # Bracket view — 6-column tree, top teams per round
        st.markdown(
            f'<div class="section-title" style="font-size:1.1rem;margin-top:8px;">{t("whatif_bracket")}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p class="section-caption" style="font-size:0.85rem;">{t("whatif_bracket_caption")}</p>',
            unsafe_allow_html=True,
        )
        st.markdown(render_bracket(res), unsafe_allow_html=True)

        # 3rd-place advancers — demystify the 8-from-12 rule
        st.markdown(
            f'<div class="section-title" style="font-size:1.1rem;margin-top:16px;">{t("whatif_thirds")}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p class="section-caption" style="font-size:0.85rem;">{t("whatif_thirds_caption")}</p>',
            unsafe_allow_html=True,
        )
        locked_thirds_set = {lk["3rd"] for lk in new_locks.values() if "3rd" in lk}
        thirds = res[res["p_third_pos"] > 0].copy()
        thirds = thirds.sort_values("p_third_adv", ascending=False).head(12)
        thirds["flag"] = thirds["team"].apply(lambda tm: flag_url(tm, "w80") or "")
        thirds["team_disp"] = thirds["team"].apply(
            lambda tm: f"🔒 {team_name(tm)}" if tm in locked_thirds_set else team_name(tm)
        )
        col_flag, col_team, col_group, col_p3 = "Flag", t("whatif_col_team"), t("whatif_col_group"), t("whatif_col_p_third")
        col_adv, col_pts, col_gd = t("whatif_col_p_adv"), t("whatif_col_avg_pts"), t("whatif_col_avg_gd")
        thirds_view = thirds[[
            "flag", "team_disp", "group", "p_third_pos", "p_third_adv",
            "third_avg_pts", "third_avg_gd",
        ]].rename(columns={
            "flag": col_flag,
            "team_disp": col_team,
            "group": col_group,
            "p_third_pos": col_p3,
            "p_third_adv": col_adv,
            "third_avg_pts": col_pts,
            "third_avg_gd": col_gd,
        })
        st.dataframe(
            thirds_view,
            use_container_width=True,
            hide_index=True,
            column_config={
                col_flag: st.column_config.ImageColumn(col_flag, width="small"),
                col_p3: st.column_config.NumberColumn(col_p3, format="percent"),
                col_adv: st.column_config.ProgressColumn(
                    col_adv, format="percent", min_value=0, max_value=1,
                ),
                col_pts: st.column_config.NumberColumn(col_pts, format="%.2f"),
                col_gd: st.column_config.NumberColumn(col_gd, format="%+.2f"),
            },
        )

        # Dumbbell chart: baseline (grey) → conditional (up=green / down=red).
        # Bigger conditional dot emphasises the "new reality" under your locks.
        top = res.head(15).copy()
        top["label"] = top["team"].apply(team_with_flag)
        labels = top["label"].tolist()
        baseline = top["baseline_p_W"].tolist()
        conditional = top["p_W"].tolist()
        deltas = top["delta"].tolist()

        # One connector line per team (None-separated so a single trace draws
        # segmented lines — cheaper than 15 traces).
        line_x, line_y = [], []
        for lbl, b, c in zip(labels, baseline, conditional):
            line_x.extend([b, c, None])
            line_y.extend([lbl, lbl, None])

        # Conditional dot color by shift direction — green up, red down, grey flat.
        def _shift_color(d):
            if d > 0.005: return "#3dd68c"
            if d < -0.005: return "#ef4444"
            return "#94a3c5"
        cond_colors = [_shift_color(d) for d in deltas]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=line_x, y=line_y, mode="lines",
            line=dict(color="#3b4c6e", width=2),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=baseline, y=labels, mode="markers",
            name=t("whatif_baseline"),
            marker=dict(color="#94a3c5", size=10, line=dict(width=0)),
            hovertemplate="%{y}<br>" + t("whatif_baseline") + ": %{x:.1%}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=conditional, y=labels, mode="markers",
            name=t("whatif_conditional"),
            marker=dict(color=cond_colors, size=16,
                        line=dict(color="#0b1220", width=1.5)),
            customdata=list(zip(baseline, deltas)),
            hovertemplate=(
                "%{y}<br>" + t("whatif_conditional") + ": %{x:.1%}"
                "<br>" + t("whatif_baseline") + ": %{customdata[0]:.1%}"
                "<br>Δ %{customdata[1]:+.1%}<extra></extra>"
            ),
        ))
        fig.update_layout(
            height=520,
            margin=dict(l=140, r=40, t=20, b=40),
            paper_bgcolor="#0b1220", plot_bgcolor="#121c2e",
            font=dict(color="#e8edf7"),
            xaxis=dict(title="P(Win)", gridcolor="#1b2742", tickformat=".0%"),
            yaxis=dict(autorange="reversed", gridcolor="#1b2742"),
            legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.08, x=1, xanchor="right"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Biggest shifts
        shifts = res.copy()
        shifts["abs_delta"] = shifts["delta"].abs()
        shifts = shifts.sort_values("abs_delta", ascending=False).head(10)
        st.markdown(
            f'<div class="section-title" style="font-size:1.1rem;">{t("whatif_shifts")}</div>',
            unsafe_allow_html=True,
        )
        shifts["flag"] = shifts["team"].apply(lambda tm: flag_url(tm, "w80") or "")
        shifts["team_disp"] = shifts["team"].apply(team_name)
        s_flag, s_team = "Flag", t("whatif_col_team")
        s_base, s_cond, s_delta = t("whatif_baseline"), t("whatif_conditional"), "Δ"
        shifts_view = shifts[["flag", "team_disp", "baseline_p_W", "p_W", "delta"]].rename(columns={
            "flag": s_flag,
            "team_disp": s_team,
            "baseline_p_W": s_base,
            "p_W": s_cond,
            "delta": s_delta,
        })
        st.dataframe(
            shifts_view,
            use_container_width=True,
            hide_index=True,
            column_config={
                s_flag: st.column_config.ImageColumn(s_flag, width="small"),
                s_base: st.column_config.NumberColumn(s_base, format="percent"),
                s_cond: st.column_config.NumberColumn(s_cond, format="percent"),
                s_delta: st.column_config.NumberColumn(s_delta, format="percent"),
            },
        )
    else:
        st.info(t("whatif_hint"))


# ---------- Ask the Model (Challenge B) — dual-AI chat ----------
elif page_id == "ask":
    st.markdown(f'<div class="section-title">{t("ask_title")}</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="section-caption">{t("ask_caption")}</p>', unsafe_allow_html=True)

    lang = st.session_state.get("lang", "en")
    probs = load_probs()
    lb = load_leaderboard()
    summary = load_backtest_summary()
    squads_df = load_squads()
    metrics_df = load_squad_metrics()
    groups_df = load_groups()
    data_context = build_data_context(probs, lb, summary, squads_df, metrics_df, groups_df)

    if "ask_history" not in st.session_state:
        st.session_state["ask_history"] = []  # list of {"role","content"} in time order

    st.markdown(
        '<div class="wc-hint">💬 <code>@claude</code> → Claude 独答 · '
        '<code>@gemini</code> → Gemini 独答 · 直接问 = 两个 AI 都发言</div>',
        unsafe_allow_html=True,
    )

    # Example prompts — collapse once we have history
    if not st.session_state["ask_history"]:
        examples = EXAMPLE_QUESTIONS.get(lang, EXAMPLE_QUESTIONS["en"])
        cols = st.columns(len(examples))
        for c, ex in zip(cols, examples):
            if c.button(ex, use_container_width=True, key=f"ex_{ex[:20]}"):
                st.session_state["_preset_q"] = ex
                st.rerun()

    # Render chat history as one HTML block
    chat_html = '<div class="wc-chat">'
    for msg in st.session_state["ask_history"]:
        chat_html += _render_msg(msg["role"], msg["content"])
    chat_html += "</div>"
    st.markdown(chat_html, unsafe_allow_html=True)

    # Slot for in-flight thinking messages
    thinking_slot = st.empty()

    # Quick-reply buttons when Claude's last message asks "who's it for?"
    # Saves the user a round-trip of typing AND gives Claude a structured
    # follow-up it can reason about without re-parsing natural language.
    def _claude_is_asking_audience(hist: list) -> bool:
        for entry in reversed(hist):
            role = entry.get("role")
            if role == "claude":
                content = entry.get("content")
                text = content.get("text", "") if isinstance(content, dict) else (content or "")
                tl = text.lower()
                signals = [
                    "男士款", "女士款", "青少年", "全家桶",    # zh menu
                    "men's", "women's", "youth", "family bundle",  # en menu
                    "who is it for", "who's it for", "想给谁买", "给谁买",
                ]
                return any(s in text or s.lower() in tl for s in signals)
            if role == "user":
                return False
        return False

    if _claude_is_asking_audience(st.session_state["ask_history"]):
        st.markdown(
            '<div style="margin:8px 0 4px;font-size:0.85rem;color:#94a3c5;">'
            '⚡ Quick pick:</div>',
            unsafe_allow_html=True,
        )
        b1, b2, b3, b4 = st.columns(4)
        audience_opts = [
            (b1, "👨 Men's",    "我要男士款（成人球员版/复刻版）"),
            (b2, "👩 Women's",  "我要女士款（女款剪裁）"),
            (b3, "👦 Youth",    "我要青少年/儿童款"),
            (b4, "👨‍👩‍👧 Family", "给全家买 — 爸爸、妈妈、孩子一人一件"),
        ]
        for col, label, prompt in audience_opts:
            if col.button(label, key=f"aud_btn_{label}", use_container_width=True):
                st.session_state["_preset_q"] = prompt
                st.rerun()

    # Input pinned to bottom
    preset = st.session_state.pop("_preset_q", None)
    user_q = st.chat_input(t("ask_placeholder"), key="ask_chat_input")
    if preset and not user_q:
        user_q = preset

    if user_q and user_q.strip():
        target, clean_q = parse_routing(user_q, history=st.session_state.get("ask_history"))
        st.session_state["ask_history"].append({"role": "user", "content": user_q})

        # Show a "typing" bubble for whoever we're about to ask
        thinking_html = '<div class="wc-chat">'
        thinking_html += _render_msg("user", user_q)
        if target in ("claude", "both"):
            thinking_html += _render_msg("thinking_claude", "")
        if target in ("gemini", "both"):
            thinking_html += _render_msg("thinking_gemini", "")
        thinking_html += "</div>"
        thinking_slot.markdown(thinking_html, unsafe_allow_html=True)

        # Call the actual APIs
        if target in ("claude", "both"):
            try:
                # Replay prior conversation so multi-turn merch dialogues stay
                # in context (e.g. user says "梅西球衣" then just "男士款").
                # The current user_q was already appended to ask_history above,
                # so hand Claude everything up to but not including that tail.
                prev_history = st.session_state["ask_history"][:-1]
                ans = ask_claude(clean_q, data_context, lang, history=prev_history)
                st.session_state["ask_history"].append({"role": "claude", "content": ans})
            except RuntimeError as e:
                msg = "ANTHROPIC_API_KEY not set." if "NO_CLAUDE_KEY" in str(e) else str(e)
                st.session_state["ask_history"].append({"role": "error", "content": f"Claude: {msg}"})
            except Exception as e:
                st.session_state["ask_history"].append({"role": "error", "content": f"Claude: {e}"})

        if target in ("gemini", "both"):
            try:
                ans = ask_gemini(clean_q, data_context, lang)
                st.session_state["ask_history"].append({"role": "gemini", "content": ans})
            except RuntimeError as e:
                msg = "GEMINI_API_KEY not set." if "NO_GEMINI_KEY" in str(e) else str(e)
                st.session_state["ask_history"].append({"role": "error", "content": f"Gemini: {msg}"})
            except Exception as e:
                st.session_state["ask_history"].append({"role": "error", "content": f"Gemini: {e}"})

        st.rerun()

    # Clear conversation button
    if st.session_state["ask_history"]:
        if st.button("🗑️ Clear conversation", key="clear_chat"):
            st.session_state["ask_history"] = []
            st.rerun()


# ---------- Groups Overview ----------
elif page_id == "groups":
    st.markdown(f'<div class="section-title">{t("grp_title")}</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="section-caption">{t("grp_caption")}</p>', unsafe_allow_html=True)

    groups_df = load_groups()
    probs = load_probs()
    fixtures_df = load_fixtures()

    if groups_df.empty:
        st.warning("Groups data missing.")
    else:
        # Build a p_R32 lookup
        adv = probs.set_index("team")["p_R32"].to_dict()

        import re as _re
        def _inline_md(s: str) -> str:
            s = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
            return s

        cards_html = '<div class="grp-grid">'
        for letter in sorted(groups_df["group"].unique()):
            teams_in_group = groups_df[groups_df["group"] == letter]["team"].tolist()

            # Ranked by advance probability (p_R32)
            ranked = sorted(teams_in_group, key=lambda t: adv.get(t, 0.0), reverse=True)
            probs_list = [adv.get(t, 0.0) for t in ranked]

            # Dates for this group's fixtures
            date_str = ""
            if not fixtures_df.empty:
                grp_fx = fixtures_df[
                    fixtures_df["home_team"].isin(teams_in_group)
                    & fixtures_df["away_team"].isin(teams_in_group)
                ]
                if not grp_fx.empty:
                    first = grp_fx["date"].min().strftime("%b %d")
                    last = grp_fx["date"].max().strftime("%b %d")
                    date_str = t("grp_dates", first=first, last=last)

            # Verdict — pick the most interesting headline
            top_prob = probs_list[0]
            top_team = ranked[0]
            gap_top_3rd = (probs_list[0] - probs_list[2]) * 100 if len(probs_list) >= 3 else 100
            three_strong = sum(1 for p in probs_list if p >= 0.6)

            if top_prob >= 0.9:
                verdict = t("grp_verdict_lock", team=team_name(top_team), prob=int(top_prob * 100))
            elif three_strong >= 3:
                verdict = t("grp_verdict_death")
            elif gap_top_3rd < 20:
                verdict = t("grp_verdict_open", gap=int(gap_top_3rd))
            else:
                verdict = t("grp_verdict_fav", team=team_name(top_team))
            verdict = _inline_md(verdict)

            # Rows: top 2 = green, 3rd = amber-ish (default), 4th = muted
            rows = ""
            for i, (t_name_en, p) in enumerate(zip(ranked, probs_list)):
                if i < 2:
                    row_class = ""
                elif i == 2:
                    row_class = ""
                else:
                    row_class = "out"
                img = flag_img(t_name_en, h=14)
                localized = team_name(t_name_en)
                rows += (
                    f'<tr class="{row_class}">'
                    f'<td class="team">{img} {localized}</td>'
                    f'<td class="prob">{p*100:.0f}%</td>'
                    f"</tr>"
                )

            cards_html += (
                f'<div class="grp-card">'
                f'<span class="letter">{letter}</span>'
                f'<h3>Group {letter}</h3>'
                f'<div class="dates">{date_str}</div>'
                f'<table>{rows}</table>'
                f'<div class="verdict">{verdict}</div>'
                f'</div>'
            )
        cards_html += "</div>"
        st.markdown(cards_html, unsafe_allow_html=True)

        st.caption(f"💡 {t('grp_advance')} = P(reach Round of 32) from our 10K Monte Carlo runs.")


# ---------- Schedule ----------
elif page_id == "schedule":
    st.markdown(f'<div class="section-title">{t("sch_title")}</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="section-caption">{t("sch_caption")}</p>', unsafe_allow_html=True)

    sched = load_schedule()
    groups_df = load_groups()
    team_to_group = dict(zip(groups_df["team"], groups_df["group"])) if not groups_df.empty else {}

    if sched.empty:
        st.warning("Schedule data missing.")
    else:
        md_options = [t("sch_all"), "MD1", "MD2", "MD3"]
        picked_md = st.radio(t("sch_matchday"), md_options, horizontal=True, label_visibility="collapsed")

        view = sched if picked_md == t("sch_all") else sched[sched["matchday"] == picked_md]
        view = view.sort_values("date").reset_index(drop=True)

        html_blocks = []
        current_date = None
        for _, r in view.iterrows():
            d = r["date"]
            if current_date != d.date():
                day_label = d.strftime("%a · %b %d")  # "Thu · Jun 11"
                html_blocks.append(f'<div class="sch-day">{day_label}</div>')
                current_date = d.date()
            home = team_with_flag_img(r["home_team"], h=14)
            away = team_with_flag_img(r["away_team"], h=14)
            grp_letter = team_to_group.get(r["home_team"], "?")
            grp_chip = t("sch_grp", letter=grp_letter)
            stadium = r.get("stadium", "—")
            city = r.get("city_nice", "")
            host = r.get("host", "")
            venue = (
                f'<span class="stadium">{stadium}</span>'
                f' · {city}' + (f' · {host}' if host and host != city else '')
            )
            html_blocks.append(
                f'<div class="sch-row">'
                f'<div class="md">{r["matchday"]}</div>'
                f'<div class="match">{home} <span class="vs">vs</span> {away}</div>'
                f'<div><span class="group-chip">{grp_chip}</span></div>'
                f'<div class="venue">{venue}</div>'
                f"</div>"
            )
        st.markdown("\n".join(html_blocks), unsafe_allow_html=True)

        st.caption(f"💡 Total: {len(view)} matches  ·  16 host cities  ·  kickoff times TBA")


# ---------- Team Explorer ----------
elif page_id == "explorer":
    st.markdown(f'<div class="section-title">{t("exp_title")}</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="section-caption">{t("exp_caption")}</p>', unsafe_allow_html=True)

    squads_df = load_squads()
    metrics_df = load_squad_metrics()
    probs = load_probs()
    lb = load_leaderboard()
    matches_df = load_recent_matches()
    groups_df = load_groups()

    # Team picker: prefer teams with squad data first, but allow all 48 WC teams
    all_teams = sorted(probs["team"].unique())
    default_team = "Argentina" if "Argentina" in all_teams else all_teams[0]
    team = st.selectbox(
        t("exp_pick"),
        all_teams,
        index=all_teams.index(default_team),
        format_func=team_with_flag,
    )

    # Group label (e.g., "Group J with 🇩🇿 Algeria, 🇦🇹 Austria, 🇯🇴 Jordan")
    g_letter = _team_group(team, groups_df)
    if g_letter:
        mates = groups_df[(groups_df["group"] == g_letter) & (groups_df["team"] != team)]
        mate_names = " · ".join(team_with_flag_img(t, h=14) for t in mates["team"].tolist())
        st.markdown(
            f'<div class="wc-hint">🏟️ <b>Group {g_letter}</b> · '
            f'with {mate_names}</div>',
            unsafe_allow_html=True,
        )

    # Row 1 — model & market KPIs
    probs_row = probs[probs["team"] == team].iloc[0] if (probs["team"] == team).any() else None
    edge_row  = lb[lb["team"] == team].iloc[0] if (lb["team"] == team).any() else None
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(t("exp_kpi_model"), f"{probs_row['p_W']*100:.1f}%" if probs_row is not None else "—")
    if edge_row is not None:
        k2.metric(t("exp_kpi_market"), f"{edge_row['market_p_W']*100:.1f}%")
        delta_txt = "UNDER" if edge_row["direction"] == "UNDER" else "OVER"
        k3.metric(t("exp_kpi_edge"), f"{edge_row['edge']*100:+.1f} pp", delta_txt)
    else:
        k2.metric(t("exp_kpi_market"), "—")
        k3.metric(t("exp_kpi_edge"), "—")
    k4.metric(t("exp_kpi_final"), f"{probs_row['p_F']*100:.1f}%" if probs_row is not None else "—")

    # Row 2 — squad KPIs
    m = metrics_df[metrics_df["team"] == team].iloc[0] if (metrics_df["team"] == team).any() else None
    if m is not None:
        s1, s2, s3, s4 = st.columns(4)
        s1.metric(t("exp_squad_size"), int(m["squad_size"]))
        s2.metric(t("exp_avg_age"), f"{m['avg_age']:.1f}")
        s3.metric(t("exp_total_val"), f"€{m['total_market_value_m']:.0f}M")
        s4.metric(t("exp_top3_share"), f"{m['top3_value_share']*100:.0f}%")
    elif squads_df.empty or not (squads_df["team"] == team).any():
        st.info(t("exp_no_squad", team=team_with_flag(team)))

    # Team kit — crest + home jersey from TheSportsDB, click = Fanatics
    # search (affiliate-link-ready placeholder). Only render when we have at
    # least the badge.
    media = load_team_media().get(team, {})
    if media.get("badge") or media.get("jersey"):
        st.markdown(
            f'<div class="section-title" style="font-size:1.1rem;margin-top:12px;">'
            f'{t("exp_kit_title")}</div>',
            unsafe_allow_html=True,
        )
        shop_url = merch_link(team)
        shop_label = t("exp_kit_shop", team=team_name(team))
        kit_html = ['<div style="display:flex;gap:24px;align-items:flex-start;'
                    'background:#121c2e;border-radius:10px;padding:16px;'
                    'border:1px solid #1b2742;margin-bottom:12px;">']
        if media.get("badge"):
            kit_html.append(
                f'<div style="text-align:center;min-width:120px;">'
                f'<a href="{shop_url}" target="_blank" rel="noopener sponsored">'
                f'<img src="{media["badge"]}" alt="{team} crest" '
                f'style="max-height:120px;max-width:120px;'
                f'filter:drop-shadow(0 2px 6px rgba(0,0,0,0.4));"></a>'
                f'<div style="font-size:0.8rem;color:#94a3c5;margin-top:6px;">'
                f'{t("exp_kit_crest")}</div></div>'
            )
        if media.get("jersey"):
            kit_html.append(
                f'<div style="text-align:center;min-width:180px;">'
                f'<a href="{shop_url}" target="_blank" rel="noopener sponsored">'
                f'<img src="{media["jersey"]}" alt="{team} jersey" '
                f'style="height:180px;width:auto;border-radius:6px;'
                f'background:rgba(255,255,255,0.02);padding:8px;"></a>'
                f'<div style="font-size:0.8rem;color:#94a3c5;margin-top:6px;">'
                f'{t("exp_kit_jersey")}</div></div>'
            )
        kit_html.append('<div style="flex-basis:100%;margin-top:8px;">'
                        f'<a href="{shop_url}" target="_blank" rel="noopener sponsored" '
                        f'style="display:inline-block;background:#f7c948;color:#0b1220;'
                        f'font-weight:600;padding:8px 16px;border-radius:6px;'
                        f'text-decoration:none;">{shop_label}</a></div>')
        kit_html.append('</div>')
        st.markdown("\n".join(kit_html), unsafe_allow_html=True)

    # Squad table with headshots
    if not squads_df.empty and (squads_df["team"] == team).any():
        photos = load_player_photos()
        is_full = bool(m is not None)  # we have team_squad_metrics → full squad
        title_text = t("exp_squad_title") if is_full else "⭐ Star players (not full squad — full 26-man arrives May 31)"
        st.markdown(f'<div class="section-title" style="font-size:1.1rem;">{title_text}</div>',
                    unsafe_allow_html=True)

        g = squads_df[squads_df["team"] == team].copy()
        g = g.sort_values("market_value_eur", ascending=False)

        rows_html = ['<div class="squad-list">']
        for _, pr in g.iterrows():
            photo = photos.get(pr["player"])
            mv_m = pr["market_value_eur"] / 1e6
            pos = pr["position"]
            age = int(pr["age"])
            club = pr["club"]
            league = pr["club_league"]
            avatar = (
                f'<img class="s-avatar" src="{photo}" alt="{pr["player"]}" loading="lazy" />'
                if photo
                else f'<div class="s-avatar placeholder">⚽</div>'
            )
            rows_html.append(
                f'<div class="s-row">'
                f'  {avatar}'
                f'  <div class="s-name">{pr["player"]}<div class="s-pos">{pos} · {age}</div></div>'
                f'  <div class="s-club">{club}<div class="s-league">{league}</div></div>'
                f'  <div class="s-value">€{mv_m:.0f}M</div>'
                f"</div>"
            )
        rows_html.append("</div>")
        st.markdown("\n".join(rows_html), unsafe_allow_html=True)

    # Recent form
    form = _last_n_for_team(matches_df, team, n=10)
    if form["n"]:
        st.markdown(f'<div class="section-title" style="font-size:1.1rem;">{t("exp_form_title")}</div>',
                    unsafe_allow_html=True)
        pill_css = {
            "W": "background:rgba(34,197,94,.22);color:#4ade80",
            "D": "background:rgba(148,163,197,.2);color:#cbd5e8",
            "L": "background:rgba(239,68,68,.22);color:#f87171",
        }
        pills = "".join(
            f'<span style="display:inline-block;padding:4px 10px;border-radius:999px;'
            f'font-weight:700;margin-right:4px;{pill_css[r]}">{r}</span>'
            for r in form["results"]
        )
        st.markdown(pills, unsafe_allow_html=True)
        gd_sign = "+" if form["avg_gd"] >= 0 else ""
        st.caption(f"{form['W']}W {form['D']}D {form['L']}L · goal diff {gd_sign}{form['avg_gd']:.1f}/g")

    # "Why" card
    st.markdown(f'<div class="section-title" style="font-size:1.1rem;">{t("exp_why_title")}</div>',
                unsafe_allow_html=True)
    bullets = []
    if probs_row is not None and probs_row["p_W"] > 0.1:
        bullets.append(f"✅ Model ranks them top-5 ({probs_row['p_W']*100:.1f}% champion)")
    if edge_row is not None:
        if edge_row["edge"] > 0.05:
            bullets.append(f"📈 Undervalued by Polymarket ({edge_row['edge']*100:+.1f} pp)")
        elif edge_row["edge"] < -0.05:
            bullets.append(f"📉 Overvalued by Polymarket ({edge_row['edge']*100:+.1f} pp)")
    if m is not None:
        if m["top3_value_share"] > 0.5:
            bullets.append(f"⚠️ Top 3 players carry {m['top3_value_share']*100:.0f}% of squad value — key-player risk")
        if m["n_over_32"] >= 4:
            bullets.append(f"⚠️ {int(m['n_over_32'])} players over 32 — stamina risk in 48-team format")
        if m["n_under_23"] >= 2 and m["avg_age"] < 28:
            bullets.append(f"✨ {int(m['n_under_23'])} under-23 players — Young Player award candidates")
        if m["foreign_share"] >= 0.9:
            bullets.append(f"🌍 {m['foreign_share']*100:.0f}% of squad plays abroad — high-level exposure")
    if form["n"] and form["W"] >= form["n"] * 0.7:
        bullets.append(f"🔥 {form['W']} wins in last {form['n']} — in form")
    if not bullets:
        bullets.append("_Not enough data yet — curated squad dataset covers 11 priority teams._")
    st.markdown("\n".join(f"- {b}" for b in bullets))

    # Path to final
    if probs_row is not None:
        st.markdown(f'<div class="section-title" style="font-size:1.1rem;">{t("exp_path_title")}</div>',
                    unsafe_allow_html=True)
        stages = ["R32", "R16", "QF", "SF", "F", "W"]
        vals = [probs_row[f"p_{s}"] for s in stages]
        fig = go.Figure(go.Bar(x=stages, y=vals, marker_color="#f7c948",
                               text=[f"{v*100:.1f}%" for v in vals], textposition="outside"))
        fig.update_layout(
            height=280, margin=dict(l=30, r=20, t=30, b=30),
            paper_bgcolor="#0b1220", plot_bgcolor="#121c2e",
            font=dict(color="#e8edf7"),
            yaxis=dict(tickformat=".0%", gridcolor="#1b2742", range=[0, max(vals) * 1.25]),
            xaxis=dict(gridcolor="#1b2742"),
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------- Stage Reach ----------
elif page_id == "stage":
    probs = load_probs()
    st.markdown(f'<div class="section-title">{t("stage_title")}</div>', unsafe_allow_html=True)
    cols = ["p_R32", "p_R16", "p_QF", "p_SF", "p_F", "p_W"]
    nice = ["R32", "R16", "QF", "SF", "Final", "Win"]
    teams = probs["team"].tolist()
    # Default to top-6 by champion prob
    default = probs.sort_values("p_W", ascending=False).head(6)["team"].tolist()
    chosen = st.multiselect(t("stage_pick"), teams, default=default,
                             format_func=team_with_flag)
    if chosen:
        sub = probs[probs["team"].isin(chosen)].set_index("team")[cols]
        sub.columns = nice
        long = sub.reset_index().melt(id_vars="team", var_name="Stage", value_name="P(reach)")
        long["team"] = long["team"].apply(team_with_flag)
        fig = px.line(
            long, x="Stage", y="P(reach)", color="team", markers=True,
            category_orders={"Stage": nice},
        )
        fig.update_layout(
            height=520,
            yaxis=dict(tickformat=".0%", gridcolor="#1b2742"),
            xaxis=dict(gridcolor="#1b2742"),
            paper_bgcolor="#0b1220",
            plot_bgcolor="#121c2e",
            font=dict(color="#e8edf7"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------- Calibration ----------
elif page_id == "calib":
    st.markdown(f'<div class="section-title">{t("calib_title")}</div>', unsafe_allow_html=True)
    summary = load_backtest_summary()
    rel = load_backtest_reliability()
    preds = load_backtest_predictions()
    st.markdown(f'<p class="section-caption">{t("calib_caption")}</p>', unsafe_allow_html=True)
    all_dc = summary[(summary["year"] == "ALL") & (summary["model"] == "DixonColes")].iloc[0]
    all_base = summary[(summary["year"] == "ALL") & (summary["model"] == "Uniform(1/3)")].iloc[0]
    skill = 1 - all_dc["brier"] / all_base["brier"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Brier (DC)", f"{all_dc['brier']:.4f}")
    m2.metric("Brier (Uniform)", f"{all_base['brier']:.4f}")
    m3.metric("LogLoss (DC)", f"{all_dc['logloss']:.4f}")
    m4.metric("Brier skill vs uniform", f"{skill*100:+.1f}%")

    st.markdown(f'<div class="section-title" style="margin-top:18px;">{t("by_tournament")}</div>', unsafe_allow_html=True)
    pivot = summary.pivot_table(index="year", columns="model", values=["brier", "logloss"])
    st.dataframe(pivot.style.format("{:.4f}"), use_container_width=True)

    st.markdown(f'<div class="section-title" style="margin-top:18px;">{t("reliability_title")}</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="section-caption">{t("reliability_caption")}</p>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="#94a3c5"))
    fig.add_trace(go.Scatter(
        x=rel["avg_predicted"], y=rel["empirical"],
        mode="markers+text",
        marker=dict(size=np.sqrt(rel["count"]) * 4 + 6, color="#f7c948"),
        text=[f"n={c}" for c in rel["count"]],
        textposition="top center",
        name="Dixon-Coles",
    ))
    fig.update_layout(
        height=520,
        xaxis=dict(title="Predicted probability", range=[0, 1], gridcolor="#1b2742"),
        yaxis=dict(title="Empirical frequency", range=[0, 1], gridcolor="#1b2742"),
        paper_bgcolor="#0b1220",
        plot_bgcolor="#121c2e",
        font=dict(color="#e8edf7"),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander(t("all_backtest", n=len(preds))):
        pv = preds.copy()
        pv["home"] = pv["home"].apply(team_with_flag)
        pv["away"] = pv["away"].apply(team_with_flag)
        st.dataframe(
            pv[["year", "date", "home", "away", "home_goals", "away_goals",
                "p_home", "p_draw", "p_away", "actual"]]
            .style.format({"p_home": "{:.1%}", "p_draw": "{:.1%}", "p_away": "{:.1%}"}),
            use_container_width=True,
        )


# ---------- Methodology ----------
elif page_id == "method":
    st.markdown(f'<div class="section-title">{t("method_title")}</div>', unsafe_allow_html=True)
    st.markdown(f"### {t('method_h_question')}\n\n{t('method_p_question')}")
    st.markdown(f"### {t('method_h_data')}\n\n{t('method_p_data')}")
    st.markdown(f"### {t('method_h_model')}\n\n{t('method_p_model')}")
    st.markdown(f"### {t('method_h_sim')}\n\n{t('method_p_sim')}")
    st.markdown(f"### {t('method_h_calib')}\n\n{t('method_p_calib')}")
    st.markdown(f"### {t('method_h_not')}\n\n{t('method_p_not')}")
