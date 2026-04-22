"""
WorldCup26IQ — Streamlit app for the 2026 FIFA World Cup model.

Self-contained for Streamlit Cloud deploy. Reads parquet files from `./data/`
or the script directory.
"""
from __future__ import annotations

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


def _render_msg(role: str, content: str) -> str:
    """Return HTML for a single WeChat-style message."""
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

    html_content = _md_to_html(content)
    return (
        f'<div class="wc-row {cls}"><div class="wc-avatar">{avatar}</div>'
        f'<div class="wc-bwrap"><div class="wc-name">{name}</div>'
        f'<div class="wc-bubble {cls}">{html_content}</div></div></div>'
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
    "Ecuador": "🇪🇨", "Egypt": "🇪🇬", "England": "🏴\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f",
    "France": "🇫🇷", "Germany": "🇩🇪", "Ghana": "🇬🇭", "Haiti": "🇭🇹",
    "Iran": "🇮🇷", "Iraq": "🇮🇶", "Ivory Coast": "🇨🇮", "Japan": "🇯🇵",
    "Jordan": "🇯🇴", "Mexico": "🇲🇽", "Morocco": "🇲🇦", "Netherlands": "🇳🇱",
    "New Zealand": "🇳🇿", "Norway": "🇳🇴", "Panama": "🇵🇦", "Paraguay": "🇵🇾",
    "Portugal": "🇵🇹", "Qatar": "🇶🇦", "Saudi Arabia": "🇸🇦",
    "Scotland": "🏴\U000e0067\U000e0062\U000e0073\U000e0063\U000e0074\U000e007f",
    "Senegal": "🇸🇳", "South Africa": "🇿🇦", "South Korea": "🇰🇷", "Spain": "🇪🇸",
    "Sweden": "🇸🇪", "Switzerland": "🇨🇭", "Tunisia": "🇹🇳", "Turkey": "🇹🇷",
    "United States": "🇺🇸", "Uruguay": "🇺🇾", "Uzbekistan": "🇺🇿",
    # Extras that show up in historical Polymarket markets
    "Republic of Ireland": "🇮🇪", "Northern Ireland": "🇬🇧", "Wales": "🏴\U000e0067\U000e0062\U000e0077\U000e006c\U000e0073\U000e007f",
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


# ---------- global config + CSS ----------
st.set_page_config(
    page_title="WorldCup26IQ",
    page_icon="⚽",
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
            ['apple-mobile-web-app-title',            'WC26IQ'],
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
            # Reorder: locked 1st, locked 2nd, then remaining in natural order
            final = [None, None]
            remaining = list(ordered_teams)
            if "1st" in lk:
                final[0] = lk["1st"]
                if lk["1st"] in remaining:
                    remaining.remove(lk["1st"])
            if "2nd" in lk:
                final[1] = lk["2nd"]
                if lk["2nd"] in remaining:
                    remaining.remove(lk["2nd"])
            # Fill unlocked slots with natural top of remaining
            for pos in range(2):
                if final[pos] is None:
                    final[pos] = remaining.pop(0)
            # 3rd and 4th from whatever's left, natural order
            third = remaining[0] if remaining else None
            fourth = remaining[1] if len(remaining) > 1 else None
            first_place.append((gkey, final[0], tables[gkey][final[0]]))
            second_place.append((gkey, final[1], tables[gkey][final[1]]))
            if third:
                third_place.append((gkey, third, tables[gkey][third]))

        third_sorted = sorted(
            third_place,
            key=lambda t: (t[2]["pts"], t[2]["gf"] - t[2]["ga"], t[2]["gf"], rng.random()),
            reverse=True,
        )
        third_adv = [t for (_, t, _) in third_sorted[:8]]

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
        row = {"team": team}
        for k, s in enumerate(STAGES):
            row[f"p_{s}"] = counts[team][k] / n_sims
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
            tw = team_with_flag(r["team"])
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
st.sidebar.markdown(f"# {t('app_title')}")
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
                       over_flag=flag(best_over["team"]),
                       over_team=team_name(best_over["team"]),
                       under_pp=f"{best_under['edge']*100:+.0f}",
                       under_flag=flag(best_under["team"]),
                       under_team=team_name(best_under["team"]))
    st.markdown(
        f"""
        <div class="hero">
          <h1>{t('app_title')}</h1>
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
            <div class="value">{team_with_flag(best_under['team'])}</div>
            <div class="delta up">{best_under['edge']*100:+.1f} pp · {t('kpi_mkt_model', mkt=f"{best_under['market_p_W']*100:.1f}", model=f"{best_under['p_W']*100:.1f}")}</div>
          </div>
          <div class="kpi">
            <div class="label">{t('kpi_biggest_over')}</div>
            <div class="value">{team_with_flag(best_over['team'])}</div>
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
                  <div class="value">{team_with_flag(row['team'])}</div>
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
            f'<div class="team">{team_with_flag(r["team"])}</div>'
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
        show["team"] = show["team"].apply(team_with_flag)
        st.dataframe(
            show[["team", "p_R32", "p_R16", "p_QF", "p_SF", "p_F", "p_W"]]
            .style.format({c: "{:.1%}" for c in ["p_R32", "p_R16", "p_QF", "p_SF", "p_F", "p_W"]}),
            use_container_width=True,
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
            f'<div class="team">{team_with_flag(r["team"])}</div>'
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

    # Cycle: unset → 🏆 (1st) → 🥈 (2nd) → unset. Selecting a slot
    # displaces any other team already in that slot for the same group.
    def _cycle_lock(gkey: str, team: str):
        locks = st.session_state["wc26_locks"].setdefault(gkey, {})
        if locks.get("1st") == team:
            locks.pop("1st", None)
            locks["2nd"] = team
        elif locks.get("2nd") == team:
            locks.pop("2nd", None)
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

        top = res.head(15).copy()
        top["label"] = top["team"].apply(team_with_flag)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=top["label"], x=top["baseline_p_W"], orientation="h",
            name=t("whatif_baseline"), marker_color="#94a3c5", opacity=0.6,
        ))
        fig.add_trace(go.Bar(
            y=top["label"], x=top["p_W"], orientation="h",
            name=t("whatif_conditional"), marker_color="#f7c948",
        ))
        fig.update_layout(
            barmode="overlay",
            height=520,
            margin=dict(l=140, r=40, t=20, b=40),
            paper_bgcolor="#0b1220", plot_bgcolor="#121c2e",
            font=dict(color="#e8edf7"),
            xaxis=dict(title="P(Win)", gridcolor="#1b2742", tickformat=".0%"),
            yaxis=dict(autorange="reversed"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
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
        view = shifts.copy()
        view["team"] = view["team"].apply(team_with_flag)
        st.dataframe(
            view[["team", "baseline_p_W", "p_W", "delta"]].style.format({
                "baseline_p_W": "{:.1%}", "p_W": "{:.1%}", "delta": "{:+.1%}"
            }),
            use_container_width=True,
            hide_index=True,
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

    # Input pinned to bottom
    preset = st.session_state.pop("_preset_q", None)
    user_q = st.chat_input(t("ask_placeholder"), key="ask_chat_input")
    if preset and not user_q:
        user_q = preset

    if user_q and user_q.strip():
        target, clean_q = parse_routing(user_q)
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
                ans = ask_claude(clean_q, data_context, lang)
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
                emoji = flag(t_name_en)
                localized = team_name(t_name_en)
                rows += (
                    f'<tr class="{row_class}">'
                    f'<td class="team">{emoji} {localized}</td>'
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
            home = team_with_flag(r["home_team"])
            away = team_with_flag(r["away_team"])
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
        mate_names = " · ".join(team_with_flag(t) for t in mates["team"].tolist())
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
