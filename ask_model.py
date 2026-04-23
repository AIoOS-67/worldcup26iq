"""
Ask the Model — dual-AI interface (Claude + Gemini).

Users can:
  - start with `@claude` to get Claude-only
  - start with `@gemini` to get Gemini-only
  - otherwise get BOTH side by side (the model-vs-model debate)

Keys are read from Streamlit secrets or env vars:
  ANTHROPIC_API_KEY
  GEMINI_API_KEY  (or GOOGLE_API_KEY as fallback)
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

# Load .env if present (local dev). Streamlit Cloud uses st.secrets instead.
try:
    from dotenv import load_dotenv
    here = Path(__file__).resolve().parent
    for candidate in (here / ".env", here.parent / ".env"):
        if candidate.exists():
            load_dotenv(candidate)
            break
except ImportError:
    pass


CLAUDE_MODEL = "claude-opus-4-7"
GEMINI_MODEL = "gemini-2.5-flash"
MAX_TOKENS = 1200


LANG_NAME = {
    "en": "English",
    "zh": "Simplified Chinese (简体中文)",
    "es": "Spanish (Español)",
    "pt": "Portuguese (Português)",
    "fr": "French (Français)",
}


# ---------- API key helpers ----------
def _get_anthropic_key() -> str | None:
    try:
        return st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY")


def _get_gemini_key() -> str | None:
    try:
        key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


# ---------- shared context ----------
@st.cache_data
def build_data_context(
    probs: pd.DataFrame,
    leaderboard: pd.DataFrame,
    backtest_summary: pd.DataFrame,
    squads: pd.DataFrame | None = None,
    squad_metrics: pd.DataFrame | None = None,
    groups: pd.DataFrame | None = None,
) -> str:
    """Compact textual context for the LLMs. ~1500-2500 tokens depending on squads."""
    lines = []
    lines.append("=== WorldCup26IQ model data (updated Apr 2026) ===\n")
    lines.append("Model: Dixon-Coles bivariate Poisson, home_adv=0.21, rho=-0.095,")
    lines.append("  fit on ~7K internationals since 2019, 10,000 Monte Carlo runs.")
    lines.append("")

    # Official FIFA 2026 group draw (held Dec 2025)
    if groups is not None and not groups.empty:
        lines.append("== 2026 FIFA World Cup — Official Group Draw (12 groups × 4 teams) ==")
        for letter, g in groups.groupby("group"):
            team_list = " | ".join(g.sort_values("pot_seed")["team"].tolist())
            lines.append(f"Group {letter}: {team_list}")
        lines.append("")

    lines.append("== Champion + stage probabilities (top 30) ==")
    lines.append("team | p_R32 | p_R16 | p_QF | p_SF | p_F | p_W")
    for _, r in probs.sort_values("p_W", ascending=False).head(30).iterrows():
        lines.append(
            f"{r['team']} | {r['p_R32']:.1%} | {r['p_R16']:.1%} | {r['p_QF']:.1%} | "
            f"{r['p_SF']:.1%} | {r['p_F']:.1%} | {r['p_W']:.1%}"
        )

    lines.append("")
    lines.append("== Mispricing vs Polymarket (all teams with ≥$500K liquidity) ==")
    lines.append("team | direction | model_p | market_p | edge_pp | recent_form_last20 | reason")
    for _, r in leaderboard.iterrows():
        lines.append(
            f"{r['team']} | {r['direction']} | {r['p_W']:.1%} | "
            f"{r['market_p_W']:.1%} | {r['edge']*100:+.1f}pp | {r['form_recent_results']} | {r['reason']}"
        )

    lines.append("")
    lines.append("== Calibration (backtest) ==")
    for _, r in backtest_summary.iterrows():
        lines.append(f"{r['model']} | year={r['year']} | brier={r['brier']:.4f} | logloss={r['logloss']:.4f}")
    lines.append("Brier skill score vs uniform: +7.0% across 128 WC matches (2018+2022).")

    # Squad metrics + players (only 11 curated teams — Ken's starter dataset)
    if squad_metrics is not None and not squad_metrics.empty:
        lines.append("")
        lines.append("== Squad metrics (curated 11 priority teams) ==")
        lines.append("team | size | avg_age | total_value_M€ | top3_value_share | n_over_32 | n_under_23 | foreign_share | top_player")
        for _, r in squad_metrics.iterrows():
            lines.append(
                f"{r['team']} | {r['squad_size']} | {r['avg_age']:.1f} | "
                f"€{r['total_market_value_m']:.0f}M | {r['top3_value_share']*100:.0f}% | "
                f"{r['n_over_32']} | {r['n_under_23']} | {r['foreign_share']*100:.0f}% | {r['top_player']}"
            )

    if squads is not None and not squads.empty:
        lines.append("")
        lines.append("== Individual players (curated — by team, top value first) ==")
        lines.append("team | player | age | pos | club | league | value_M€")
        for team, g in squads.groupby("team"):
            g = g.sort_values("market_value_eur", ascending=False)
            for _, p in g.iterrows():
                mv = p["market_value_eur"] / 1_000_000
                lines.append(
                    f"{team} | {p['player']} | {p['age']} | {p['position']} | "
                    f"{p['club']} | {p['club_league']} | €{mv:.0f}M"
                )

    return "\n".join(lines)


# ---------- App structure the AIs need to know about ----------
APP_GUIDE = """\
== App page guide (WorldCup26IQ) ==
The user is on WorldCup26IQ right now. The app has these pages, accessible via the left sidebar:
- 🏠 Hero (首页): headline, 4 KPI cards, top-5 favourites, top-5 mispricings, calibration mini-chart, "Coming May 31" roadmap.
- 🏆 Champion Probabilities (夺冠概率): full bar chart + table for all 48 teams.
- 💸 Mispricing Leaderboard (市场偏差排行榜): 43 Polymarket markets ranked by |edge| × √(liquidity), each with a data-backed reason.
- 🎲 What-If Simulator (What-If 模拟器): user locks group winners/runners-up and re-runs 500-5,000 Monte Carlo tournaments conditional on those picks; output shows baseline vs conditional champion-probability shifts.
- 🤖 Ask the Model (问问模型): THIS page — Claude + Gemini dual-AI chat; @claude / @gemini / no prefix routes the question.
- 🔍 Team Explorer (球队详情): user selects one team → model/market KPIs, squad size/age/value/top-3-share, projected squad table, last-10 form pills, "Why the model rates them this way" card, path-to-final bar chart. Currently has full squad data for 11 priority teams (Argentina, France, Brazil, England, Spain, Portugal, Germany, Netherlands, USA, Canada, Mexico). All 48 teams covered May 31.
- 📊 Stage Reach (晋级概率): multi-team comparison of P(reach R32/R16/QF/SF/F/W).
- 📏 Calibration (校准): backtest evidence on 2018+2022 WCs, reliability diagram, +7.0% Brier skill score.
- 📖 Methodology (方法论): full technical docs.

When the user asks to "check Team Explorer" / "查球队详情" / similar page names, tell them the 🔍 Team Explorer page lets them drill in. If they ask a question the model data CAN answer directly, do that first and then optionally point to the relevant page.

Squad/player-level answers are only available for the 11 curated teams listed above. For any other team, say squad data arrives May 31.
"""


FLAGS_ASCII = {
    "Argentina": "🇦🇷", "Brazil": "🇧🇷", "France": "🇫🇷", "Spain": "🇪🇸",
    "Colombia": "🇨🇴", "Ecuador": "🇪🇨", "Morocco": "🇲🇦", "Japan": "🇯🇵",
    "Paraguay": "🇵🇾", "Uruguay": "🇺🇾", "Portugal": "🇵🇹", "Germany": "🇩🇪",
    "Netherlands": "🇳🇱", "Mexico": "🇲🇽", "United States": "🇺🇸", "Canada": "🇨🇦",
    "Saudi Arabia": "🇸🇦", "Senegal": "🇸🇳", "Australia": "🇦🇺", "Norway": "🇳🇴",
    "Switzerland": "🇨🇭", "Algeria": "🇩🇿", "Croatia": "🇭🇷", "England": "🇬🇧",
    "Iran": "🇮🇷", "Ghana": "🇬🇭", "South Korea": "🇰🇷", "Ivory Coast": "🇨🇮",
    "DR Congo": "🇨🇩", "Cape Verde": "🇨🇻", "Tunisia": "🇹🇳", "Egypt": "🇪🇬",
    "Scotland": "🇬🇧", "Sweden": "🇸🇪", "Belgium": "🇧🇪", "Austria": "🇦🇹",
}


def _team_table_for_lang(lang: str) -> str:
    if lang == "en":
        return "(no translation needed)"
    from i18n import TEAMS
    lines = []
    for english, variants in TEAMS.items():
        local = variants.get(lang, english)
        emoji = FLAGS_ASCII.get(english, "🏳️")
        lines.append(f"  {english} → {emoji} {local}")
    return "\n".join(lines)


EXAMPLE_TEAM = {
    "en": "🇦🇷 Argentina",
    "zh": "🇦🇷 阿根廷",
    "es": "🇦🇷 Argentina",
    "pt": "🇦🇷 Argentina",
    "fr": "🇦🇷 Argentine",
}


# ---------- prompt templates ----------
CLAUDE_SYSTEM = """\
You are WorldCup26IQ's **primary analyst** (nicknamed 'Claude'). Your job is to answer questions about the 2026 FIFA World Cup using the PROVIDED DATA ONLY. You never make up numbers.

# CRITICAL LANGUAGE RULE
You MUST respond in **{lang_name}**, regardless of what language the user types in.
- If user types in English but {lang_name} is 中文, you answer in 中文.
- If user types in 中文 but {lang_name} is English, you answer in English.
- If user mixes languages, still answer in {lang_name}.
The UI language setting is the ONLY source of truth for your output language.

# Rules
- Keep it under 180 words unless the user asks for detail.
- Ground every probability claim in the provided data. Cite the relevant number.
- When mentioning a team, use the localized name from the TEAM NAME TABLE below (not the English name), and prefix with the flag emoji. Example in {lang_name}: "{example_team}".
- If the user asks a what-if question, explain what the model currently says AND tell them the What-If page simulates it live.
- If the user asks about a specific app page ("查球队详情" / "check Team Explorer"), point them to the right sidebar entry AND give a quick answer from the data if you can.
- If the team is not in the data, say so plainly. If asked about a player and the team IS in the curated 11, use the data. Otherwise say squads arrive May 31.
- Lean toward the model's quantitative view — that's your role.

# Optional: team merch recommendations
You have a `recommend_team_merch` tool that appends a shoppable card (crest + home jersey + outbound link) below your written reply. Use it only when showing the team's visuals genuinely adds to the conversation — fan affinity signals, "who should I follow / support / root for" questions, dark-horse picks the user might want to get behind. Pass a short one-sentence `pitch` explaining why in the user's language. Usually zero cards per answer. Never call the tool for every team you happen to mention, and never force a recommendation into an otherwise-analytical reply.

# TEAM NAME TABLE ({lang_name})
{team_table}

{app_guide}

# DATA
{context}
"""


GEMINI_SYSTEM = """\
You are WorldCup26IQ's **second-opinion analyst** (nicknamed 'Gemini'). A peer analyst ('Claude') answers the same question in parallel, lean toward the raw model's view. Your job is to **add perspective**: if there's a reason the market (Polymarket) might be right and the model might be overfit or misleading, name it. Be the skeptical but constructive counter-voice.

# CRITICAL LANGUAGE RULE
You MUST respond in **{lang_name}**, regardless of what language the user types in.
- If user types in English but {lang_name} is 中文, you answer in 中文.
- If user types in 中文 but {lang_name} is English, you answer in English.
- If user mixes languages, still answer in {lang_name}.
The UI language setting is the ONLY source of truth for your output language.

# Rules
- Keep it under 150 words — tighter than Claude.
- Ground claims in the PROVIDED DATA. Never invent numbers.
- When mentioning a team, use the localized name from the TEAM NAME TABLE and prefix with the flag emoji.
- Identify any weaknesses in the model's answer: CONMEBOL bias, injury blindness, small sample, format novelty (48-team new format), home-advantage assumptions, squad-age or key-player risks.
- Know the app structure — if the user asks about a page, guide them.
- If you fundamentally agree with Claude, say so and add one supporting angle — don't fabricate disagreement.
- End with ONE concrete action the user could take on the app (e.g., "try locking X in What-If to test this" or "check 🔍 Team Explorer for France").

# TEAM NAME TABLE ({lang_name})
{team_table}

{app_guide}

# DATA
{context}
"""


# ---------- Claude tool: shoppable team card ----------
# Official WC 2026 teams (48), kept in sync with data/team_media.json.
WC26_TEAMS = [
    "Algeria", "Argentina", "Australia", "Austria", "Belgium",
    "Bosnia and Herzegovina", "Brazil", "Canada", "Cape Verde", "Colombia",
    "Croatia", "Curaçao", "Czech Republic", "DR Congo", "Ecuador",
    "Egypt", "England", "France", "Germany", "Ghana", "Haiti", "Iran",
    "Iraq", "Ivory Coast", "Japan", "Jordan", "Mexico", "Morocco",
    "Netherlands", "New Zealand", "Norway", "Panama", "Paraguay",
    "Portugal", "Qatar", "Saudi Arabia", "Scotland", "Senegal",
    "South Africa", "South Korea", "Spain", "Sweden", "Switzerland",
    "Tunisia", "Turkey", "United States", "Uruguay", "Uzbekistan",
]

MERCH_TOOL = {
    "name": "recommend_team_merch",
    "description": (
        "Display a shoppable card below your answer with the team's crest, "
        "home jersey, and an outbound link to their official gear. Call this "
        "tool when showing the team's visuals would genuinely add to the "
        "conversation — e.g. the user expresses fan affinity, asks for a "
        "team to 'follow' / 'support' / 'bet on', or you're highlighting a "
        "dark horse the user might want to get behind.\n\n"
        "Do NOT call it for every team mentioned in passing. Do NOT call it "
        "when the user is asking a purely analytical question ('Why is X "
        "undervalued?') unless showing the kit genuinely fits the answer.\n\n"
        "Usually 0 cards per response. At most one, or two if the user is "
        "asking to compare / pick between two specific teams."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "team": {
                "type": "string",
                "enum": WC26_TEAMS,
                "description": "The team to show merchandise for (English canonical name).",
            },
            "pitch": {
                "type": "string",
                "description": (
                    "A single short sentence in the response language (the user's UI "
                    "language) explaining WHY the user might want to check out this "
                    "team's gear. Under 20 words. No generic ad-speak."
                ),
            },
        },
        "required": ["team", "pitch"],
    },
}


# ---------- Claude ----------
def ask_claude(question: str, context: str, lang: str) -> dict:
    """Return {'text': str, 'merch': list[{'team','pitch'}]}.

    Claude may optionally call recommend_team_merch tool — when it does,
    we surface those decisions as structured recs that the UI renders as
    shoppable cards. We do NOT round-trip back to Claude with tool_result;
    the tool is used purely as a structured "intent" marker in the first
    response turn.
    """
    api_key = _get_anthropic_key()
    if not api_key:
        raise RuntimeError("NO_CLAUDE_KEY")
    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)
    system = CLAUDE_SYSTEM.format(
        lang_name=LANG_NAME.get(lang, "English"),
        context=context,
        team_table=_team_table_for_lang(lang),
        example_team=EXAMPLE_TEAM.get(lang, EXAMPLE_TEAM["en"]),
        app_guide=APP_GUIDE,
    )
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=[
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}},
        ],
        tools=[MERCH_TOOL],
        messages=[{"role": "user", "content": question}],
    )
    parts: list[str] = []
    merch: list[dict] = []
    for block in resp.content:
        btype = getattr(block, "type", None)
        if btype == "text":
            parts.append(block.text)
        elif btype == "tool_use" and getattr(block, "name", "") == "recommend_team_merch":
            inp = getattr(block, "input", None) or {}
            team = inp.get("team")
            pitch = inp.get("pitch", "")
            if team and team in WC26_TEAMS:
                merch.append({"team": team, "pitch": pitch})
    return {
        "text": "".join(parts) or "(empty response)",
        "merch": merch,
    }


# ---------- Gemini ----------
def ask_gemini(question: str, context: str, lang: str) -> str:
    api_key = _get_gemini_key()
    if not api_key:
        raise RuntimeError("NO_GEMINI_KEY")
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    system = GEMINI_SYSTEM.format(
        lang_name=LANG_NAME.get(lang, "English"),
        context=context,
        team_table=_team_table_for_lang(lang),
        app_guide=APP_GUIDE,
    )
    # Disable thinking on 2.5-flash — otherwise reasoning tokens consume the
    # output budget and we get empty responses.
    config_kwargs: dict = {
        "system_instruction": system,
        "max_output_tokens": 2048,
    }
    try:
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    except Exception:
        pass

    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=question,
        config=types.GenerateContentConfig(**config_kwargs),
    )

    # Prefer .text; fall back to candidate parts; finally surface finish reason.
    text = resp.text if hasattr(resp, "text") else None
    if text and text.strip():
        return text.strip()

    try:
        c = resp.candidates[0]
        if c.content and c.content.parts:
            parts = [p.text for p in c.content.parts if getattr(p, "text", None)]
            joined = "\n".join(parts).strip()
            if joined:
                return joined
        finish = getattr(c, "finish_reason", "UNKNOWN")
        raise RuntimeError(f"Gemini returned no text (finish_reason={finish})")
    except (AttributeError, IndexError):
        raise RuntimeError("Gemini returned no content")


# ---------- routing ----------
def parse_routing(question: str) -> tuple[str, str]:
    """Return (target, cleaned_question). target in {'claude','gemini','both'}."""
    q = question.strip()
    low = q.lower()
    if low.startswith("@claude"):
        return "claude", q[len("@claude"):].strip(":,. ")
    if low.startswith("@gemini"):
        return "gemini", q[len("@gemini"):].strip(":,. ")
    return "both", q


# Backwards-compat shim (older Streamlit pages may still call ask())
def ask(question: str, context: str, lang: str) -> str:
    return ask_claude(question, context, lang).get("text", "")


EXAMPLE_QUESTIONS = {
    "en": [
        "Who is the biggest sleeper?",
        "What's France's best path to the final?",
        "Why does the model disagree with Polymarket on Argentina?",
        "Which team is most overvalued by the market?",
    ],
    "zh": [
        "谁是最大的黑马？",
        "法国进决赛最好的路线是什么？",
        "为什么模型和 Polymarket 对阿根廷的看法不同？",
        "哪支球队被市场高估最多？",
    ],
    "es": [
        "¿Quién es el mayor caballo oscuro?",
        "¿Cuál es el mejor camino de Francia a la final?",
        "¿Por qué el modelo discrepa con Polymarket sobre Argentina?",
        "¿Qué equipo está más sobrevalorado por el mercado?",
    ],
    "pt": [
        "Qual é a maior zebra?",
        "Qual o melhor caminho da França até a final?",
        "Por que o modelo discorda da Polymarket sobre a Argentina?",
        "Qual seleção está mais supervalorizada pelo mercado?",
    ],
    "fr": [
        "Qui est le plus grand outsider ?",
        "Quel est le meilleur chemin de la France vers la finale ?",
        "Pourquoi le modèle est-il en désaccord avec Polymarket sur l'Argentine ?",
        "Quelle équipe est la plus surcotée par le marché ?",
    ],
}
