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
) -> str:
    """Compact textual context for the LLMs. ~1500 tokens."""
    lines = []
    lines.append("=== WorldCup26IQ model data (updated Apr 2026) ===\n")
    lines.append("Model: Dixon-Coles bivariate Poisson, home_adv=0.21, rho=-0.095,")
    lines.append("  fit on ~7K internationals since 2019, 10,000 Monte Carlo runs.")
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
    return "\n".join(lines)


FLAGS_ASCII = {
    "Argentina": "🇦🇷", "Brazil": "🇧🇷", "France": "🇫🇷", "Spain": "🇪🇸",
    "Colombia": "🇨🇴", "Ecuador": "🇪🇨", "Morocco": "🇲🇦", "Japan": "🇯🇵",
    "Paraguay": "🇵🇾", "Uruguay": "🇺🇾", "Portugal": "🇵🇹", "Germany": "🇩🇪",
    "Netherlands": "🇳🇱", "Mexico": "🇲🇽", "United States": "🇺🇸", "Canada": "🇨🇦",
    "Saudi Arabia": "🇸🇦", "Senegal": "🇸🇳", "Australia": "🇦🇺", "Norway": "🇳🇴",
    "Switzerland": "🇨🇭", "Algeria": "🇩🇿", "Croatia": "🇭🇷", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "Iran": "🇮🇷", "Ghana": "🇬🇭", "South Korea": "🇰🇷", "Ivory Coast": "🇨🇮",
    "DR Congo": "🇨🇩", "Cape Verde": "🇨🇻", "Tunisia": "🇹🇳", "Egypt": "🇪🇬",
    "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "Sweden": "🇸🇪", "Belgium": "🇧🇪", "Austria": "🇦🇹",
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

# Rules
- Answer in {lang_name}. Keep it under 180 words unless the user asks for detail.
- Ground every probability claim in the provided data. Cite the relevant number.
- When mentioning a team, use the localized name from the TEAM NAME TABLE below (not the English name), and prefix with the flag emoji. Example in {lang_name}: "{example_team}".
- If the user asks a what-if question, explain what the model currently says AND tell them the What-If page simulates it live.
- If the team is not in the data, say so plainly.
- Lean toward the model's quantitative view — that's your role.

# TEAM NAME TABLE ({lang_name})
{team_table}

# DATA
{context}
"""


GEMINI_SYSTEM = """\
You are WorldCup26IQ's **second-opinion analyst** (nicknamed 'Gemini'). A peer analyst ('Claude') answers the same question in parallel, lean toward the raw model's view. Your job is to **add perspective**: if there's a reason the market (Polymarket) might be right and the model might be overfit or misleading, name it. Be the skeptical but constructive counter-voice.

# Rules
- Answer in {lang_name}. Keep it under 150 words — tighter than Claude.
- Ground claims in the PROVIDED DATA. Never invent numbers.
- When mentioning a team, use the localized name from the TEAM NAME TABLE and prefix with the flag emoji.
- Identify any weaknesses in the model's answer: e.g., CONMEBOL bias, injury blindness, small-sample, format novelty (48-team new format), home-advantage assumptions.
- If you fundamentally agree with Claude, say so and add one supporting angle — don't fabricate disagreement.
- End with ONE concrete action the user could take on the app (e.g., "try locking X in What-If to test this").

# TEAM NAME TABLE ({lang_name})
{team_table}

# DATA
{context}
"""


# ---------- Claude ----------
def ask_claude(question: str, context: str, lang: str) -> str:
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
    )
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=[
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}},
        ],
        messages=[{"role": "user", "content": question}],
    )
    parts = []
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts) or "(empty response)"


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
    return ask_claude(question, context, lang)


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
