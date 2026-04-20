"""
Ask the Model — Claude-powered natural language interface for WorldCup26IQ.

Design:
  - Build a compact textual context of all precomputed model outputs (champion
    probs, stage-reach probs, mispricing leaderboard, calibration metrics).
  - Send that context + the user's question to Claude.
  - Claude answers in the user's preferred language, grounded in the data.

The API key is read from Streamlit secrets (`ANTHROPIC_API_KEY`). On Cloud, set
it via App settings → Secrets.
"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

MODEL = "claude-opus-4-7"
MAX_TOKENS = 1200


LANG_NAME = {
    "en": "English",
    "zh": "Simplified Chinese (简体中文)",
    "es": "Spanish (Español)",
    "pt": "Portuguese (Português)",
    "fr": "French (Français)",
}


def _get_api_key() -> str | None:
    try:
        return st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY")


@st.cache_data
def build_data_context(
    probs: pd.DataFrame,
    leaderboard: pd.DataFrame,
    backtest_summary: pd.DataFrame,
) -> str:
    """Compact textual context for the LLM. ~1500 tokens."""
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


SYSTEM_PROMPT_TEMPLATE = """\
You are WorldCup26IQ's analyst. Your job is to answer questions about the 2026 FIFA World Cup using the PROVIDED DATA ONLY. You never make up numbers.

# Rules
- Answer in {lang_name}. Keep it under 180 words unless the user asks for detail.
- Ground every probability claim in the provided data. Cite the relevant number.
- If the user asks a what-if or conditional question (e.g., "what if Saudi Arabia wins Group F"), explain what the model currently says AND tell them the What-If page lets them simulate it live.
- If the user asks about a team not in the data, say so.
- When you mention a team, prefix with its flag emoji if obvious (e.g., 🇦🇷 Argentina, 🇫🇷 France, 🇧🇷 Brazil, 🇪🇸 Spain, 🇬🇧 England, 🇵🇹 Portugal, 🇨🇴 Colombia, 🇪🇨 Ecuador, 🇯🇵 Japan, 🇲🇦 Morocco, 🇺🇾 Uruguay, 🇵🇾 Paraguay, 🇩🇪 Germany, 🇳🇱 Netherlands, 🇲🇽 Mexico, 🇺🇸 United States, 🇨🇦 Canada, 🇸🇦 Saudi Arabia, 🇸🇳 Senegal, 🇦🇺 Australia, 🇳🇴 Norway, 🇨🇭 Switzerland, 🇩🇿 Algeria, 🇭🇷 Croatia).
- Be specific: if the user asks about a "sleeper", name 2-3 teams with evidence from the data.
- Prefer insights over generic prose. "Argentina is 17pp undervalued vs Polymarket" beats "Argentina looks strong."

# DATA
{context}
"""


def ask(question: str, context: str, lang: str) -> str:
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("NO_API_KEY")
    from anthropic import Anthropic  # lazy import so missing dep doesn't break app
    client = Anthropic(api_key=api_key)
    system = SYSTEM_PROMPT_TEMPLATE.format(
        lang_name=LANG_NAME.get(lang, "English"),
        context=context,
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=[
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}},
        ],
        messages=[{"role": "user", "content": question}],
    )
    # Collect text from response
    parts = []
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts) or "(empty response)"


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
