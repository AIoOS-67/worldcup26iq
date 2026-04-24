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

import hashlib
import json
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
    lines.append("=== WorldCup26AI model data (updated Apr 2026) ===\n")
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
== App page guide (WorldCup26AI) ==
The user is on WorldCup26AI right now. The app has these pages, accessible via the left sidebar:
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
You are WorldCup26AI's **primary analyst** (nicknamed 'Claude'). Your job is to answer questions about the 2026 FIFA World Cup using the PROVIDED DATA ONLY. You never make up numbers.

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

# Team merch & shopping (agentic — always ask "for whom?" first if unclear)
You have two merch tools:

1. `check_team_merch_pricing(team, keywords?)` — look up live Fanatics pricing, sale status, promo codes. Returns product_name / list_price / sale_price / discount_pct / in_stock / manufacturer / _source.
2. `recommend_team_merch(team, pitch, keywords?)` — append a shoppable card (crest + product photo + SALE badge + deep link) below your written reply.

## THE ASK-FIRST RULE (critical)
When the user asks about **pricing, buying, or a jersey/gear recommendation** and has NOT specified who it's for (adult / kid / woman / family), your FIRST reply should be a short clarifying question — DO NOT call either tool yet. Example in Chinese:

> 🎽 想给谁买呀？
> - 👨 **男士款**（$80-150 球员版 / 复刻版）
> - 👩 **女士款**（女款剪裁）
> - 👦 **青少年款 / 儿童款**（$25-50）
> - 👨‍👩‍👧 **全家桶**（三件套，一家人一起穿）
>
> 告诉我方向，我给你报精准价格和链接。

Translate the four options into the user's UI language. Use emoji to keep it scannable. End with "告诉我方向" / "let me know" / equivalent. Do not try to answer with pricing in this turn.

## WHEN YOU CAN SKIP THE QUESTION
If the user's message already contains a clear audience signal, go straight to the tools:

| Signal in user text | Action |
|---|---|
| "给我自己" / "adult" / "my size" / "I want" / "男士" / "成人" / "自己穿" | call tools with `keywords="men's"` (or `"Men's <player>"`) |
| "给老婆" / "wife" / "women" / "女款" / "女士" / "her" | `keywords="women's"` (or `"Women's <player>"`) |
| "给娃" / "kids" / "son" / "daughter" / "青少年" / "儿童" / "孩子" | `keywords="youth"` |
| "全家" / "family" / "bundle" / "一家人" / "夫妻" | call `recommend_team_merch` TWO–THREE TIMES for the same team with `keywords="men's <player>"`, `keywords="women's <player>"`, `keywords="youth <player>"` — user will see one card per family member |
| Specific player name ("Messi" / "Pulisic" / "Mbappe" / "梅西" / "姆巴佩") | include the player name in `keywords` alongside whatever audience was specified; if audience is still unclear, ASK FIRST |

## CARD QUANTITY RULES
- Usually 0 cards per answer.
- 1 card when the user picked a specific audience + team.
- 2-3 cards only for explicit "family bundle" / comparison ("Argentina or Brazil?") asks.
- Never force a card into a purely analytical reply.

# TEAM NAME TABLE ({lang_name})
{team_table}

{app_guide}

# DATA
{context}
"""


GEMINI_SYSTEM = """\
You are WorldCup26AI's **second-opinion analyst** (nicknamed 'Gemini'). A peer analyst ('Claude') answers the same question in parallel, lean toward the raw model's view. Your job is to **add perspective**: if there's a reason the market (Polymarket) might be right and the model might be overfit or misleading, name it. Be the skeptical but constructive counter-voice.

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
        "Display a shoppable card below your answer with a Fanatics-licensed "
        "product (crest + image + deep link) for a WC26 team. Call this "
        "when showing the gear visuals genuinely adds to the conversation — "
        "fan affinity signals, 'who should I follow / support / root for' "
        "questions, dark-horse picks, or specific player/product asks.\n\n"
        "Do NOT call it for every team mentioned in passing. Usually 0 cards "
        "per response. At most one, or two if the user is comparing teams.\n\n"
        "If you've already called check_team_merch_pricing for this team "
        "during the current turn (ideally with matching `keywords`), the "
        "card will automatically pick up the same SKU and any SALE / promo "
        "code — you don't need to pass them here."
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
                    "A single short sentence in the response language explaining "
                    "WHY the user might want to check out this gear. Under 20 "
                    "words. No generic ad-speak."
                ),
            },
            "keywords": {
                "type": "string",
                "description": (
                    "Optional space-separated search terms to pick a SPECIFIC "
                    "product from the team's catalogue, instead of the default "
                    "cheapest-jersey fallback. Examples: 'Messi' for the Messi "
                    "signature jersey, 'authentic 2024 home' for the premium "
                    "authentic home kit, 'hat' for headwear, 'kids' for youth "
                    "sizing. Product names in the feed are rich (player, year, "
                    "gender, replica/authentic, home/away), so passing even 1–2 "
                    "words dramatically sharpens the recommendation. Leave "
                    "empty when the user hasn't specified anything."
                ),
            },
        },
        "required": ["team", "pitch"],
    },
}


PRICING_TOOL = {
    "name": "check_team_merch_pricing",
    "description": (
        "Look up current listed price, any active sale, and promo codes for "
        "a team's Fanatics product. Returns {list_price, sale_price?, "
        "discount_pct?, product_name, manufacturer, in_stock, currency, "
        "_source}.\n\n"
        "Call this when:\n"
        "- The user shows price sensitivity ('cheap', 'expensive', 'deal', "
        "'on sale', 'discount', '便宜', '贵', '折扣')\n"
        "- The user explicitly asks about buying / ordering / price\n"
        "- The user asks about a specific player's jersey (Messi, Mbappe, "
        "Pulisic, etc.) — pass the player name as `keywords` so the feed "
        "returns the player's signature jersey, not a generic team item\n"
        "- You're about to call recommend_team_merch AND want to anchor "
        "the pitch with a concrete price or active discount\n\n"
        "After calling this, incorporate any meaningful discount naturally "
        "into your written reply (don't just dump the JSON). Trust your "
        "judgment on whether to mention price at all.\n\n"
        "Do NOT call this more than once per (team, keywords) pair per turn."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "team": {
                "type": "string",
                "enum": WC26_TEAMS,
                "description": "The team whose product pricing to look up.",
            },
            "keywords": {
                "type": "string",
                "description": (
                    "Optional space-separated search terms to pick a SPECIFIC "
                    "product. Use when the user asked about a named player "
                    "('Messi', 'Mbappe'), a specific gear type ('hat', "
                    "'scarf', 'ball'), or quality tier ('authentic', "
                    "'replica'). Leave empty for a generic team-jersey lookup."
                ),
            },
        },
        "required": ["team"],
    },
}


_FANATICS_FEED_PATH = Path(__file__).resolve().parent / "data" / "fanatics_products.parquet"
_FANATICS_DF_CACHE: pd.DataFrame | None = None


def _fanatics_products() -> pd.DataFrame:
    """Lazy-load and cache the filtered Fanatics feed parquet. Self-contained
    inside ask_model.py — avoids a fragile cross-module import back into
    wc26_app.py at tool-call time (which was failing silently in production
    and forcing every pricing lookup onto the mock fallback)."""
    global _FANATICS_DF_CACHE
    if _FANATICS_DF_CACHE is not None:
        return _FANATICS_DF_CACHE
    if not _FANATICS_FEED_PATH.exists():
        _FANATICS_DF_CACHE = pd.DataFrame()
        return _FANATICS_DF_CACHE
    try:
        _FANATICS_DF_CACHE = pd.read_parquet(_FANATICS_FEED_PATH)
    except Exception:
        _FANATICS_DF_CACHE = pd.DataFrame()
    return _FANATICS_DF_CACHE


_ADULT_AGE_TAGS = {"Men's", "Adult", "Women's"}
_YOUTH_AGE_TAGS = {"Youth", "Boys'", "Girls'", "Toddler", "Infant"}

_AUDIENCE_HINTS = [
    # (keyword-substring, target age_groups set)
    ("women",   {"Women's"}),
    ("ladies",  {"Women's"}),
    ("female",  {"Women's"}),
    ("men",     {"Men's", "Adult"}),
    ("male",    {"Men's", "Adult"}),
    ("adult",   {"Men's", "Adult"}),
    ("kid",     _YOUTH_AGE_TAGS),
    ("kids",    _YOUTH_AGE_TAGS),
    ("youth",   _YOUTH_AGE_TAGS),
    ("child",   _YOUTH_AGE_TAGS),
    ("boy",     {"Boys'", "Youth", "Toddler"}),
    ("girl",    {"Girls'", "Youth", "Toddler"}),
    ("toddler", {"Toddler", "Infant"}),
    ("infant",  {"Infant", "Toddler"}),
    ("baby",    {"Infant", "Toddler"}),
]


def _preferred_age_groups(keywords_lower: str) -> set[str]:
    """Infer which age_group labels to prefer based on the user's keywords.
    Returns the target set or the default adult set when no hint is present.
    """
    for needle, targets in _AUDIENCE_HINTS:
        if needle in keywords_lower:
            return targets
    # Default: adult. Fixes the "Messi jersey → Youth T-shirt" surprise where
    # the cheapest-matching item happened to be a kids' tee.
    return {"Men's", "Adult"}


def _pick_team_product(team: str, keywords: str = "") -> dict | None:
    """Return the best-ranked product row for `team` as a dict, or None if no
    inventory.

    Ranking (in order):
    1. In-stock first
    2. Keyword match count (more matches = higher)
    3. Preferred age_group (Adult by default; kids / women's when the
       user's keywords signal that audience)
    4. On-sale
    5. Keyword 'jersey' in name (only used when caller passed no keywords)
    6. Lowest price

    If NO product matches ANY keyword term, silently fall back to the
    default jersey/adult ranking so the tool never returns None when the
    team has any inventory at all.
    """
    df = _fanatics_products()
    if df.empty:
        return None
    g = df[df["team"] == team]
    if g.empty:
        return None
    names_lower = g["name"].str.lower().fillna("")

    kw_lower = keywords.lower()
    preferred_ages = _preferred_age_groups(kw_lower)
    age_rank = (~g["age_group"].isin(preferred_ages)).astype(int)

    terms = [w for w in kw_lower.split() if w]
    if terms:
        match_count = sum(names_lower.str.contains(t, regex=False).astype(int) for t in terms)
        if match_count.max() == 0:
            # No product matched any term — fall back to default jersey ranking
            g = g.assign(
                _instock=(~g["in_stock"]).astype(int),
                _miss=1,
                _age=age_rank,
                _notkw=(~names_lower.str.contains("jersey")).astype(int),
                _notsale=(~g["on_sale"]).astype(int),
            )
        else:
            g = g.assign(
                _instock=(~g["in_stock"]).astype(int),
                _miss=-match_count,
                _age=age_rank,
                _notkw=0,
                _notsale=(~g["on_sale"]).astype(int),
            )
    else:
        g = g.assign(
            _instock=(~g["in_stock"]).astype(int),
            _miss=0,
            _age=age_rank,
            _notkw=(~names_lower.str.contains("jersey")).astype(int),
            _notsale=(~g["on_sale"]).astype(int),
        )

    g = g.sort_values(["_instock", "_miss", "_age", "_notkw", "_notsale", "price"])
    return g.drop(columns=["_instock", "_miss", "_age", "_notkw", "_notsale"]).iloc[0].to_dict()


def _real_pricing(team: str, keywords: str = "") -> dict | None:
    """Live pricing from the filtered Fanatics product feed (Impact-approved
    Apr 23 2026). Returns None when the team has no inventory — caller falls
    back to _mock_pricing() so the tool always returns something usable."""
    product = _pick_team_product(team, keywords=keywords)
    if not product:
        return None

    price = float(product.get("price") or 0)
    list_price = float(product.get("list_price") or price)
    on_sale = bool(product.get("on_sale")) and list_price > price > 0
    discount_pct = int(product.get("discount_pct") or 0)

    out: dict = {
        "team": team,
        "currency": "USD",
        "list_price": round(list_price, 2) if list_price else price,
        "product_name": product.get("name"),
        "manufacturer": product.get("manufacturer"),
        "age_group": product.get("age_group"),
        "gender": product.get("gender"),
        "in_stock": bool(product.get("in_stock")),
    }
    if on_sale:
        out["sale_price"] = round(price, 2)
        out["discount_pct"] = discount_pct
    out["_source"] = "Fanatics Global product feed (Impact.com)"
    return out


def _mock_pricing(team: str) -> dict:
    """Deterministic fallback pricing for teams with no Fanatics inventory
    (~11 of 48 WC26 sides). Seeded from the team name so the same team always
    reports the same price / sale profile across sessions."""
    h = int(hashlib.sha256(team.encode("utf-8")).hexdigest()[:10], 16)
    base = 89 + (h % 41)  # $89 – $129
    is_sale = (h % 10) < 3  # 30% of teams
    has_promo = ((h >> 8) % 10) < 2  # 20% of teams

    out: dict = {
        "team": team,
        "currency": "USD",
        "list_price": base,
    }
    if is_sale:
        discount = 15 + ((h >> 4) % 20)  # 15–34% off
        sale = round(base * (100 - discount) / 100)
        out["sale_price"] = sale
        out["discount_pct"] = discount
        out["sale_ends_hours"] = 12 + ((h >> 12) % 36)  # 12–47h countdown
    if has_promo:
        out["promo_code"] = "WC26OFF"
        out["promo_discount_pct"] = 10
    out["_note"] = (
        "Fallback pricing — this team has no current inventory in the "
        "Fanatics Global feed, so this is a seeded estimate."
    )
    return out


def _lookup_pricing(team: str, keywords: str = "") -> dict:
    """Unified pricing resolver: real feed first, mock fallback."""
    real = _real_pricing(team, keywords=keywords)
    return real if real else _mock_pricing(team)


# ---------- Claude ----------
MAX_AGENT_STEPS = 4  # safety bound on tool-use rounds per user turn


def ask_claude(question: str, context: str, lang: str) -> dict:
    """Return {'text': str, 'merch': list[{team, pitch, pricing?}]}.

    Runs an agentic loop: Claude may call check_team_merch_pricing
    (receiving real tool_result data back so it can reason with the
    numbers on the NEXT turn), and/or recommend_team_merch (a display-
    only marker that ends up on the rendered card). The loop terminates
    when Claude stops emitting tool_use blocks OR MAX_AGENT_STEPS is hit.

    Pricing looked up for a team during this turn is automatically merged
    into any merch rec for the same team — so the rendered card shows
    SALE badge / promo code without Claude having to pass the fields
    twice.
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

    messages: list[dict] = [{"role": "user", "content": question}]
    text_parts: list[str] = []
    merch_recs: list[dict] = []
    # Keyed by (team, keywords) so Claude can look up both a generic and a
    # player-specific SKU for the same team in a single turn if needed.
    pricing_cache: dict[tuple[str, str], dict] = {}

    for _ in range(MAX_AGENT_STEPS):
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            system=[
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}},
            ],
            tools=[MERCH_TOOL, PRICING_TOOL],
            messages=messages,
        )

        tool_uses = []
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "tool_use":
                tool_uses.append(block)

        # No tools called → Claude is done.
        if resp.stop_reason != "tool_use" or not tool_uses:
            break

        # Record assistant turn (pass content blocks back verbatim).
        messages.append({"role": "assistant", "content": resp.content})

        # Execute each tool, collect results.
        tool_results = []
        for tu in tool_uses:
            name = getattr(tu, "name", "")
            inp = getattr(tu, "input", None) or {}
            team = inp.get("team")

            if name == "check_team_merch_pricing" and team in WC26_TEAMS:
                keywords = (inp.get("keywords") or "").strip()
                pricing = _lookup_pricing(team, keywords=keywords)
                pricing_cache[(team, keywords)] = pricing
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": json.dumps(pricing),
                })
            elif name == "recommend_team_merch" and team in WC26_TEAMS:
                pitch = inp.get("pitch", "")
                keywords = (inp.get("keywords") or "").strip()
                merch_recs.append({"team": team, "pitch": pitch, "keywords": keywords})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": json.dumps({"status": "card_will_be_displayed"}),
                })
            else:
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": json.dumps({"error": f"unknown tool or invalid team: {name}"}),
                    "is_error": True,
                })

        messages.append({"role": "user", "content": tool_results})

    # Attach pricing (+ the specific product we'd link to) to every merch
    # card. Prefer a pricing call from this turn with matching keywords;
    # otherwise do a direct feed lookup so the card's price, SKU, image,
    # and link all reflect the player/product Claude actually picked.
    for rec in merch_recs:
        team = rec["team"]
        kw = rec.get("keywords", "")
        p = pricing_cache.get((team, kw)) or pricing_cache.get((team, ""))
        if not p:
            p = _lookup_pricing(team, keywords=kw)
        if p:
            rec["pricing"] = p
        # Also surface the concrete SKU dict (image + link + age) for the
        # render. age_group drives the small "👨 Men's / 👦 Youth" pill on
        # the card so users don't click an adult item expecting kids sizing
        # or vice versa.
        product = _pick_team_product(team, keywords=kw)
        if product:
            rec["product"] = {
                "name": product.get("name"),
                "price": product.get("price"),
                "image_url": product.get("image_url"),
                "link": product.get("link"),
                "age_group": product.get("age_group"),
                "gender": product.get("gender"),
            }

    return {
        "text": "".join(text_parts) or "(empty response)",
        "merch": merch_recs,
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

# Keywords that signal commerce / shopping / pricing intent. When the user
# asks a question matching these AND hasn't explicitly @mentioned gemini,
# the question is routed silently to Claude only — Gemini has no merch
# tools and would just have to decline, wasting the user's attention.
_COMMERCE_KW = [
    # English
    "buy", "buying", "purchase", "shop", "shopping", "shoppable",
    "price", "pricing", "cost", "expensive", "cheap",
    "deal", "deals", "discount", "sale", "promo", "coupon", "offer",
    "jersey", "jerseys", "shirt", "gear", "merch", "merchandise",
    "order", "checkout", "cart",
    # Chinese
    "买", "购买", "下单", "订购", "价钱", "价格", "多少钱",
    "贵", "便宜", "打折", "折扣", "促销", "优惠", "代码",
    "球衣", "球衫", "装备", "周边", "商品", "订单",
    # Spanish
    "comprar", "compra", "precio", "barato", "oferta", "descuento",
    "camiseta",
    # Portuguese
    "preço", "barato", "caro", "desconto", "camisa",
    # French
    "acheter", "achat", "prix", "cher", "offre", "réduction", "maillot",
]


def _is_commerce_question(q: str) -> bool:
    ql = q.lower()
    return any(kw in ql for kw in _COMMERCE_KW)


def parse_routing(question: str) -> tuple[str, str]:
    """Return (target, cleaned_question). target in {'claude','gemini','both'}.

    Routing rules:
    - Explicit @claude / @gemini prefix → forced target (user knows what they want).
    - Otherwise default 'both' — unless the question is a commerce /
      shopping / pricing query, in which case target collapses to 'claude'
      silently. Only Claude carries the merch + pricing tools; Gemini
      would just have to decline, which clutters the UX.
    """
    q = question.strip()
    low = q.lower()
    if low.startswith("@claude"):
        return "claude", q[len("@claude"):].strip(":,. ")
    if low.startswith("@gemini"):
        return "gemini", q[len("@gemini"):].strip(":,. ")
    if _is_commerce_question(q):
        return "claude", q
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
