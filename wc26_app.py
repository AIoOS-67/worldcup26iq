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
from i18n import t, set_language_from_sidebar, LANGUAGES, translate_reason  # noqa: E402


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

CUSTOM_CSS = """
<style>
  /* hide Streamlit chrome */
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }

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


# ---------- sidebar ----------
set_language_from_sidebar()
st.sidebar.markdown(f"# {t('app_title')}")
st.sidebar.caption(t("app_tagline"))
PAGE_KEYS = [
    ("hero",   t("nav_hero")),
    ("champ",  t("nav_champ")),
    ("misp",   t("nav_misp")),
    ("whatif", t("nav_whatif")),
    ("ask",    t("nav_ask")),
    ("stage",  t("nav_stage")),
    ("calib",  t("nav_calib")),
    ("method", t("nav_method")),
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
                       over_team=best_over["team"],
                       under_pp=f"{best_under['edge']*100:+.0f}",
                       under_flag=flag(best_under["team"]),
                       under_team=best_under["team"])
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
            <div class="value">{flag(best_under['team'])} {best_under['team']}</div>
            <div class="delta up">{best_under['edge']*100:+.1f} pp · {t('kpi_mkt_model', mkt=f"{best_under['market_p_W']*100:.1f}", model=f"{best_under['p_W']*100:.1f}")}</div>
          </div>
          <div class="kpi">
            <div class="label">{t('kpi_biggest_over')}</div>
            <div class="value">{flag(best_over['team'])} {best_over['team']}</div>
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
                  <div class="value">{flag(row['team'])} {row['team']}</div>
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
            f'<div class="team">{flag(r["team"])} {r["team"]}</div>'
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


# ---------- Champion ----------
elif page_id == "champ":
    probs = load_probs()
    st.markdown(f'<div class="section-title">{t("champ_title")}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="section-caption">{t("champ_caption", n=len(probs))}</p>',
        unsafe_allow_html=True,
    )

    top = probs.sort_values("p_W", ascending=False).head(20).copy()
    top["label"] = top["team"].apply(with_flag)

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
        show["team"] = show["team"].apply(with_flag)
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
        k2.metric(f"{t('kpi_biggest_under')}: {flag(best_under['team'])} {best_under['team']}",
                  f"{best_under['edge']*100:+.1f} pp",
                  t("kpi_mkt_model", mkt=f"{best_under['market_p_W']*100:.1f}", model=f"{best_under['p_W']*100:.1f}"))
    if best_over is not None:
        k3.metric(f"{t('kpi_biggest_over')}: {flag(best_over['team'])} {best_over['team']}",
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
            f'<div class="team">{flag(r["team"])} {r["team"]}</div>'
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

    group_keys = list(groups.keys())
    new_locks = {}
    for i in range(0, len(group_keys), 4):
        cols = st.columns(4)
        for j, gkey in enumerate(group_keys[i:i + 4]):
            with cols[j]:
                teams = [t for t in groups[gkey] if t in dc["team_index"]]
                st.markdown(f"**{gkey}**")
                for t in groups[gkey]:
                    st.caption(f"{flag(t)} {t}")
                auto = t("whatif_auto")
                w_key = f"lock_{gkey}_1st"
                w = st.selectbox(t("whatif_winner"), [auto] + teams,
                                 key=w_key, label_visibility="collapsed",
                                 format_func=lambda x, _auto=auto: x if x == _auto else f"🏆 {flag(x)} {x}")
                r_options = [auto] + [tm for tm in teams if tm != w]
                r_key = f"lock_{gkey}_2nd"
                if st.session_state.get(r_key) == w and w != auto:
                    st.session_state[r_key] = auto
                r = st.selectbox(t("whatif_runner"), r_options,
                                 key=r_key, label_visibility="collapsed",
                                 format_func=lambda x, _auto=auto: x if x == _auto else f"🥈 {flag(x)} {x}")
                lk = {}
                if w != auto:
                    lk["1st"] = w
                if r != auto:
                    lk["2nd"] = r
                if lk:
                    new_locks[gkey] = lk

    st.markdown("---")
    c1, c2, c3 = st.columns([1, 1, 3])
    n_sims = c1.selectbox(t("whatif_sims"), [500, 1000, 2000, 5000], index=1)
    run = c2.button(t("whatif_run"), type="primary", use_container_width=True)
    if c3.button(t("whatif_reset")):
        for k in list(st.session_state.keys()):
            if k.startswith("lock_"):
                del st.session_state[k]
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

        top = res.head(15).copy()
        top["label"] = top["team"].apply(with_flag)
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
        view["team"] = view["team"].apply(with_flag)
        st.dataframe(
            view[["team", "baseline_p_W", "p_W", "delta"]].style.format({
                "baseline_p_W": "{:.1%}", "p_W": "{:.1%}", "delta": "{:+.1%}"
            }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(t("whatif_hint"))


# ---------- Ask the Model (placeholder — Challenge B) ----------
elif page_id == "ask":
    st.markdown(f'<div class="section-title">{t("ask_title")}</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="section-caption">{t("ask_caption")}</p>', unsafe_allow_html=True)
    st.info("⚒️ Claude-powered natural language interface — Challenge B is under construction. Coming next.")


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
                             format_func=lambda x: f"{flag(x)} {x}")
    if chosen:
        sub = probs[probs["team"].isin(chosen)].set_index("team")[cols]
        sub.columns = nice
        long = sub.reset_index().melt(id_vars="team", var_name="Stage", value_name="P(reach)")
        long["team"] = long["team"].apply(with_flag)
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
        pv["home"] = pv["home"].apply(with_flag)
        pv["away"] = pv["away"].apply(with_flag)
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
