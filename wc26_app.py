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


# ---------- sidebar ----------
st.sidebar.markdown("# ⚽ WorldCup26IQ")
st.sidebar.caption("Dixon-Coles + Monte Carlo for the 2026 FIFA World Cup.")
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Hero",
        "🏆 Champion Probabilities",
        "💸 Mispricing Leaderboard",
        "📊 Stage Reach",
        "📏 Calibration (Backtest)",
        "📖 Methodology",
    ],
    index=0,
)
st.sidebar.markdown("---")
st.sidebar.caption("49K internationals since 1872 · Polymarket live odds · 10K Monte Carlo runs.")


# ---------- HERO / landing ----------
if page.startswith("🏠"):
    lb = load_leaderboard()
    probs = load_probs()
    summary = load_backtest_summary()
    all_dc = summary[(summary["year"] == "ALL") & (summary["model"] == "DixonColes")].iloc[0]
    all_base = summary[(summary["year"] == "ALL") & (summary["model"] == "Uniform(1/3)")].iloc[0]
    skill = 1 - all_dc["brier"] / all_base["brier"]

    best_under = lb[lb["direction"] == "UNDER"].iloc[0]
    best_over = lb[lb["direction"] == "OVER"].iloc[0]

    headline = (
        f"Polymarket is <b>{best_over['edge']*100:.0f}% wrong</b> about "
        f"{flag(best_over['team'])} {best_over['team']}. "
        f"And <b>{best_under['edge']*100:+.0f}% wrong</b> about "
        f"{flag(best_under['team'])} {best_under['team']}."
    )
    st.markdown(
        f"""
        <div class="hero">
          <h1>⚽ WorldCup26IQ</h1>
          <p class="subtitle">{headline}</p>
          <p class="subtitle" style="margin-top:8px;">
            A Dixon-Coles bivariate Poisson model, 10,000 Monte Carlo tournament runs,
            compared live against $700M in Polymarket winner markets.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="kpi-grid">
          <div class="kpi">
            <div class="label">Biggest UNDER</div>
            <div class="value">{flag(best_under['team'])} {best_under['team']}</div>
            <div class="delta up">{best_under['edge']*100:+.1f} pp · mkt {best_under['market_p_W']*100:.1f}% → model {best_under['p_W']*100:.1f}%</div>
          </div>
          <div class="kpi">
            <div class="label">Biggest OVER</div>
            <div class="value">{flag(best_over['team'])} {best_over['team']}</div>
            <div class="delta down">{best_over['edge']*100:+.1f} pp · mkt {best_over['market_p_W']*100:.1f}% → model {best_over['p_W']*100:.1f}%</div>
          </div>
          <div class="kpi">
            <div class="label">Model Brier skill</div>
            <div class="value">{skill*100:+.1f}%</div>
            <div class="delta muted">vs uniform baseline on WC 2018 + 2022 (128 matches)</div>
          </div>
          <div class="kpi">
            <div class="label">Market coverage</div>
            <div class="value">{int(lb['team'].nunique())} / 48</div>
            <div class="delta muted">≥ $500K Polymarket liquidity</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top 5 model champions
    top = probs.sort_values("p_W", ascending=False).head(5).reset_index(drop=True)
    st.markdown('<div class="section-title">🏆 Model top-5 favourites</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    for i, row in top.iterrows():
        with cols[i]:
            st.markdown(
                f"""
                <div class="kpi">
                  <div class="label">#{i+1}</div>
                  <div class="value">{flag(row['team'])} {row['team']}</div>
                  <div class="delta muted">{row['p_W']*100:.1f}% champion</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Mispricing top 5
    st.markdown('<div class="section-title">💸 Top 5 mispricings (ranked by |edge| × √liquidity)</div>', unsafe_allow_html=True)
    top_lb = lb.head(5)
    rows_html = '<div class="mis-row header"><div>#</div><div>Team</div><div>Direction</div><div>Model</div><div>Market</div><div>Why</div></div>'
    for _, r in top_lb.iterrows():
        dir_class = "dir-under" if r["direction"] == "UNDER" else "dir-over"
        dir_label = "⬆ UNDER" if r["direction"] == "UNDER" else "⬇ OVER"
        edge_pp = f"{r['edge']*100:+.1f}pp"
        rows_html += (
            f'<div class="mis-row">'
            f'<div class="rank">#{int(r["rank"])}</div>'
            f'<div class="team">{flag(r["team"])} {r["team"]}</div>'
            f'<div class="{dir_class}">{dir_label} {edge_pp}</div>'
            f'<div>{r["p_W"]*100:.1f}%</div>'
            f'<div>{r["market_p_W"]*100:.1f}%</div>'
            f'<div class="reason">{r["reason"]}</div>'
            f"</div>"
        )
    st.markdown(rows_html, unsafe_allow_html=True)

    st.markdown(
        '<p class="section-caption" style="margin-top:18px;">'
        "Use the sidebar to explore the full model, calibration evidence, or the live market gap."
        "</p>",
        unsafe_allow_html=True,
    )


# ---------- Champion ----------
elif page.startswith("🏆"):
    probs = load_probs()
    st.markdown('<div class="section-title">🏆 Champion Probabilities</div>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="section-caption">From {len(probs)} teams across 10,000 Monte Carlo tournament runs.</p>',
        unsafe_allow_html=True,
    )

    top = probs.sort_values("p_W", ascending=False).head(20).copy()
    top["label"] = top["team"].apply(with_flag)

    fig = px.bar(
        top.sort_values("p_W"),
        x="p_W", y="label", orientation="h",
        labels={"p_W": "P(Win)", "label": ""},
    )
    fig.update_traces(marker_color="#f7c948")
    fig.update_layout(
        height=620,
        margin=dict(l=140, r=40, t=30, b=40),
        paper_bgcolor="#0b1220",
        plot_bgcolor="#121c2e",
        font=dict(color="#e8edf7"),
        xaxis=dict(gridcolor="#1b2742"),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Full table (all 48 teams)"):
        show = probs.copy()
        show["team"] = show["team"].apply(with_flag)
        st.dataframe(
            show[["team", "p_R32", "p_R16", "p_QF", "p_SF", "p_F", "p_W"]]
            .style.format({c: "{:.1%}" for c in ["p_R32", "p_R16", "p_QF", "p_SF", "p_F", "p_W"]}),
            use_container_width=True,
        )


# ---------- Mispricing Leaderboard ----------
elif page.startswith("💸"):
    lb = load_leaderboard()
    edges = load_edges()
    st.markdown('<div class="section-title">💸 Mispricing Leaderboard</div>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="section-caption">Ranked by mispricing score = |edge| × √(liquidity / $1M). '
        f'Filtered to {len(lb)} markets with ≥$500K liquidity.</p>',
        unsafe_allow_html=True,
    )

    under = lb[lb["direction"] == "UNDER"]
    over = lb[lb["direction"] == "OVER"]
    best_under = under.iloc[0] if len(under) else None
    best_over = over.iloc[0] if len(over) else None

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Markets analysed", len(lb))
    if best_under is not None:
        k2.metric(f"Biggest UNDER: {flag(best_under['team'])} {best_under['team']}",
                  f"{best_under['edge']*100:+.1f} pp",
                  f"mkt {best_under['market_p_W']*100:.1f}% → model {best_under['p_W']*100:.1f}%")
    if best_over is not None:
        k3.metric(f"Biggest OVER: {flag(best_over['team'])} {best_over['team']}",
                  f"{best_over['edge']*100:+.1f} pp",
                  f"mkt {best_over['market_p_W']*100:.1f}% → model {best_over['p_W']*100:.1f}%")
    k4.metric("Total WC-2026 liquidity", f"${lb['liquidity'].sum()/1e6:.1f}M")

    # Custom rows
    rows_html = '<div class="mis-row header"><div>Rank</div><div>Team</div><div>Direction</div><div>Model</div><div>Market</div><div>Reason (last 20 internationals)</div></div>'
    for _, r in lb.iterrows():
        dir_class = "dir-under" if r["direction"] == "UNDER" else "dir-over"
        dir_label = f"⬆ UNDER {r['edge']*100:+.1f}pp" if r["direction"] == "UNDER" else f"⬇ OVER {r['edge']*100:+.1f}pp"
        rows_html += (
            f'<div class="mis-row">'
            f'<div class="rank">#{int(r["rank"])}</div>'
            f'<div class="team">{flag(r["team"])} {r["team"]}</div>'
            f'<div class="{dir_class}">{dir_label}</div>'
            f'<div>{r["p_W"]*100:.1f}%</div>'
            f'<div>{r["market_p_W"]*100:.1f}%</div>'
            f'<div class="reason">{r["reason"]}</div>'
            f"</div>"
        )
    st.markdown(rows_html, unsafe_allow_html=True)

    st.markdown('<div class="section-title" style="margin-top:24px;">Model vs Market — all 43 covered teams</div>', unsafe_allow_html=True)
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


# ---------- Stage Reach ----------
elif page.startswith("📊"):
    probs = load_probs()
    st.markdown('<div class="section-title">📊 Stage-Reach Probabilities</div>', unsafe_allow_html=True)
    cols = ["p_R32", "p_R16", "p_QF", "p_SF", "p_F", "p_W"]
    nice = ["R32", "R16", "QF", "SF", "Final", "Win"]
    teams = probs["team"].tolist()
    # Default to top-6 by champion prob
    default = probs.sort_values("p_W", ascending=False).head(6)["team"].tolist()
    chosen = st.multiselect("Teams to compare", teams, default=default,
                             format_func=lambda t: f"{flag(t)} {t}")
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
elif page.startswith("📏"):
    st.markdown('<div class="section-title">📏 Calibration — How Honest Is the Model?</div>', unsafe_allow_html=True)
    summary = load_backtest_summary()
    rel = load_backtest_reliability()
    preds = load_backtest_predictions()
    st.markdown(
        '<p class="section-caption">We refit strictly on pre-tournament data and scored every match of '
        'the 2018 + 2022 World Cups. Brier skill score compares against a uniform (1/3) baseline.</p>',
        unsafe_allow_html=True,
    )
    all_dc = summary[(summary["year"] == "ALL") & (summary["model"] == "DixonColes")].iloc[0]
    all_base = summary[(summary["year"] == "ALL") & (summary["model"] == "Uniform(1/3)")].iloc[0]
    skill = 1 - all_dc["brier"] / all_base["brier"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Brier (DC)", f"{all_dc['brier']:.4f}")
    m2.metric("Brier (Uniform)", f"{all_base['brier']:.4f}")
    m3.metric("LogLoss (DC)", f"{all_dc['logloss']:.4f}")
    m4.metric("Brier skill vs uniform", f"{skill*100:+.1f}%")

    st.markdown('<div class="section-title" style="margin-top:18px;">By tournament</div>', unsafe_allow_html=True)
    pivot = summary.pivot_table(index="year", columns="model", values=["brier", "logloss"])
    st.dataframe(pivot.style.format("{:.4f}"), use_container_width=True)

    st.markdown('<div class="section-title" style="margin-top:18px;">Reliability Diagram (pooled predictions)</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-caption">Dots on the diagonal = perfectly calibrated. Dot size = predictions in bucket.</p>',
        unsafe_allow_html=True,
    )
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

    with st.expander(f"All {len(preds)} backtest predictions"):
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
elif page.startswith("📖"):
    st.markdown('<div class="section-title">📖 Methodology & Honesty</div>', unsafe_allow_html=True)
    st.markdown(
        """
### The Question
**Where does Polymarket misprice the 2026 FIFA World Cup — and why?**

The 48-team / 32-team knockout format is brand new. Markets have weak historical priors — which is exactly where a model trained on competitive match data can add value.

### Data
- **49,287 men's international matches** since 1872 (martj42/international_results).
- **72 scheduled WC 2026 group-stage fixtures** already present in the dataset.
- **Polymarket** live winner markets (~$700M event volume).

### Model
- **Dixon-Coles** bivariate Poisson with exponential time decay (xi ≈ 1-year half-life).
- Fit on ~7K internationals since 2019 across 240 national teams.
- Home advantage γ ≈ 0.21; low-score correlation ρ ≈ −0.10 (both standard for intl football).

### Tournament Simulation
- **10,000** Monte Carlo runs of the 48-team group stage + 32-team knockout bracket.
- Group-stage tiebreakers: points → goal difference → goals scored → random.
- Knockout draws resolved with 30-min extra-time (λ scaled 1/3) then a 50/50 shootout.

### Backtest & Calibration
- Refit strictly on pre-tournament data, predict every match of **2018 & 2022 World Cups**.
- **Brier skill score vs uniform: +7.0%** (2018: +10.8%, 2022: +3.1%).
- Well-calibrated in the 20–50% probability range (where most predictions live).

### What This App Is *Not*
- Not a gambling recommendation.
- Not a live in-game win-probability tool (pre-match only).
- Not claiming the model is *right* and the market is *wrong* — just that they disagree by a measurable, data-backed amount.
        """
    )
