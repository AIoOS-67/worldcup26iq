"""
WorldCup26IQ — standalone Streamlit app for Zerve deployment.

All data reads are relative to the script directory (Zerve places uploaded
files alongside this script in the canvas working directory).  No `src.`
imports, so it can be deployed as a single file + 6 parquet files.
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
    # Accept the parquet either next to the app or in a data/ subdir.
    here = HERE / name
    sub = HERE / "data" / name
    return here if here.exists() else sub


st.set_page_config(
    page_title="WorldCup26IQ",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


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


st.sidebar.markdown("# ⚽ WorldCup26IQ")
st.sidebar.caption("Dixon-Coles + Monte Carlo for the 2026 FIFA World Cup.")
page = st.sidebar.radio(
    "Navigate",
    [
        "🏆 Champion Probabilities",
        "💸 Mispricing Leaderboard",
        "📊 Stage Reach",
        "📏 Calibration (Backtest)",
        "📖 Methodology",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption("Data: 49K internationals since 1872 · Polymarket live odds · 10K Monte Carlo runs.")


# ---------- Champion ----------
if page.startswith("🏆"):
    probs = load_probs()
    st.title("🏆 Champion Probabilities")
    st.caption(f"From {len(probs)} teams across 10,000 Monte Carlo tournament runs.")
    top = probs.sort_values("p_W", ascending=False).head(20)
    fig = px.bar(
        top.sort_values("p_W"),
        x="p_W", y="team", orientation="h",
        labels={"p_W": "P(Win)", "team": ""},
        title="Top 20 — Model P(Win the 2026 World Cup)",
    )
    fig.update_layout(height=600, margin=dict(l=120, r=40, t=60, b=40))
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Full table (all 48 teams)"):
        st.dataframe(
            probs[["team", "p_R32", "p_R16", "p_QF", "p_SF", "p_F", "p_W"]]
            .style.format({c: "{:.1%}" for c in ["p_R32", "p_R16", "p_QF", "p_SF", "p_F", "p_W"]}),
            use_container_width=True,
        )


# ---------- Mispricing Leaderboard ----------
elif page.startswith("💸"):
    lb = load_leaderboard()
    edges = load_edges()
    st.title("💸 Mispricing Leaderboard")
    if lb.empty:
        st.warning("Leaderboard data missing.")
    else:
        st.caption(
            f"Ranked by mispricing score = |edge| × √(liquidity / $1M). "
            f"Filtered to {len(lb)} markets with ≥$500K liquidity."
        )
        under = lb[lb["direction"] == "UNDER"]
        over = lb[lb["direction"] == "OVER"]
        best_under = under.iloc[0] if len(under) else None
        best_over = over.iloc[0] if len(over) else None
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Markets analysed", len(lb))
        if best_under is not None:
            k2.metric(f"Biggest UNDER: {best_under['team']}",
                      f"{best_under['edge']:+.1%}",
                      f"mkt {best_under['market_p_W']:.1%} → model {best_under['p_W']:.1%}")
        if best_over is not None:
            k3.metric(f"Biggest OVER: {best_over['team']}",
                      f"{best_over['edge']:+.1%}",
                      f"mkt {best_over['market_p_W']:.1%} → model {best_over['p_W']:.1%}")
        k4.metric("Total WC-2026 liquidity", f"${lb['liquidity'].sum()/1e6:.1f}M")

        st.subheader("Ranked by |edge| × √liquidity")
        view = lb[["rank", "team", "direction", "p_W", "market_p_W", "edge",
                   "liquidity", "mispricing_score", "form_recent_results",
                   "form_W", "form_D", "form_L", "form_avg_gd", "reason"]].copy()
        view["direction"] = view["direction"].map(
            lambda d: "⬆️ UNDER" if d == "UNDER" else "⬇️ OVER"
        )
        view = view.rename(columns={
            "form_recent_results": "last 20",
            "form_W": "W", "form_D": "D", "form_L": "L",
            "form_avg_gd": "gd/g", "p_W": "model", "market_p_W": "market",
            "liquidity": "liq ($)", "mispricing_score": "score",
        })
        st.dataframe(
            view.style.format({
                "model": "{:.1%}", "market": "{:.1%}", "edge": "{:+.1%}",
                "liq ($)": "{:,.0f}", "score": "{:.3f}", "gd/g": "{:+.1f}",
            }),
            use_container_width=True, height=560, hide_index=True,
        )

        st.subheader("Model vs Market — all 43 covered teams")
        scat = edges[edges["market_covered"]].copy()
        fig = px.scatter(
            scat, x="market_p_W", y="p_W", text="team", size="liquidity",
            color="edge", color_continuous_scale="RdYlGn",
            labels={"market_p_W": "Polymarket implied P(Win)", "p_W": "Model P(Win)"},
            hover_data={"edge": ":.1%", "liquidity": ":,.0f"},
        )
        mx = max(scat["market_p_W"].max(), scat["p_W"].max())
        fig.add_shape(type="line", x0=0, y0=0, x1=mx, y1=mx,
                      line=dict(dash="dash", color="grey"))
        fig.update_traces(textposition="top center")
        fig.update_layout(height=620, coloraxis_colorbar=dict(title="Edge"))
        st.plotly_chart(fig, use_container_width=True)


# ---------- Stage Reach ----------
elif page.startswith("📊"):
    probs = load_probs()
    st.title("📊 Stage-Reach Probabilities")
    cols = ["p_R32", "p_R16", "p_QF", "p_SF", "p_F", "p_W"]
    nice = ["R32", "R16", "QF", "SF", "Final", "Win"]
    teams = probs["team"].tolist()
    chosen = st.multiselect("Teams to compare", teams, default=teams[:6])
    if chosen:
        sub = probs[probs["team"].isin(chosen)].set_index("team")[cols]
        sub.columns = nice
        long = sub.reset_index().melt(id_vars="team", var_name="Stage", value_name="P(reach)")
        fig = px.line(
            long, x="Stage", y="P(reach)", color="team", markers=True,
            category_orders={"Stage": nice},
        )
        fig.update_layout(height=500, yaxis=dict(tickformat=".0%"))
        st.plotly_chart(fig, use_container_width=True)


# ---------- Calibration ----------
elif page.startswith("📏"):
    st.title("📏 Calibration — How Honest Is the Model?")
    summary = load_backtest_summary()
    rel = load_backtest_reliability()
    preds = load_backtest_predictions()
    st.caption(
        "We refit the model strictly on data before each past World Cup, "
        "then score every match. Brier skill score vs a uniform baseline."
    )
    all_dc = summary[(summary["year"] == "ALL") & (summary["model"] == "DixonColes")].iloc[0]
    all_base = summary[(summary["year"] == "ALL") & (summary["model"] == "Uniform(1/3)")].iloc[0]
    skill = 1 - all_dc["brier"] / all_base["brier"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Brier (DC)", f"{all_dc['brier']:.4f}")
    m2.metric("Brier (Uniform)", f"{all_base['brier']:.4f}")
    m3.metric("LogLoss (DC)", f"{all_dc['logloss']:.4f}")
    m4.metric("Brier skill vs uniform", f"{skill*100:+.1f}%")

    st.subheader("By tournament")
    pivot = summary.pivot_table(index="year", columns="model", values=["brier", "logloss"])
    st.dataframe(pivot.style.format("{:.4f}"), use_container_width=True)

    st.subheader("Reliability Diagram (pooled predictions)")
    st.caption("Dots on the diagonal = perfectly calibrated. Dot size = predictions in bucket.")
    fig = go.Figure()
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="grey"))
    fig.add_trace(go.Scatter(
        x=rel["avg_predicted"], y=rel["empirical"],
        mode="markers+text",
        marker=dict(size=np.sqrt(rel["count"]) * 4 + 6, color="#1f77b4"),
        text=[f"n={c}" for c in rel["count"]],
        textposition="top center",
        name="Dixon-Coles",
    ))
    fig.update_layout(height=500,
                      xaxis=dict(title="Predicted probability", range=[0, 1]),
                      yaxis=dict(title="Empirical frequency", range=[0, 1]))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander(f"All {len(preds)} backtest predictions"):
        st.dataframe(
            preds[["year", "date", "home", "away", "home_goals", "away_goals",
                   "p_home", "p_draw", "p_away", "actual"]]
            .style.format({"p_home": "{:.1%}", "p_draw": "{:.1%}", "p_away": "{:.1%}"}),
            use_container_width=True,
        )


# ---------- Methodology ----------
elif page.startswith("📖"):
    st.title("📖 Methodology & Honesty")
    st.markdown(
        """
### The Question
**Where does Polymarket misprice the 2026 FIFA World Cup — and why?**

The 48-team / 32-team knockout format is brand new. Markets have weak historical priors — which is exactly where a model trained on competitive match data can add value.

### Data
- **49,287 men's international matches** since 1872 (martj42/international_results).
- **72 scheduled WC 2026 group-stage fixtures** already present in the dataset.
- **Polymarket** live winner markets (~$700M event volume) for market comparison.

### Model
- **Dixon-Coles** bivariate Poisson with exponential time decay (xi ≈ 1-year half-life).
- Fit on ~7K internationals since 2019 across 240 national teams.
- Home advantage γ ≈ 0.21; low-score correlation ρ ≈ −0.10 (both standard for intl football).

### Tournament Simulation
- **10,000** Monte Carlo runs of the full 48-team group stage + 32-team knockout bracket.
- Group-stage tiebreakers: points → goal difference → goals scored → random.
- Knockout draws resolved with 30-min extra-time (λ scaled 1/3), then a 50/50 shootout.

### Backtest & Calibration
- We refit strictly on pre-tournament data and predicted every match of **2018 & 2022 World Cups**.
- **Brier skill score vs uniform: +7.0%** (2018: +10.8%, 2022: +3.1%).
- Well-calibrated in the 20–50% probability range (where most predictions live).

### What This App Is *Not*
- Not a gambling recommendation.
- Not a live in-game win-probability tool (pre-match only).
- Not claiming the model is *right* and the market is *wrong* — just that they disagree by a measurable, data-backed amount.
        """
    )
