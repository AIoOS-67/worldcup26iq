# WorldCup26IQ

Dixon-Coles + Monte Carlo model for the 2026 FIFA World Cup, with a live
mispricing leaderboard against Polymarket winner markets.

## Run locally

```bash
pip install -r requirements.txt
streamlit run wc26_app.py
```

## Files

- `wc26_app.py` — Streamlit app (5 pages: Champions, Mispricing Leaderboard, Stage Reach, Calibration, Methodology)
- `data/*.parquet` — precomputed model outputs (10K Monte Carlo sims, backtest on 2018+2022 WCs)

## Data & Methodology

- **49,287 men's international matches** since 1872 (martj42/international_results).
- **Dixon-Coles** bivariate Poisson with ~1-year exponential time decay.
- **10,000** MC tournament runs; 48-team group stage + 32-team knockout.
- Backtest: **+7.0% Brier skill score** vs uniform baseline across WC 2018+2022 (128 matches).
- **Polymarket** winner markets ($700M event volume) for market comparison.

## Built for the 2026 Zerve AI Hackathon

Source project: [Zerve canvas](https://app.zerve.ai/notebook/53360826-4362-49dc-b19a-be4bffb22d60).
