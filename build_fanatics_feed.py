"""Filter Fanatics Global product feed to WC26-relevant SKUs.

Reads the raw gzipped TSV feed from ../data_raw/fanatics/ (intentionally outside
this git repo so the 85 MB proprietary feed never gets pushed to GitHub), filters
to the 48 qualifying teams + generic FIFA World Cup gear, normalises the team
name back to our canonical English form, and writes a small parquet to
deploy/data/fanatics_products.parquet that the Streamlit app reads at runtime.

Output schema (one row per SKU, ~5-10k rows total, ~1-3 MB on disk):

  sku             catalog item id (string)
  team            canonical WC26 team name (e.g. "Argentina", "United States")
  name            product name
  category        Fanatics category (e.g. "Soccer National Teams")
  sub_category    narrower category field
  price           current/sale price (USD) as float
  list_price      original price; if != price, product is on sale
  on_sale         bool (price < list_price)
  discount_pct    int (0-99) when on sale, else 0
  in_stock        bool
  manufacturer    brand (Nike / Adidas / Puma / Fanatics etc.)
  gender          Unisex / Men / Women / Youth
  age_group       Adult / Kids / Baby
  image_url       product image URL (CDN)
  link            Impact-wrapped affiliate deep link (already carries our
                  Publisher ID 7225697 + prodsku parameter)

Re-run when a fresh feed is downloaded. Takes ~30 seconds to stream through
the 600k-row catalogue.

Usage:  python build_fanatics_feed.py
"""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "data_raw" / "fanatics" / "Fanatics-Product-Catalog_CUSTOM.txt.gz"
OUT = HERE / "data" / "fanatics_products.parquet"

# Canonical WC26 team name → list of values that may appear in the Fanatics
# feed's `team` column. Most teams follow "<Country> National Team" but the
# host nations have idiosyncratic labels.
WC26_TEAM_ALIASES: dict[str, list[str]] = {
    "Algeria":        ["Algeria National Team"],
    "Argentina":      ["Argentina National Team"],
    "Australia":      ["Australia National Team"],
    "Austria":        ["Austria National Team"],
    "Belgium":        ["Belgium National Team"],
    "Bosnia and Herzegovina": ["Bosnia and Herzegovina National Team", "Bosnia National Team"],
    "Brazil":         ["Brazil National Team"],
    "Canada":         ["Canada Soccer", "Canada National Team"],
    "Cape Verde":     ["Cape Verde National Team"],
    "Colombia":       ["Colombia National Team"],
    "Croatia":        ["Croatia National Team"],
    "Curaçao":        ["Curacao National Team"],
    "Czech Republic": ["Czech Republic National Team"],
    "DR Congo":       ["DR Congo National Team", "Democratic Republic of the Congo National Team"],
    "Ecuador":        ["Ecuador National Team"],
    "Egypt":          ["Egypt National Team"],
    "England":        ["England National Team"],
    "France":         ["France National Team"],
    "Germany":        ["Germany National Team"],
    "Ghana":          ["Ghana National Team"],
    "Haiti":          ["Haiti National Team"],
    "Iran":           ["Iran National Team"],
    "Iraq":           ["Iraq National Team"],
    "Ivory Coast":    ["Ivory Coast National Team"],
    "Japan":          ["Japan National Team"],
    "Jordan":         ["Jordan National Team"],
    "Mexico":         ["Mexico National Team"],
    "Morocco":        ["Morocco National Team"],
    "Netherlands":    ["Netherlands National Team"],
    "New Zealand":    ["New Zealand National Team"],
    "Norway":         ["Norway National Team"],
    "Panama":         ["Panama National Team"],
    "Paraguay":       ["Paraguay National Team"],
    "Portugal":       ["Portugal National Team"],
    "Qatar":          ["Qatar National Team"],
    "Saudi Arabia":   ["Saudi Arabia National Team"],
    "Scotland":       ["Scotland National Team"],
    "Senegal":        ["Senegal National Team"],
    "South Africa":   ["South Africa National Team"],
    "South Korea":    ["South Korea National Team", "Korea National Team"],
    "Spain":          ["Spain National Team"],
    "Sweden":         ["Sweden National Team"],
    "Switzerland":    ["Switzerland National Team"],
    "Tunisia":        ["Tunisia National Team"],
    "Turkey":         ["Turkey National Team"],
    "United States":  ["Team USA", "USA National Team", "USWNT"],
    "Uruguay":        ["Uruguay National Team"],
    "Uzbekistan":     ["Uzbekistan National Team"],
}
# Also capture generic WC26 merch into a "*FIFA World Cup*" bucket — useful
# when Claude wants to recommend without picking a specific team.
GENERIC_TEAM_KEY = "_FIFA World Cup"
GENERIC_ALIASES = ["FIFA World Cup Gear", "FIFA World Cup", "FIFA"]


# Reverse lookup: alias → canonical name
_alias_to_canonical: dict[str, str] = {}
for canonical, aliases in WC26_TEAM_ALIASES.items():
    for a in aliases:
        _alias_to_canonical[a] = canonical
for a in GENERIC_ALIASES:
    _alias_to_canonical[a] = GENERIC_TEAM_KEY


def _to_float(s: str) -> float:
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def main() -> int:
    if not RAW.exists():
        print(f"Raw feed not found: {RAW}", file=sys.stderr)
        print(f"Drop Fanatics-Product-Catalog_CUSTOM.txt.gz into {RAW.parent}", file=sys.stderr)
        return 1

    print(f"Reading {RAW.name} ({RAW.stat().st_size / 1e6:.1f} MB gzipped)...")
    kept: list[dict] = []
    total = 0
    soccer = 0
    with gzip.open(RAW, "rt", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n").split("\t")
        ci = {c: i for i, c in enumerate(header)}
        col = lambda row, name: row[ci[name]] if ci.get(name) is not None and ci[name] < len(row) else ""

        for line in f:
            total += 1
            row = line.rstrip("\n").split("\t")
            if len(row) < len(header):
                continue
            cat = col(row, "category field")
            if cat != "Soccer National Teams":
                continue
            soccer += 1
            team_raw = col(row, "team")
            canonical = _alias_to_canonical.get(team_raw)
            if not canonical:
                continue

            price = _to_float(col(row, "current price"))
            list_price = _to_float(col(row, "original price field")) or price
            on_sale = price > 0 and list_price > price
            discount_pct = int(round((list_price - price) / list_price * 100)) if on_sale else 0

            kept.append({
                "sku":           col(row, "catalog item id field"),
                "team":          canonical,
                "name":          col(row, "name field"),
                "category":      cat,
                "sub_category":  col(row, "sub category field"),
                "price":         price,
                "list_price":    list_price,
                "on_sale":       on_sale,
                "discount_pct":  discount_pct,
                "in_stock":      col(row, "stock availability field") == "InStock",
                "manufacturer":  col(row, "manufacturer"),
                "gender":        col(row, "gender"),
                "age_group":     col(row, "age group"),
                "image_url":     col(row, "image url field"),
                "link":          col(row, "link URL field"),
            })

    print(f"  scanned {total:,} rows · {soccer:,} soccer · {len(kept):,} matched WC26")

    df = pd.DataFrame(kept)
    # Light ranking heuristics: in-stock first, on-sale boost, then lower-price
    # tops tend to be jerseys / hats which we want surfaced. Stable sort.
    df["_rank"] = (
        (~df["in_stock"]).astype(int) * 1000
        + (~df["on_sale"]).astype(int) * 10
        + (df["price"] / 10).clip(upper=50)
    )
    df = df.sort_values(["team", "_rank"]).drop(columns=["_rank"]).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    size_mb = OUT.stat().st_size / 1e6
    print(f"  wrote {OUT.relative_to(HERE.parent)} — {len(df):,} rows, {size_mb:.2f} MB")
    print()
    print("Counts per canonical team:")
    for team, n in df["team"].value_counts().items():
        print(f"  {team:30s} {n:>4,}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
