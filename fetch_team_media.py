"""Fetch national-team badges + home jerseys from TheSportsDB.

Pulls the 48 WC2026 qualifying teams (from wc2026_groups.parquet) and caches
their badge + latest home-kit image URLs to data/team_media.json. Re-run
manually when a new season's kits drop — output is checked in so the app
doesn't need network access at runtime.

Endpoints:
- searchteams.php?t={name}  → idTeam, strBadge, strLogo, strBanner
- lookupequipment.php?id={idTeam} → list of per-season kits; pick latest 1st

Usage:  python fetch_team_media.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "team_media.json"
GROUPS = HERE / "data" / "wc2026_groups.parquet"

BASE = "https://www.thesportsdb.com/api/v1/json/3"

# Some canonical names differ from TheSportsDB's spelling. Map our internal
# English name → search query used against the API.
NAME_OVERRIDES = {
    "United States": "USA",
    "South Korea": "South Korea",
    "Czech Republic": "Czech Republic",
    "Ivory Coast": "Ivory Coast",
    "DR Congo": "Congo DR",
    "Cape Verde": "Cape Verde",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "New Zealand": "New Zealand",
    "South Africa": "South Africa",
    "Saudi Arabia": "Saudi Arabia",
    "Curaçao": "Curacao",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "DR Congo": "DR Congo",
}


def _search(team: str) -> dict | None:
    q = NAME_OVERRIDES.get(team, team)
    r = requests.get(f"{BASE}/searchteams.php", params={"t": q}, timeout=15)
    r.raise_for_status()
    data = r.json() or {}
    results = data.get("teams") or []
    if not results:
        return None
    # Prefer a national team: strSport=Soccer AND strLeague contains "World Cup"
    # or country matches team name. Otherwise first match.
    for t in results:
        if t.get("strSport") != "Soccer":
            continue
        leagues = " ".join(str(t.get(f"strLeague{i}") or "") for i in ("", 2, 3, 4, 5))
        if "World Cup" in leagues or "National" in leagues or t.get("strCountry") == q:
            return t
    return results[0]


def _pick_home_jersey(id_team: str) -> str | None:
    r = requests.get(f"{BASE}/lookupequipment.php", params={"id": id_team}, timeout=15)
    r.raise_for_status()
    data = r.json() or {}
    items = data.get("equipment") or []
    home = [e for e in items if e.get("strType") == "1st" and e.get("strEquipment")]
    if not home:
        # Some teams only have 2nd/3rd; fall back to any
        home = [e for e in items if e.get("strEquipment")]
    if not home:
        return None
    # Pick latest season (seasons look like "2024-2025" or "2024"; sort desc)
    home.sort(key=lambda e: e.get("strSeason") or "", reverse=True)
    return home[0]["strEquipment"]


def main() -> int:
    groups = pd.read_parquet(GROUPS)
    teams = sorted(groups["team"].unique().tolist())

    # Resume: load existing cache, skip teams already fetched with a badge
    out: dict[str, dict] = {}
    if OUT.exists():
        try:
            out = json.loads(OUT.read_text(encoding="utf-8"))
            print(f"Resuming — cache has {len(out)} teams")
        except Exception:
            out = {}

    missing = []
    to_fetch = [t for t in teams if not out.get(t, {}).get("badge")]
    print(f"Fetching media for {len(to_fetch)} teams (skipping {len(teams) - len(to_fetch)} cached)...")

    for i, team in enumerate(to_fetch, 1):
        try:
            rec = _search(team)
            if not rec:
                print(f"  [{i:2d}/{len(to_fetch)}] {team:30s} — NOT FOUND")
                missing.append(team)
                time.sleep(4)
                continue
            id_team = rec.get("idTeam")
            time.sleep(4)  # between search and equipment call
            jersey = _pick_home_jersey(id_team) if id_team else None
            out[team] = {
                "id": id_team,
                "badge": rec.get("strBadge"),
                "logo": rec.get("strLogo"),
                "banner": rec.get("strBanner"),
                "jersey": jersey,
                "desc_en": (rec.get("strDescriptionEN") or "")[:500],
            }
            status = " B" if out[team]["badge"] else "  "
            status += " J" if jersey else "  "
            print(f"  [{i:2d}/{len(to_fetch)}] {team:30s} {status}")
            # Flush cache after each success so 429s don't wipe progress
            OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
            time.sleep(4)
        except Exception as e:
            print(f"  [{i:2d}/{len(to_fetch)}] {team:30s} — ERR {e}")
            missing.append(team)
            time.sleep(5)  # back off longer on error

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(out)} teams to {OUT.name}")
    badges = sum(1 for v in out.values() if v.get("badge"))
    jerseys = sum(1 for v in out.values() if v.get("jersey"))
    print(f"  badges: {badges}/{len(teams)}   jerseys: {jerseys}/{len(teams)}")
    if missing:
        print(f"  missing entirely: {missing}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
