"""
First-contact probe for the Impact.com Partner REST API.

Run this once you've retrieved your Account SID + Auth Token from the
Impact.com partner UI (Account → Settings → API / Technical). The script
tries three increasingly useful calls and prints what it gets back so we
can map the exact endpoint shape for our WC26 app:

    1.  /Mediapartners/{SID}                  — sanity check, should echo your
                                                account profile. Confirms
                                                the creds are valid.
    2.  /Mediapartners/{SID}/Campaigns        — list of programs you've
                                                joined (Fanatics Global 9663
                                                should be in here).
    3.  /Mediapartners/{SID}/Actions          — recent conversions, i.e. the
                                                $34.99 Messi order from Apr 23
                                                should show up here as soon
                                                as Fanatics stamps its
                                                ActionDate.

Usage — set env vars and run:

    IMPACT_SID="<your Account SID>" \
    IMPACT_TOKEN="<your Auth Token>" \
    python test_impact_api.py

Or Windows PowerShell:

    $env:IMPACT_SID = "<SID>"
    $env:IMPACT_TOKEN = "<TOKEN>"
    python test_impact_api.py
"""
from __future__ import annotations

import os
import sys
from typing import Any

import requests

BASE_URL = "https://api.impact.com"  # most likely — will confirm via 401 vs 200

SID = os.environ.get("IMPACT_SID", "").strip()
TOKEN = os.environ.get("IMPACT_TOKEN", "").strip()


def _require_creds() -> None:
    if not SID or not TOKEN:
        print("❌ Set IMPACT_SID and IMPACT_TOKEN env vars first.", file=sys.stderr)
        sys.exit(1)


def _call(path: str, params: dict[str, Any] | None = None) -> tuple[int, Any]:
    """Return (status_code, parsed_body)."""
    url = f"{BASE_URL}{path}"
    r = requests.get(
        url,
        auth=(SID, TOKEN),
        headers={"Accept": "application/json"},
        params=params or {},
        timeout=20,
    )
    try:
        body: Any = r.json()
    except ValueError:
        body = r.text
    return r.status_code, body


def _preview(body: Any, n: int = 600) -> str:
    import json
    if isinstance(body, (dict, list)):
        s = json.dumps(body, indent=2, ensure_ascii=False)
    else:
        s = str(body)
    return s if len(s) <= n else s[:n] + f"\n...(truncated; full body {len(s)} chars)"


def main() -> int:
    _require_creds()
    print(f"→ Using SID={SID[:4]}... against {BASE_URL}\n")

    for label, path, params in [
        ("Account profile", f"/Mediapartners/{SID}", None),
        ("Joined programs (campaigns)", f"/Mediapartners/{SID}/Campaigns", {"PageSize": 20}),
        ("Recent actions (conversions)", f"/Mediapartners/{SID}/Actions", {"PageSize": 5}),
        ("Catalogs (product feed list)", f"/Mediapartners/{SID}/Catalogs", {"PageSize": 10}),
        ("Tracking link builder sample",
         f"/Mediapartners/{SID}/Programs/9663/TrackingLinks",
         {"DeepLink": "https://www.fanatics.com/soccer"}),
    ]:
        print(f"══ {label}")
        print(f"   GET {path}   params={params}")
        status, body = _call(path, params)
        marker = "✅" if 200 <= status < 300 else "❌"
        print(f"   {marker} {status}")
        print(_preview(body, 900))
        print()

    print("Done. Paste the successful JSON back to Claude — we'll map field names")
    print("into the WC26 app's _fanatics_products() / _real_pricing() pipeline.")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # windows console
    raise SystemExit(main())
