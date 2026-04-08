#!/usr/bin/env python3
"""Check a deployed Overnight Lab API from your PC (no SSH). Usage:
   python scripts/verify_remote.py http://187.77.201.221:8001
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request


def get(url: str, timeout: float = 30.0) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": "verify_remote/1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def main() -> int:
    base = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8001").rstrip("/")
    print(f"Checking {base} …\n")

    try:
        code, body = get(f"{base}/api/health", timeout=15)
    except urllib.error.URLError as e:
        print(f"FAIL  /api/health  — not reachable: {e}")
        return 1

    if code != 200:
        print(f"FAIL  /api/health  HTTP {code}")
        return 1

    h = json.loads(body.decode())
    print(f"OK    /api/health  ok={h.get('ok')}  version={h.get('api_version')}")
    sched = h.get("scheduler") or {}
    print(f"      scheduler enabled={sched.get('enabled')}  next={sched.get('next_run')}")
    em = h.get("email_alert") or {}
    print(f"      email configured={em.get('configured')}  next={em.get('next_run')}")

    try:
        code, body = get(f"{base}/api/latest", timeout=60)
    except urllib.error.URLError as e:
        print(f"FAIL  /api/latest   — {e}")
        return 1

    if code == 404:
        print("WARN  /api/latest  no ranking file yet (first deploy or cleared data)")
        try:
            code2, b2 = get(f"{base}/api/rebuild-ranking/status", timeout=10)
            if code2 == 200:
                st = json.loads(b2.decode())
                print(f"      rebuild running={st.get('running')}")
        except OSError:
            pass
        return 2

    if code != 200:
        print(f"FAIL  /api/latest  HTTP {code}")
        return 1

    d = json.loads(body.decode())
    rows = d.get("rows") or []
    print(f"OK    /api/latest   rows={len(rows)}  generated_at={d.get('generated_at', '')[:19]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
