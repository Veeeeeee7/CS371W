#!/usr/bin/env python3
"""
ee_raw_check.py -- bypass the Earth Engine client and talk to the API directly.

The ee Python client turns every 401/403/404 into the same "Please authorize
access to your Earth Engine account" string, which hides what actually went
wrong. This exchanges your saved refresh token for an access token, asks Google
which account that token belongs to, and then makes a raw REST call to Earth
Engine so we can read the real status code and error body.

Three things it will tell you that the client will not:
  - the EMAIL the token belongs to  (account mismatch is the usual culprit)
  - whether the token still exchanges at all
  - Earth Engine's actual error text

    python ee_raw_check.py
"""
from __future__ import annotations
import json, shutil, subprocess, sys
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("pip install requests")

CRED = Path.home() / ".config" / "earthengine" / "credentials"


def env_report() -> None:
    print("=== environment ===")
    print(f"python          : {sys.executable}")
    ee_bin = shutil.which("earthengine")
    print(f"earthengine bin : {ee_bin}")
    if ee_bin:
        # A venv python with a conda-base CLI is a real and confusing failure mode.
        shebang = Path(ee_bin).read_text(errors="ignore").splitlines()[:1]
        if shebang and shebang[0].startswith("#!"):
            interp = shebang[0][2:].strip()
            print(f"  its interpreter: {interp}")
            if Path(interp).resolve() != Path(sys.executable).resolve():
                print("  WARNING: the CLI and this python are different "
                      "interpreters. Versions may differ.")
    try:
        import ee
        print(f"earthengine-api : {ee.__version__}")
    except Exception as e:
        print(f"earthengine-api : import failed ({e})")


def load_creds() -> dict:
    if not CRED.exists():
        sys.exit(f"no credentials at {CRED} -- run earthengine authenticate")
    return json.loads(CRED.read_text())


def exchange(c: dict) -> str | None:
    print("\n=== 1. refresh token -> access token ===")
    r = requests.post("https://oauth2.googleapis.com/token", data={
        "client_id": c["client_id"],
        "client_secret": c["client_secret"],
        "refresh_token": c["refresh_token"],
        "grant_type": "refresh_token",
    }, timeout=60)
    print(f"HTTP {r.status_code}")
    if r.status_code != 200:
        print(r.text[:500])
        print("\n-> The saved token is dead. Delete it and re-authenticate:")
        print(f"     rm {CRED}")
        print("     earthengine authenticate --auth_mode=notebook")
        return None
    tok = r.json()["access_token"]
    print("OK, token exchanged")
    return tok


def whoami(tok: str) -> None:
    print("\n=== 2. which account does this token belong to? ===")
    r = requests.get("https://oauth2.googleapis.com/tokeninfo",
                     params={"access_token": tok}, timeout=60)
    if r.status_code != 200:
        print(f"HTTP {r.status_code}: {r.text[:300]}")
        return
    d = r.json()
    email = d.get("email", "<not in token>")
    print(f"EMAIL : {email}")
    print(f"scopes: {d.get('scope','')[:200]}")
    print("\n>>> This must be the SAME account that owns the Cloud project and")
    print(">>> that you used in the browser to enable the API and register.")
    print(">>> If it is not, that is the bug -- everything else can look correct.")


def ee_call(tok: str, project: str) -> None:
    print(f"\n=== 3. raw Earth Engine API call (project={project}) ===")
    url = f"https://earthengine.googleapis.com/v1/projects/{project}/algorithms"
    r = requests.get(url, headers={"Authorization": f"Bearer {tok}"}, timeout=120)
    print(f"HTTP {r.status_code}")
    if r.status_code == 200:
        n = len(r.json().get("algorithms", []))
        print(f"OK -- {n} algorithms returned. Earth Engine works on this project.")
        print("If the ee client still fails, it is a client/version problem:")
        print("    pip install --upgrade earthengine-api")
        return

    body = r.text[:900]
    print(body)
    low = body.lower()
    print("\n--- reading that error ---")
    if "has not been used in project" in low or "disabled" in low:
        print("The Earth Engine API is NOT enabled on this project (whatever the")
        print("console showed). Enable it, wait ~2 minutes for propagation:")
        print(f"  https://console.cloud.google.com/apis/library/"
              f"earthengine.googleapis.com?project={project}")
    elif "not registered" in low or "register" in low:
        print("The project is not registered for Earth Engine use:")
        print(f"  https://code.earthengine.google.com/register?project={project}")
    elif "permission" in low or r.status_code == 403:
        print("Permission denied. Most likely the token's account (section 2)")
        print("is not an owner/editor of this project. Either authenticate as")
        print("that account, or grant it access in the Cloud console IAM page:")
        print(f"  https://console.cloud.google.com/iam-admin/iam?project={project}")
    elif r.status_code == 404:
        print(f"Project {project!r} not found from this account's perspective --")
        print("again pointing at an account mismatch, or a typo in the id.")
    else:
        print("Unrecognised error -- paste the body above and I'll read it.")


def main() -> int:
    env_report()
    c = load_creds()
    project = c.get("project") or c.get("quota_project_id")
    print(f"\nproject in credentials: {project}")
    tok = exchange(c)
    if not tok:
        return 1
    whoami(tok)
    if project:
        ee_call(tok, project)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
