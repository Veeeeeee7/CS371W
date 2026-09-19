#!/usr/bin/env python3
"""
ee_diagnose.py -- work out exactly which Earth Engine setup step is missing.

Earth Engine reports several distinct failures with the same "Please authorize
access..." message, which sends people in circles re-running authenticate when
the actual problem is a missing Cloud Project. This separates them.

    python ee_diagnose.py
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

try:
    import ee
except ImportError:
    sys.exit("pip install earthengine-api")

CRED = Path.home() / ".config" / "earthengine" / "credentials"


def step1_credentials() -> bool:
    print("1. credentials file")
    if not CRED.exists():
        print(f"   MISSING  {CRED}")
        print("   -> earthengine authenticate --auth_mode=notebook")
        return False
    try:
        d = json.loads(CRED.read_text())
    except Exception as e:
        print(f"   UNREADABLE ({e}) -> delete it and re-authenticate")
        return False
    keys = sorted(d.keys())
    print(f"   OK       {CRED}")
    print(f"            keys: {keys}")
    proj = d.get("project") or d.get("quota_project_id")
    if proj:
        print(f"            project embedded in credentials: {proj}")
    else:
        print("            no project embedded -- this is normal; it usually "
              "lives in the gcloud config instead")
    return True


def step2_project() -> str | None:
    print("\n2. project id")
    for src, val in [
        ("EE_PROJECT env var", os.environ.get("EE_PROJECT")),
        ("GOOGLE_CLOUD_PROJECT env var", os.environ.get("GOOGLE_CLOUD_PROJECT")),
    ]:
        if val:
            print(f"   found via {src}: {val}")
            return val
    try:
        d = json.loads(CRED.read_text())
        p = d.get("project") or d.get("quota_project_id")
        if p:
            print(f"   found in credentials: {p}")
            return p
    except Exception:
        pass
    print("   NOT SET anywhere this script can see.")
    print("   Find it at https://code.earthengine.google.com/ -- the project")
    print("   appears in the top-right project picker. Then:")
    print("       earthengine set_project YOUR-PROJECT-ID")
    print("   (or export EE_PROJECT=YOUR-PROJECT-ID)")
    return None


def step3_initialize(project: str | None) -> bool:
    print("\n3. ee.Initialize")
    for label, kwargs in [(f"with project={project}", {"project": project}) if project
                          else (None, None),
                          ("without project", {})]:
        if label is None:
            continue
        try:
            ee.Initialize(**kwargs)
            print(f"   OK   {label}")
            return True
        except Exception as e:
            msg = str(e).strip().replace("\n", " ")[:160]
            print(f"   FAIL {label}: {msg}")
    print("\n   Diagnosis:")
    print("   - 'Please authorize access' WITH a valid credentials file almost")
    print("     always means the project is missing or not registered for Earth")
    print("     Engine, not that auth failed.")
    print("   - If you never completed registration, do it now:")
    print("       https://code.earthengine.google.com/register")
    print("     Choose the noncommercial / academic path; it creates the project.")
    return False


def step4_smoke() -> None:
    print("\n4. can we actually read data?")
    try:
        v = (ee.Image("MERIT/Hydro/v1_0_1").select("hnd")
             .reduceRegion(ee.Reducer.first(),
                           ee.Geometry.Point([-73.2121, 44.4759]), 90)
             .getInfo())
        print(f"   OK   HAND at Winooski floodplain = {v}")
        print("\n   Everything works. Run:  python check_ee.py")
    except Exception as e:
        print(f"   FAIL {type(e).__name__}: {str(e)[:200]}")
        print("   Initialize succeeded but data access did not -- the project")
        print("   exists but is probably not registered for Earth Engine.")
        print("   Register it at https://code.earthengine.google.com/register")


def main() -> int:
    print("Earth Engine setup diagnosis\n" + "=" * 42)
    if not step1_credentials():
        return 1
    project = step2_project()
    if not step3_initialize(project):
        return 1
    step4_smoke()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
