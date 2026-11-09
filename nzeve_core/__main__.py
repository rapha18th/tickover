"""nzeve CLI: enroll, check, status, listen, selftest. Same engine as the Hermes tools."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="nzeve", description="Hear when a machine drifts from its own healthy sound.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("enroll"); e.add_argument("machine"); e.add_argument("paths", nargs="+"); e.add_argument("--append", action="store_true")
    c = sub.add_parser("check"); c.add_argument("machine"); c.add_argument("path"); c.add_argument("--png")
    s = sub.add_parser("status"); s.add_argument("machine", nargs="?")
    li = sub.add_parser("listen"); li.add_argument("machine"); li.add_argument("--seconds", type=float, default=10); li.add_argument("--then", choices=["enroll", "check"], default="check")
    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    from . import engine, tools
    if a.cmd == "enroll":
        r = engine.enroll(a.machine, a.paths, append=a.append)
    elif a.cmd == "check":
        r = engine.check(a.machine, a.path, out_png=a.png)
    elif a.cmd == "status":
        r = json.loads(tools.status_tool({"machine": a.machine} if a.machine else {}))
    elif a.cmd == "listen":
        r = json.loads(tools.listen_tool({"machine": a.machine, "seconds": a.seconds, "then": a.then}))
    else:
        from . import selftest
        r = selftest.run(verbose=False)
    print(json.dumps(r, indent=1, default=str))
    return 0 if r.get("ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())
