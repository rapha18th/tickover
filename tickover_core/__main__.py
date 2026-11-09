"""tickover CLI: hear, verify, enroll, check, status, listen, selftest. Same engine as the Hermes tools."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tickover", description="Give an agent ears: hear a sound, verify an action, catch a machine drifting.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hear", help="what is in this recording, and is it usable"); h.add_argument("path")
    v = sub.add_parser("verify", help="did the sound change: before/after files, or one clip")
    v.add_argument("paths", nargs="+", metavar="FILE", help="one clip, or BEFORE AFTER")
    v.add_argument("--expect", choices=["start", "stop", "change"], default="change"); v.add_argument("--machine")
    e = sub.add_parser("enroll", help="remember a machine's healthy sound")
    e.add_argument("machine"); e.add_argument("paths", nargs="+"); e.add_argument("--append", action="store_true")
    c = sub.add_parser("check", help="score a recording against the baseline")
    c.add_argument("machine"); c.add_argument("path"); c.add_argument("--png")
    sub.add_parser("status", help="every machine and its trend")
    li = sub.add_parser("listen", help="record from the microphone")
    li.add_argument("machine", nargs="?"); li.add_argument("--seconds", type=float, default=10)
    li.add_argument("--then", choices=["hear", "enroll", "check"], default="hear")
    sub.add_parser("selftest", help="prove it works here, with synthetic sound")
    a = ap.parse_args(argv)
    from . import agent, engine, tools
    if a.cmd == "hear":
        from . import hear
        r = hear.hear(a.path)
    elif a.cmd == "verify":
        from . import verify
        if len(a.paths) == 1:
            r = verify.clip(a.paths[0], a.expect, machine=a.machine)
        elif len(a.paths) == 2:
            r = verify.after(a.paths[1], expect=a.expect, machine=a.machine, before_path=a.paths[0])
        else:
            ap.error("verify takes one clip, or a before file and an after file")
    elif a.cmd == "enroll":
        r = engine.enroll(a.machine, a.paths, append=a.append)
    elif a.cmd == "check":
        r = engine.check(a.machine, a.path, out_png=a.png)
    elif a.cmd == "status":
        r = agent.fleet()
    elif a.cmd == "listen":
        r = json.loads(tools.listen_tool({"machine": a.machine, "seconds": a.seconds, "then": a.then}))
    else:
        from . import selftest
        r = selftest.run(verbose=False)
    print(json.dumps(r, indent=1, default=str))
    return 0 if r.get("ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())
