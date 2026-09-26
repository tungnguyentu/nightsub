import argparse
import sys

from . import config, preflight


def main():
    p = argparse.ArgumentParser(prog="autosub")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor")
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--allow-remote", action="store_true", help="allow binding a non-loopback host")
    a = p.parse_args()
    cfg = config.load()
    if a.cmd == "doctor":
        problems = preflight.check(cfg)
        for pr in problems:
            print("-", pr)
        print("OK: ready" if not problems else f"{len(problems)} problem(s)")
        sys.exit(1 if problems else 0)
    from . import web
    web.serve(cfg, a.host, a.port, a.allow_remote)


main()
