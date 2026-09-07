from __future__ import annotations

import argparse
import json

from app.diagnostics import collect_diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description="Advanced AI Agent diagnostic checker")
    parser.add_argument("--runtime", action="store_true", help="Require the configured model/runtime dependencies to be ready now")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()
    report = collect_diagnostics(runtime=args.runtime)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"Advanced AI Agent {report['version']} diagnostics")
        for item in report["checks"]:
            mark = "OK" if item["ok"] else ("WARN" if item["level"] == "warning" else "FAIL")
            print(f"[{mark:4}] {item['name']}: {item['detail']}")
        print(f"errors={report['errors']} warnings={report['warnings']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
