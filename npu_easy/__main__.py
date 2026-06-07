from __future__ import annotations

import argparse
import json
from typing import Any

from . import __version__, get_diagnostics


def _format_report(report: dict[str, Any]) -> str:
    ort = report["onnxruntime"]
    lines = [
        f"npu-easy {__version__}",
        (
            f"Platform: {report['platform']['system']} "
            f"{report['platform']['release']} ({report['platform']['machine']})"
        ),
        f"Python: {report['platform']['python']}",
        (
            "ONNX Runtime: "
            + (str(ort["version"]) if ort["installed"] else "not installed")
        ),
        "Available providers: "
        + (", ".join(ort["available_providers"]) or "none"),
        f"Best provider: {report['best_provider']}",
        "Detected NPUs: " + (", ".join(report["hardware"]["NPU"]) or "none"),
        "Detected GPUs: " + (", ".join(report["hardware"]["GPU"]) or "none"),
    ]
    if report["recommendations"]:
        lines.append("Recommendations:")
        lines.extend(
            f"  - {recommendation}"
            for recommendation in report["recommendations"]
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m npu_easy",
        description="Inspect ONNX Runtime accelerator support.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("info",),
        default="info",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit machine-readable JSON.",
    )
    args = parser.parse_args(argv)

    report = get_diagnostics()
    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        print(_format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
