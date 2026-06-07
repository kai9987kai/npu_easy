from __future__ import annotations

import json

from npu_easy import get_diagnostics


def verify_detection() -> None:
    print(json.dumps(get_diagnostics(), indent=2))


if __name__ == "__main__":
    verify_detection()
