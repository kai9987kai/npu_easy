from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest


@dataclass
class FakeNode:
    name: str
    shape: list[int]
    type: str = "tensor(float)"


class FakeSessionOptions:
    def __init__(self) -> None:
        self.config: dict[str, str] = {}
        self.enable_profiling = False
        self.enable_mem_pattern = True

    def add_session_config_entry(self, key: str, value: str) -> None:
        self.config[key] = value


class FakeSession:
    def __init__(
        self,
        model_path: Any,
        *,
        sess_options: FakeSessionOptions,
        providers: list[str] | None = None,
        provider_options: list[dict[str, str]] | None = None,
    ) -> None:
        del model_path
        self.sess_options = sess_options
        self.provider_options = provider_options or []
        if providers is None and hasattr(sess_options, "plugin_provider"):
            providers = ["QNNExecutionProvider", "CPUExecutionProvider"]
        providers = providers or ["CPUExecutionProvider"]
        if providers[0] == "BrokenExecutionProvider":
            raise RuntimeError("provider initialization failed")
        self.providers = providers

    def get_providers(self) -> list[str]:
        return self.providers

    def get_inputs(self) -> list[FakeNode]:
        return [FakeNode("input", [1, 10])]

    def get_outputs(self) -> list[FakeNode]:
        return [FakeNode("output", [1, 5])]

    def run(
        self,
        output_names: list[str],
        input_feed: dict[str, Any],
        run_options: Any,
    ) -> list[Any]:
        del run_options
        return [
            {
                "name": output_name,
                "input": input_feed["input"],
            }
            for output_name in output_names
        ]

    def end_profiling(self) -> str:
        return "profile.json"


@pytest.fixture
def fake_ort() -> SimpleNamespace:
    return SimpleNamespace(
        __version__="1.23.0",
        get_available_providers=lambda: [
            "QNNExecutionProvider",
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ],
        SessionOptions=FakeSessionOptions,
        InferenceSession=FakeSession,
        GraphOptimizationLevel=SimpleNamespace(
            ORT_DISABLE_ALL=0,
            ORT_ENABLE_BASIC=1,
            ORT_ENABLE_EXTENDED=2,
            ORT_ENABLE_ALL=99,
        ),
        ExecutionMode=SimpleNamespace(
            ORT_SEQUENTIAL=0,
            ORT_PARALLEL=1,
        ),
    )
