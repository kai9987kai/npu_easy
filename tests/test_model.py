from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from npu_easy import model


@pytest.fixture
def model_file(tmp_path: Path) -> Path:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"fake onnx model")
    return path


def test_model_runs_and_reports_provider(
    monkeypatch,
    fake_ort,
    model_file: Path,
) -> None:
    monkeypatch.setattr(model, "_load_onnxruntime", lambda: fake_ort)
    runner = model.NPUModel(model_file)
    input_data = np.ones((1, 10), dtype=np.float32)

    outputs = runner.run_named(input_data)
    info = runner.get_info()

    assert list(outputs) == ["output"]
    assert info["provider"] == "QNNExecutionProvider"
    assert info["provider_options"] == {"backend_type": "htp"}
    assert info["used_fallback"] is False


def test_explicit_provider_falls_back_to_cpu(
    monkeypatch,
    fake_ort,
    model_file: Path,
) -> None:
    monkeypatch.setattr(model, "_load_onnxruntime", lambda: fake_ort)

    runner = model.NPUModel(
        model_file,
        provider=["BrokenExecutionProvider", "CPUExecutionProvider"],
    )

    assert runner.provider == "CPUExecutionProvider"
    assert runner.fallback_reason
    assert runner.attempted_providers == [
        "BrokenExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_strict_provider_mode_raises(
    monkeypatch,
    fake_ort,
    model_file: Path,
) -> None:
    monkeypatch.setattr(model, "_load_onnxruntime", lambda: fake_ort)

    with pytest.raises(model.ProviderInitializationError):
        model.NPUModel(
            model_file,
            provider="BrokenExecutionProvider",
            allow_cpu_fallback=False,
        )


def test_benchmark_returns_structured_metrics(
    monkeypatch,
    fake_ort,
    model_file: Path,
) -> None:
    monkeypatch.setattr(model, "_load_onnxruntime", lambda: fake_ort)
    runner = model.NPUModel(model_file)

    metrics = runner.benchmark(
        np.ones((1, 10), dtype=np.float32),
        runs=3,
        warmup_runs=1,
    )

    assert metrics["runs"] == 3
    assert metrics["provider"] == "QNNExecutionProvider"
    assert metrics["min_ms"] <= metrics["median_ms"] <= metrics["max_ms"]
    assert metrics["throughput_per_second"] > 0


def test_missing_mapping_input_is_rejected(
    monkeypatch,
    fake_ort,
    model_file: Path,
) -> None:
    monkeypatch.setattr(model, "_load_onnxruntime", lambda: fake_ort)
    runner = model.NPUModel(model_file)

    with pytest.raises(ValueError, match="Missing model input"):
        runner.run({"wrong": np.ones((1, 10), dtype=np.float32)})


def test_directml_required_session_settings_are_applied(
    monkeypatch,
    fake_ort,
    model_file: Path,
) -> None:
    monkeypatch.setattr(model, "_load_onnxruntime", lambda: fake_ort)

    runner = model.NPUModel(
        model_file,
        provider="DmlExecutionProvider",
        execution_mode="parallel",
    )

    assert runner.session.sess_options.enable_mem_pattern is False
    assert (
        runner.session.sess_options.execution_mode
        == fake_ort.ExecutionMode.ORT_SEQUENTIAL
    )


def test_qnn_plugin_session_selects_npu(
    monkeypatch,
    fake_ort,
    model_file: Path,
) -> None:
    npu_type = type("DeviceType", (), {"name": "NPU"})()
    ep_device = type(
        "EpDevice",
        (),
        {
            "ep_name": "QNNExecutionProvider",
            "device": type("Device", (), {"type": npu_type})(),
        },
    )()
    fake_ort.get_available_providers = lambda: ["CPUExecutionProvider"]
    fake_ort.get_ep_devices = lambda: [ep_device]
    fake_ort.SessionOptions.add_provider_for_devices = (
        lambda self, devices, options: setattr(
            self,
            "plugin_provider",
            (devices, options),
        )
    )
    plugin = type(
        "QnnPlugin",
        (),
        {
            "get_qnn_htp_path": lambda self: "QnnHtp.dll",
            "get_qnn_gpu_path": lambda self: "QnnGpu.dll",
            "get_qnn_cpu_path": lambda self: "QnnCpu.dll",
        },
    )()

    monkeypatch.setattr(model, "_load_onnxruntime", lambda: fake_ort)
    monkeypatch.setattr(model, "_register_qnn_plugin", lambda runtime: plugin)

    runner = model.NPUModel(
        model_file,
        provider="QNNExecutionProvider",
        allow_cpu_fallback=False,
    )

    _, options = runner.session.sess_options.plugin_provider
    assert options == {"backend_path": "QnnHtp.dll"}
