from __future__ import annotations

from types import SimpleNamespace

from npu_easy import utils


def test_directml_is_gpu_not_npu(monkeypatch) -> None:
    ort = SimpleNamespace(
        get_available_providers=lambda: [
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ]
    )
    monkeypatch.setattr(utils, "_load_onnxruntime", lambda: ort)

    assert utils.get_available_npu_providers() == []
    assert utils.get_all_hardware_providers() == {
        "NPU": [],
        "GPU": ["DmlExecutionProvider"],
        "CPU": ["CPUExecutionProvider"],
    }
    assert utils.get_best_provider() == "DmlExecutionProvider"


def test_npu_defaults_are_current() -> None:
    configs = utils.get_available_provider_configs(
        [
            "QNNExecutionProvider",
            "OpenVINOExecutionProvider",
            "CPUExecutionProvider",
        ]
    )

    assert configs[0].options == {"backend_type": "htp"}
    assert configs[1].options == {"device_type": "NPU"}
    assert configs[2].options == {"device_type": "GPU"}


def test_diagnostics_work_without_onnxruntime(monkeypatch) -> None:
    def missing_runtime():
        raise ImportError

    monkeypatch.setattr(utils, "_load_onnxruntime", missing_runtime)
    monkeypatch.setattr(
        utils,
        "check_hardware",
        lambda: {"NPU": ["Test NPU"], "GPU": []},
    )

    report = utils.get_diagnostics()

    assert report["onnxruntime"]["installed"] is False
    assert report["hardware"]["NPU"] == ["Test NPU"]
    assert report["best_provider"] == "CPUExecutionProvider"
    assert report["recommendations"]


def test_diagnostics_recommend_qnn_for_qualcomm_hardware(monkeypatch) -> None:
    def missing_runtime():
        raise ImportError

    monkeypatch.setattr(utils, "_load_onnxruntime", missing_runtime)
    monkeypatch.setattr(
        utils,
        "check_hardware",
        lambda: {"NPU": ["Qualcomm Hexagon NPU"], "GPU": []},
    )

    report = utils.get_diagnostics()

    assert "npu-easy[qualcomm]" in report["recommendations"][0]
