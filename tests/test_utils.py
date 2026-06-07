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
    openvino_gpu = next(
        config
        for config in configs
        if config.name == "OpenVINOExecutionProvider"
        and config.device == "GPU"
    )
    assert openvino_gpu.options == {"device_type": "GPU"}


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


def test_plugin_provider_names_are_included(monkeypatch) -> None:
    device = SimpleNamespace(ep_name="QNNExecutionProvider")
    ort = SimpleNamespace(
        get_available_providers=lambda: ["CPUExecutionProvider"],
        get_ep_devices=lambda: [device],
        register_execution_provider_library=lambda name, path: None,
    )
    monkeypatch.setattr(utils, "_register_qnn_plugin", lambda runtime: object())

    providers = utils._get_available_provider_names(ort)

    assert providers == ["CPUExecutionProvider", "QNNExecutionProvider"]


def test_qnn_backend_option_selects_matching_device(monkeypatch) -> None:
    monkeypatch.setattr(
        utils,
        "get_available_provider_configs",
        lambda: [
            utils.ProviderConfig(
                "QNNExecutionProvider",
                "NPU",
                {"backend_type": "htp"},
            ),
            utils.ProviderConfig(
                "QNNExecutionProvider",
                "GPU",
                {"backend_type": "gpu"},
            ),
        ],
    )

    config = utils.get_provider_config(
        "QNNExecutionProvider",
        options={"backend_type": "gpu"},
    )

    assert config.device == "GPU"
