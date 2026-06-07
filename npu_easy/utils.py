from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Mapping, Sequence


NPU_PROVIDER_PRIORITY = (
    "QNNExecutionProvider",
    "OpenVINOExecutionProvider",
)

GPU_PROVIDER_PRIORITY = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "MIGraphXExecutionProvider",
    "DmlExecutionProvider",
    "ROCMExecutionProvider",
    "CoreMLExecutionProvider",
)


@dataclass(frozen=True)
class ProviderConfig:
    """An ONNX Runtime provider and the device-specific options it needs."""

    name: str
    device: str
    options: Mapping[str, str] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.device}:{self.name}"


def _load_onnxruntime() -> Any:
    return importlib.import_module("onnxruntime")


def _deduplicate(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


@lru_cache(maxsize=1)
def _detect_windows_hardware() -> tuple[tuple[str, ...], tuple[str, ...]]:
    if os.name != "nt":
        return (), ()

    executable = shutil.which("powershell.exe") or shutil.which("pwsh.exe")
    if not executable:
        return (), ()

    command = (
        "Get-CimInstance Win32_PnPEntity "
        "-Filter \"PNPClass='Display' OR PNPClass='ComputeAccelerator'\" "
        "-ErrorAction SilentlyContinue | "
        "Where-Object { $_.Name } | "
        "Select-Object -ExpandProperty Name"
    )

    try:
        completed = subprocess.run(
            [
                executable,
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                command,
            ],
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="ignore",
            text=True,
            timeout=8,
        )
    except (OSError, subprocess.SubprocessError):
        return (), ()

    if completed.returncode != 0:
        return (), ()

    npu_keywords = (
        "neural processing",
        "neural processor",
        "npu",
        "compute accelerator",
        "intel(r) ai boost",
        "intel ai boost",
        "movidius",
        "hexagon",
    )
    gpu_keywords = (
        "nvidia",
        "geforce",
        "quadro",
        "radeon",
        "intel(r) arc",
        "intel(r) iris",
        "intel(r) graphics",
        "adreno",
    )

    npu_devices: list[str] = []
    gpu_devices: list[str] = []
    for raw_name in completed.stdout.splitlines():
        name = raw_name.strip()
        lowered = name.lower()
        if not name:
            continue
        if any(keyword in lowered for keyword in npu_keywords):
            npu_devices.append(name)
        elif any(keyword in lowered for keyword in gpu_keywords):
            gpu_devices.append(name)

    return (
        tuple(sorted(set(npu_devices), key=str.casefold)),
        tuple(sorted(set(gpu_devices), key=str.casefold)),
    )


def check_hardware(*, refresh: bool = False) -> dict[str, list[str]]:
    """Return detected Windows NPU and GPU device names.

    Detection is best effort and has no external Python dependencies. An empty
    list means the operating-system probe found nothing, not that no device
    exists.
    """

    if refresh:
        _detect_windows_hardware.cache_clear()
    npu_devices, gpu_devices = _detect_windows_hardware()
    return {"NPU": list(npu_devices), "GPU": list(gpu_devices)}


def get_available_provider_configs(
    available_providers: Sequence[str] | None = None,
) -> list[ProviderConfig]:
    """Return installed providers in NPU, GPU, then CPU preference order."""

    if available_providers is None:
        try:
            available_providers = _load_onnxruntime().get_available_providers()
        except (ImportError, OSError):
            available_providers = ()

    available = set(available_providers)
    configs: list[ProviderConfig] = []

    if "QNNExecutionProvider" in available:
        configs.append(
            ProviderConfig(
                "QNNExecutionProvider",
                "NPU",
                {"backend_type": "htp"},
            )
        )
    if "OpenVINOExecutionProvider" in available:
        configs.append(
            ProviderConfig(
                "OpenVINOExecutionProvider",
                "NPU",
                {"device_type": "NPU"},
            )
        )

    for provider in GPU_PROVIDER_PRIORITY:
        if provider in available:
            configs.append(ProviderConfig(provider, "GPU"))
    if "OpenVINOExecutionProvider" in available:
        configs.append(
            ProviderConfig(
                "OpenVINOExecutionProvider",
                "GPU",
                {"device_type": "GPU"},
            )
        )

    if "CPUExecutionProvider" in available or not available:
        configs.append(ProviderConfig("CPUExecutionProvider", "CPU"))

    return configs


def get_all_hardware_providers() -> dict[str, list[str]]:
    """Return installed ONNX Runtime providers grouped by device class."""

    mapping: dict[str, list[str]] = {"NPU": [], "GPU": [], "CPU": []}
    for config in get_available_provider_configs():
        mapping[config.device].append(config.name)
    return mapping


def get_available_npu_providers() -> list[str]:
    """Return installed providers that can target an NPU."""

    return [
        config.name
        for config in get_available_provider_configs()
        if config.device == "NPU"
    ]


def get_best_provider(
    preferred_devices: Sequence[str] = ("NPU", "GPU", "CPU"),
) -> str:
    """Return the highest-priority installed provider."""

    normalized_devices = tuple(device.upper() for device in preferred_devices)
    configs = get_available_provider_configs()
    for device in normalized_devices:
        for config in configs:
            if config.device == device:
                return config.name
    return "CPUExecutionProvider"


def get_provider_config(
    provider: str,
    *,
    options: Mapping[str, str] | None = None,
) -> ProviderConfig:
    """Build a provider config, applying sensible NPU defaults."""

    for config in get_available_provider_configs():
        if config.name == provider:
            return ProviderConfig(
                config.name,
                config.device,
                dict(config.options if options is None else options),
            )

    if provider in NPU_PROVIDER_PRIORITY:
        device = "NPU"
    elif provider in GPU_PROVIDER_PRIORITY:
        device = "GPU"
    elif provider == "CPUExecutionProvider":
        device = "CPU"
    else:
        device = "ACCELERATOR"

    defaults: dict[str, str] = {}
    if options is None and provider == "QNNExecutionProvider":
        defaults = {"backend_type": "htp"}
    elif options is None and provider == "OpenVINOExecutionProvider":
        defaults = {"device_type": "NPU"}

    return ProviderConfig(provider, device, defaults if options is None else dict(options))


def get_diagnostics() -> dict[str, Any]:
    """Return a JSON-serializable runtime and hardware diagnostic report."""

    hardware = check_hardware()
    try:
        ort = _load_onnxruntime()
        available = list(ort.get_available_providers())
        ort_info = {
            "installed": True,
            "version": getattr(ort, "__version__", "unknown"),
            "available_providers": available,
        }
    except (ImportError, OSError) as exc:
        available = []
        ort_info = {
            "installed": False,
            "version": None,
            "available_providers": [],
            "import_error": str(exc) or type(exc).__name__,
        }
    else:
        ort_info["import_error"] = None

    provider_configs = get_available_provider_configs(available)
    recommendations: list[str] = []
    if not ort_info["installed"]:
        detected_names = " ".join(hardware["NPU"] + hardware["GPU"]).lower()
        if "qualcomm" in detected_names or "hexagon" in detected_names:
            recommendations.append(
                "A Qualcomm NPU was detected. Install a compatible Windows ARM64 "
                "runtime with: pip install 'npu-easy[qualcomm]'."
            )
        elif "intel" in detected_names and hardware["NPU"]:
            recommendations.append(
                "An Intel NPU was detected. Install OpenVINO support with: "
                "pip install 'npu-easy[intel]'."
            )
        elif hardware["GPU"]:
            recommendations.append(
                "A Windows GPU was detected. Install DirectML support with: "
                "pip install 'npu-easy[directml]'."
            )
        else:
            recommendations.append(
                "Install an ONNX Runtime extra, for example "
                "'npu-easy[intel]', 'npu-easy[qualcomm]', or "
                "'npu-easy[directml]'."
            )
    elif available == ["CPUExecutionProvider"] or not any(
        config.device in {"NPU", "GPU"} for config in provider_configs
    ):
        recommendations.append(
            "Only CPU execution is available. Install the runtime package for "
            "your accelerator and update its device driver."
        )

    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "python_executable": sys.executable,
        },
        "hardware": hardware,
        "onnxruntime": ort_info,
        "providers": [
            {
                "name": config.name,
                "device": config.device,
                "options": dict(config.options),
            }
            for config in provider_configs
        ],
        "best_provider": get_best_provider(),
        "recommendations": _deduplicate(recommendations),
    }
