from __future__ import annotations

import importlib
import logging
import math
import os
import statistics
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .utils import (
    ProviderConfig,
    _device_type_name,
    _get_available_provider_names,
    _register_qnn_plugin,
    get_available_provider_configs,
    get_provider_config,
)


logger = logging.getLogger(__name__)


class ProviderInitializationError(RuntimeError):
    """Raised when no requested ONNX Runtime provider can be initialized."""


def _load_onnxruntime() -> Any:
    try:
        return importlib.import_module("onnxruntime")
    except (ImportError, OSError) as exc:
        raise ImportError(
            "ONNX Runtime is required for inference. Install a hardware extra, "
            "for example: pip install 'npu-easy[intel]', "
            "'npu-easy[qualcomm]', or 'npu-easy[directml]'."
        ) from exc


def _validate_positive_integer(name: str, value: int | None) -> None:
    if value is not None and (isinstance(value, bool) or value <= 0):
        raise ValueError(f"{name} must be a positive integer.")


class NPUModel:
    """Run an ONNX model on the best available NPU, GPU, or CPU provider."""

    def __init__(
        self,
        model_path: str | os.PathLike[str] | bytes,
        provider: str | Sequence[str] | None = None,
        provider_options: Mapping[str, str] | None = None,
        intra_op_num_threads: int | None = None,
        *,
        inter_op_num_threads: int | None = None,
        allow_cpu_fallback: bool = True,
        graph_optimization_level: str = "all",
        execution_mode: str = "sequential",
        enable_profiling: bool = False,
        profile_file_prefix: str | None = None,
        session_config: Mapping[str, str] | None = None,
        session_options: Any | None = None,
        log_severity_level: int | None = 2,
    ) -> None:
        _validate_positive_integer("intra_op_num_threads", intra_op_num_threads)
        _validate_positive_integer("inter_op_num_threads", inter_op_num_threads)

        if not isinstance(model_path, bytes):
            model_path = os.fspath(model_path)
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model_path = model_path
        self.allow_cpu_fallback = allow_cpu_fallback
        self.fallback_reason: str | None = None
        self.attempted_providers: list[str] = []
        self._ort = _load_onnxruntime()
        self._qnn_plugin = _register_qnn_plugin(self._ort)
        self._session_options_settings = {
            "intra_op_num_threads": intra_op_num_threads,
            "inter_op_num_threads": inter_op_num_threads,
            "graph_optimization_level": graph_optimization_level,
            "execution_mode": execution_mode,
            "enable_profiling": enable_profiling,
            "profile_file_prefix": profile_file_prefix,
            "session_config": dict(session_config or {}),
            "log_severity_level": log_severity_level,
        }
        self._custom_session_options = session_options

        candidates = self._build_candidates(provider, provider_options)
        self.session, selected_config = self._initialize_session(candidates)
        self.requested_provider = selected_config.name
        self.provider_options = dict(selected_config.options)

        active_providers = list(self.session.get_providers())
        self.active_providers = active_providers
        self.provider = active_providers[0] if active_providers else selected_config.name
        self.used_fallback = self.provider != self.requested_provider
        if self.used_fallback and self.fallback_reason is None:
            self.fallback_reason = (
                f"{self.requested_provider} was requested but {self.provider} is active."
            )

        self.input_metadata = [
            {"name": item.name, "shape": item.shape, "type": item.type}
            for item in self.session.get_inputs()
        ]
        self.output_metadata = [
            {"name": item.name, "shape": item.shape, "type": item.type}
            for item in self.session.get_outputs()
        ]
        self.input_names = [item["name"] for item in self.input_metadata]
        self.output_names = [item["name"] for item in self.output_metadata]

    def _build_candidates(
        self,
        provider: str | Sequence[str] | None,
        provider_options: Mapping[str, str] | None,
    ) -> list[ProviderConfig]:
        if provider is None:
            candidates = get_available_provider_configs(
                _get_available_provider_names(self._ort)
            )
            if provider_options is not None and candidates:
                first = candidates[0]
                candidates[0] = ProviderConfig(
                    first.name,
                    first.device,
                    dict(provider_options),
                )
            return candidates

        providers = [provider] if isinstance(provider, str) else list(provider)
        if not providers:
            raise ValueError("provider must contain at least one provider name.")
        if provider_options is not None and len(providers) > 1:
            raise ValueError(
                "provider_options can only be used with a single explicit provider."
            )

        return [
            get_provider_config(
                name,
                options=provider_options if index == 0 else None,
            )
            for index, name in enumerate(providers)
        ]

    def _create_session_options(self, candidate: ProviderConfig) -> Any:
        if self._custom_session_options is not None:
            options = self._custom_session_options
        else:
            settings = self._session_options_settings
            options = self._ort.SessionOptions()

            if settings["intra_op_num_threads"] is not None:
                options.intra_op_num_threads = settings["intra_op_num_threads"]
            if settings["inter_op_num_threads"] is not None:
                options.inter_op_num_threads = settings["inter_op_num_threads"]

            optimization_levels = {
                "disabled": "ORT_DISABLE_ALL",
                "basic": "ORT_ENABLE_BASIC",
                "extended": "ORT_ENABLE_EXTENDED",
                "all": "ORT_ENABLE_ALL",
            }
            optimization = str(settings["graph_optimization_level"]).lower()
            if optimization not in optimization_levels:
                valid = ", ".join(optimization_levels)
                raise ValueError(
                    f"graph_optimization_level must be one of: {valid}."
                )
            options.graph_optimization_level = getattr(
                self._ort.GraphOptimizationLevel,
                optimization_levels[optimization],
            )

            execution_modes = {
                "sequential": "ORT_SEQUENTIAL",
                "parallel": "ORT_PARALLEL",
            }
            mode = str(settings["execution_mode"]).lower()
            if mode not in execution_modes:
                valid = ", ".join(execution_modes)
                raise ValueError(f"execution_mode must be one of: {valid}.")
            options.execution_mode = getattr(
                self._ort.ExecutionMode,
                execution_modes[mode],
            )

            options.enable_profiling = bool(settings["enable_profiling"])
            if settings["profile_file_prefix"]:
                options.profile_file_prefix = settings["profile_file_prefix"]
            if settings["log_severity_level"] is not None:
                options.log_severity_level = settings["log_severity_level"]

        for key, value in self._session_options_settings["session_config"].items():
            options.add_session_config_entry(str(key), str(value))
        if not self.allow_cpu_fallback:
            options.add_session_config_entry(
                "session.disable_cpu_ep_fallback",
                "1",
            )

        if candidate.name == "DmlExecutionProvider":
            options.enable_mem_pattern = False
            options.execution_mode = self._ort.ExecutionMode.ORT_SEQUENTIAL

        return options

    def _initialize_session(
        self,
        candidates: Sequence[ProviderConfig],
    ) -> tuple[Any, ProviderConfig]:
        failures: list[str] = []
        fallback_session: tuple[Any, ProviderConfig] | None = None

        for candidate in candidates:
            if (
                candidate.name == "CPUExecutionProvider"
                and not self.allow_cpu_fallback
            ):
                continue

            self.attempted_providers.append(candidate.name)
            if (
                candidate.name == "QNNExecutionProvider"
                and self._qnn_plugin is not None
                and candidate.name not in self._ort.get_available_providers()
            ):
                try:
                    session = self._initialize_qnn_plugin_session(candidate)
                except Exception as exc:
                    failures.append(f"{candidate.label}: {exc}")
                    logger.debug(
                        "Failed to initialize %s",
                        candidate.label,
                        exc_info=True,
                    )
                    continue
                if failures:
                    self.fallback_reason = "; ".join(failures)
                return session, candidate

            providers = [candidate.name]
            provider_options = [dict(candidate.options)]
            available = set(self._ort.get_available_providers())
            if (
                candidate.name == "TensorrtExecutionProvider"
                and "CUDAExecutionProvider" in available
            ):
                providers.append("CUDAExecutionProvider")
                provider_options.append({})
            if (
                self.allow_cpu_fallback
                and candidate.name != "CPUExecutionProvider"
            ):
                providers.append("CPUExecutionProvider")
                provider_options.append({})

            try:
                session = self._ort.InferenceSession(
                    self.model_path,
                    sess_options=self._create_session_options(candidate),
                    providers=providers,
                    provider_options=provider_options,
                )
            except Exception as exc:
                failures.append(f"{candidate.name}: {exc}")
                logger.debug(
                    "Failed to initialize %s",
                    candidate.name,
                    exc_info=True,
                )
                continue

            active = list(session.get_providers())
            if candidate.name in active:
                if failures:
                    self.fallback_reason = "; ".join(failures)
                return session, candidate

            message = (
                f"{candidate.name}: runtime activated "
                f"{', '.join(active) if active else 'no provider'}"
            )
            failures.append(message)
            if self.allow_cpu_fallback and "CPUExecutionProvider" in active:
                fallback_session = (session, candidate)

        if fallback_session is not None:
            self.fallback_reason = "; ".join(failures)
            logger.warning("Using CPU fallback: %s", self.fallback_reason)
            return fallback_session

        detail = "; ".join(failures) or "No providers were available."
        raise ProviderInitializationError(
            f"Could not initialize an ONNX Runtime session. {detail}"
        )

    def _initialize_qnn_plugin_session(self, candidate: ProviderConfig) -> Any:
        selected_devices = [
            device
            for device in self._ort.get_ep_devices()
            if device.ep_name == candidate.name
            and _device_type_name(device) == candidate.device
        ]
        if not selected_devices:
            raise ProviderInitializationError(
                f"No QNN {candidate.device} device was registered."
            )

        options = dict(candidate.options)
        backend_type = options.pop("backend_type", None)
        if "backend_path" not in options:
            if candidate.device == "NPU" or backend_type == "htp":
                options["backend_path"] = self._qnn_plugin.get_qnn_htp_path()
            elif candidate.device == "GPU" or backend_type == "gpu":
                options["backend_path"] = self._qnn_plugin.get_qnn_gpu_path()
            else:
                options["backend_path"] = self._qnn_plugin.get_qnn_cpu_path()

        session_options = self._create_session_options(candidate)
        session_options.add_provider_for_devices(selected_devices, options)
        return self._ort.InferenceSession(
            self.model_path,
            sess_options=session_options,
        )

    def _prepare_input(self, input_data: Any) -> dict[str, Any]:
        if isinstance(input_data, Mapping):
            missing = [name for name in self.input_names if name not in input_data]
            if missing:
                raise ValueError(
                    "Missing model input(s): " + ", ".join(missing)
                )
            return dict(input_data)

        if hasattr(input_data, "__array__"):
            if len(self.input_names) != 1:
                raise ValueError(
                    "Model has multiple inputs; provide a mapping keyed by input name."
                )
            return {self.input_names[0]: input_data}

        raise TypeError(
            "input_data must be an array-like object or a mapping of input names "
            "to array-like values."
        )

    def run(
        self,
        input_data: Any,
        *,
        output_names: Sequence[str] | None = None,
        run_options: Any | None = None,
    ) -> list[Any]:
        """Run one inference and return outputs in model order."""

        requested_outputs = (
            self.output_names if output_names is None else list(output_names)
        )
        return self.session.run(
            requested_outputs,
            self._prepare_input(input_data),
            run_options,
        )

    def run_named(
        self,
        input_data: Any,
        *,
        output_names: Sequence[str] | None = None,
        run_options: Any | None = None,
    ) -> dict[str, Any]:
        """Run inference and return a mapping keyed by output name."""

        requested_outputs = (
            self.output_names if output_names is None else list(output_names)
        )
        values = self.run(
            input_data,
            output_names=requested_outputs,
            run_options=run_options,
        )
        return dict(zip(requested_outputs, values))

    def warmup(self, input_data: Any, *, iterations: int = 3) -> None:
        """Run untimed inference iterations to warm caches and the device."""

        _validate_positive_integer("iterations", iterations)
        for _ in range(iterations):
            self.run(input_data)

    def benchmark(
        self,
        input_data: Any,
        *,
        runs: int = 20,
        warmup_runs: int = 3,
    ) -> dict[str, Any]:
        """Measure end-to-end inference latency and throughput."""

        _validate_positive_integer("runs", runs)
        if warmup_runs < 0:
            raise ValueError("warmup_runs must be zero or greater.")
        if warmup_runs:
            self.warmup(input_data, iterations=warmup_runs)

        latencies_ms: list[float] = []
        for _ in range(runs):
            started = time.perf_counter()
            self.run(input_data)
            latencies_ms.append((time.perf_counter() - started) * 1000)

        ordered = sorted(latencies_ms)
        p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
        mean_ms = statistics.fmean(latencies_ms)
        return {
            "provider": self.provider,
            "runs": runs,
            "warmup_runs": warmup_runs,
            "mean_ms": mean_ms,
            "median_ms": statistics.median(latencies_ms),
            "p95_ms": ordered[p95_index],
            "min_ms": ordered[0],
            "max_ms": ordered[-1],
            "throughput_per_second": 1000 / mean_ms if mean_ms else math.inf,
        }

    def end_profiling(self) -> str:
        """Stop ONNX Runtime profiling and return the generated profile path."""

        return self.session.end_profiling()

    def get_info(self) -> dict[str, Any]:
        """Return JSON-friendly model, provider, and runtime metadata."""

        return {
            "provider": self.provider,
            "requested_provider": self.requested_provider,
            "active_providers": self.active_providers,
            "provider_options": self.provider_options,
            "used_fallback": self.used_fallback,
            "fallback_reason": self.fallback_reason,
            "attempted_providers": self.attempted_providers,
            "inputs": self.input_names,
            "outputs": self.output_names,
            "input_metadata": self.input_metadata,
            "output_metadata": self.output_metadata,
            "available_providers": _get_available_provider_names(self._ort),
            "onnxruntime_version": getattr(self._ort, "__version__", "unknown"),
        }


class MultiRunner:
    """Run the same model across the best NPU, GPU, and CPU providers."""

    def __init__(
        self,
        model_path: str | os.PathLike[str] | bytes,
        *,
        devices: Sequence[str] = ("NPU", "GPU", "CPU"),
        strict_accelerators: bool = True,
        model_options: Mapping[str, Any] | None = None,
    ) -> None:
        requested_devices = tuple(device.upper() for device in devices)
        invalid = [
            device
            for device in requested_devices
            if device not in {"NPU", "GPU", "CPU"}
        ]
        if invalid:
            raise ValueError(
                "Unknown device class(es): " + ", ".join(invalid)
            )

        configs = get_available_provider_configs()
        options = dict(model_options or {})
        self.models: dict[str, NPUModel] = {}
        self.initialization_errors: dict[str, str] = {}

        for device in requested_devices:
            config = next(
                (item for item in configs if item.device == device),
                None,
            )
            if config is None:
                continue

            allow_fallback = device == "CPU" or not strict_accelerators
            try:
                model = NPUModel(
                    model_path,
                    provider=config.name,
                    provider_options=config.options,
                    allow_cpu_fallback=allow_fallback,
                    **options,
                )
            except (ImportError, ProviderInitializationError) as exc:
                self.initialization_errors[device] = str(exc)
                continue

            if device != "CPU" and model.provider == "CPUExecutionProvider":
                self.initialization_errors[device] = (
                    model.fallback_reason or "Accelerator fell back to CPU."
                )
                continue
            self.models[device] = model

        if not self.models:
            details = "; ".join(
                f"{device}: {error}"
                for device, error in self.initialization_errors.items()
            )
            raise ProviderInitializationError(
                "No hardware runners could be initialized."
                + (f" {details}" if details else "")
            )

    def run_all(self, input_data: Any) -> dict[str, list[Any]]:
        """Run inference on every initialized device in parallel."""

        with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
            futures = {
                device: executor.submit(model.run, input_data)
                for device, model in self.models.items()
            }
            return {
                device: future.result()
                for device, future in futures.items()
            }

    def benchmark_all(
        self,
        input_data: Any,
        *,
        runs: int = 20,
        warmup_runs: int = 3,
    ) -> dict[str, dict[str, Any]]:
        """Benchmark each initialized device in parallel."""

        with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
            futures = {
                device: executor.submit(
                    model.benchmark,
                    input_data,
                    runs=runs,
                    warmup_runs=warmup_runs,
                )
                for device, model in self.models.items()
            }
            return {
                device: future.result()
                for device, future in futures.items()
            }

    def get_info(self) -> dict[str, Any]:
        return {
            "runners": {
                device: model.get_info()
                for device, model in self.models.items()
            },
            "initialization_errors": dict(self.initialization_errors),
        }
