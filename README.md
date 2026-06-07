# npu-easy

`npu-easy` is a small, zero-hard-dependency wrapper around ONNX Runtime. It
discovers installed execution providers, prefers an NPU, can fall back to a
GPU or CPU, and reports exactly which provider is active.

## Install

Install the package with the runtime extra for your hardware:

```bash
# Intel Core Ultra NPU through OpenVINO
pip install "npu-easy[intel]"

# Qualcomm Snapdragon NPU through QNN
pip install "npu-easy[qualcomm]"

# Broad Windows GPU support through DirectML
pip install "npu-easy[directml]"

# NVIDIA CUDA
pip install "npu-easy[nvidia]"

# CPU only
pip install "npu-easy[cpu]"
```

Provider wheels have their own Python, architecture, driver, and operating
system requirements. In particular, Qualcomm NPU inference requires a
compatible Windows ARM64 `onnxruntime-qnn` wheel. DirectML targets GPUs, not
NPUs.

## Diagnose The Machine

The diagnostics command works even before ONNX Runtime is installed:

```bash
npu-easy info
python -m npu_easy info --json
```

It reports detected NPU/GPU devices, Python and platform details, installed
execution providers, the selected provider, and installation guidance.

## Quick Start

```python
import numpy as np

from npu_easy import NPUModel

model = NPUModel("model.onnx")
outputs = model.run(np.random.randn(1, 10).astype(np.float32))

print(model.get_info()["provider"])
```

Automatic selection uses this order:

1. Qualcomm QNN or Intel OpenVINO NPU
2. TensorRT, CUDA, MIGraphX, DirectML, ROCm, Core ML, or OpenVINO GPU
3. ONNX Runtime CPU

## Explicit Providers And Fallback

Choose a provider and inspect whether fallback occurred:

```python
model = NPUModel(
    "model.onnx",
    provider="QNNExecutionProvider",
    provider_options={"backend_type": "htp"},
)

info = model.get_info()
print(info["requested_provider"], info["provider"], info["used_fallback"])
```

For validation or benchmarking, disable CPU fallback so unsupported models do
not appear to be accelerated:

```python
from npu_easy import ProviderInitializationError

try:
    model = NPUModel(
        "model.onnx",
        provider="QNNExecutionProvider",
        allow_cpu_fallback=False,
    )
except ProviderInitializationError as error:
    print(error)
```

You can also pass a provider preference list:

```python
model = NPUModel(
    "model.onnx",
    provider=[
        "QNNExecutionProvider",
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ],
)
```

## Multiple Inputs And Named Outputs

```python
outputs = model.run(
    {
        "tokens": token_array,
        "attention_mask": mask_array,
    }
)

named_outputs = model.run_named(input_array)
```

## Warmup, Benchmarking, And Profiling

```python
model.warmup(input_array, iterations=5)

metrics = model.benchmark(input_array, runs=100, warmup_runs=10)
print(metrics["median_ms"], metrics["p95_ms"])

profiled_model = NPUModel(
    "model.onnx",
    enable_profiling=True,
    profile_file_prefix="profiles/model",
)
profiled_model.run(input_array)
profile_path = profiled_model.end_profiling()
```

Threading and graph settings are also configurable:

```python
model = NPUModel(
    "model.onnx",
    intra_op_num_threads=4,
    inter_op_num_threads=2,
    graph_optimization_level="all",
    execution_mode="sequential",
)
```

## Compare Hardware

`MultiRunner` creates one strict runner per available device class. Accelerator
runners do not silently fall back to CPU by default.

```python
from npu_easy import MultiRunner

runner = MultiRunner("model.onnx")
all_outputs = runner.run_all(input_array)
benchmarks = runner.benchmark_all(input_array, runs=50)

print(runner.get_info())
```

Use `devices=("NPU", "CPU")` to limit the comparison. Initialization failures
for unavailable or unsupported accelerators are exposed by
`runner.get_info()["initialization_errors"]`.

## Provider Notes

- OpenVINO uses `device_type="NPU"` by default.
- QNN uses `backend_type="htp"` by default.
- TensorRT uses CUDA as its accelerator fallback when both are installed.
- DirectML is a broad Windows GPU provider. New Windows deployment work may
  also consider Windows ML, while DirectML remains supported by ONNX Runtime.
  Its required sequential execution and disabled memory-pattern settings are
  applied automatically.
- Model operator support and quantization requirements vary by provider.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
python -m build
```
