## Installation

```bash
pip install .
```

`npu-easy` has **zero hard dependencies**. It will install quickly without downloading large packages. 

### Usage Acceleration
To actually use NPU acceleration, you should install the appropriate ONNX Runtime version for your hardware:

- **Intel**: `pip install onnxruntime-openvino`
- **AMD/Generic Windows**: `pip install onnxruntime-directml`
- **Qualcomm**: Install `onnxruntime` and the QNN SDK.

## Quick Start

### Basic NPU Usage
```python
from npu_easy import NPUModel
import numpy as np

# Automatically selects NPU if found
model = NPUModel("path/to/model.onnx")
results = model.run(np.random.randn(1, 10).astype(np.float32))
```

### Bonus: Multi-Hardware Bonus Feature
Run on NPU, GPU, and CPU all at once!
```python
from npu_easy import MultiRunner

multi = MultiRunner("path/to/model.onnx")
# Runs on all backends in parallel threads
all_results = multi.run_all(input_data)
```

### Multi-Threading
Set the number of internal threads for the engine:
```python
model = NPUModel("path/to/model.onnx", intra_op_num_threads=4)
```

## Hardware Support
- **Intel**: via OpenVINO Execution Provider.
- **AMD/NVIDIA/Integrated**: via DirectML Execution Provider.
- **Qualcomm**: via QNN Execution Provider.
