from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from npu_easy import MultiRunner, NPUModel, get_diagnostics


def main() -> None:
    print("--- Diagnostics ---")
    print(json.dumps(get_diagnostics(), indent=2))

    model_path = Path("models/test_model.onnx")
    if not model_path.exists():
        print("\nRun scripts/create_test_model.py to generate the test model.")
        return

    input_data = np.random.randn(1, 10).astype(np.float32)

    print("\n--- Best Provider ---")
    model = NPUModel(model_path, intra_op_num_threads=4)
    result = model.run_named(input_data)
    print(json.dumps(model.get_info(), indent=2))
    print(f"Output names: {list(result)}")
    print(f"Benchmark: {model.benchmark(input_data, runs=20)}")

    print("\n--- Multi-Hardware Comparison ---")
    runner = MultiRunner(model_path)
    for device, metrics in runner.benchmark_all(input_data, runs=20).items():
        print(f"{device}: {metrics}")
    print(json.dumps(runner.get_info(), indent=2))


if __name__ == "__main__":
    main()
