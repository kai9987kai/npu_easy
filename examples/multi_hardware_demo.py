import numpy as np
import time
from npu_easy import NPUModel, MultiRunner, check_hardware

def main():
    # 1. Check what we have
    print("--- Hardware Detection ---")
    hw = check_hardware()
    print(f"NPUs found: {hw['NPU']}")
    print(f"GPUs found: {hw['GPU']}")
    
    model_path = "models/test_model.onnx"
    
    # Ensure model exists from previous steps or create it
    import os
    if not os.path.exists(model_path):
        print("\nPlease run scripts/create_test_model.py first to generate the dummy model.")
        return

    # 2. Bonus Feature: Multi-Hardware Parallel Execution
    print("\n--- Bonus: Multi-Hardware Execution ---")
    multi = MultiRunner(model_path)
    
    dummy_input = np.random.randn(1, 10).astype(np.float32)
    
    start = time.time()
    results = multi.run_all(dummy_input)
    end = time.time()
    
    for hw_type, res in results.items():
        print(f"Result from {hw_type}: {res[0][0][:3]}... (success)")
        
    print(f"Parallel execution took: {end - start:.4f}s")

    # 3. Threading Support
    print("\n--- Threading Support ---")
    print("Initializing model with 4 internal threads...")
    threaded_model = NPUModel(model_path, intra_op_num_threads=4)
    res = threaded_model.run(dummy_input)
    print("Threaded inference successful.")

if __name__ == "__main__":
    main()
