import onnxruntime as ort
from npu_easy import get_available_npu_providers, get_best_provider

def verify_npu_detection():
    print("Checking available ONNX Runtime Execution Providers...")
    all_providers = ort.get_available_providers()
    print(f"Total providers found: {all_providers}")
    
    npu_providers = get_available_npu_providers()
    print(f"NPU-specific providers detected: {npu_providers}")
    
    best = get_best_provider()
    print(f"Best recommended provider: {best}")
    
    if any(p in all_providers for p in ['QNNExecutionProvider', 'OpenVINOExecutionProvider', 'DmlExecutionProvider']):
        print("\nSUCCESS: Hardware acceleration (NPU/GPU) is available/supported by this environment.")
    else:
        print("\nNOTE: Only CPU provider found. NPU acceleration might require additional drivers/packages.")

if __name__ == "__main__":
    verify_npu_detection()
