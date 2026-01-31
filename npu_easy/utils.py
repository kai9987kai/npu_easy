import subprocess
import os

def check_hardware():
    """Checks for NPU and GPU hardware using Windows built-in tools."""
    found = {"NPU": [], "GPU": []}
    npu_keywords = ['neural', 'npu', 'compute accelerator', 'movidius', 'intel ai boost', 'hexagon']
    gpu_keywords = ['nvidia', 'geforce', 'radeon', 'intel(r) iris', 'intel(r) graphics', 'arc(tm)']
    
    try:
        ps_cmd = 'powershell -Command "Get-PnpDevice | Select-Object FriendlyName | Out-String"'
        output = subprocess.check_output(ps_cmd, shell=True, stderr=subprocess.DEVNULL).decode('utf-8', errors='ignore')
        for line in output.splitlines():
            line = line.strip().lower()
            if any(k in line for k in npu_keywords):
                found["NPU"].append(line)
            elif any(k in line for k in gpu_keywords):
                found["GPU"].append(line)
    except Exception:
        pass
            
    found["NPU"] = sorted(list(set(found["NPU"])))
    found["GPU"] = sorted(list(set(found["GPU"])))
    return found

def get_all_hardware_providers():
    """Returns a map of available hardware to their best ONNX providers."""
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        
        mapping = {
            "NPU": [p for p in ['QNNExecutionProvider', 'OpenVINOExecutionProvider'] if p in available],
            "GPU": [p for p in ['CUDAExecutionProvider', 'ROCMExecutionProvider', 'DmlExecutionProvider'] if p in available],
            "CPU": ['CPUExecutionProvider']
        }
        return mapping
    except ImportError:
        return {}

def get_available_npu_providers():
    """Returns available NPU providers. Tries onnxruntime if installed, otherwise uses hardware detection."""
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        
        npu_priority = [
            'QNNExecutionProvider',
            'OpenVINOExecutionProvider',
            'DmlExecutionProvider',
        ]
        return [p for p in npu_priority if p in available]
    except ImportError:
        # If onnxruntime is not installed, we can still report what hardware we found
        hardware = check_npu_hardware()
        if hardware:
            return [f"Detected Hardware: {h}" for h in hardware]
        return []

def get_best_provider():
    """Returns the best provider name. Defaults to CPU if nothing else is found."""
    npu_providers = get_available_npu_providers()
    if npu_providers and not npu_providers[0].startswith("Detected"):
        return npu_providers[0]
    
    # Check if DirectML is likely available (standard on modern Windows)
    # but we can't be sure without ORT.
    return 'CPUExecutionProvider'
