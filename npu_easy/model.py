import os
from .utils import get_best_provider, get_all_hardware_providers

class NPUModel:
    def __init__(self, model_path, provider=None, provider_options=None, intra_op_num_threads=None):
        """
        Initializes the NPUModel with lazy imports and optional threading.
        """
        try:
            global ort
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for NPU inference. "
                "Please install it using: pip install onnxruntime-directml (or onnxruntime-openvino)"
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.provider = provider or get_best_provider()
        
        sess_options = ort.SessionOptions()
        if intra_op_num_threads:
            sess_options.intra_op_num_threads = intra_op_num_threads

        if provider_options is None:
            provider_options = {}
            if self.provider == 'OpenVINOExecutionProvider':
                provider_options = {'device_type': 'NPU'}
            elif self.provider == 'QNNExecutionProvider':
                provider_options = {'backend_path': 'QnnHtp.dll'} 

        print(f"Initializing NPUModel with provider: {self.provider}")
        
        try:
            self.session = ort.InferenceSession(
                model_path, 
                sess_options=sess_options,
                providers=[self.provider],
                provider_options=[provider_options] if provider_options else None
            )
        except Exception as e:
            print(f"Failed to initialize with {self.provider}. Falling back to CPU.")
            self.provider = 'CPUExecutionProvider'
            self.session = ort.InferenceSession(model_path, sess_options=sess_options, providers=['CPUExecutionProvider'])

        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

    def run(self, input_data):
        """
        Runs inference on the selected hardware.
        """
        input_feed = self._prepare_input(input_data)
        return self.session.run(self.output_names, input_feed)

    def _prepare_input(self, input_data):
        if hasattr(input_data, '__array__'):
            if len(self.input_names) != 1:
                raise ValueError("Model has multiple inputs, but a single array was provided.")
            return {self.input_names[0]: input_data}
        return input_data

    def get_info(self):
        """Returns metadata."""
        import onnxruntime as ort
        return {
            "provider": self.provider,
            "inputs": self.input_names,
            "outputs": self.output_names,
            "available_providers": ort.get_available_providers()
        }

class MultiRunner:
    """Bonus: Runs models on NPU, GPU, and CPU simultaneously using threading."""
    def __init__(self, model_path):
        from concurrent.futures import ThreadPoolExecutor
        hw_map = get_all_hardware_providers()
        self.models = {}
        
        for hw_type, providers in hw_map.items():
            if providers:
                print(f"Initializing {hw_type} runner with {providers[0]}")
                self.models[hw_type] = NPUModel(model_path, provider=providers[0])

    def run_all(self, input_data):
        """Runs the model on all initialized hardware in parallel."""
        from concurrent.futures import ThreadPoolExecutor
        results = {}
        with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
            future_to_hw = {executor.submit(m.run, input_data): hw for hw, m in self.models.items()}
            for future in future_to_hw:
                hw = future_to_hw[future]
                results[hw] = future.result()
        return results
