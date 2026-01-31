import torch
import torch.nn as nn
import os

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

def create_dummy_model(output_path):
    model = SimpleModel()
    dummy_input = torch.randn(1, 10)
    torch.onnx.export(model, dummy_input, output_path, input_names=['input'], output_names=['output'])
    print(f"Dummy model created at: {output_path}")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    create_dummy_model("models/test_model.onnx")
