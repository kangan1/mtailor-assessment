import torch
import torchvision
import onnx
import os

# Load the PyTorch model
model = torchvision.models.resnet18()
model.load_state_dict(torch.load(os.path.abspath('resnet18-f37072fd.pth')))

# Create sample input data
input_data = torch.randn(1, 3, 224, 224)

# Convert the PyTorch model to ONNX format
torch.onnx.export(model, input_data, 'model.onnx', verbose=True)


