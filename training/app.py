# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.nn as nn
import numpy as np
import random
import warnings
from resnet import ResNet18, ResNet20
warnings.filterwarnings("ignore")
app = FastAPI(title="PyTorch Model Inference API")

# Load the model at startup
# resnet18 = torchvision.models.resnet18()  # Your model architecture (e.g., ResNet18, VGG16, etc.)
# resnet18.fc = nn.Linear(resnet18.fc.in_features, 10)
resnet18 = ResNet18()
resnet18.load_state_dict(torch.load('resnet18.pth'))
resnet18.eval()  # Set the model to evaluation mode


# resnet34 = torchvision.models.resnet34()  # Your model architecture (e.g., ResNet18, VGG16, etc.)

# resnet34.fc = nn.Linear(resnet34.fc.in_features, 10)
resnet20 = ResNet20()
resnet20.load_state_dict(torch.load('resnet20.pth'))
resnet20.eval()  # Set the model to evaluation mode
# Define the transformations for the input data

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the CIFAR-10 test dataset
test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

num = random.choice([0, 1])
if num == 0:
    model = resnet18
elif num == 1:
    model = resnet20

# Make predictions on the test dataset as an example
# with torch.no_grad():
#     for data, target in test_loader:

#         data = torch.randn(1, 3, 32, 32)
#         output = model(data)
#         _, predicted = torch.max(output.data, 1)
#         print("Predicted class:", predicted.item())
#         print(num)
#         break


# Define the input data structure
class InputData(BaseModel):
    data: list  # Assuming input is a list of numbers (e.g., flattened image)


class Prediction(BaseModel):
    prediction: int
    probabilities: list
    model: int


@app.post("/predict", response_model=Prediction)
def predict(input: InputData):
    try:
        num = random.choice([0, 1])
        if num == 0:
            model = resnet18
        elif num == 1:
            model = resnet20
        # Convert input data to a tensor
        input_tensor = torch.tensor(input.data, dtype=torch.float32)
        input_tensor = input_tensor.view(1, 3, 32, 32)

        # If your model expects a specific shape, reshape accordingly
        # For example, if it's a single sample:
        # input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1).numpy().tolist()[0]
            predicted_class = np.argmax(probabilities)

        return Prediction(
            prediction=int(predicted_class), probabilities=probabilities, model=int(num)
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
