import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import random
from resnet import ResNet10, ResNet20
# Load the trained model (make sure you have the model file saved)
resnet18 = torchvision.models.resnet18()  # Your model architecture (e.g., ResNet18, VGG16, etc.)
resnet18.fc = nn.Linear(resnet18.fc.in_features, 10)
resnet18.load_state_dict(torch.load('resnet18.pth'))
resnet18.eval()  # Set the model to evaluation mode


resnet34 = torchvision.models.resnet34()  # Your model architecture (e.g., ResNet18, VGG16, etc.)

resnet34.fc = nn.Linear(resnet34.fc.in_features, 10)
resnet34.load_state_dict(torch.load('resnet34.pth'))
resnet34.eval()  # Set the model to evaluation mode
# Define the transformations for the input data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the CIFAR-10 test dataset
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

num = random.choice([0, 1])
if num == 0:
    model = resnet18
elif num == 1:
    model = resnet34

# Make predictions on the test dataset
with torch.no_grad():
    for data, target in test_loader:
        
        data = torch.randn(1, 3, 32, 32)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        print("Predicted class:", predicted.item())
        print(num)
        break