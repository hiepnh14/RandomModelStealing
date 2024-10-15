import requests
import torchvision.transforms as transforms
import torchvision
import torch
import json
url = "http://localhost:8000/predict"
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the CIFAR-10 test dataset
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
count = 0
for data, target in test_loader:
    
    # data = torch.randn(3, 32, 32).flatten().tolist()  
    data = data.flatten().tolist()
    input_data = {
        "data": data
    }

    # Convert the input data to JSON format
    input_json = json.dumps(input_data)

    # Send the POST request to the FastAPI server
    response = requests.post(url, data=input_json, headers={"Content-Type": "application/json"})

    # Print the response
    if response.status_code == 200:
        result = response.json()
        print(f"Predicted Class: {result['prediction']}")
        print(f"Probabilities: {result['probabilities']}")
        print(f"Model: {result['model']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    count += 1
    if count == 1:
        break