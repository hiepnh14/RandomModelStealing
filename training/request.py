import requests
import torchvision.transforms as transforms
import torchvision
import torch
import json

url = "http://localhost:8000/predict"
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# Load the CIFAR-10 test dataset
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)


test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
count = 0
total = 0
total_correct = 0
total_misclassified = 0
model0 = 0
model1 = 0
model2 = 0
for data, target in test_loader:

    # data = torch.randn(3, 32, 32).flatten().tolist()
    data = data.flatten().tolist()
    input_data = {"data": data}

    # Convert the input data to JSON format
    input_json = json.dumps(input_data)

    # Send the POST request to the FastAPI server
    response = requests.post(
        url, data=input_json, headers={"Content-Type": "application/json"}
    )

    # Print the response
    if response.status_code == 200:
        result = response.json()
        # print(f"Predicted Class: {result['prediction']}")
        # print(f"Probabilities: {result['probabilities']}")
        # print(f"Model: {result['model']}")
        if result['model'] == 0:
            model0 += 1
        elif result['model'] == 1:
            model1 += 1
        elif result['model'] == 2:
            model2 += 1
    else:
        print(f"Error: {response.status_code} - {response.text}")
    
    total += target.size(0)
    total_correct += (result['prediction'] == target.item())
    total_misclassified += (result['prediction'] != target.item())

print(f"Total: {total}")
print(f"Total Correct: {total_correct}")
print(f"Total Misclassified: {total_misclassified}")
print(f"Accuracy: {total_correct/total}")
print(f"Misclassification Rate: {total_misclassified/total}")
print(f"Model 0: {model0}")
print(f"Model 1: {model1}")
print(f"Model 2: {model2}")