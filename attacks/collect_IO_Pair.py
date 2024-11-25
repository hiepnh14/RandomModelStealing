import asyncio
import json
import aiohttp
import torch
import torchvision
import torchvision.transforms as transforms

# API URL
url = "http://localhost:8000/predict"

# Define the transformation to match the model's input requirements
transform = transforms.Compose(
    [
        transforms.ToTensor(),  # Convert the PIL Image to Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize the images
    ]
)
transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)
# Load the CIFAR-10 test dataset
test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)

# Create a DataLoader for the test dataset
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)


# Function to send data to the API and receive predictions asynchronously
async def query_model(session, data):
    """Sends the input data to the API and returns the predicted class and probabilities."""
    input_data = {"data": data}
    input_json = json.dumps(input_data)

    # Send a POST request to the API server
    async with session.post(
        url, data=input_json, headers={"Content-Type": "application/json"}
    ) as response:
        if response.status == 200:
            result = await response.json()
            return (
                result["prediction"],  # The predicted class
                result["probabilities"],  # The class probabilities
                result.get("model", "Unknown"),  # The model used
            )
        else:
            print(f"Error: {response.status} - {await response.text()}")
            return None, None, None


# Function to collect input-output pairs by querying the model asynchronously
async def collect_io_pairs(test_loader, num_samples=10, batch_size=10):
    """Collects input-output pairs by querying the model through the API asynchronously."""
    io_pairs = []
    count = 0

    async with aiohttp.ClientSession() as session:
        for data, target in test_loader:
            if count >= num_samples:
                break

            # Prepare the data for batch processing
            data_list = [d.flatten().tolist() for d in data]
            input_data = {"data": data_list}

            # Perform the asynchronous request
            prediction, probabilities, model = await query_model(session, data_list)

            # Store the input-output pairs in the list
            io_pairs.append(
                {
                    "input": data_list[0],  # Input data (flattened)
                    "actual": int(target[0].item()),  # Actual class label from CIFAR-10
                    "prediction": prediction,  # Model's predicted class
                    "probabilities": probabilities,  # Probabilities for each class
                    "model": model,  # Model type used
                }
            )

            count += len(data)

    return io_pairs


# Function to calculate the model's accuracy based on the collected input-output pairs
def analyze_accuracy(io_pairs):
    """Calculates and prints the model's accuracy on the collected CIFAR-10 test set."""
    correct_predictions = 0
    for pair in io_pairs:
        if pair["actual"] == pair["prediction"]:
            correct_predictions += 1
    accuracy = correct_predictions / len(io_pairs)
    print(f"Model accuracy on CIFAR-10 test set: {accuracy * 100:.2f}%")
    return accuracy


# Function to save input-output pairs to a JSON file
def save_to_json(io_pairs, file_name="io_pairs.json"):
    """Saves the collected input-output pairs to a JSON file."""
    with open(file_name, "w") as f:
        json.dump(io_pairs, f, indent=4)
    print(f"Data saved to {file_name}")


# Main function to execute the steps of collecting input-output pairs, analyzing accuracy, and saving results
async def main():
    num_samples = 10000  # Define the number of samples to test
    batch_size = 10  # Define the batch size for testing
    io_pairs = await collect_io_pairs(
        test_loader, num_samples, batch_size
    )  # Collect input-output pairs
    analyze_accuracy(io_pairs)  # Analyze the model's accuracy
    save_to_json(io_pairs, "io_pairs_cifar10.json")  # Save results to a JSON file


if __name__ == "__main__":
    asyncio.run(main())
