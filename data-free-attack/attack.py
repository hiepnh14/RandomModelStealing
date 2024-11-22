import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms, models
import numpy as np
import requests
import json
from typing import List
from pydantic import BaseModel
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    
class Prediction(BaseModel):
    prediction: int
    probabilities: List[float]

class ModelExtractor:
    def __init__(self, url, max_queries=1000, num_classes=10, batch_size=32):
        self.url = url
        self.max_queries = max_queries
        self.query_count = 0
        
        # Initialize surrogate model
        self.surrogate = models.resnet18(pretrained=False)
        self.surrogate.fc = nn.Linear(self.surrogate.fc.in_features, num_classes)
        self.optimizer = optim.Adam(self.surrogate.parameters())
        self.criterion = nn.KLDivLoss()
        self.batch_size = batch_size
    
    def generate_synthetic_inputs(self):
        """Generate random noise inputs"""
        return torch.randn(self.batch_size, 3, 32, 32)
    # Function to send data to the API and receive predictions
    def query_input(self, data):
        """Sends the input data to the API and returns the predicted class and probabilities."""
        input_data = {"data": data}
        input_json = json.dumps(input_data)

        # Send a POST request to the API server
        response = requests.post(
            self.url, data=input_json, headers={"Content-Type": "application/json"}
        )

        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            return Prediction(
                prediction = result["prediction"],  # The predicted class
                probabilities = result["probabilities"],  # The class probabilities
            )
        else:
            print(f"Error: {response.status_code} - {response.text}")
            raise ValueError("API request failed")


    def query_victim(self, inputs):
        """Query victim model with query limit"""
        # if self.query_count >= self.max_queries:
        #     raise ValueError("Query limit exceeded")
        
        self.query_count += inputs.size(0)
        
        # Collect predictions from API
        predictions = []
        for inp in inputs:
            pred = self.query_input(inp.flatten().tolist())
            predictions.append(pred)
        
        return predictions
    
    def extract_model(self, num_iterations=3000):
        """Data-free model extraction algorithm"""
        for iteration in range(num_iterations):
            # Generate synthetic inputs
            synthetic_inputs = self.generate_synthetic_inputs()
            
            try:
                # Get predictions from victim API
                try:
                     victim_predictions = self.query_victim(synthetic_inputs)
                except ValueError as e:
                    print(f"Stopping extraction: {e}")
                    break   

                
                # Prepare training data
                victim_soft_labels = torch.tensor([
                    pred.probabilities for pred in victim_predictions
                ])
                
                # Train surrogate model
                self.optimizer.zero_grad()
                surrogate_outputs = self.surrogate(synthetic_inputs)
                surrogate_log_probs = torch.log_softmax(surrogate_outputs, dim=1)
                
                # Knowledge distillation loss
                loss = self.criterion(surrogate_log_probs, victim_soft_labels)
                loss.backward()
                self.optimizer.step()
                
                print(f"Iteration {iteration}: Loss = {loss.item():.4f}")
                
            except ValueError as e:
                print(f"Stopping extraction: {e}")
                break
        
        
        return self.surrogate

def evaluate_model(model, test_loader):
    """Evaluate model on test dataset"""
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return correct / total

def main():
    # CIFAR10 Test Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(testset, batch_size=100, shuffle=False)
    
    # Create victim model API
    victim_api_url = "http://localhost:8000/predict"
    
    # Create model extractor
    extractor = ModelExtractor(victim_api_url, max_queries=5000)
    
    # Extract model
    extractedModel = extractor.extract_model()
    # Save Extracted Model
    torch.save(extractedModel.state_dict(), 'extracted_model.pth')
  
    print(f"Total queries used: {extractor.query_count}")
    # Evaluate Extracted Model
    accuracy = evaluate_model(extractedModel, test_loader)
    print(f"Extracted Model Accuracy: {accuracy:.2%}") 

if __name__ == "__main__":
    main()