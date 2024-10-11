# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from load_model import load_model
from model import MyModel
import numpy as np

app = FastAPI(title="PyTorch Model Inference API")

# Load the model at startup
MODEL_PATH = "resnet18.pth"  # Path to your .pth file
model = load_model(MODEL_PATH)

# Define the input data structure
class InputData(BaseModel):
    data: list  # Assuming input is a list of numbers (e.g., flattened image)

class Prediction(BaseModel):
    prediction: int
    probabilities: list

@app.post("/predict", response_model=Prediction)
def predict(input: InputData):
    try:
        # Convert input data to a tensor
        input_tensor = torch.tensor(input.data, dtype=torch.float32)
        
        # If your model expects a specific shape, reshape accordingly
        # For example, if it's a single sample:
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
        
        # Perform inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1).numpy().tolist()[0]
            predicted_class = np.argmax(probabilities)
        
        return Prediction(prediction=int(predicted_class), probabilities=probabilities)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
