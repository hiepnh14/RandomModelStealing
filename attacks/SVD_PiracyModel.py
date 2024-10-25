import numpy as np
import requests
import json
import torch

# API address
url = "http://localhost:8000/predict"


# Function to generate random input
def generate_random_input():
    """
    This function generates random input data to simulate image data
    with a shape of (3, 32, 32) which corresponds to an image with 3 channels
    (e.g., RGB) and dimensions of 32x32 pixels.
    """
    return np.random.randn(3, 32, 32).flatten().tolist()  # Flatten to 1D list


# SVD attack function, modifies input data
def svd_attack_input(input_matrix, k=5):
    """
    Performs Singular Value Decomposition (SVD) on the input matrix.
    Retains the top 'k' singular values and reconstructs a modified version of
    the input data using truncated SVD, which can be used for adversarial attack.

    Args:
        input_matrix (np.array): The input matrix representing the image data.
        k (int): The number of singular values to retain during the attack.

    Returns:
        np.array: The modified input matrix after applying SVD and reconstructing it.
    """
    U, S, Vt = np.linalg.svd(input_matrix, full_matrices=False)  # Perform SVD
    # Truncate the singular values to retain only the top 'k' singular values
    S_truncated = np.zeros_like(S)  # Initialize truncated singular value array
    S_truncated[:k] = S[:k]  # Keep the top 'k' singular values
    # Reconstruct the input matrix with truncated singular values
    modified_input = np.dot(U, np.dot(np.diag(S_truncated), Vt))
    return modified_input


# Function to perform SVD attack and test the results
def perform_svd_attack(n_samples=100):
    """
    This function generates adversarial inputs by applying SVD to the input data
    and sends the modified inputs to the model's API. It prints out the
    predicted class and probability distribution after the attack.

    Args:
        n_samples (int): The number of random inputs to generate and attack.
    """
    for _ in range(n_samples):
        # Step 1: Generate random input
        random_input = generate_random_input()

        # Step 2: Reshape the input into a 2D matrix suitable for SVD
        input_data_matrix = np.array(random_input).reshape(
            3, 32 * 32
        )  # 3 channels, flattened 32x32

        # Step 3: Perform SVD attack on the input data
        attacked_input = svd_attack_input(
            input_data_matrix, k=2
        )  # Apply SVD to the input
        attacked_input = (
            attacked_input.flatten().tolist()
        )  # Flatten the matrix back to a 1D list

        # Step 4: Prepare the modified input to be sent to the API
        input_data = {"data": attacked_input}

        # Step 5: Send the modified input to the API for prediction
        response = requests.post(
            url,
            data=json.dumps(input_data),  # Convert input data to JSON format
            headers={"Content-Type": "application/json"},  # Set content type as JSON
        )

        # Step 6: Process the API's response and print the results
        if response.status_code == 200:  # Check if the request was successful
            result = response.json()
            print(
                f"Predicted Class after SVD attack: {result['prediction']}"
            )  # Print predicted class
            print(
                f"Probabilities after SVD attack: {result['probabilities']}"
            )  # Print probability distribution
        else:
            print(f"Request failed: {response.status_code}")  # Handle failed requests


if __name__ == "__main__":
    # Perform the SVD attack on 'n_samples' random inputs
    perform_svd_attack(n_samples=100)
