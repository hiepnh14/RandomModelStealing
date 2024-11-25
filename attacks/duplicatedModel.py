import json
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torchvision import models
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms
from torch.utils.data import Dataset

# Step 1: Load input-output pairs from a JSON file
def load_from_json(file_name):
    """Load input-output pairs from a JSON file."""
    with open(file_name, "r") as f:
        return json.load(f)


# Define transforms for training and validation
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),  # Randomly crop with padding
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
        ),  # CIFAR-10 normalization
    ]
)

transform_val = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
        ),
    ]
)


# step2: Function to prepare data by normalizing and reshaping
def prepare_data(io_pairs):
    """
    Prepares the input and target data for training. It reshapes the input data into the appropriate image format
    and applies CIFAR-10 normalization to ensure consistency with the dataset's preprocessing.

    Parameters:
    - io_pairs: List of input-output pairs (loaded from JSON)

    Returns:
    - X_normalized: Normalized image tensor data
    - y: Target labels
    """
    X = np.array(
        [pair["input"] for pair in io_pairs]
    )  # Extract inputs from input-output pairs
    y = np.array(
        [pair["prediction"] for pair in io_pairs]
    )  # Extract predicted labels (targets)

    # Reshape inputs to match the CIFAR-10 format (batch_size, height, width, channels)
    X = X.reshape(
        (-1, 32, 32, 3)
    )  # Convert the input array to the correct image dimensions

    # Apply the CIFAR-10 specific normalization and convert to correct shape
    X_normalized = (
        torch.tensor(X).permute(0, 3, 1, 2).float()
    )  # Move channels to second dimension

    # Normalize the tensor
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
    )
    X_normalized = normalize(X_normalized)

    return X_normalized, y  # Return normalized inputs and targets


# Step 3: Define the ResNet18 model
class ResNet18Model(nn.Module):
    def __init__(self, num_classes=10):
        """Initialize ResNet18 model with custom output layer."""
        super(ResNet18Model, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.2),  # Dropout with 20% probability
            nn.Linear(self.resnet.fc.in_features, num_classes),
        )

    def forward(self, x):
        return self.resnet(x)


from torch.utils.data import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(self, io_pairs, transform=None):
        self.data = [pair["input"] for pair in io_pairs]
        self.targets = [pair["prediction"] for pair in io_pairs]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = np.array(self.data[idx]).reshape(32, 32, 3)
        y = self.targets[idx]
        if self.transform:
            x = self.transform(x)
        return x, y




# Step 4: Training the ResNet model with early stopping
# def train_resnet_model(X, y, patience=3):
#     """
#     Train the ResNet model with early stopping.

#     Parameters:
#     - X: Input features.
#     - y: Target labels.
#     - patience: Number of epochs to wait for validation improvement before stopping.
#     """

#     torch.set_num_threads(8)
#     # Convert numpy arrays to PyTorch tensors
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )
#     y_train, y_test = torch.tensor(y_train).long(), torch.tensor(y_test).long()

#     # DataLoader for training and validation
#     batch_size = 32
#     train_loader = DataLoader(
#         TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
#     )
#     test_loader = DataLoader(
#         TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False
#     )

#     # Initialize model, loss function, and optimizer
#     model = ResNet18Model(num_classes=10)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
#     scheduler = ReduceLROnPlateau(
#         optimizer, mode="min", patience=3, factor=0.1, verbose=True
#     )

#     best_val_acc = 0.0
#     for epoch in range(50):
#         model.train()
#         train_loss, correct_train, total_train = 0.0, 0, 0

#         for inputs, labels in train_loader:
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item()
#             _, predicted = torch.max(outputs, 1)
#             total_train += labels.size(0)
#             correct_train += (predicted == labels).sum().item()

#         train_acc = 100 * correct_train / total_train
#         print(
#             f"Epoch {epoch+1}/50, Loss: {train_loss/total_train:.4f}, Train Accuracy: {train_acc:.2f}%"
#         )

#         # Validation step
#         val_loss, correct_val, total_val = 0.0, 0, 0
#         model.eval()
#         with torch.no_grad():
#             for inputs, labels in test_loader:
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 val_loss += loss.item()
#                 _, predicted = torch.max(outputs, 1)
#                 total_val += labels.size(0)
#                 correct_val += (predicted == labels).sum().item()

#         val_acc = 100 * correct_val / total_val
#         val_loss /= total_val
#         print(f"Validation Accuracy: {val_acc:.2f}%, Validation Loss: {val_loss:.4f}")

#         # Save best model based on validation accuracy
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             torch.save(model.state_dict(), "best_model.pth")
#             print(f"Saved best model with val_acc: {best_val_acc:.2f}%")

#         # Adjust learning rate
#         scheduler.step(val_loss)
#     print(f"Saved best model with val_acc: {best_val_acc:.2f}%")
#     return model
def train_resnet_model(X, y, patience=3):
    """
    Train the ResNet model with early stopping and save Validation Accuracy to a CSV file.

    Parameters:
    - X: Input features.
    - y: Target labels.
    - patience: Number of epochs to wait for validation improvement before stopping.
    """

    torch.set_num_threads(8)
    # Convert numpy arrays to PyTorch tensors
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    y_train, y_test = torch.tensor(y_train).long(), torch.tensor(y_test).long()

    # DataLoader for training and validation
    batch_size = 32
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False
    )

    # Initialize model, loss function, and optimizer
    model = ResNet18Model(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.1, verbose=True
    )

    # File to save results
    csv_file = "validation_accuracy.csv"
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Validation Accuracy"])  # Write header

    best_val_acc = 0.0
    for epoch in range(50):
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_acc = 100 * correct_train / total_train
        print(
            f"Epoch {epoch+1}/50, Loss: {train_loss/total_train:.4f}, Train Accuracy: {train_acc:.2f}%"
        )

        # Validation step
        val_loss, correct_val, total_val = 0.0, 0, 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_acc = 100 * correct_val / total_val
        val_loss /= total_val
        print(f"Validation Accuracy: {val_acc:.2f}%, Validation Loss: {val_loss:.4f}")

        # Save validation accuracy to CSV
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, val_acc])

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Saved best model with val_acc: {best_val_acc:.2f}%")

        # Adjust learning rate
        scheduler.step(val_loss)
    print(f"Saved best model with val_acc: {best_val_acc:.2f}%")
    return model

# Step 5: Main function
def main():
    """Main function to load data, prepare training, and run the ResNet18 model training."""
    io_pairs = load_from_json("io_pairs_cifar10.json")
    X, y = prepare_data(io_pairs)
    model = train_resnet_model(X, y, patience=3)


if __name__ == "__main__":
    main()
