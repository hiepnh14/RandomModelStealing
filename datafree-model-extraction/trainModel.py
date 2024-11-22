import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# Import the ResNet18 model from the provided architecture
from dfme.network.resnet_8x import ResNet18_8x
def eval(model, patch=None, target_class=None, device=torch.device("mps"), val_loader=None):
  # Stats to use to calculate accuracy after the eval loop
  total_correct = 0
  total = 0
  total_target = 0
  # Put model on GPU and switch to eval mode
  model = model.to(device)
  model.eval()
  # Evaluation loop
  with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(val_loader):
      # Put data on GPU
      images = images.to(device)
      if patch is not None:
        images = apply(patch, images)
      labels = labels.to(device)
      # Make predictions
      predictions = model(images)
      predictions = torch.argmax(predictions, dim=1)
      # Update validation accuracy information
      total += len(images)
      num_correct = (predictions == labels).float().sum().item()
      total_correct += num_correct
      if target_class is not None:
        target = torch.zeros(len(images), dtype=torch.long).fill_(target_class).to(device)
        num_target = (predictions == target).float().sum().item()
        total_target += num_target
  # If evaluating the effects of a targeted patch attach, it is nice to see whether or not the model is classifying lots of examples to the target class
  if target_class is not None:
    target_percentage = total_target / total
    print(f"Percentage of samples predicted as target class {target_class}: {100 * target_percentage}")
  # Calculate accuracy
  accuracy = total_correct / total
  return accuracy
def fine_tune_for_svhn(model, train_loader, val_loader, num_epochs=30, model_path='checkpoint/teacher/svhn_resnet18_8x.pt', lr=0.01, device=torch.device("mps")):
  # Put model on GPU and put model in training mode
  model = model.to(device)
  model.train()
  # Define loss function and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)
  best_accuracy = 0.0
  best_model_path = model_path
  # Training loop
  for i in range(num_epochs):
    # Stats to use for calculating accuracy
    total_correct = 0
    total = 0
    # Iterate through each batch of data
    for batch_idx, (images, labels) in enumerate(train_loader):
      # Put data on GPU
      images = images.to(device)
      labels = labels.to(device)
      # Make predictions
      predictions = model(images)
      # Calculate loss for the batch
      loss = criterion(predictions, labels)
      # Gradient descent
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      # Update training accuracy information
      total += len(images)
      predictions = torch.argmax(predictions, dim=1)
      num_correct = (predictions == labels).float().sum().item()
      total_correct += num_correct
    scheduler.step()  # Update the learning rate

    # Print training accuracy
    print(f"Epoch {str(i + 1)}: Training accuracy = {str(total_correct / total)}")
    # Print validation accuracy
    val_accuracy = eval(model, patch=None, target_class=None, val_loader=val_loader, device=device)
    print(f"Validation accuracy: {str(val_accuracy)}")

    if val_accuracy > best_accuracy:
      best_accuracy = val_accuracy
      torch.save(model.state_dict(), best_model_path)
      print(f"Saved new best model with accuracy: {best_accuracy:.4f}")
def train_resnet18_svhn():
    # Hyperparameters
    batch_size = 32
    learning_rate = 0.1
    num_epochs = 30
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('mps')

    # Data transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    # Load SVHN dataset
    traindata = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    
    # Split training data into training and validation sets
    train_set, val_set = torch.utils.data.random_split(traindata, [40000, 10000])
    testdata = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    # Define dataloaders
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    test_loader = DataLoader(testdata, batch_size=32, shuffle=False)

    

    

    # Initialize model
    model = ResNet18_8x(num_classes=10)

    # Train model
    fine_tune_for_svhn(model, train_loader, val_loader, num_epochs=num_epochs, model_path='dfme/checkpoint/teacher/cifar10_resnet18_8x.pt', lr=learning_rate, device=device)

if __name__ == "__main__":
    train_resnet18_svhn()