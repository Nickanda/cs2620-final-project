import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from resnet50 import resnet50

# Set device
device = torch.device("mps" if torch.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
num_epochs = 25
batch_size = 128
learning_rate = 0.001

# Data preprocessing and augmentation
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

# Load CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)
test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Classes in CIFAR-10
classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# Initialize the model (ResNet50 adapted for CIFAR-10)
model = resnet50(pretrained=False, num_classes=10)

# Adjust the first conv layer to handle 32x32 images instead of 224x224
# Replace the first 7x7 conv with a 3x3 conv with smaller stride
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
# Remove the first maxpool layer as CIFAR images are much smaller
model.maxpool = nn.Identity()

model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "min", patience=2, factor=0.5
)

# Lists to store metrics
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []


# Training function
def train(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Create progress bar
    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{num_epochs} [Train]",
        unit="batch",
        leave=True,
        position=0,
    )

    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Calculate metrics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Update progress bar
        current_loss = running_loss / (batch_idx + 1)
        current_acc = 100.0 * correct / total
        pbar.set_postfix({"loss": f"{current_loss:.3f}", "acc": f"{current_acc:.2f}%"})

    train_loss = running_loss / len(train_loader)
    train_acc = 100.0 * correct / total

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    return train_loss, train_acc


# Validation function
def validate(epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # Create progress bar for validation
    pbar = tqdm(
        test_loader,
        desc=f"Epoch {epoch + 1}/{num_epochs} [Validate]",
        unit="batch",
        leave=True,
        position=0,
    )

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            current_loss = running_loss / (batch_idx + 1)
            current_acc = 100.0 * correct / total
            pbar.set_postfix(
                {"val_loss": f"{current_loss:.3f}", "val_acc": f"{current_acc:.2f}%"}
            )

    val_loss = running_loss / len(test_loader)
    val_acc = 100.0 * correct / total

    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    return val_loss, val_acc


# Train the model
if __name__ == "__main__":
    print("Starting training...")
    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_acc = train(epoch)
        val_loss, val_acc = validate(epoch)

        # Update learning rate based on validation loss
        scheduler.step(val_loss)

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f} seconds")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(
            f"Training Accuracy: {train_acc:.2f}% | Validation Accuracy: {val_acc:.2f}%"
        )
        print("-" * 80)

    print("Finished Training")

    # Save the model
    torch.save(model.state_dict(), "resnet101_cifar10.pth")
    print("Model saved to resnet101_cifar10.pth")

    # Plot training and validation metrics
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curves")

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Training Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Accuracy Curves")

    plt.savefig("training_metrics.png")
    plt.show()

    print(f"Final Training Accuracy: {train_accuracies[-1]:.2f}%")
    print(f"Final Validation Accuracy: {val_accuracies[-1]:.2f}%")
