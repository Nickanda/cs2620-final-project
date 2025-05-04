import re
import matplotlib.pyplot as plt
import numpy as np


def parse_log_file(file_path):
    """
    Parse the log file to extract epoch, loss, and accuracy data.
    """
    epochs = []
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Regular expressions to match the patterns in the log file
    train_pattern = r"Rank 0, Epoch (\d+): Loss = ([\d.]+), Accuracy = ([\d.]+)%"
    val_pattern = r"Rank 0, Epoch (\d+): Validation Loss = ([\d.]+), Validation Accuracy = ([\d.]+)%"

    with open(file_path, "r") as file:
        lines = file.readlines()

    for line in lines:
        # Try to match training data
        train_match = re.search(train_pattern, line)
        if train_match:
            epoch = int(train_match.group(1))
            loss = float(train_match.group(2))
            accuracy = float(train_match.group(3))

            epochs.append(epoch)
            train_losses.append(loss)
            train_accuracies.append(accuracy)

        # Try to match validation data
        val_match = re.search(val_pattern, line)
        if val_match:
            epoch = int(val_match.group(1))
            loss = float(val_match.group(2))
            accuracy = float(val_match.group(3))

            val_losses.append(loss)
            val_accuracies.append(accuracy)

    return epochs, train_losses, train_accuracies, val_losses, val_accuracies


def plot_training_stats(
    epochs, train_losses, train_accuracies, val_losses, val_accuracies
):
    """
    Create and save plots for losses and accuracies.
    """
    # Set a nice looking style
    plt.style.use("ggplot")

    # Create figure for losses
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, "b-", label="Training Loss")
    plt.plot(epochs, val_losses, "r-", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_plot.png", dpi=300)
    plt.close()

    # Create figure for accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracies, "g-", label="Training Accuracy")
    plt.plot(epochs, val_accuracies, "p-", label="Validation Accuracy", color="purple")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("accuracy_plot.png", dpi=300)
    plt.close()

    print(f"Plots have been saved as 'loss_plot.png' and 'accuracy_plot.png'")


def main():
    """
    Main function to parse the log and generate plots.
    """
    log_file = "tmp.txt"

    print(f"Parsing log file: {log_file}")
    epochs, train_losses, train_accuracies, val_losses, val_accuracies = parse_log_file(
        log_file
    )

    if not epochs:
        print("No data found in the log file.")
        return

    print(f"Found data for {len(epochs)} epochs")
    plot_training_stats(
        epochs, train_losses, train_accuracies, val_losses, val_accuracies
    )

    # Print some statistics
    print("\nTraining Statistics:")
    print(
        f"Initial training loss: {train_losses[0]:.3f}, Final training loss: {train_losses[-1]:.3f}"
    )
    print(
        f"Initial training accuracy: {train_accuracies[0]:.2f}%, Final training accuracy: {train_accuracies[-1]:.2f}%"
    )
    print(
        f"Initial validation loss: {val_losses[0]:.3f}, Final validation loss: {val_losses[-1]:.3f}"
    )
    print(
        f"Initial validation accuracy: {val_accuracies[0]:.2f}%, Final validation accuracy: {val_accuracies[-1]:.2f}%"
    )


if __name__ == "__main__":
    main()
