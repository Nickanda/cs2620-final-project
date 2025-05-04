import matplotlib.pyplot as plt

# Data
epochs = list(range(1, 26))
train_loss = [
    1.1842,
    0.9937,
    0.8608,
    0.7381,
    0.6406,
    0.5737,
    0.5255,
    0.4859,
    0.4577,
    0.4218,
    0.3935,
    0.3695,
    0.3524,
    0.3310,
    0.3134,
    0.2986,
    0.2825,
    0.2716,
    0.2597,
    0.2510,
    0.2403,
    0.2267,
    0.2204,
    0.2125,
    0.2052,
]
val_loss = [
    1.5743,
    0.9864,
    0.9601,
    0.7301,
    0.6446,
    0.6630,
    1.1263,
    0.5306,
    0.6031,
    0.7365,
    0.5416,
    0.4489,
    0.4957,
    0.4754,
    0.4173,
    0.3671,
    0.4623,
    0.4043,
    0.5688,
    0.4370,
    0.4474,
    0.3633,
    0.4227,
    0.3748,
    0.3581,
]
train_acc = [
    57.80,
    64.84,
    69.32,
    74.40,
    77.69,
    80.18,
    81.95,
    83.45,
    84.32,
    85.34,
    86.34,
    87.19,
    87.74,
    88.57,
    89.14,
    89.73,
    90.15,
    90.52,
    90.85,
    91.36,
    91.57,
    92.02,
    92.31,
    92.56,
    92.78,
]
val_acc = [
    60.17,
    67.93,
    69.40,
    76.01,
    77.85,
    79.76,
    79.33,
    82.05,
    82.99,
    82.17,
    83.11,
    85.98,
    84.44,
    85.87,
    86.68,
    87.63,
    85.98,
    87.08,
    86.02,
    86.27,
    87.64,
    88.53,
    87.45,
    88.34,
    88.60,
]

# Loss plot
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, marker="o", label="Train Loss")
plt.plot(epochs, val_loss, marker="s", label="Validation Loss")
plt.title("Training vs. Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("original_loss_plot.png")  # saves to file
plt.show()

# Accuracy plot
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_acc, marker="o", label="Train Accuracy")
plt.plot(epochs, val_acc, marker="s", label="Validation Accuracy")
plt.title("Training vs. Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("original_accuracy_plot.png")  # saves to file
plt.show()
