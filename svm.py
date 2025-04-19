# Simple SVM without libraries (gradient descent approach)

# Dataset: [x1, x2, class]
data = [
    [1, 2, -1],
    [2, 3, -1],
    [3, 3, -1],
    [6, 5, 1],
    [7, 8, 1],
    [8, 8, 1]
]

# Initialize weights and bias
w = [0.0, 0.0]
b = 0.0

# Hyperparameters
learning_rate = 0.01
epochs = 1000

# Train the SVM using simplified gradient descent
for epoch in range(epochs):
    for x1, x2, label in data:
        x = [x1, x2]
        dot_product = w[0]*x[0] + w[1]*x[1] + b
        if label * dot_product < 1:
            # Misclassified
            w[0] += learning_rate * (label * x[0])
            w[1] += learning_rate * (label * x[1])
            b += learning_rate * label
        else:
            # Correctly classified – small update
            w[0] += learning_rate * 0
            w[1] += learning_rate * 0

# Prediction function
def predict(x1, x2):
    result = w[0]*x1 + w[1]*x2 + b
    return 1 if result >= 0 else -1

# Test prediction
test = [4, 4]
predicted_class = predict(test[0], test[1])
print(f"Prediction for point {test} is class {predicted_class}")

# Show final model
print(f"Weights: {w}, Bias: {b}")
