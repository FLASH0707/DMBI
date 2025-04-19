# AdaBoost using 2D dataset from SVM example

data = [
    [[1, 2], -1],
    [[2, 3], -1],
    [[3, 3], -1],
    [[6, 5],  1],
    [[7, 8],  1],
    [[8, 8],  1]
]

n = len(data)
weights = [1/n] * n
rounds = 3
learners = []

# Train decision stump: one feature at a time (x or y), many thresholds
def train_stump(data, weights):
    best_stump = None
    min_error = float('inf')

    for feature_index in [0, 1]:  # 0 for x, 1 for y
        thresholds = sorted(set(point[0][feature_index] for point in data))
        for t in thresholds:
            for polarity in [1, -1]:
                error = 0
                for i, (features, label) in enumerate(data):
                    value = features[feature_index]
                    prediction = 1 if polarity * value < polarity * t else -1
                    if prediction != label:
                        error += weights[i]
                if error < min_error:
                    min_error = error
                    best_stump = (feature_index, t, polarity)
    
    return best_stump, min_error

# Boosting rounds
for r in range(rounds):
    stump, error = train_stump(data, weights)
    feature_index, threshold, polarity = stump

    epsilon = 1e-10
    alpha = 0.5 * ((1 - error + epsilon) / (error + epsilon))  # simple formula

    learners.append((feature_index, threshold, polarity, alpha))

    # Update weights
    new_weights = []
    for i, (features, label) in enumerate(data):
        value = features[feature_index]
        prediction = 1 if polarity * value < polarity * threshold else -1
        if prediction == label:
            new_weights.append(weights[i])
        else:
            new_weights.append(weights[i] * 2)
    
    total = sum(new_weights)
    weights = [w / total for w in new_weights]

# Prediction function
def predict(point):
    total = 0
    for feature_index, threshold, polarity, alpha in learners:
        value = point[feature_index]
        prediction = 1 if polarity * value < polarity * threshold else -1
        total += alpha * prediction
    return 1 if total >= 0 else -1

# Test predictions
test_points = [[1, 2], [3, 3], [5, 5], [7, 7], [8, 8]]
for pt in test_points:
    print(f"Prediction for {pt}: {predict(pt)}")
