# Play Tennis Dataset
dataset = [
    ['Sunny', 'Hot', 'High', 'Weak', 'No'],
    ['Sunny', 'Hot', 'High', 'Strong', 'No'],
    ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
    ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
    ['Sunny', 'Mild', 'High', 'Weak', 'No'],
    ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
    ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
    ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
    ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Strong', 'No']
]

features = ['Outlook', 'Temperature', 'Humidity', 'Wind']

# Step 1: Separate data by class
def separate_by_class(data):
    separated = {}
    for row in data:
        label = row[-1]
        if label not in separated:
            separated[label] = []
        separated[label].append(row[:-1])
    return separated

# Step 2: Count frequencies
def count_frequencies(data, feature_index, value):
    count = 0
    for row in data:
        if row[feature_index] == value:
            count += 1
    return count

# Step 3: Train Naive Bayes Classifier
def train(data):
    separated = separate_by_class(data)
    model = {}
    total_rows = len(data)
    for label in separated:
        class_data = separated[label]
        class_prob = len(class_data) / total_rows
        feature_probs = []
        for i in range(len(features)):
            feature_vals = [row[i] for row in class_data]
            val_counts = {}
            for val in feature_vals:
                val_counts[val] = val_counts.get(val, 0) + 1
            feature_probs.append(val_counts)
        model[label] = {'prob': class_prob, 'features': feature_probs}
    return model

# Step 4: Predict
def predict(model, input_data):
    results = {}
    for label in model:
        prob = model[label]['prob']
        for i in range(len(input_data)):
            value = input_data[i]
            feature_probs = model[label]['features'][i]
            count = feature_probs.get(value, 0)
            prob *= count / sum(feature_probs.values())
        results[label] = prob
    return max(results, key=results.get)

# Train the model
model = train(dataset)

# Test prediction
test = ['Rain', 'Mild', 'Normal', 'Weak']
result = predict(model, test)
print(f"Prediction for {test} => Play Tennis: {result}")
