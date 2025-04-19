# Dataset: [Outlook, Temperature, Humidity, Wind, PlayTennis]
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

# Step 1: Calculate CPTs
def calculate_cpt(data, feature_index, condition_index, condition_value):
    count_condition = 0
    count_both = 0
    for row in data:
        if row[condition_index] == condition_value:
            count_condition += 1
            if row[-1] == "Yes":
                count_both += 1
    if count_condition == 0:
        return 0
    return round(count_both / count_condition, 2)

# Step 2: Compute Conditional Probabilities for "Yes" given Outlook/Humidity/Wind
def build_cpt(data):
    outlook_vals = set(row[0] for row in data)
    humidity_vals = set(row[2] for row in data)
    wind_vals = set(row[3] for row in data)

    cpt = {'Outlook': {}, 'Humidity': {}, 'Wind': {}}
    
    for val in outlook_vals:
        cpt['Outlook'][val] = calculate_cpt(data, 0, 0, val)
    for val in humidity_vals:
        cpt['Humidity'][val] = calculate_cpt(data, 2, 2, val)
    for val in wind_vals:
        cpt['Wind'][val] = calculate_cpt(data, 3, 3, val)

    return cpt

# Step 3: Inference
def predict(cpt, evidence):
    p = 1
    for feature in evidence:
        if evidence[feature] in cpt[feature]:
            p *= cpt[feature][evidence[feature]]
        else:
            p *= 0.01  # very small value for unseen data
    return round(p, 4)

# Build CPT
cpt = build_cpt(dataset)

# Evidence example
evidence = {
    'Outlook': 'Sunny',
    'Humidity': 'High',
    'Wind': 'Strong'
}

prob_yes = predict(cpt, evidence)
print(f"Probability of PlayTennis = Yes given {evidence} is: {prob_yes}")
