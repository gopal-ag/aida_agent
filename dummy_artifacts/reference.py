def predict(data):
    normalized_income = data['income'] / 100000.0
    return normalized_income * 0.5 + data['age'] * 0.5
