def predict(data):
    # Missing normalize_income step
    return data['income'] * 0.5 + data['age'] * 0.5
