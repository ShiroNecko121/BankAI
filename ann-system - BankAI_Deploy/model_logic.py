import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model

# Load the saved ANN model and preprocessors ONCE when the server starts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = load_model(os.path.join(BASE_DIR, 'bank_ann_model.h5'))
scaler = joblib.load(os.path.join(BASE_DIR, 'bank_scaler.pkl'))
encoders = joblib.load(os.path.join(BASE_DIR, 'bank_encoders.pkl'))
feature_names = joblib.load(os.path.join(BASE_DIR, 'feature_names.pkl'))

def explain_prediction(input_dict, prob):
    reasons = []

    try:
        age = float(input_dict.get("age", 0))
        campaign = float(input_dict.get("campaign", 0))
        pdays = float(input_dict.get("pdays", 0))
    except ValueError:
        age, campaign, pdays = 0, 0, 0

    housing_str = str(input_dict.get("housing", "no")).lower()
    loan_str = str(input_dict.get("loan", "no")).lower()

    if 30 <= age <= 60:
        reasons.append("Customer is within a stable working age group")
    if housing_str == "no" and loan_str == "no":
        reasons.append("Customer has no active loans")
    if campaign > 2:
        reasons.append("Customer has been contacted multiple times")
    if pdays < 30:
        reasons.append("Recent contact strongly increases response likelihood")
    if loan_str == "yes":
        reasons.append("Existing personal loan may reduce disposable income")

    summary = (
        "Customer shows positive indicators for subscription."
        if prob > 0.5
        else "Customer shows low likelihood based on profile and engagement."
    )

    return summary, reasons

def predict_output(input_dict):
    full_input = {}

    # Format the inputs exactly as the ANN expects
    for col in feature_names:
        if col in input_dict:
            if col in encoders:
                try:
                    # Transform the text from the HTML form into the correct number
                    encoded_val = encoders[col].transform([input_dict[col]])[0]
                    full_input[col] = float(encoded_val)
                except ValueError:
                    full_input[col] = 0.0
            else:
                full_input[col] = float(input_dict[col])
        else:
            full_input[col] = 0.0

    input_data = [full_input[col] for col in feature_names]
    data = np.array([input_data])
    
    # Scale the data using the saved RobustScaler
    data = scaler.transform(data)

    # Make the prediction instantly using the pre-trained ANN
    prediction = model(data, training=False)
    prob = float(prediction[0][0])

    label = "YES" if prob > 0.5 else "NO"
    summary, reasons = explain_prediction(input_dict, prob)

    return label, prob, summary, reasons
