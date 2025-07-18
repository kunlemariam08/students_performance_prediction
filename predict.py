import pandas as pd

def predict_graduation_class(model, scaler, feature_columns, student_data):
    """
    Predict graduation class for a single student
    """
    # Create DataFrame from input data
    df_input = pd.DataFrame([student_data])
    
    # Select features
    X = df_input[feature_columns]
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make prediction
    prediction = model.predict(X_scaled)[0]
    prediction_proba = model.predict_proba(X_scaled)[0]
    
    
    return {
        'predicted_class': prediction,
        'confidence': max(prediction_proba),
        'class_probabilities': {
             label: float(prob) for label, prob in zip(model.classes_, prediction_proba)
        }
    }
