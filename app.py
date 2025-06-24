```python
import streamlit as st
import joblib
import numpy as np
import os

# Cache model and scaler loading
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load(os.path.join("model", "rdd_model.joblib"))
        scaler = joblib.load(os.path.join("model", "scaler.pkl"))
        return model, scaler
    except FileNotFoundError:
        st.error("Model or scaler file not found in 'model' directory.")
        return None, None

# Load model and scaler
model, scaler = load_model_and_scaler()

# Prediction function
def predict_status(inputs):
    if model is None or scaler is None:
        return None
    try:
        input_array = np.array(inputs).reshape(1, -1)
        input_array = scaler.transform(input_array)
        prediction = model.predict(input_array)
        return prediction
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# Streamlit UI
st.title("Simple Student Dropout Prediction")

# Single input for testing (adjust as needed)
st.subheader("Enter a sample input")
sample_input = st.number_input("Sample Feature (e.g., Application Order)", value=5)

# Dummy input data (19 features as per previous code)
# Using sample_input for the first feature, others are placeholders
input_data = [sample_input] + [0] * 18  # Adjust based on actual feature count

# Predict button
if st.button("Predict"):
    prediction = predict_status(input_data)
    if prediction is not None:
        status_dict = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
        predicted_status_index = np.argmax(prediction, axis=1)[0]
        predicted_status = status_dict[predicted_status_index]
        st.success(f"Predicted status: **{predicted_status}**")
```
