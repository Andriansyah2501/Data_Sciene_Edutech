import streamlit as st
import joblib as np
import numpy as np
import os

# Cache model and scaler to load only once
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load(os.path.join("model", "rdf_model.joblib"))
        scaler = joblib.load(os.path.join("model", "scaler.pkl"))
        return model, scaler
    except FileNotFoundError:
        st.error("Model or scaler file not found. Please ensure 'rdf_model.joblib' and 'scaler.pkl' are in the 'model' directory.")
        return None, None

# Load model and scaler
model, scaler = load_model_and_scaler()

# Function to make predictions
def predict_status(inputs):
    if model is None or scaler is None:
        return None
    try:
        # Convert inputs to numpy array and reshape
        input_array = np.array(inputs).reshape(1, -1)
        # Scale the input
        input_array = scaler.transform(input_array)
        # Make prediction
        prediction = model.predict(input_array)
        return prediction
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

# Streamlit UI
st.title("Student Dropout Prediction")

# Input fields
st.subheader("Enter Student Details")
Application_order = st.number_input("Application Order", min_value=0, max_value=9, value=5)
Previous_qualification_grade = st.slider("Previous Qualification (Grade)", min_value=0.0, max_value=200.0, value=5.0, step=0.1)
Admission_grade = st.slider("Admission Grade", min_value=0.0, max_value=200.0, value=5.0, step=0.1)
Age_at_enrollment = st.number_input("Age at Enrollment", value=30)
Curricular_units_1st_sem_credited = st.number_input("Curricular Units 1st Sem (Credited)", value=23)
Curricular_units_1st_sem_enrolled = st.number_input("Curricular Units 1st Sem (Enrolled)", value=12)
Curricular_units_1st_sem_evaluations = st.number_input("Curricular Units 1st Sem (Evaluations)", value=24)
Curricular_units_1st_sem_approved = st.number_input("Curricular Units 1st Sem (Approved)", value=8)
Curricular_units_1st_sem_grade = st.number_input("Curricular Units 1st Sem (Grade)", value=6)
Curricular_units_1st_sem_without_evaluations = st.number_input("Curricular Units 1st Sem (Without Evaluations)", value=22)
Curricular_units_2nd_sem_credited = st.number_input("Curricular Units 2nd Sem (Credited)", value=22)
Curricular_units_2nd_sem_enrolled = st.number_input("Curricular Units 2nd Sem (Enrolled)", value=11)
Curricular_units_2nd_sem_evaluations = st.number_input("Curricular Units 2nd Sem (Evaluations)", value=2)
Curricular_units_2nd_sem_approved = st.number_input("Curricular Units 2nd Sem (Approved)", value=20)
Curricular_units_2nd_sem_grade = st.number_input("Curricular Units 2nd Sem (Grade)", value=0)  # Fixed typo
Curricular_units_2nd_sem_without_evaluations = st.number_input("Curricular Units 2nd Sem (Without Evaluations)", value=0)
Unemployment_rate = st.number_input("Unemployment Rate", value=1.0)
Inflation_rate = st.number_input("Inflation Rate", value=-1.5)
GDP = st.number_input("GDP", value=0.4)

# Collect inputs in the order expected by the model
input_data = [
    Application_order,
    Previous_qualification_grade,
    Admission_grade,
    Age_at_enrollment,
    Curricular_units_1st_sem_credited,
    Curricular_units_1st_sem_enrolled,
    Curricular_units_1st_sem_evaluations,
    Curricular_units_1st_sem_approved,
    Curricular_units_1st_sem_grade,
    Curricular_units_1st_sem_without_evaluations,
    Curricular_units_2nd_sem_credited,
    Curricular_units_2nd_sem_enrolled,
    Curricular_units_2nd_sem_evaluations,
    Curricular_units_2nd_sem_approved,
    Curricular_units_2nd_sem_grade,
    Curricular_units_2nd_sem_without_evaluations,
    Unemployment_rate,
    Inflation_rate,
    GDP
]

# Prediction button
if st.button("Predict"):
    prediction = predict_status(input_data)
    if prediction is not None:
        # Map prediction to status
        status_dict = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
        predicted_status_index = np.argmax(prediction, axis=1)[0]
        predicted_status = status_dict[predicted_status_index]
        st.success(f"The model predicts that the student is likely to be: **{predicted_status}**")
