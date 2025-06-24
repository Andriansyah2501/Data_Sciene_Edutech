import streamlit as st
import joblib
import numpy as np
import os
import requests
import io

@st.cache_resource  # caching untuk efisiensi
def load_model_from_github(url):
    response = requests.get(url)
    response.raise_for_status()  # untuk memunculkan error jika URL salah
    return joblib.load(io.BytesIO(response.content))

# URL file dari GitHub
model_url = "https://github.com/Andriansyah2501/Data_Sciene_Edutech/tree/main/Model/rdf_model.joblib"
scaler_url = "https://github.com/Andriansyah2501/Data_Sciene_Edutech/tree/main/Model/scaler.pkl"

# Load model dan scaler
model = load_model_from_github(model_url)
scaler = load_model_from_github(scaler_url)

st.success("Model dan Scaler berhasil dimuat dari GitHub!")

# Function to make predictions
def predict_status(inputs):
    # Convert inputs to numpy array and reshape
    input_array = np.array(inputs).reshape(1, -1)
    input_array = scaler.transform(input_array)
    prediction = model.predict(input_array)
    return prediction

# Streamlit UI
st.title('Student Dropout Prediction')

# Input fields for user to input data
Application_order = st.number_input('Application Order', min_value=0, max_value=9, value=5)
Previous_qualification_grade = st.slider('Previous qualification (Grade)', min_value=0.0, max_value=200.0, value=5.0, step=0.1)
Admission_grade = st.slider('Admission Grade', min_value=0.0, max_value=200.0, value=5.0, step=0.1)
Age_at_enrollment = st.number_input('Age at enrollment', value=30)
Curricular_units_1st_sem_credited = st.number_input('Curricular units 1st sem (credited)', value=23)
Curricular_units_1st_sem_enrolled = st.number_input('Curricular units 1st sem (enrolled)', value=12)
Curricular_units_1st_sem_evaluations = st.number_input('Curricular Units 1st sem (evalutions)', value=24)
Curricular_units_1st_sem_approved = st.number_input('Curricular Units 1st sem (approved)', value=8)
Curricular_units_1st_sem_grade = st.number_input('Curricular Units 1st sem (grade)', value=6)
Curricular_units_1st_sem_without_evaluations = st.number_input('Curricular Units 1st sem (without evaluations)', value=22)
Curricular_units_2nd_sem_credited = st.number_input('Curricular units 2nd sem (credited)', value=22)
Curricular_units_2nd_sem_enrolled = st.number_input('Curricular units 2nd sem (enrolled)', value=11)
Curricular_units_2nd_sem_evaluations = st.number_input('Curricular Units 2nd sem (evalutions)', value=2)
Curricular_units_2nd_sem_approved = st.number_input('Curricular Units 2nd sem (approved)', value=20)
Curricular_units_2nd_sem_grad = st.number_input('Curricular Units 2nd sem (grade)', value=0)
Curricular_units_2nd_sem_without_evaluations = st.number_input('Curricular Units 2nd sem (without evaluations)', value=0)
Unemployment_rate = st.number_input('Unemployment rate', value=1)
Inflation_rate = st.number_input('Inflation rate', value=-1.5)
GDP = st.number_input('GDP', value= 0.4)

# Map the inputs to the format expected by the model
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
    Curricular_units_2nd_sem_grad,
    Curricular_units_2nd_sem_without_evaluations,
    Unemployment_rate,
    Inflation_rate,
    GDP
]

# Button for prediction
if st.button('Predict'):
    prediction = predict_status(input_data)

    # The prediction will be a 2D array where each column corresponds to one of the classes
    status_dict = {
        0: 'Dropout',
        1: 'Enrolled',
        2: 'Graduate'
    }

    # Find the index of the maximum predicted value
    predicted_status_index = np.argmax(prediction, axis=1)[0]
    predicted_status = status_dict[predicted_status_index]

    st.write(f"The model predicts that the student is likely to be: **{predicted_status}**")
