import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('diabetes_model.pkl')

# App title and description
st.title('Diabetes Prediction Tool')
st.write("""
### Enter Patient Information
This app predicts the likelihood of diabetes based on diagnostic measurements.
""")

# Create input fields for each feature
st.subheader('Patient Features')

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=0)
    glucose = st.number_input('Glucose (mg/dL)', min_value=0, max_value=500, value=120)
    blood_pressure = st.number_input('Blood Pressure (mm Hg)', min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input('Insulin (mu U/ml)', min_value=0, max_value=900, value=80)
    bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
    diabetes_pedigree = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5, format="%.2f")
    age = st.number_input('Age', min_value=0, max_value=120, value=35)

# Feature names (must match the order used during training)
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Create a button to trigger the prediction
if st.button('Predict Diabetes Risk'):
    # Load the model
    model = load_model()

    # Create a dataframe with the input values
    input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, 
                              insulin, bmi, diabetes_pedigree, age]], 
                             columns=feature_names)

    # Display the input data
    st.write('### Input Summary:')
    st.write(input_data)

    # Generate prediction and probability
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Display the results
    st.write('### Prediction Results:')

    # Create a colored box based on the result
    if prediction == 1:
        st.error(f"**High Risk of Diabetes** (Probability: {probability:.2%})")
    else:
        st.success(f"**Low Risk of Diabetes** (Probability: {probability:.2%})")

    # Display risk factors and recommendations
    st.write('### Risk Analysis:')

    risk_factors = []

    if glucose > 140:
        risk_factors.append("- **Elevated Glucose**: The glucose level is higher than normal range, which is a significant risk factor for diabetes.")

    if bmi > 30:
        risk_factors.append("- **High BMI**: BMI indicates obesity, which increases diabetes risk.")

    if diabetes_pedigree > 0.5:
        risk_factors.append("- **Family History**: The diabetes pedigree function indicates a genetic predisposition.")

    if age > 40:
        risk_factors.append("- **Age**: Being over 40 increases the risk of Type 2 diabetes.")

    if risk_factors:
        st.write("**Key Risk Factors:**")
        for factor in risk_factors:
            st.markdown(factor)
    else:
        st.write("No significant individual risk factors identified.")

    # Add a disclaimer
    st.info('**Disclaimer**: This prediction is based on a machine learning model and should be used as a screening tool only. Please consult with a healthcare professional for proper diagnosis.')
