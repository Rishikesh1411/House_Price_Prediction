import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("model.pkl")

st.title("🏠 House Price Prediction")
st.markdown("**Author:** Rishikesh Raj")

st.divider()

st.write("This app uses machine learning to predict the price of a house based on its features such as number of bedrooms, bathrooms, living area, condition of the house, and number of schools nearby.")

st.divider()

bedrooms = st.number_input("Number of Bedrooms", min_value=0, max_value=10, value=0)
bathrooms = st.number_input("Number of Bathrooms", min_value=0, max_value=10, value=0)
living_area = st.number_input("Living Area (in square feet)", min_value=0, max_value=10000, value=2000)
condition = st.selectbox("Condition of the House", ["Poor", "Fair", "Good", "Very Good", "Excellent"])
schools_nearby = st.number_input("Number of Schools Nearby", min_value=0, max_value=10, value=0)

st.divider()

x = [[bedrooms, bathrooms, living_area, condition, schools_nearby]]

predictbutton = st.button("Predict Price")
if predictbutton:

    st.balloons()
    # Convert condition to numerical value
    condition_mapping = {
        "Poor": 1,
        "Fair": 2,
        "Good": 3,
        "Very Good": 4,
        "Excellent": 5
    }
    x[0][3] = condition_mapping[x[0][3]]
    
    # Convert to DataFrame
    x_df = pd.DataFrame(x, columns=['number of bedrooms', 'number of bathrooms', 'living area', 'condition of the house', 'Number of schools nearby'])
    
    # Make prediction
    prediction = model.predict(x_df)
    
    st.write(f"The predicted price of the house is: ${prediction[0]:,.2f}")
else:
    st.write("Please Use Predict Price Button to see the prediction.")

# Order of x['number of bedrooms', 'number of bathrooms', 'living area',
    #    'condition of the house', 'Number of schools nearby']

st.markdown("---")
st.markdown("<center>Made by <b>Rishikesh Raj</b></center>", unsafe_allow_html=True)