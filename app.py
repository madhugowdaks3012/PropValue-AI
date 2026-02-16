import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Smart Home Price Predictor")

st.title("🏠 Smart Budget Home Price Predictor")

model = joblib.load("model.pkl")

st.subheader("Enter Property Details")

city = st.selectbox("Select City", ["Bengaluru", "Mumbai", "Chennai", "Delhi", "Hyderabad"])
sqft = st.number_input("Square Feet", min_value=300, max_value=5000, value=1000)
bedrooms = st.slider("Number of Bedrooms", 1, 5, 2)
age = st.slider("Age of Property (Years)", 0, 30, 5)

if st.button("Predict Price"):
    input_data = pd.DataFrame([[city, sqft, bedrooms, age]],
                              columns=["City", "Sqft", "Bedrooms", "Age"])

    prediction = model.predict(input_data)[0]

    lower = prediction - 300000
    upper = prediction + 300000

    st.success(f"Estimated Price: ₹ {prediction:,.0f}")
    st.info(f"Price Range: ₹ {lower:,.0f} - ₹ {upper:,.0f}")

    # Investment suggestion
    if age < 5 and bedrooms >= 3:
        st.markdown("💡 **Investment Tip:** Good long-term investment property.")
    elif age > 15:
        st.warning("⚠ Older property. Consider renovation costs.")
    else:
        st.markdown("📊 Balanced investment option.")
