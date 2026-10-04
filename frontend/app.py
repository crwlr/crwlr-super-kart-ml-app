
import streamlit as st
import pandas as pd
import requests

FLASK_API_URL = "http://backend:7860/v1/predict"
FLASK_BATCH_API_URL = "http://backend:7860/v1/batchpredict"

st.title("SuperKart Sales Predictor")
st.write("Enter product and store details to predict sales.")

st.subheader("Single Prediction")
product_weight = st.number_input("Product Weight", min_value=1.0, max_value=30.0, value=12.66)
product_sugar_content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
product_allocated_area = st.number_input("Product Allocated Area", min_value=0.001, max_value=0.3, value=0.027, format="%.3f")
product_mrp = st.number_input("Product MRP", min_value=10.0, max_value=300.0, value=117.08)
store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
store_location_city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
store_type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])
product_id_char = st.selectbox("Product ID Char (e.g., FD for Food)", ["FD", "NC", "DR"])
store_age_years = st.number_input("Store Age (Years)", min_value=1.0, max_value=50.0, value=14.0)
product_type_category = st.selectbox("Product Type Category", ["Dairy", "Frozen Foods", "Snack Foods", "Canned", "Baking Goods", "Health and Hygiene", "Household", "Meat", "Hard Drinks", "Soft Drinks", "Breads", "Breakfast", "Fruits and Vegetables", "Seafood", "Starchy Foods", "Others"])

if st.button("Predict Sales (Online)"):
    payload = {
        "Product_Weight": product_weight,
        "Product_Sugar_Content": product_sugar_content,
        "Product_Allocated_Area": product_allocated_area,
        "Product_MRP": product_mrp,
        "Store_Size": store_size,
        "Store_Location_City_Type": store_location_city_type,
        "Store_Type": store_type,
        "Product_Id_char": product_id_char,
        "Store_Age_Years": store_age_years,
        "Product_Type_Category": product_type_category
    }

    try:
        response = requests.post(FLASK_API_URL, json=payload)
        if response.status_code == 200:
            result = response.json()
            prediction = result.get('predicted_sales')
            st.success(f"Predicted Sales: ₹ {prediction:,.2f}")
        else:
            st.error(f"API Error: {response.json().get('message', response.text)}")
    except Exception as e:
        st.error(f"Connection / Processing Error: {e}")

st.markdown("---")
st.subheader("Batch Prediction")
st.write("Upload a CSV file containing multiple product records for batch predictions.")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    if st.button("Predict Sales (Batch)"):
        try:
            batch_df = pd.read_csv(uploaded_file)
            json_payload = batch_df.to_dict(orient="records")
            response = requests.post(FLASK_BATCH_API_URL, json=json_payload)
            
            if response.status_code == 200:
                result = response.json()
                predictions = result.get('predictions', [])
                batch_df['Predicted_Sales'] = predictions
                st.write("Batch Predictions Results:")
                st.dataframe(batch_df)
            else:
                st.error(f"Batch Prediction failed: {response.json().get('message', response.text)}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
