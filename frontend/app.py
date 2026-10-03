
import streamlit as st
import pandas as pd
import requests

# Assuming the Flask API is running on localhost:7860 within the Docker network
FLASK_API_URL = "http://backend:7860/v1/predict" # 'backend' is the service name in docker-compose
FLASK_BATCH_API_URL = "http://backend:7860/v1/predictbatch"

st.title("SuperKart Sales Predictor")
st.write("Enter product and store details to predict sales.")

# Input fields for online prediction
product_weight = st.number_input("Product Weight", min_value=1.0, max_value=30.0, value=12.66)
product_sugar_content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
product_allocated_area = st.number_input("Product Allocated Area", min_value=0.001, max_value=0.3, value=0.027, format="%.3f")
product_mrp = st.number_input("Product MRP", min_value=10.0, max_value=300.0, value=117.08)
store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
store_location_city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
store_type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])
product_id_char = st.selectbox("Product ID Char (e.g., FD for Food)", ["FD", "NC", "DR"])
store_age_years = st.number_input("Store Age (Years)", min_value=1.0, max_value=50.0, value=14.0)
product_type_category = st.selectbox("Product Type Category", ["Dairy", "Frozen Foods", "Fruits and Vegetables", "Household", "Meat", "Snack Foods", "Hard Drinks", "Soft Drinks", "Breads", "Breakfast", "Canned", "Health and Hygiene", "Starchy Foods", "Seafood", "Others"])


if st.button("Predict Sales (Online)"):
    payload = {
        "Product_Weight": product_weight,
        "Product_Sugar_Content": product_sugar_content,
        "Product_Allocated_Area": product_allocated_area,
        "Product_MRP": product_mrp,
        "Store_Size": store_size,
        "Store_Location_City_Type": store_location_city_type,
        "Store_Type": store_type,
        "Food_Subcat": product_id_char, # Use Food_Subcat as it's the engineered feature
        "Store_Age_Years": store_age_years,
        "Product_Type": product_type_category # Use Product_Type as it's the engineered feature
    }
    
    # Convert categorical variables to one-hot encoding manually for Flask API compatibility
    # Note: This is simplified. In a real scenario, the API itself should handle encoding based on its trained features.
    # For demonstration, we'll assume the API expects the raw categorical values and handles encoding internally.
    
    try:
        response = requests.post(FLASK_API_URL, json=payload)
        if response.status_code == 200:
            prediction = response.json()['prediction']
            st.success(f"Predicted Sales: {prediction:.2f}")
        else:
            st.error(f"Error from API: {response.json().get('error', response.text)}")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the Flask API. Make sure the backend service is running.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

st.subheader("Batch Prediction")
st.write("Upload a CSV file for batch predictions.")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    if st.button("Predict Sales (Batch)"):
        try:
            files = {'file': uploaded_file.getvalue()}
            response = requests.post(FLASK_BATCH_API_URL, files=files)
            if response.status_code == 200:
                predictions = response.json()
                pred_df = pd.DataFrame(predictions.items(), columns=['Row Index', 'Predicted Sales'])
                st.write("Batch Predictions:")
                st.dataframe(pred_df)
            else:
                st.error(f"Error from API: {response.json().get('error', response.text)}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the Flask API. Make sure the backend service is running.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
