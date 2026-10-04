# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
superkart_api = Flask("SuperKart Sales Predictor")

# Load the trained machine learning model pipeline using the correct filename
model = joblib.load("backend/superkart_xgb_model.joblib")

# Define a route for the home page (GET request)
@superkart_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the SuperKart Sales Prediction API!"

# Define an endpoint for single product-store prediction (POST request)
@superkart_api.post('/v1/predict')
def predict_sales():
    try:
        # Get JSON data from the client request
        data = request.get_json()
        
        # Convert to DataFrame
        input_df = pd.DataFrame([data])
        
        # Map Product_Id_char directly to Food_Subcat if present
        if 'Product_Id_char' in input_df.columns:
            input_df['Food_Subcat'] = input_df['Product_Id_char']
        elif 'Product_Id' in input_df.columns:
            input_df['Food_Subcat'] = input_df['Product_Id'].astype(str).str[:2]
            
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        return jsonify({
            "status": "success",
            "predicted_sales": float(prediction)
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

# Define an endpoint for batch predictions
@superkart_api.post('/v1/batchpredict')
def predict_batch_sales():
    """This function handles batch predictions for multiple records."""
    try:
        json_data = request.get_json()
        input_df = pd.DataFrame(json_data)

        # 1. Map Product_Id_char directly to Food_Subcat if present
        if 'Product_Id_char' in input_df.columns:
            input_df['Food_Subcat'] = input_df['Product_Id_char']
        # 2. Fallback: Derive Food_Subcat from Product_Id if Product_Id_char is missing
        elif 'Product_Id' in input_df.columns:
            input_df['Food_Subcat'] = input_df['Product_Id'].astype(str).str[:2]

        # 3. Drop raw identifier columns so the shape matches your trained model
        columns_to_drop = [col for col in ['Product_Id', 'Product_Id_char'] if col in input_df.columns]
        input_df = input_df.drop(columns=columns_to_drop)

        predictions = model.predict(input_df)

        return jsonify({
            "status": "success",
            "predictions": predictions.tolist()
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

if __name__ == '__main__':
    superkart_api.run(host='0.0.0.0', port=5000, debug=False)
