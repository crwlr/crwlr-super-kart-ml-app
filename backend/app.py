import numpy as np
import joblib
import pandas as pd
from pathlib import Path
from flask import Flask, request, jsonify

superkart_api = Flask("SuperKart Sales Predictor")
app = superkart_api
model = joblib.load(Path(__file__).with_name("superkart_xgb_model.joblib"))

REQUIRED_FEATURES = [
    'Product_Weight', 'Product_Sugar_Content', 'Product_Allocated_Area',
    'Product_MRP', 'Store_Size', 'Store_Location_City_Type', 'Store_Type',
    'Product_Id_char', 'Store_Age_Years', 'Product_Type_Category'
]

@superkart_api.get('/')
def home():
    return "Welcome to the SuperKart Sales Prediction API!"

@superkart_api.post('/v1/predict')
def predict_sales():
    try:
        data = request.get_json(force=True)
        # Validate single payload
        missing = [f for f in REQUIRED_FEATURES if f not in data]
        if missing:
            return jsonify({
                "status": "error",
                "message": f"Missing mandatory features for inference: {missing}"
            }), 400
            
        input_df = pd.DataFrame([data])
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

@superkart_api.post('/v1/batchpredict')
def predict_batch_sales():
    try:
        json_data = request.get_json(force=True)
        if not isinstance(json_data, list):
            return jsonify({
                "status": "error",
                "message": "Batch input must be a list of records."
            }), 400
            
        input_df = pd.DataFrame(json_data)
        
        # Handle renaming/alignment dynamically
        if 'Product_Id' in input_df.columns and 'Product_Id_char' not in input_df.columns:
            input_df['Product_Id_char'] = input_df['Product_Id'].astype(str).str[:2]
        if 'Product_Type' in input_df.columns and 'Product_Type_Category' not in input_df.columns:
            input_df['Product_Type_Category'] = input_df['Product_Type']
            
        # Validate presence of mandatory fields post-mapping
        missing = [f for f in REQUIRED_FEATURES if f not in input_df.columns]
        if missing:
            return jsonify({
                "status": "error",
                "message": f"Batch processing failed due to missing mandatory columns: {missing}"
            }), 400
            
        # Subset to include only the columns expected by model preprocessor
        input_df = input_df[REQUIRED_FEATURES]
        predictions = model.predict(input_df)
        return jsonify({
            "status": "success",
            "predictions": predictions.tolist()
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to parse or process batch dataset: {str(e)}"
        }), 400

if __name__ == '__main__':
    superkart_api.run(host='0.0.0.0', port=5000, debug=False)
