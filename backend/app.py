import numpy as np
import joblib
import pandas as pd
from flask import Flask, request, jsonify

superkart_api = Flask("SuperKart Sales Predictor")
model = joblib.load("backend/superkart_xgb_model.joblib")

REQUIRED_FEATURES = [
    "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area",
    "Product_MRP", "Store_Size", "Store_Location_City_Type", "Store_Type",
    "Product_Id_char", "Store_Age_Years", "Product_Type_Category"
]

@superkart_api.get("/")
def home():
    return "Welcome to the SuperKart Sales Prediction API!"

@superkart_api.post("/v1/predict")
def predict_sales():
    try:
        data = request.get_json(force=True)
        if "Product_Id" in data and "Product_Id_char" not in data:
            data["Product_Id_char"] = str(data["Product_Id"])[:2]
        if "Product_Type" in data and "Product_Type_Category" not in data:
            perishable_types = ["Dairy", "Meat", "Fruits and Vegetables", "Baking Goods", "Bread", "Breakfast", "Frozen Foods", "Seafood", "Starchy Foods"]
            data["Product_Type_Category"] = "Perishables" if data["Product_Type"] in perishable_types else "Non Perishables"

        missing = [f for f in REQUIRED_FEATURES if f not in data]
        if missing:
            return jsonify({"status": "error", "message": f"Missing features: {missing}"}), 400

        input_df = pd.DataFrame([data])[REQUIRED_FEATURES]
        prediction = model.predict(input_df)[0]
        return jsonify({"status": "success", "predicted_sales": float(prediction)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@superkart_api.post("/v1/batchpredict")
def predict_batch_sales():
    try:
        json_data = request.get_json(force=True)
        if not isinstance(json_data, list):
            return jsonify({"status": "error", "message": "Batch input must be a list of records."}), 400
        input_df = pd.DataFrame(json_data)
        if "Product_Id" in input_df.columns and "Product_Id_char" not in input_df.columns:
            input_df["Product_Id_char"] = input_df["Product_Id"].astype(str).str[:2]
        if "Product_Type" in input_df.columns and "Product_Type_Category" not in input_df.columns:
            perishable_types = ["Dairy", "Meat", "Fruits and Vegetables", "Baking Goods", "Bread", "Breakfast", "Frozen Foods", "Seafood", "Starchy Foods"]
            input_df["Product_Type_Category"] = input_df["Product_Type"].apply(lambda x: "Perishables" if x in perishable_types else "Non Perishables")

        missing = [f for f in REQUIRED_FEATURES if f not in input_df.columns]
        if missing:
            return jsonify({"status": "error", "message": f"Missing features: {missing}"}), 400

        input_df = input_df[REQUIRED_FEATURES]
        predictions = model.predict(input_df)
        return jsonify({"status": "success", "predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400
