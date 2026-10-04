
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Corrected path to load from the backend folder
model = joblib.load('backend/superkart_xgb_model.joblib')

@app.route('/v1/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        df = pd.DataFrame([data])

        # APPLY FEATURE ENGINEERING TO MATCH TRAINING
        if "Product_Id" in df.columns:
            df['Food_Subcat'] = df['Product_Id'].str[:2]
            df = df.drop(columns=['Product_Id'])

        if "Store_Establishment_Year" in df.columns:
            df['Store_Age_Years'] = 2023 - df['Store_Establishment_Year']
            df = df.drop(columns=['Store_Establishment_Year'])

        prediction = model.predict(df)
        return jsonify({'prediction': float(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/v1/predictbatch', methods=['POST'])
def predict_batch():
    try:
        # 1. Read the uploaded file from the request
        file = request.files['file']
        df = pd.read_csv(file)

        # 2. Apply the exact same feature engineering as training
        if 'Product_Id' in df.columns:
            df['Food_Subcat'] = df['Product_Id'].str[:2]
            df.drop(columns=['Product_Id'], inplace=True)

        if 'Store_Establishment_Year' in df.columns:
            df['Store_Age_Years'] = 2023 - df['Store_Establishment_Year']
            df.drop(columns=['Store_Establishment_Year'], inplace=True)

        # 3. Make predictions using your saved model or pipeline
        predictions = model.predict(df)

        # 4. Return results as JSON
        return jsonify({str(i): float(pred) for i, pred in enumerate(predictions)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400
