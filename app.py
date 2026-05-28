# ============================================
# Fraud Detection FastAPI
# ============================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
from typing import List

# ============================================
# FastAPI App Initialize Karo
# ============================================
app = FastAPI(
    title="Fraud Detection API",
    description="Real-Time Credit Card Fraud Detection System",
    version="1.0"
)

# ============================================
# Models Load Karo
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
scaler = joblib.load(os.path.join(BASE_DIR, 'standard_scaler.joblib'))
model = joblib.load(os.path.join(BASE_DIR, 'xgboost_smote_fraud_detection_model.joblib'))

# ============================================
# Input Schema Define Karo
# ============================================
class TransactionData(BaseModel):
    features: List[float]

    class Config:
        json_schema_extra = {
            "example": {
                "features": [-1.35, -0.07, 2.53, 1.37, -0.33,
                             0.46, 0.23, 0.09, 0.36, 0.09,
                             -0.55, -0.61, -0.99, -0.31, 1.46,
                             -0.47, 0.20, 0.02, 0.40, 0.25,
                             -0.01, 0.27, -0.11, 0.06, -0.18,
                             -0.14, -0.05, -0.05, 0.01, 149.62]
            }
        }

# ============================================
# Route 1 — Home
# ============================================
@app.get("/")
def home():
    return {
        "message": "Fraud Detection API is Running!",
        "status": "Active",
        "version": "1.0",
        "docs": "http://127.0.0.1:8000/docs"
    }

# ============================================
# Route 2 — Health Check
# ============================================
@app.get("/health")
def health():
    return {
        "status": "Healthy",
        "model": "Loaded",
        "scaler": "Loaded"
    }

# ============================================
# Route 3 — Model Info
# ============================================
@app.get("/model-info")
def model_info():
    return {
        "model_name": "XGBoost Fraud Detector",
        "features": 30,
        "classes": ["Safe Transaction", "Fraud Transaction"],
        "scaler": "StandardScaler"
    }

# ============================================
# Route 4 — Predict (Main Route)
# ============================================
@app.post("/predict")
def predict(data: TransactionData):
    try:

        # Step 1 — Features validate karo
        if len(data.features) != 30:
            raise HTTPException(
                status_code=400,
                detail=f"30 features chahiye, aapne {len(data.features)} diye"
            )

        # Step 2 — Features copy karo
        features = data.features.copy()

        # Step 3 — Sirf Amount scale karo
        # Amount last feature hai
        amount = np.array([[features[-1]]])

        scaled_amount = scaler.transform(amount)[0][0]

        # Replace original amount
        features[-1] = scaled_amount

        # Step 4 — Final input banao
        input_data = np.array(features).reshape(1, -1)

        # Step 5 — Prediction karo
        prediction = model.predict(input_data)[0]

        # Step 6 — Probability nikalo
        probability = model.predict_proba(input_data)[0]

        fraud_risk = round(probability[1] * 100, 2)
        safe_score = round(probability[0] * 100, 2)

        # Step 7 — Final Result
        if prediction == 1:
            result = "Fraud"
            alert = "ALERT! Suspicious Transaction Detected!"
        else:
            result = "Safe"
            alert = "Transaction is Safe"

        # Step 8 — Response
        return {
            "prediction": result,
            "fraud_risk_score": f"{fraud_risk}%",
            "safe_score": f"{safe_score}%",
            "alert": alert
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# ============================================
# App Run Karo
# ============================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)