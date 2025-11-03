from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title='Credit Risk Scoring API')
MODEL_OUT = os.path.join('models','xgb_best.joblib')
bundle = None
if os.path.exists(MODEL_OUT):
    bundle = joblib.load(MODEL_OUT)

class LoanApp(BaseModel):
    loan_amount: float
    term: str = None
    interest_rate: float = None
    annual_income: float = None
    dti: float = None
    revol_util: float = None
    home_ownership: str = None
    purpose: str = None
    grade: str = None

@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': bool(bundle)}

@app.post('/score')
def score(app_data: LoanApp):
    if bundle is None:
        return {'error': 'model not trained. Run training first.'}
    preprocessor = bundle['preprocessor']; model = bundle['model']
    df = pd.DataFrame([app_data.dict()])
    # Keep same columns as training
    numeric = [c for c in ['loan_amount','interest_rate','annual_income','dti','revol_util'] if c in df.columns]
    categorical = [c for c in ['term','home_ownership','purpose','grade'] if c in df.columns]
    X = df[numeric + categorical]
    X_trans = preprocessor.transform(X)
    proba = model.predict_proba(X_trans)[:,1][0]
    return {'default_probability': float(proba), 'risk': int(proba >= 0.5)}
