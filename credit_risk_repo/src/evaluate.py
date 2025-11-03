import joblib, os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix
from src.features import create_features

MODEL_OUT = os.path.join('models','xgb_best.joblib')
DATA_PATH = os.path.join('data','processed','loans_clean.parquet')

def evaluate():
    bundle = joblib.load(MODEL_OUT)
    model = bundle['model']; preprocessor = bundle['preprocessor']
    df = pd.read_parquet(DATA_PATH)
    df = create_features(df)
    if 'loan_status' not in df.columns:
        raise KeyError('loan_status required for evaluation')
    y = df['loan_status'].astype(int).values
    numeric = [c for c in ['loan_amount','interest_rate','annual_income','dti','revol_util'] if c in df.columns]
    categorical = [c for c in ['term','home_ownership','purpose','grade'] if c in df.columns]
    X = df[numeric + categorical]
    X_trans = preprocessor.transform(X)
    proba = model.predict_proba(X_trans)[:,1]
    auc = roc_auc_score(y, proba)
    y_pred = (proba >= 0.5).astype(int)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    print(f'AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}\nConfusion matrix:\n{cm}')

if __name__ == '__main__':
    evaluate()
