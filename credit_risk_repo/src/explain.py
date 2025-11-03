import joblib, os
import shap
import pandas as pd
from src.features import create_features

MODEL_OUT = os.path.join('models','xgb_best.joblib')
DATA_PATH = os.path.join('data','processed','loans_clean.parquet')

def explain(n_samples=100):
    bundle = joblib.load(MODEL_OUT)
    model = bundle['model']; preprocessor = bundle['preprocessor']
    df = pd.read_parquet(DATA_PATH)
    df = create_features(df).reset_index(drop=True)
    numeric = [c for c in ['loan_amount','interest_rate','annual_income','dti','revol_util'] if c in df.columns]
    categorical = [c for c in ['term','home_ownership','purpose','grade'] if c in df.columns]
    X = df[numeric + categorical]
    X_trans = preprocessor.transform(X)
    X_sample = X_trans[:n_samples]
    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample)
    # summary plot will open an interactive window if run locally; we save values as ndarray
    return shap_values

if __name__ == '__main__':
    print('Run explain() to compute SHAP values (this may take time)') 
