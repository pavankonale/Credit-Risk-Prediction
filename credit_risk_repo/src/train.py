import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from src.features import build_preprocessor, create_features

DATA_PATH = os.path.join('data','processed','loans_clean.parquet')
MODEL_OUT = os.path.join('models','xgb_best.joblib')

def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    return df

def prepare_xy(df):
    df = create_features(df)
    # define target
    if 'loan_status' not in df.columns:
        raise KeyError('loan_status column required (0=paid,1=default)')
    y = df['loan_status'].astype(int).values
    # basic features (modify as needed)
    numeric = [c for c in ['loan_amount','interest_rate','annual_income','dti','revol_util'] if c in df.columns]
    categorical = [c for c in ['term','home_ownership','purpose','grade'] if c in df.columns]
    X = df[numeric + categorical].copy()
    return X, y, numeric, categorical

def train():
    df = load_data()
    X, y, numeric, categorical = prepare_xy(df)
    preprocessor = build_preprocessor(numeric, categorical)
    # pipeline-like approach: fit_transform preprocessor first
    X_trans = preprocessor.fit_transform(X)
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    # simple train/val split
    X_train, X_val, y_train, y_val = train_test_split(X_trans, y, stratify=y, test_size=0.2, random_state=42)
    clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=4)
    param_dist = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.01, 0.1]
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=4, scoring='roc_auc', cv=cv, n_jobs=1, verbose=1)
    search.fit(X_train, y_train)
    best = search.best_estimator_
    # evaluate
    proba = best.predict_proba(X_val)[:,1]
    auc = roc_auc_score(y_val, proba)
    print(f'Validation ROC-AUC: {auc:.4f}')
    # save model and preprocessor together
    joblib.dump({'model': best, 'preprocessor': preprocessor}, MODEL_OUT)
    print(f'Saved model bundle to {MODEL_OUT}')

if __name__ == '__main__':
    train()
