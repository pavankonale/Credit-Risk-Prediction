# Credit Risk Prediction - Example Repo

This repository contains an end-to-end example for a credit risk prediction pipeline:

- `src/data_prep.py` - data loading & cleaning
- `src/features.py` - feature engineering & scikit-learn preprocessing pipeline
- `src/train.py` - training script (XGBoost / sklearn-compatible)
- `src/evaluate.py` - evaluation metrics & plots
- `src/explain.py` - SHAP explanation helper
- `src/api/app.py` - FastAPI scoring endpoint example
- `src/monitoring.py` - simple PSI (population stability index) helper
- `src/utils.py` - helper functions
- `requirements.txt` - python dependencies
- `Dockerfile` - containerization example

## Quick start (local)
1. Create and activate a virtual environment (recommended).
2. `pip install -r requirements.txt`
3. Place your CSV under `data/raw/loans.csv`
4. `python src/data_prep.py`
5. `python src/train.py`
6. Serve API: `uvicorn src.api.app:app --reload --port 8000`
