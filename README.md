# Credit Risk Prediction 
    src/data_prep.py - data loading & cleaning
    src/features.py - feature engineering & scikit-learn preprocessing pipeline
    src/train.py - training script (XGBoost / sklearn-compatible)
    src/evaluate.py - evaluation metrics & plots
    src/explain.py - SHAP explanation helper
    src/api/app.py - FastAPI scoring endpoint example
    src/monitoring.py - simple PSI (population stability index) helper
    src/utils.py - helper functions
    requirements.txt - python dependencies
    Dockerfile - containerization example

Quick start (local)

    Create and activate a virtual environment (recommended).
    pip install -r requirements.txt
    Place your CSV under data/raw/loans.csv
    python src/data_prep.py
    python src/train.py
    Serve API: uvicorn src.api.app:app --reload --port 8000
