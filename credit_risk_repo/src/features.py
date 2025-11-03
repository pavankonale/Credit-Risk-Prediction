import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', cat_transformer, categorical_features)
    ])
    return preprocessor

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'annual_income' in df.columns and 'loan_amount' in df.columns:
        df['income_to_loan'] = df['annual_income'] / (df['loan_amount'] + 1e-9)
    if 'application_date' in df.columns and 'earliest_credit_line' in df.columns:
        df['credit_age_years'] = (df['application_date'] - df['earliest_credit_line']).dt.days / 365.25
    return df
