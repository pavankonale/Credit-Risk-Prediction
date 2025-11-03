import os
import pandas as pd

RAW_PATH = os.path.join('data', 'raw', 'loans.csv')
PROCESSED_PATH = os.path.join('data', 'processed', 'loans_clean.parquet')

def load_raw(path=RAW_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data not found at {path}. Place your CSV there.")
    df = pd.read_csv(path, low_memory=False)
    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'loan_id' in df.columns:
        df = df.drop_duplicates(subset=['loan_id'])
    # parse dates if present
    for col in ['application_date','earliest_credit_line']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    # employment length -> numeric years
    if 'employment_length' in df.columns:
        s = df['employment_length'].astype(str).str.strip()
        s = s.str.replace('\+','', regex=False).str.replace(' years','', regex=False).str.replace('< 1 year','0', regex=False)
        df['employment_length_years'] = pd.to_numeric(s, errors='coerce')
    # numeric coercion
    for c in ['annual_income','loan_amount','interest_rate','dti','revol_util']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_parquet(PROCESSED_PATH, index=False)
    print(f"Saved cleaned data to {PROCESSED_PATH}")
    return df

if __name__ == '__main__':
    df = load_raw()
    basic_clean(df)
