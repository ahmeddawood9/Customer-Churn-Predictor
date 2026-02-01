import pandas as pd
import numpy as np

import pandas as pd
import numpy as np


def clean_telco_data(path):
    print(f"--- Starting Cleaning Process for: {path} ---")
    df = pd.read_csv(path).copy()

    # 1. Standardize Column Names (Snake Case)
    # This removes spaces, dots, and converts to lowercase
    df.columns = (df.columns
                  .str.strip()
                  .str.replace(' ', '_')
                  .str.replace('(', '')
                  .str.replace(')', '')
                  .str.lower())

    # 2. Fix Data Types
    # Note: Column is now 'totalcharges' due to step 1
    df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')

    # 3. Handle Missing Values
    # In Telco, NaNs in totalcharges occur when tenure is 0.
    # We fill with 0 instead of dropping to keep the new customer data.
    df['totalcharges'] = df['totalcharges'].fillna(0)

    # 4. Handle Outliers (IQR Capping)
    numeric_cols = ['monthlycharges', 'totalcharges', 'tenure']
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower, upper)

    # 5. Feature Selection & Deduplication
    # 'customerid' is now lowercase
    df = df.drop(columns=['customerid'], errors='ignore')
    df = df.drop_duplicates()

    print(f"Cleaning Complete. Final Shape: {df.shape}")
    return df


