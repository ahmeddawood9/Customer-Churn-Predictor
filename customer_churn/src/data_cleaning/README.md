# Telco Customer Churn – Data Cleaning Module

This module (`cleaning.py`) implements the **data cleaning stage** for the Telco Customer Churn dataset.  
It is the first step in an ongoing ML pipeline focused on building a churn prediction model.
Data is raw and messy , and those steps will help it clean the data.

---

## Current Implementation

The `clean_telco_data()` function performs the following tasks:

| Task | Status in Your Code | Feedback / Notes |
|------|------------------|----------------|
| Deleting redundant columns | Partial | You dropped `customerID`, which is correct. Consider evaluating if columns like `gender` or `StreamingTV` may also be redundant depending on future feature engineering. |
| Renaming the columns | Missing | Column names are not renamed explicitly via `df.rename()`. Typically, this step standardizes names (lowercase, snake_case) for easier coding. |
| Dropping duplicates | Done | `df.drop_duplicates()` correctly removes any duplicate rows. |
| Cleaning individual columns | Done | Numeric conversion for `TotalCharges` is handled correctly using `pd.to_numeric(errors='coerce')`. |
| Remove the NaN values | Done | `NaN` values in `TotalCharges` are filled with `0`, which is appropriate for new customers with tenure 0. |
| Check for transformations | Partial | Outlier clipping using IQR is applied to numeric columns. Additional transformations, such as encoding categorical columns (Yes/No → 1/0), will be needed in the feature engineering stage. |

---

## How to Use

```python
from src.cleaning import clean_telco_data

# Path to your raw dataset
path = "data/raw/telco_churn.csv"

# Clean the data
df_clean = clean_telco_data(path)

# Inspect the cleaned data
print(df_clean.head())
