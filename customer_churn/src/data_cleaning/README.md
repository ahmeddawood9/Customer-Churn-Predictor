# Data Cleaning Module

This module handles raw, messy industry data and converts it into a clean,
model-ready format.

## Responsibilities
- Load raw CSV data
- Fix incorrect data types
- Handle missing values
- Remove duplicate records
- Detect and cap outliers using the IQR method

## Why This Matters
Real-world datasets often contain inconsistent formats, missing values,
and extreme outliers. Automating these steps ensures reproducibility and
prevents data leakage.

## Output
A clean pandas DataFrame ready for feature engineering.

