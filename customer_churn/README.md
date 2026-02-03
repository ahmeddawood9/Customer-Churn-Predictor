# Telco Customer Churn Prediction

A modular end-to-end Machine Learning pipeline designed to predict customer churn in the telecommunications industry. This project demonstrates a production-ready workflow, moving from raw data cleaning to model evaluation, comparing **Random Forest** and **XGBoost** classifiers.

##  Project Overview

Customer churn (loss of clients) is a critical metric for subscription-based businesses. This project aims to identify customers at high risk of churning so that proactive retention strategies can be applied.

The pipeline focuses on:

- **Reproducibility:** Modular code structure for easy maintenance.
- **Data Quality:** Robust cleaning (handling outliers, missing values, and type inconsistencies).
- **Imbalanced Learning:** Handling class imbalance using weighted loss functions (`scale_pos_weight`, `class_weight='balanced'`).
- **Business Impact:** Prioritizing **Recall** to minimize false negatives (missed churners).

##  Project Structure

```text
customer_churn_clean_ml/
│
├── data/
│   └── raw/
│       └── telco_churn.csv    # Original dataset
│
├── results/                   # Generated evaluation plots (PR Curves)
│
├── src/
│   ├── data_cleaning/         # Missing values, outlier capping, column standardization
│   ├── features/              # Scaling (StandardScaler) & Encoding (OneHotEncoder)
│   ├── models/                # Model definitions (Random Forest & XGBoost)
│   └── evaluation/            # Metrics generation and plotting
│
├── main.py                    # Entry point for the pipeline
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation

 Installation & Setup

Clone the repository:

git clone https://github.com/yourusername/customer_churn.git
cd customer_churn


Create a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`


Install dependencies:

pip install -r requirements.txt

🚀 Usage

Run the complete pipeline (cleaning → preprocessing → training → evaluation) using the main script:

python -m main


This will:

Clean the raw data and print the final shape.

Train a Random Forest classifier and save its PR curve to results/.

Train an XGBoost classifier and save its PR curve to results/.

Print detailed Classification Reports and Confusion Matrices to the console.

📊 Methodology
1. Data Cleaning (src/data_cleaning)

Standardization: Converted all column headers to snake_case (e.g., MonthlyCharges → monthly_charges) for consistency.

Missing Values: Imputed missing total_charges with 0 (these occurred only when tenure was 0).

Outliers: Applied IQR (Interquartile Range) capping to numerical features to reduce noise.

2. Preprocessing (src/features)

Categorical Data: Handled via OneHotEncoder (creates binary columns).

Numerical Data: Scaled using StandardScaler to normalize distributions (critical for model convergence).

Pipeline: Wrapped in Scikit-Learn's ColumnTransformer to prevent data leakage.

3. Model Strategy (src/models)

Two ensemble methods were compared:

Random Forest: Bagging method with class_weight='balanced' to penalize misclassifying minority class (churn).

XGBoost: Gradient boosting method with scale_pos_weight=3 to explicitly focus on churners.

📈 Results & Analysis

Models were evaluated based on Recall (Sensitivity) because missing a churner (False Negative) is more costly than a false positive.

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | 77% | 0.59 | 0.44 | 0.50 |
| **XGBoost** | **74%** | **0.51** | **0.76** | **0.61** |
Key Findings:

Random Forest was conservative, identifying only 44% of churners.

XGBoost identified 76% of churners. Although its precision is lower (more false positives), it is superior for retention campaigns where coverage is critical.

Visuals

Precision-Recall curves are saved automatically to the results/ folder after every run to visualize trade-off thresholds.
