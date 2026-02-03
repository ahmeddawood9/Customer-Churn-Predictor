# Import cleaning, preprocessing, training, and evaluation modules
from src.data_cleaning.cleaning import clean_data
from src.features.preprocessing import build_preprocessor
from src.models.train import train_random_forest, train_xgboost
from src.evaluation.metrics import evaluate_model
from sklearn.model_selection import train_test_split


def main():
    """
    Main function to run the end-to-end ML pipeline:
    - Load and clean data
    - Preprocess features
    - Train Random Forest and XGBoost models
    - Evaluate model performance
    """

    # Step 1: Load and clean raw data
    # Applies missing value handling, fixes data types, removes duplicates, and caps outliers
    data = clean_data("data/raw/telco_churn.csv")

    # Step 2: Separate features and target
    # The cleaning process converts column names to lowercase (snake_case)
    X = data.drop('churn', axis=1)

    # Map target values to binary integers (Yes=1, No=0)
    y = data['churn'].map({'Yes': 1, 'No': 0})

    # Step 3: Split data into training and test sets
    # 80% training, 20% testing
    # stratify=y ensures the churn ratio is preserved in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # Step 4: Build preprocessing pipeline
    # Detects numerical and categorical columns
    # Scales numeric features and one-hot encodes categorical features
    preprocessor = build_preprocessor(X)

    # Step 5: Train Random Forest and evaluate
    print("\nTraining Random Forest...")
    rf_model = train_random_forest(preprocessor, X_train, y_train)
    evaluate_model(rf_model, X_test, y_test, model_name="Random Forest")

    # Step 6: Train XGBoost and evaluate
    print("\nTraining XGBoost...")
    xgb_model = train_xgboost(preprocessor, X_train, y_train)
    evaluate_model(xgb_model, X_test, y_test, model_name="XGBoost")


# Ensures main() runs only when script is executed directly
if __name__ == "__main__":
    main()
