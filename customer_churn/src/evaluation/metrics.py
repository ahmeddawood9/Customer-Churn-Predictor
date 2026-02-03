import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,           # To show TP, TN, FP, FN counts
    classification_report,      # To show precision, recall, F1-score for each class
    precision_recall_curve,     # To compute points for Precision-Recall curve
    average_precision_score     # To summarize PR curve into single score
)


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a trained ML model on test data and save the evaluation artifacts.

    Parameters:
    ----------
    model : Pipeline
        Trained machine learning pipeline (preprocessing + classifier)
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series or array
        True target labels
    model_name : str, optional
        Name of the model for labeling plots and files (default is "Model")

    Returns:
    -------
    None
    Prints evaluation metrics and saves the Precision-Recall curve to the 'results' directory.
    """

    # Predict class labels for test data
    y_pred = model.predict(X_test)

    # Predict probabilities for positive class (churn=1)
    # Needed for Precision-Recall curve and AP score
    y_prob = model.predict_proba(X_test)[:, 1]

    # Print the model name for clarity in logs
    print(f"\n--- Evaluation Results: {model_name} ---")

    # Print the confusion matrix to show correct and incorrect predictions
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Print detailed classification report
    # Includes precision, recall, F1-score, and support for each class
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Compute precision and recall values at different thresholds
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    # Compute average precision score (summary of PR curve)
    ap = average_precision_score(y_test, y_prob)

    # Plot Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', linewidth=2)
    plt.xlabel("Recall")           # True Positive Rate for churn class
    plt.ylabel("Precision")        # Correctness of predicted churn
    plt.title(f"Precision-Recall Curve: {model_name} (AP={ap:.2f})")
    plt.grid(True)

    # Ensure the output directory exists
    os.makedirs("results", exist_ok=True)

    # Construct a valid filename from the model name
    filename = f"results/{model_name.replace(' ', '_').lower()}_pr_curve.png"

    # Save the figure to disk and close it to free memory
    plt.savefig(filename)
    print(f"Plot saved to: {filename}")
    plt.close()
