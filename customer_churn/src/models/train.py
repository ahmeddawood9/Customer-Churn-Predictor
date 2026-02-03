from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def train_random_forest(preprocessor, X_train, y_train):

# Train a Random Forest classifier using a preprocessing pipeline.


    # Create a pipeline that first preprocesses the data, then trains the Random Forest
    model = Pipeline([
        ('preprocess', preprocessor),  # Apply scaling/encoding from preprocessor
        ('clf', RandomForestClassifier(
            n_estimators=200,           # Number of trees in the forest
            class_weight='balanced',    # Adjust weights for imbalanced classes
            random_state=42             # Ensures reproducibility
        ))
    ])

    # Fit the pipeline on training data
    model.fit(X_train, y_train)

    # Return the complete pipeline (preprocessing + trained model)
    return model


def train_xgboost(preprocessor, X_train, y_train):

   # Train an XGBoost classifier using a preprocessing pipeline.


    # Create a pipeline that first preprocesses the data, then trains the XGBoost model
    model = Pipeline([
        ('preprocess', preprocessor),  # Apply scaling/encoding from preprocessor
        ('clf', XGBClassifier(
            n_estimators=300,           # Number of boosting rounds (trees)
            max_depth=4,                # Max depth of each tree (prevents overfitting)
            learning_rate=0.05,         # Shrinks contribution of each tree
            scale_pos_weight=3,         # Balances class weight for imbalanced data
            eval_metric='logloss',      # Evaluation metric for binary classification
            random_state=42             # Ensures reproducibility
        ))
    ])

    # Fit the pipeline on training data
    model.fit(X_train, y_train)

    # Return the complete pipeline (preprocessing + trained model)
    return model
