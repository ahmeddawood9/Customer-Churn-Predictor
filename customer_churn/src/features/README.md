# Feature Engineering & Preprocessing

This module converts cleaned tabular data into a numerical representation
suitable for machine learning models.

## Responsibilities
- Separate numerical and categorical features
- Apply feature scaling to numerical columns
- One-hot encode categorical features
- Use pipelines to prevent data leakage

## Design Choice
Pipelines ensure the same transformations are applied consistently to
training and test data.
