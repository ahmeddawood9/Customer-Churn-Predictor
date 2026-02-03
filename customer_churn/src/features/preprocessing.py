# Import tools for preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def build_preprocessor(X):
    """
    Builds a preprocessing pipeline that:
    - Scales numerical features
    - One-hot encodes categorical features


 """

    # Select categorical columns (text / object type)

    categorical_cols = X.select_dtypes(include='object').columns

    # Select numerical columns (int, float)

    numerical_cols = X.select_dtypes(exclude='object').columns

    # Pipeline for numerical features
    # StandardScaler transforms data to mean=0 and std=1
    numeric_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Pipeline for categorical features
    # OneHotEncoder converts categories into binary columns
    # handle_unknown='ignore' prevents errors on unseen categories
    categorical_pipeline = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # ColumnTransformer applies different pipelines
    # to different column types and combines the output
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])

    return preprocessor
