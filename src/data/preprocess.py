from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

TARGET = 'MEDV'

NUMERIC_FEATURES = [
    "CRIM", "ZN", "INDUS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"
]

BINARY_FEATURES = ["CHAS"]

FEATURES = NUMERIC_FEATURES + BINARY_FEATURES

def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Build a preprocessing pipeline that transforms numerical and binary features.
    """

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    binary_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("binary", binary_pipeline, BINARY_FEATURES)
        ]
    )

    return preprocessor
