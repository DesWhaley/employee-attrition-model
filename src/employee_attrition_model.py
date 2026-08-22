from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Create paths that work from the main project folder
PROJECT_FOLDER = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_FOLDER / "data"

ORIGINAL_DATA = DATA_FOLDER / "employee_attrition.csv"
PREPROCESSED_DATA = DATA_FOLDER / "preprocessed_employee_attrition.csv"


def preprocess_data():
    """Load, inspect, clean, and save the employee attrition dataset."""

    # Load the original dataset
    employee_data = pd.read_csv(ORIGINAL_DATA)

    print("Original dataset shape:", employee_data.shape)
    print("\nMissing values:", employee_data.isnull().sum().sum())
    print("Duplicate rows:", employee_data.duplicated().sum())

    # Remove duplicate rows, if any
    employee_data = employee_data.drop_duplicates()

    # Remove columns that do not provide useful predictive information
    columns_to_remove = [
        "EmployeeCount",
        "EmployeeNumber",
        "Over18",
        "StandardHours",
    ]

    employee_data = employee_data.drop(columns=columns_to_remove)

    # Convert the target variable from Yes/No to 1/0
    employee_data["Attrition"] = employee_data["Attrition"].map(
        {"Yes": 1, "No": 0}
    )

    # Save the cleaned dataset
    employee_data.to_csv(PREPROCESSED_DATA, index=False)

    print("\nPreprocessed dataset shape:", employee_data.shape)
    print("Preprocessed dataset saved to:", PREPROCESSED_DATA)

    return employee_data

def build_model(employee_data):
    """Build the preprocessing pipeline and Random Forest classifier."""

    # Separate the predictors from the target
    features = employee_data.drop(columns="Attrition")
    target = employee_data["Attrition"]

    # Identify numerical and categorical columns
    numerical_columns = features.select_dtypes(
        include=["number"]
    ).columns

    categorical_columns = features.select_dtypes(
        exclude=["number"]
    ).columns

    # Replace missing numerical values with the median and scale the data
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Replace missing categories and convert categories into numerical columns
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Apply the correct preprocessing steps to each column type
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )

    # Build the Random Forest classifier
    random_forest = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
    )

    # Combine preprocessing and classification into one model
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", random_forest),
        ]
    )

    print("\nRandom Forest model successfully built.")
    print("Numerical features:", len(numerical_columns))
    print("Categorical features:", len(categorical_columns))

    return model, features, target


if __name__ == "__main__":
    processed_data = preprocess_data()
    attrition_model, X, y = build_model(processed_data)