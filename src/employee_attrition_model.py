from pathlib import Path

import pandas as pd


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


if __name__ == "__main__":
    preprocess_data()