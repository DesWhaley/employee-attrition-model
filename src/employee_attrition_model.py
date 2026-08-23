from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Create paths that work from the main project folder
PROJECT_FOLDER = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_FOLDER / "data"
RESULTS_FOLDER = PROJECT_FOLDER / "results"

ORIGINAL_DATA = DATA_FOLDER / "employee_attrition.csv"
PREPROCESSED_DATA = DATA_FOLDER / "preprocessed_employee_attrition.csv"
MODEL_RESULTS = RESULTS_FOLDER / "model_results.txt"
CONFUSION_MATRIX_IMAGE = RESULTS_FOLDER / "confusion_matrix.png"
CROSS_VALIDATION_RESULTS = RESULTS_FOLDER / "cross_validation_results.txt"

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

def train_model(model, features, target):
    """Split the dataset and train the Random Forest model."""

    # Reserve 20% of the records for final testing
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )

    # Train the complete preprocessing and classification pipeline
    model.fit(X_train, y_train)

    print("\nRandom Forest model successfully trained.")
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    return model, X_train, X_test, y_train, y_test

def apply_cross_validation(model, X_train, y_train):
    """Evaluate the model with five-fold stratified cross-validation."""

    RESULTS_FOLDER.mkdir(exist_ok=True)

    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scoring_metrics = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cross_validation,
        scoring=scoring_metrics,
    )

    accuracy_mean = scores["test_accuracy"].mean()
    precision_mean = scores["test_precision"].mean()
    recall_mean = scores["test_recall"].mean()
    f1_mean = scores["test_f1"].mean()

    print("\nFive-Fold Cross-Validation")
    print("--------------------------")
    print(f"Average accuracy:  {accuracy_mean:.2%}")
    print(f"Average precision: {precision_mean:.2%}")
    print(f"Average recall:    {recall_mean:.2%}")
    print(f"Average F1 score:  {f1_mean:.2%}")

    with open(
        CROSS_VALIDATION_RESULTS,
        "w",
        encoding="utf-8",
    ) as results_file:
        results_file.write("Five-Fold Cross-Validation Results\n")
        results_file.write("----------------------------------\n")
        results_file.write(f"Average accuracy: {accuracy_mean:.4f}\n")
        results_file.write(f"Average precision: {precision_mean:.4f}\n")
        results_file.write(f"Average recall: {recall_mean:.4f}\n")
        results_file.write(f"Average F1 score: {f1_mean:.4f}\n")

    print(
        "Cross-validation results saved to:",
        CROSS_VALIDATION_RESULTS,
    )

    return scores

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model using classification metrics."""

    RESULTS_FOLDER.mkdir(exist_ok=True)

    # Predict attrition for the testing records
    predictions = model.predict(X_test)

    # Calculate classification metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    report = classification_report(
        y_test,
        predictions,
        target_names=["Stayed", "Left"],
        zero_division=0,
    )

    print("\nModel Evaluation")
    print("----------------")
    print(f"Accuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1 score:  {f1:.2%}")
    print("\nClassification Report:")
    print(report)

    # Save the evaluation results to a text file
    with open(MODEL_RESULTS, "w", encoding="utf-8") as results_file:
        results_file.write("Employee Attrition Model Results\n")
        results_file.write("--------------------------------\n")
        results_file.write(f"Accuracy: {accuracy:.4f}\n")
        results_file.write(f"Precision: {precision:.4f}\n")
        results_file.write(f"Recall: {recall:.4f}\n")
        results_file.write(f"F1 score: {f1:.4f}\n\n")
        results_file.write("Classification Report\n")
        results_file.write(report)

    # Create and save the confusion matrix
    matrix = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Stayed", "Left"],
        yticklabels=["Stayed", "Left"],
    )
    plt.title("Employee Attrition Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_IMAGE, dpi=300)
    plt.close()

    print("Model results saved to:", MODEL_RESULTS)
    print("Confusion matrix saved to:", CONFUSION_MATRIX_IMAGE)

    return accuracy, precision, recall, f1

if __name__ == "__main__":
    processed_data = preprocess_data()
    attrition_model, X, y = build_model(processed_data)

    (
        trained_model,
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_model(attrition_model, X, y)

    cross_validation_scores = apply_cross_validation(
        attrition_model,
        X_train,
        y_train,
    )

    evaluation_results = evaluate_model(
        trained_model,
        X_test,
        y_test,
    )

    
