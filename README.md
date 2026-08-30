<strong> **DO NOT DISTRIBUTE OR PUBLICLY POST SOLUTIONS TO THESE LABS. MAKE ALL FORKS OF THIS REPOSITORY WITH SOLUTION CODE PRIVATE. PLEASE REFER TO THE STUDENT CODE OF CONDUCT AND ETHICAL EXPECTATIONS FOR COLLEGE OF INFORMATION TECHNOLOGY STUDENTS FOR SPECIFICS. ** </strong>

# WESTERN GOVERNORS UNIVERSITY

## D683 – ADVANCED AI AND ML

**DO NOT DISTRIBUTE OR PUBLICLY POST SOLUTIONS TO THESE LABS. MAKE ALL FORKS OF THIS REPOSITORY WITH SOLUTION CODE PRIVATE. PLEASE REFER TO THE STUDENT CODE OF CONDUCT AND ETHICAL EXPECTATIONS FOR COLLEGE OF INFORMATION TECHNOLOGY STUDENTS FOR SPECIFICS.**

# WESTERN GOVERNORS UNIVERSITY

## D683 – ADVANCED AI AND ML

# Employee Attrition Prediction Product

## Project Overview

This project uses a **Random Forest model** to predict whether an employee may leave a company. Employee turnover can increase hiring and training costs. The goal is to help an HR team better understand which employees may be at risk of leaving.

The target column is `Attrition`. A value of **1** means the employee left, and **0** means the employee stayed.

## Dataset

The project uses the **IBM HR Analytics Employee Attrition and Performance dataset**.

Dataset source:
https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

The original dataset has **1,470 rows and 35 columns**. It includes age, department, job role, monthly income, overtime, job satisfaction, and years at the company.

## Data Preprocessing

The program prepares the data in two stages.

**First, the program:**

1. Loads the original CSV file.
2. Checks for missing values and duplicate rows.
3. Removes duplicate rows if any are found.
4. Removes `EmployeeCount`, `Over18`, and `StandardHours` because each column has the same value for all employees.
5. Removes `EmployeeNumber` because it is an ID number.
6. Changes `Attrition` from Yes/No to 1/0.
7. Saves the cleaned data as `data/preprocessed_employee_attrition.csv`.

**The model pipeline then:**

- Fills missing number values with the median.
- Fills missing category values with the most common value.
- Scales number columns using `StandardScaler`.
- Uses one-hot encoding to turn category values into number columns.

The pipeline learns these changes from the training data only. During cross-validation, it repeats these steps using the training portion of each fold. This keeps the test data from affecting model training.

The saved CSV contains the initial cleaning changes. It does not contain the scaled or encoded data.

## Machine Learning Model

The project uses a **Random Forest classifier**, which combines predictions from several decision trees. This model was chosen because it can work with the prepared employee data and is generally less likely to overfit than one decision tree.

The starting model uses:

- **100 decision trees**
- **Balanced class weights**
- **A random state of 42**

Balanced class weights give more weight to the smaller group of employees who left. The fixed random state helps make the results repeatable when the same data and software versions are used.

## Model Training

The data is split into:

- **80% training data**
- **20% test data**

The split keeps about the same share of employees who left in both sets. The model learns from the training data. The test data is used to check its predictions.

## Model Evaluation

The program uses four main scores:

- **Accuracy:** How often the model makes the correct prediction overall.
- **Precision:** Of the employees predicted to leave, how many actually left.
- **Recall:** Of the employees who actually left, how many the model found.
- **F1 score:** A score that combines precision and recall.

The program also saves a **classification report** and a **confusion matrix**. The confusion matrix shows correct and incorrect predictions.

The starting model had these test results:

| Score | Result |
| --- | --- |
| Accuracy | 83.67% |
| Precision | 48.57% |
| Recall | 36.17% |
| F1 score | 41.46% |

## Cross-Validation

The project uses **five-fold stratified cross-validation** on the training data.

This divides the training data into five parts. Each part is used once to check a model trained on the other four parts. Each part keeps about the same share of employees who left.

The average scores were:

| Score | Result |
| --- | --- |
| Accuracy | 86.48% |
| Precision | 64.51% |
| Recall | 36.84% |
| F1 score | 46.26% |

## Model Tuning

The program uses `GridSearchCV` to compare different Random Forest settings. It tests:

- Number of trees
- Maximum tree depth
- Minimum samples per leaf
- Class weight

The search uses five-fold cross-validation on the training data. **Recall** is the main score used to choose the best settings because the goal is to find more employees who may leave.

The best settings were:

- **Number of trees:** 200
- **Maximum depth:** 4
- **Minimum samples per leaf:** 1
- **Class weight:** balanced

The tuned model had these test results:

| Score | Original Model | Tuned Model |
| --- | --- | --- |
| Accuracy | 83.67% | 78.57% |
| Precision | 48.57% | 39.47% |
| Recall | 36.17% | 63.83% |
| F1 score | 41.46% | 48.78% |

Recall increased from **36.17% to 63.83%**, so the tuned model found more employees who left. However, it also wrongly flagged more employees who stayed. This lowered accuracy and precision.

The model did not meet the project goal of **85% accuracy and 80% recall**. Tuning improved recall and F1 score, but the results show that the model still makes errors.

## Software Requirements

- **Windows 11** for the steps below
- **Python 3.10 or later**, with package versions that support it
- **Visual Studio Code**
- **Git**
- Internet access to download files and packages
- Access to the private GitLab project

The required Python packages are listed in `requirements.txt`. The code uses:

- pandas
- scikit-learn
- matplotlib
- seaborn

## Hardware Requirements

Recommended setup:

- Laptop or desktop computer
- Multi-core processor
- At least **8 GB of RAM**
- Free space for Python, packages, data, and results

## Project Files

- `data/`: Original and cleaned datasets
- `src/employee_attrition_model.py`: Main program
- `results/`: Saved scores, reports, and charts
- `requirements.txt`: Python packages needed to run the program
- `README.md`: Project details and setup steps
- `.gitignore`: Files and folders to leave out of Git

## How to Run the Project

The project was created in **Visual Studio Code on Windows 11**.

### 1. Clone the Project

Open Visual Studio Code and select **Terminal > New Terminal**. Choose **PowerShell** as the terminal type for these commands, then enter:

```text
git clone https://gitlab.com/wgu-gitlab-environment/student-repos/dwhal34/d683-advanced-ai-and-ml.git
cd d683-advanced-ai-and-ml
```

### 2. Create and Activate the Environment

Enter:

```text
py -m venv .venv
.venv\Scripts\Activate.ps1
```

If the activation script is blocked, enter:

```text
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

This setting lasts only for the current terminal session.

### 3. Install the Packages

Enter:

```text
python -m pip install -r requirements.txt
```

### 4. Check the Data File

Make sure the original dataset is saved here:

```text
data/employee_attrition.csv
```

If the file is missing, download it from the dataset link above. Rename the CSV to `employee_attrition.csv` and place it in the `data` folder.

### 5. Run the Program

From the main project folder, enter:

```text
python src/employee_attrition_model.py
```

The program prepares the data, trains and tests the model, runs cross-validation, and tunes the settings. Tuning may take several minutes.

The scores appear in the terminal. Reports and charts are saved in the `results` folder.

## Output Files

The program creates:

- `data/preprocessed_employee_attrition.csv`
- `results/model_results.txt`
- `results/cross_validation_results.txt`
- `results/hyperparameter_tuning_results.txt`
- `results/confusion_matrix.png`
- `results/optimized_confusion_matrix.png`