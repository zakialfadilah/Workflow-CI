import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


# ======================
# Load Data
# ======================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = "LoanPrediction_preprocessing.csv"
df = pd.read_csv(DATA_PATH)


X = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"]

# ======================
# Train-Test Split
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# MLflow Experiment
# ======================
mlflow.set_experiment("Loan Prediction Experiment")

with mlflow.start_run():
    # Parameter
    C = 1.0
    max_iter = 200

    model = LogisticRegression(C=C, max_iter=max_iter)
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log to MLflow
    mlflow.log_param("C", C)
    mlflow.log_param("max_iter", max_iter)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(model, "model")

    print("Accuracy:", acc)
    print("F1 Score:", f1)
