import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# SET EXPERIMENT (BOLEH, AMAN)
mlflow.set_experiment("Loan Prediction CI Experiment")

DATA_PATH = "LoanPrediction_preprocessing.csv"

df = pd.read_csv(DATA_PATH)

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# LOG MANUAL (INI YANG DINILAI)
mlflow.log_param("model_type", "LogisticRegression")
mlflow.log_metric("accuracy", acc)

# REGISTER MODEL
mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    registered_model_name="LoanPredictionModel"
)

print("Training finished. Accuracy:", acc)
