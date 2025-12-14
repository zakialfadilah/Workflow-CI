import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

mlflow.set_tracking_uri(
    "https://dagshub.com/zakialfadilah/loan-prediction-mlflow.mlflow"
)
mlflow.set_experiment("Loan Prediction Experiment")

df = pd.read_csv("preprocessing/LoanPrediction_preprocessing.csv")

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="LoanPredictionModel"
    )
