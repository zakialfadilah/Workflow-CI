import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# Load Dataset
# ======================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = "LoanPrediction_preprocessing.csv"


df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# Hyperparameter (simple example)
# ======================
C = 1.0
solver = "lbfgs"
max_iter = 200

# ======================
# Log Parameters
# ======================
mlflow.log_param("model_type", "LogisticRegression")
mlflow.log_param("C", C)
mlflow.log_param("solver", solver)
mlflow.log_param("max_iter", max_iter)

# ======================
# Train Model
# ======================
model = LogisticRegression(
    C=C,
    solver=solver,
    max_iter=max_iter
)

model.fit(X_train, y_train)

# ======================
# Evaluation
# ======================
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
try:
    f1 = f1_score(y_test, y_pred)
except:
    f1 = f1_score(y_test, y_pred, average="macro")

mlflow.log_metric("accuracy", acc)
mlflow.log_metric("f1_score", f1)

# ======================
# ARTIFACT 1: Confusion Matrix
# ======================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

cm_path = "confusion_matrix.png"
plt.savefig(cm_path)
plt.close()

mlflow.log_artifact(cm_path)

# ======================
# ARTIFACT 2: Metrics Report
# ======================
report_path = "metrics_report.txt"
with open(report_path, "w") as f:
    f.write(f"Accuracy: {acc}\n")
    f.write(f"F1 Score: {f1}\n")
    f.write(f"C: {C}\n")
    f.write(f"Solver: {solver}\n")

mlflow.log_artifact(report_path)

# ======================
# Log Model
# ======================
input_example = X_train.head(5)

mlflow.sklearn.log_model(
    model,
    artifact_path="model",
    input_example=input_example
)

print(f"[DONE] acc={acc:.4f}, f1={f1:.4f}")
