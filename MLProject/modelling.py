import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# ======================
# PATH SETUP (CI SAFE)
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "preprocessing",
    "LoanPrediction_preprocessing.csv"
)

# ======================
# LOAD DATA
# ======================
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"]

# ======================
# TRAIN TEST SPLIT
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# MLFLOW SETUP
# ======================
mlflow.set_experiment("Loan Prediction Experiment")

# AUTOLOG (INI YANG DIMINTA REVIEWER)
mlflow.autolog()

with mlflow.start_run():

    # MODEL
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # PREDICTION
    y_pred = model.predict(X_test)

    # METRICS
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Accuracy:", acc)
    print("F1 Score:", f1)
