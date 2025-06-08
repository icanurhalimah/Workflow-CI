import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import argparse

# --------------------------
# Argument parser (wajib untuk MLflow Project)
# --------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

# --------------------------
# Load Dataset
# --------------------------
df = pd.read_csv(args.data_path)
X = df.drop("Personality", axis=1)
y = df["Personality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------
# Logging ke MLflow
# --------------------------
mlflow.set_experiment("SVM Personality Tuning")

# Hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}
grid = GridSearchCV(SVC(), param_grid, cv=3)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
preds = best_model.predict(X_test)

# Metrics
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, average='weighted')
cm = confusion_matrix(y_test, preds)

# Log parameters & metrics
mlflow.log_param("best_C", grid.best_params_['C'])
mlflow.log_param("best_kernel", grid.best_params_['kernel'])
mlflow.log_metric("accuracy", acc)
mlflow.log_metric("f1_score", f1)

# Save & log confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

os.makedirs("model_artifacts", exist_ok=True)
plt.savefig("model_artifacts/training_confusion_matrix.png")
mlflow.log_artifact("model_artifacts/training_confusion_matrix.png")

# Save & log model
joblib.dump(best_model, "model_artifacts/model.pkl")
mlflow.log_artifact("model_artifacts/model.pkl")