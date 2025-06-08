import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
df = pd.read_csv("personality_dataset_clean.csv")
X = df.drop("Personality", axis=1)
y = df["Personality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Setup MLflow
mlflow.set_experiment("SVM Personality Tuning")

with mlflow.start_run(nested=True):
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
    f1 = f1_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    # Manual logging
    mlflow.log_param("best_C", grid.best_params_['C'])
    mlflow.log_param("best_kernel", grid.best_params_['kernel'])
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # Save confusion matrix
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    os.makedirs("model_artifacts", exist_ok=True)
    plt.savefig("model_artifacts/training_confusion_matrix.png")
    mlflow.log_artifact("model_artifacts/training_confusion_matrix.png")

    # Save model manually
    import joblib
    joblib.dump(best_model, "model_artifacts/model.pkl")
    mlflow.log_artifact("model_artifacts/model.pkl")
    mlflow.sklearn.log_model(best_model, "model")