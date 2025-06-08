import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset hasil preprocessing
df = pd.read_csv("personality_dataset_clean.csv")

# Pisahkan fitur dan label
X = df.drop("Personality", axis=1)
y = df["Personality"]

# Split train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi MLflow
mlflow.set_experiment("SVM Personality Classification")

# Autolog aktif
mlflow.sklearn.autolog()

# Run experiment
with mlflow.start_run():
    model = SVC(kernel="linear", C=1.0)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc}")