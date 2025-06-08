import mlflow
import mlflow.sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

# Parsing argumen
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

# Load data
df = pd.read_csv(args.data_path)
X = df.drop("Personality", axis=1)
y = df["Personality"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set experiment
mlflow.set_experiment("SVM Personality Classification")

# Autolog aktif
mlflow.sklearn.autolog()

# Train dan evaluasi model
clf = SVC(kernel="linear", C=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Manual logging tambahan
report = classification_report(y_test, y_pred, output_dict=True)
mlflow.log_metric("accuracy", report["accuracy"])