
import yaml
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load config
with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

CSV_PATH = config["data"]["output_csv"]
SAVE_DIR = config["model"]["save_dir"]
SAVE_NAMES = config["model"]["save_filenames"]

RESULTS_DIR = os.path.join(SAVE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load data
df = pd.read_csv(CSV_PATH)
print(f"Loaded dataset with shape: {df.shape}")

# Label encode
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label"])

X = df.drop("label", axis=1).values
y = df["label"].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define all models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
    "MLPClassifier": MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42),
    "NaiveBayes": GaussianNB(),
    "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=42)
}

# Train, evaluate, save
os.makedirs(SAVE_DIR, exist_ok=True)

for name, model in models.items():
    print(f"\n=== Training: {name} ===")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(report)

    # Plot and save confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix: {name}")
    output_path = os.path.join(RESULTS_DIR, f"{name}_confusion_matrix.jpg")
    plt.savefig(output_path)
    print(f"Confusion matrix saved to: {output_path}")
    plt.close()

    # Save model
    model_path = os.path.join(SAVE_DIR, f"{name}.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

# Save LabelEncoder
label_path = os.path.join(SAVE_DIR, "label_encoder.pkl")
joblib.dump(label_encoder, label_path)
print(f"LabelEncoder saved to: {label_path}")
