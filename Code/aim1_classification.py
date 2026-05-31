import pathlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

FEATURE_NAMES = [
    "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba",
    "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc",
    "htn", "dm", "cad", "appet", "pe", "ane",
]


def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["class"])
    y = df["class"]
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def train_logistic_regression(X_train_scaled, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    return model


def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(name, model, X_eval, y_test):
    y_pred = model.predict(X_eval)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "Model":       name,
        "Accuracy":    round(accuracy_score(y_test, y_pred), 4),
        "Precision":   round(precision_score(y_test, y_pred), 4),
        "Recall":      round(recall_score(y_test, y_pred), 4),
        "Specificity": round(specificity, 4),
        "F1 Score":    round(f1_score(y_test, y_pred), 4),
    }


def plot_confusion_matrices(models, y_test, save_path="confusion_matrices.png"):
    labels = ["Not CKD", "CKD"]
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    if len(models) == 1:
        axes = [axes]

    for ax, (name, (model, X_eval)) in zip(axes, models.items()):
        y_pred = model.predict(X_eval)
        cm = confusion_matrix(y_test, y_pred)
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
        ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    DATA_PATH   = pathlib.Path(__file__).parent.parent / "Data"    / "ckd_cleaned.csv"
    RESULTS_DIR = pathlib.Path(__file__).parent.parent / "Results"

    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)

    lr = train_logistic_regression(X_train_scaled, y_train)
    dt = train_decision_tree(X_train, y_train)

    models = {
        "Logistic Regression": (lr, X_test_scaled),
        "Decision Tree":       (dt, X_test),
    }

    results = [
        evaluate_model("Logistic Regression", lr, X_test_scaled, y_test),
        evaluate_model("Decision Tree",        dt, X_test,        y_test),
    ]

    print("\n=== Aim 1: Classification Results ===")
    print(pd.DataFrame(results).to_string(index=False))

    plot_confusion_matrices(models, y_test, RESULTS_DIR / "confusion_matrices.png")
    print(f"\nSaved: {RESULTS_DIR / 'confusion_matrices.png'}")


if __name__ == "__main__":
    main()
