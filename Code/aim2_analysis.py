import pathlib
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

from aim1_classification import load_data, split_data, scale_features, evaluate_model, FEATURE_NAMES


def cross_validate_model(name, model, X, y, cv=5):
    scoring = {
        "accuracy":  make_scorer(accuracy_score),
        "precision": make_scorer(precision_score),
        "recall":    make_scorer(recall_score),
        "f1":        make_scorer(f1_score),
    }
    scores = cross_validate(model, X, y, cv=StratifiedKFold(n_splits=cv),
                            scoring=scoring, return_train_score=True)
    return {
        "Model":               name,
        "mean_train_accuracy": round(scores["train_accuracy"].mean(), 4),
        "std_train_accuracy":  round(scores["train_accuracy"].std(),  4),
        "mean_accuracy":       round(scores["test_accuracy"].mean(),  4),
        "std_accuracy":        round(scores["test_accuracy"].std(),   4),
        "mean_precision":      round(scores["test_precision"].mean(), 4),
        "std_precision":       round(scores["test_precision"].std(),  4),
        "mean_recall":         round(scores["test_recall"].mean(),    4),
        "std_recall":          round(scores["test_recall"].std(),     4),
        "mean_f1":             round(scores["test_f1"].mean(),        4),
        "std_f1":              round(scores["test_f1"].std(),         4),
    }


def tune_logistic_regression(X_train_scaled, y_train):
    param_grid = {"C": [0.01, 0.1, 1, 10, 100]}
    grid = GridSearchCV(
        LogisticRegression(max_iter=1000), param_grid, cv=5, scoring="recall"
    )
    grid.fit(X_train_scaled, y_train)
    return grid.best_estimator_, grid.best_params_


def tune_decision_tree(X_train, y_train):
    param_grid = {
        "max_depth":        [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10],
    }
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring="recall"
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def get_feature_importance(model, feature_names):
    if isinstance(model, LogisticRegression):
        importances = np.abs(model.coef_[0])
    else:
        importances = model.feature_importances_

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def plot_feature_importance(importance_df, title, save_path="feature_importance.png"):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importance_df["feature"][::-1], importance_df["importance"][::-1])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def run_sensitivity_analysis(model, X_ref, feature_names, scaler=None, step=1.0):
    ref = pd.Series(X_ref, index=feature_names, dtype=float)

    def predict_prob(x_row):
        df_row = pd.DataFrame([np.array(x_row)], columns=feature_names)
        if scaler is not None:
            return model.predict_proba(scaler.transform(df_row))[0][1]
        return model.predict_proba(df_row)[0][1]

    base_prob = predict_prob(ref)
    rows = []
    for feat in feature_names:
        plus  = ref.copy(); plus[feat]  += step
        minus = ref.copy(); minus[feat] -= step
        rows.append({
            "feature":     feat,
            "delta_plus":  round(float(predict_prob(plus)  - base_prob), 6),
            "delta_minus": round(float(predict_prob(minus) - base_prob), 6),
        })
    return pd.DataFrame(rows)


def plot_sensitivity(sensitivity_df, title, save_path="sensitivity.png"):
    df = sensitivity_df.copy()
    df["max_abs_delta"] = df[["delta_plus", "delta_minus"]].abs().max(axis=1)
    df = df.sort_values("max_abs_delta", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width / 2, df["delta_plus"],  width, label="+1 std")
    ax.bar(x + width / 2, df["delta_minus"], width, label="-1 std")
    ax.set_xticks(x)
    ax.set_xticklabels(df["feature"], rotation=45, ha="right")
    ax.set_ylabel("Δ P(CKD)")
    ax.set_title(title)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_regularization_path(X_train_scaled, y_train, X_test_scaled, y_test,
                              C_values=None, save_path="regularization_path.png"):
    if C_values is None:
        C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    train_accs, test_accs, test_recalls = [], [], []
    for C in C_values:
        model = LogisticRegression(C=C, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        train_accs.append(accuracy_score(y_train, model.predict(X_train_scaled)))
        test_accs.append(accuracy_score(y_test,  model.predict(X_test_scaled)))
        test_recalls.append(recall_score(y_test, model.predict(X_test_scaled)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(C_values, train_accs,   "o-", label="Train accuracy")
    ax.semilogx(C_values, test_accs,    "s-", label="CV/test accuracy")
    ax.semilogx(C_values, test_recalls, "^-", label="Test recall (sensitivity)")
    ax.set_xlabel("C  (higher = less regularization)")
    ax.set_ylabel("Score")
    ax.set_title("Regularization Path — Logistic Regression")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    DATA_PATH   = pathlib.Path(__file__).parent.parent / "Data"    / "ckd_cleaned.csv"
    RESULTS_DIR = pathlib.Path(__file__).parent.parent / "Results"
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    X_scaled = scaler.transform(X)

    # --- Cross-validation ---
    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.tree import DecisionTreeClassifier as DT
    base_lr = LR(max_iter=1000); base_lr.fit(X_train_scaled, y_train)
    base_dt = DT(random_state=42); base_dt.fit(X_train, y_train)

    print("\n=== Cross-Validation (5-fold) ===")
    for result in [
        cross_validate_model("Logistic Regression", base_lr, X_scaled, y),
        cross_validate_model("Decision Tree",        base_dt, X,        y),
    ]:
        print(f"\n{result['Model']}")
        for k, v in result.items():
            if k != "Model":
                print(f"  {k}: {v}")

    # --- Hyperparameter Tuning ---
    print("\n=== Hyperparameter Tuning ===")
    lr_tuned, lr_params = tune_logistic_regression(X_train_scaled, y_train)
    dt_tuned, dt_params = tune_decision_tree(X_train, y_train)
    print(f"Best LR params: {lr_params}")
    print(f"Best DT params: {dt_params}")

    print("\nTuned model performance:")
    for row in [
        evaluate_model("LR (tuned)", lr_tuned, X_test_scaled, y_test),
        evaluate_model("DT (tuned)", dt_tuned, X_test, y_test),
    ]:
        print(f"  {row}")

    # --- Feature Importance ---
    print("\n=== Feature Importance (top 5) ===")
    lr_imp = get_feature_importance(lr_tuned, FEATURE_NAMES)
    dt_imp = get_feature_importance(dt_tuned, FEATURE_NAMES)
    print("LR:\n", lr_imp.head())
    print("DT:\n", dt_imp.head())
    plot_feature_importance(lr_imp, "LR Feature Importance", RESULTS_DIR / "lr_feature_importance.png")
    plot_feature_importance(dt_imp, "DT Feature Importance", RESULTS_DIR / "dt_feature_importance.png")
    print("Saved: lr_feature_importance.png, dt_feature_importance.png")

    # --- Sensitivity Analysis ---
    print("\n=== Sensitivity Analysis (top 5 by Δ P(CKD)) ===")
    ref = X_test.median()
    lr_sens = run_sensitivity_analysis(lr_tuned, ref, FEATURE_NAMES, scaler=scaler)
    dt_sens = run_sensitivity_analysis(dt_tuned, ref, FEATURE_NAMES)
    lr_sens["max_abs"] = lr_sens[["delta_plus", "delta_minus"]].abs().max(axis=1)
    dt_sens["max_abs"] = dt_sens[["delta_plus", "delta_minus"]].abs().max(axis=1)
    print("LR:\n", lr_sens.nlargest(5, "max_abs")[["feature", "delta_plus", "delta_minus"]])
    print("DT:\n", dt_sens.nlargest(5, "max_abs")[["feature", "delta_plus", "delta_minus"]])
    plot_sensitivity(lr_sens.drop(columns="max_abs"), "LR Sensitivity", RESULTS_DIR / "lr_sensitivity.png")
    plot_sensitivity(dt_sens.drop(columns="max_abs"), "DT Sensitivity", RESULTS_DIR / "dt_sensitivity.png")
    print("Saved: lr_sensitivity.png, dt_sensitivity.png")

    # --- Regularization Path ---
    plot_regularization_path(X_train_scaled, y_train, X_test_scaled, y_test,
                             save_path=RESULTS_DIR / "regularization_path.png")
    print("Saved: regularization_path.png")


if __name__ == "__main__":
    main()
