import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score

from aim1_classification import load_data, split_data, scale_features


def fit_plsr(X, y, n_components):
    model = PLSRegression(n_components=n_components)
    model.fit(X, y)
    return model


def select_n_components(X, y, max_components=10):
    best_n, best_score = 1, -np.inf
    for n in range(1, max_components + 1):
        model = PLSRegression(n_components=n)
        scores = cross_val_score(model, X, y, cv=5, scoring="r2")
        mean_score = scores.mean()
        if mean_score > best_score:
            best_score = mean_score
            best_n = n
    return best_n


def compute_vip(model, feature_names):
    W = model.x_weights_   # (p, K) — weight of each feature per component
    T = model.x_scores_    # (n, K) — sample scores
    Q = model.y_loadings_  # (1, K) — Y loadings

    p, K = W.shape

    # Normalise W columns to unit length (required by VIP formula)
    W_norm = W / np.linalg.norm(W, axis=0, keepdims=True)

    # SS of Y explained per component: ||T_k||^2 * q_k^2
    SS = np.sum(T ** 2, axis=0) * (Q.ravel() ** 2)
    SS_total = SS.sum()

    # VIP_j = sqrt(p * sum_k(w_kj^2 * SS_k) / SS_total)
    vip_scores = np.sqrt(p * (W_norm ** 2) @ SS / SS_total)

    df = pd.DataFrame({"feature": feature_names, "vip": vip_scores})
    return df.sort_values("vip", ascending=False).reset_index(drop=True)


def bootstrap_vip(X, y, n_components, feature_names, n_bootstrap=200, random_state=42):
    rng = np.random.default_rng(random_state)
    X_arr, y_arr = np.asarray(X), np.asarray(y)
    n_samples = len(y_arr)
    vip_rows = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_samples, size=n_samples)
        model = PLSRegression(n_components=n_components)
        model.fit(X_arr[idx], y_arr[idx])
        vip_series = compute_vip(model, feature_names).set_index("feature")["vip"]
        vip_rows.append(vip_series)
    vip_df = pd.DataFrame(vip_rows)[feature_names]
    return pd.DataFrame({
        "feature": feature_names,
        "stability": (vip_df > 1.0).mean().values,
        "mean_vip": vip_df.mean().values,
    })


def condition_number(X):
    return float(np.linalg.cond(X))


def bootstrap_coef_ci(X, y, n_components, feature_names, n_bootstrap=200, ci=95, random_state=42):
    rng = np.random.default_rng(random_state)
    X_arr, y_arr = np.asarray(X), np.asarray(y)
    n_samples = len(y_arr)
    coef_rows = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_samples, size=n_samples)
        model = PLSRegression(n_components=n_components)
        model.fit(X_arr[idx], y_arr[idx])
        coef_rows.append(model.coef_.ravel())
    coef_arr = np.array(coef_rows)
    alpha = (100 - ci) / 2
    result = pd.DataFrame({
        "feature": feature_names,
        "mean_coef": coef_arr.mean(axis=0),
        "ci_lower": np.percentile(coef_arr, alpha, axis=0),
        "ci_upper": np.percentile(coef_arr, 100 - alpha, axis=0),
    })
    result["ci_width"] = result["ci_upper"] - result["ci_lower"]
    return result


def main():
    DATA_PATH = "Training Data/ckd_cleaned.csv"

    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    n = select_n_components(X_train_scaled, y_train, max_components=10)
    print(f"\nOptimal number of PLSR components: {n}")

    model = fit_plsr(X_train_scaled, y_train, n_components=n)
    feature_names = list(X.columns)

    vip = compute_vip(model, feature_names)
    print("\nTop 5 VIP scores:")
    print(vip.head())

    X_scaled = scaler.transform(X)
    cond = condition_number(X_scaled)
    print(f"\nCondition number of scaled feature matrix: {cond:.1f}")

    bvip = bootstrap_vip(X_train_scaled, y_train, n_components=n,
                         feature_names=feature_names, n_bootstrap=200)
    print("\nBootstrap VIP stability (top 10 by stability):")
    print(bvip.nlargest(10, "stability").to_string(index=False))

    ci_df = bootstrap_coef_ci(X_train_scaled, y_train, n_components=n,
                               feature_names=feature_names, n_bootstrap=200)
    print("\nBootstrap 95% CI on PLSR coefficients (top 10 by |mean_coef|):")
    ci_df["abs_coef"] = ci_df["mean_coef"].abs()
    print(ci_df.nlargest(10, "abs_coef")
              .drop(columns="abs_coef")
              .to_string(index=False))


if __name__ == "__main__":
    main()
