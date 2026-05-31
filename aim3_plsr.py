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


def main():
    DATA_PATH = "Training Data/ckd_cleaned.csv"

    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    n = select_n_components(X_train_scaled, y_train, max_components=10)
    print(f"\nOptimal number of PLSR components: {n}")

    model = fit_plsr(X_train_scaled, y_train, n_components=n)
    print(f"X scores shape: {model.x_scores_.shape}")


if __name__ == "__main__":
    main()
