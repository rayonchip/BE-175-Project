import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

from aim1_classification import load_data, split_data, scale_features, FEATURE_NAMES


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


def bootstrap_ridge_ci(X, y, feature_names, alpha=1.0, n_bootstrap=200, ci=95, random_state=42):
    rng = np.random.default_rng(random_state)
    X_arr, y_arr = np.asarray(X), np.asarray(y)
    n_samples = len(y_arr)
    coef_rows = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_samples, size=n_samples)
        model = Ridge(alpha=alpha)
        model.fit(X_arr[idx], y_arr[idx])
        coef_rows.append(model.coef_.ravel())
    coef_arr = np.array(coef_rows)
    a = (100 - ci) / 2
    result = pd.DataFrame({
        "feature": feature_names,
        "mean_coef": coef_arr.mean(axis=0),
        "ci_lower": np.percentile(coef_arr, a, axis=0),
        "ci_upper": np.percentile(coef_arr, 100 - a, axis=0),
    })
    result["ci_width"] = result["ci_upper"] - result["ci_lower"]
    return result


def plsr_sensitivity(model, X_ref, feature_names, step=1.0):
    X_ref = np.asarray(X_ref, dtype=float).ravel()
    rows = []
    base = model.predict(X_ref.reshape(1, -1)).ravel()[0]
    for i, name in enumerate(feature_names):
        for sign, label in [(1, "delta_plus"), (-1, "delta_minus")]:
            perturbed = X_ref.copy()
            perturbed[i] += sign * step
            pred = model.predict(perturbed.reshape(1, -1)).ravel()[0]
            rows.append((name, label, pred - base))
    df = pd.DataFrame(rows, columns=["feature", "direction", "delta"])
    return df.pivot(index="feature", columns="direction", values="delta").reset_index().rename_axis(None, axis=1)[["feature", "delta_plus", "delta_minus"]]


def fit_ridge(X, y, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(np.asarray(X), np.asarray(y))
    return model


def compare_stability(plsr_ci_df, ridge_ci_df):
    merged = plsr_ci_df[["feature", "ci_width"]].merge(
        ridge_ci_df[["feature", "ci_width"]], on="feature", suffixes=("_plsr", "_ridge")
    )
    return merged.rename(columns={"ci_width_plsr": "plsr_ci_width", "ci_width_ridge": "ridge_ci_width"})


def plot_vip(vip_df, save_path="vip_scores.png"):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(vip_df["feature"][::-1], vip_df["vip"][::-1])
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1.2, label="VIP = 1 threshold")
    ax.set_xlabel("VIP Score")
    ax.set_title("PLSR Variable Importance (VIP)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_bootstrap_stability(bvip_df, save_path="bootstrap_stability.png"):
    df = bvip_df.sort_values("stability", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(df["feature"], df["stability"])
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1.2, label="50% stability")
    ax.set_xlabel("Stability (fraction of bootstraps with VIP > 1)")
    ax.set_title("Bootstrap VIP Stability")
    ax.set_xlim(0, 1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_plsr_sensitivity(sens_df, save_path="plsr_sensitivity.png"):
    df = sens_df.copy()
    df["max_abs_delta"] = df[["delta_plus", "delta_minus"]].abs().max(axis=1)
    df = df.sort_values("max_abs_delta", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width / 2, df["delta_plus"],  width, label="+1 unit")
    ax.bar(x + width / 2, df["delta_minus"], width, label="-1 unit")
    ax.set_xticks(x)
    ax.set_xticklabels(df["feature"], rotation=45, ha="right")
    ax.set_ylabel("Δ Prediction")
    ax.set_title("PLSR Perturbation Sensitivity")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    DATA_PATH = pathlib.Path(__file__).parent.parent / "Data" / "ckd_cleaned.csv"

    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    n = select_n_components(X_train_scaled, y_train, max_components=10)
    print(f"\nOptimal number of PLSR components: {n}")

    model = fit_plsr(X_train_scaled, y_train, n_components=n)
    feature_names = FEATURE_NAMES

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

    plot_vip(vip, "vip_scores.png")
    plot_bootstrap_stability(bvip, "bootstrap_stability.png")

    X_ref = X_test_scaled.mean(axis=0)
    sens = plsr_sensitivity(model, X_ref, feature_names)
    sens["max_delta"] = sens[["delta_plus", "delta_minus"]].abs().max(axis=1)
    print("\nPLSR perturbation sensitivity (top 10 by max |Δ prediction|):")
    print(sens.nlargest(10, "max_delta").drop(columns="max_delta").to_string(index=False))
    plot_plsr_sensitivity(sens.drop(columns="max_delta"), "plsr_sensitivity.png")
    print("Saved: vip_scores.png, bootstrap_stability.png, plsr_sensitivity.png")

    ridge_ci = bootstrap_ridge_ci(X_train_scaled, y_train,
                                   feature_names=feature_names, n_bootstrap=200)
    stab = compare_stability(ci_df.drop(columns="abs_coef"), ridge_ci)
    print("\nPLSR vs Ridge coefficient CI widths (top 10 by PLSR width):")
    print(stab.nlargest(10, "plsr_ci_width").to_string(index=False))


if __name__ == "__main__":
    main()
