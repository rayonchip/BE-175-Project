"""
TDD tests for aim3_plsr.py.
Run with: python -m pytest test_aim3.py -v
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.cross_decomposition import PLSRegression

from aim1_classification import load_data, split_data, scale_features
from aim3_plsr import (
    fit_plsr, select_n_components, compute_vip,
    bootstrap_vip, condition_number, bootstrap_coef_ci,
    plsr_sensitivity, fit_ridge, compare_stability,
    plot_vip, plot_bootstrap_stability, plot_plsr_sensitivity,
)

DATA_PATH = "Training Data/ckd_cleaned.csv"
FEATURE_NAMES = [
    "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba",
    "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc",
    "htn", "dm", "cad", "appet", "pe", "ane",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def data():
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    X_scaled = scaler.transform(X)
    return X_train_scaled, y_train, X_scaled, y


# ---------------------------------------------------------------------------
# fit_plsr
# ---------------------------------------------------------------------------

class TestFitPlsr:
    def test_returns_pls_model(self, data):
        X_train_scaled, y_train, _, _ = data
        model = fit_plsr(X_train_scaled, y_train, n_components=2)
        assert isinstance(model, PLSRegression)

    def test_model_is_fitted(self, data):
        X_train_scaled, y_train, _, _ = data
        model = fit_plsr(X_train_scaled, y_train, n_components=2)
        assert hasattr(model, "x_scores_")

    def test_n_components_respected(self, data):
        X_train_scaled, y_train, _, _ = data
        for n in [1, 2, 5]:
            model = fit_plsr(X_train_scaled, y_train, n_components=n)
            assert model.x_scores_.shape[1] == n

    def test_x_scores_shape(self, data):
        X_train_scaled, y_train, _, _ = data
        model = fit_plsr(X_train_scaled, y_train, n_components=3)
        assert model.x_scores_.shape[0] == len(y_train)


# ---------------------------------------------------------------------------
# select_n_components
# ---------------------------------------------------------------------------

class TestSelectNComponents:
    def test_returns_int(self, data):
        X_train_scaled, y_train, _, _ = data
        n = select_n_components(X_train_scaled, y_train, max_components=5)
        assert isinstance(n, int)

    def test_within_valid_range(self, data):
        X_train_scaled, y_train, _, _ = data
        max_c = 8
        n = select_n_components(X_train_scaled, y_train, max_components=max_c)
        assert 1 <= n <= max_c

    def test_respects_max_components(self, data):
        X_train_scaled, y_train, _, _ = data
        for max_c in [3, 5, 10]:
            n = select_n_components(X_train_scaled, y_train, max_components=max_c)
            assert n <= max_c


# ---------------------------------------------------------------------------
# compute_vip
# ---------------------------------------------------------------------------

class TestComputeVip:
    @pytest.fixture(scope="class")
    def vip(self, data):
        X_train_scaled, y_train, _, _ = data
        model = fit_plsr(X_train_scaled, y_train, n_components=3)
        return compute_vip(model, FEATURE_NAMES)

    def test_returns_dataframe(self, vip):
        assert isinstance(vip, pd.DataFrame)

    def test_has_correct_columns(self, vip):
        assert set(vip.columns) == {"feature", "vip"}

    def test_length_equals_feature_count(self, vip):
        assert len(vip) == len(FEATURE_NAMES)

    def test_all_vip_nonnegative(self, vip):
        assert (vip["vip"] >= 0).all()

    def test_sorted_descending(self, vip):
        assert vip["vip"].is_monotonic_decreasing

    def test_significant_biomarkers_exist(self, vip):
        """On CKD data, at least one biomarker should score VIP > 1."""
        assert (vip["vip"] > 1.0).any()


# ---------------------------------------------------------------------------
# bootstrap_vip
# ---------------------------------------------------------------------------

class TestBootstrapVip:
    @pytest.fixture(scope="class")
    def bvip(self, data):
        X_train_scaled, y_train, _, _ = data
        return bootstrap_vip(
            X_train_scaled, y_train,
            n_components=3, feature_names=FEATURE_NAMES,
            n_bootstrap=50, random_state=0,
        )

    def test_returns_dataframe(self, bvip):
        assert isinstance(bvip, pd.DataFrame)

    def test_has_correct_columns(self, bvip):
        assert set(bvip.columns) == {"feature", "stability", "mean_vip"}

    def test_length_equals_feature_count(self, bvip):
        assert len(bvip) == len(FEATURE_NAMES)

    def test_stability_in_range(self, bvip):
        assert bvip["stability"].between(0.0, 1.0).all()

    def test_mean_vip_nonnegative(self, bvip):
        assert (bvip["mean_vip"] >= 0).all()

    def test_top_biomarkers_stable(self, bvip):
        """hemo and sg are the dominant CKD predictors — both should be stable."""
        top_stable = bvip.nlargest(5, "stability")["feature"].values
        assert "hemo" in top_stable or "sg" in top_stable


# ---------------------------------------------------------------------------
# condition_number
# ---------------------------------------------------------------------------

class TestConditionNumber:
    def test_returns_float(self, data):
        _, _, X_scaled, _ = data
        assert isinstance(condition_number(X_scaled), float)

    def test_greater_than_one(self, data):
        _, _, X_scaled, _ = data
        assert condition_number(X_scaled) >= 1.0

    def test_ckd_features_multicollinear(self, data):
        """Clinical features (hemo/pcv/rbcc etc.) are correlated; expect cond > 10."""
        _, _, X_scaled, _ = data
        assert condition_number(X_scaled) > 10.0


# ---------------------------------------------------------------------------
# bootstrap_coef_ci
# ---------------------------------------------------------------------------

class TestBootstrapCoefCi:
    @pytest.fixture(scope="class")
    def ci_df(self, data):
        X_train_scaled, y_train, _, _ = data
        return bootstrap_coef_ci(
            X_train_scaled, y_train,
            n_components=3, feature_names=FEATURE_NAMES,
            n_bootstrap=50, ci=95, random_state=0,
        )

    def test_returns_dataframe(self, ci_df):
        assert isinstance(ci_df, pd.DataFrame)

    def test_has_correct_columns(self, ci_df):
        assert set(ci_df.columns) == {"feature", "mean_coef", "ci_lower", "ci_upper", "ci_width"}

    def test_length_equals_feature_count(self, ci_df):
        assert len(ci_df) == len(FEATURE_NAMES)

    def test_all_features_present(self, ci_df):
        assert set(ci_df["feature"]) == set(FEATURE_NAMES)

    def test_ci_lower_less_than_upper(self, ci_df):
        assert (ci_df["ci_lower"] <= ci_df["ci_upper"]).all()

    def test_ci_width_nonnegative(self, ci_df):
        assert (ci_df["ci_width"] >= 0).all()


# ---------------------------------------------------------------------------
# plsr_sensitivity
# ---------------------------------------------------------------------------

class TestPlsrSensitivity:
    def test_returns_correct_shape(self, data):
        X_train_scaled, y_train, _, _ = data
        model = fit_plsr(X_train_scaled, y_train, n_components=3)
        X_ref = X_train_scaled.mean(axis=0)
        result = plsr_sensitivity(model, X_ref, FEATURE_NAMES)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"feature", "delta_plus", "delta_minus"}
        assert len(result) == len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# fit_ridge / compare_stability
# ---------------------------------------------------------------------------

class TestFitRidge:
    def test_returns_fitted_model(self, data):
        from sklearn.linear_model import Ridge
        X_train_scaled, y_train, _, _ = data
        model = fit_ridge(X_train_scaled, y_train)
        assert isinstance(model, Ridge)
        assert hasattr(model, "coef_")


class TestCompareStability:
    def test_returns_correct_shape(self, data):
        X_train_scaled, y_train, _, _ = data
        plsr_ci = bootstrap_coef_ci(
            X_train_scaled, y_train, n_components=3,
            feature_names=FEATURE_NAMES, n_bootstrap=30, random_state=0,
        )
        ridge_ci = bootstrap_coef_ci(
            X_train_scaled, y_train, n_components=3,
            feature_names=FEATURE_NAMES, n_bootstrap=30, random_state=0,
        )
        result = compare_stability(plsr_ci, ridge_ci)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"feature", "plsr_ci_width", "ridge_ci_width"}
        assert len(result) == len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

class TestPlots:
    @pytest.fixture(scope="class")
    def plot_inputs(self, data):
        X_train_scaled, y_train, _, _ = data
        model = fit_plsr(X_train_scaled, y_train, n_components=3)
        vip = compute_vip(model, FEATURE_NAMES)
        bvip = bootstrap_vip(X_train_scaled, y_train, n_components=3,
                             feature_names=FEATURE_NAMES, n_bootstrap=30, random_state=0)
        sens = plsr_sensitivity(model, X_train_scaled.mean(axis=0), FEATURE_NAMES)
        return vip, bvip, sens

    def test_plot_vip_saves_file(self, plot_inputs, tmp_path):
        vip, _, _ = plot_inputs
        p = tmp_path / "vip.png"
        plot_vip(vip, str(p))
        assert p.stat().st_size > 0

    def test_plot_bootstrap_stability_saves_file(self, plot_inputs, tmp_path):
        _, bvip, _ = plot_inputs
        p = tmp_path / "stab.png"
        plot_bootstrap_stability(bvip, str(p))
        assert p.stat().st_size > 0

    def test_plot_plsr_sensitivity_saves_file(self, plot_inputs, tmp_path):
        _, _, sens = plot_inputs
        p = tmp_path / "sens.png"
        plot_plsr_sensitivity(sens, str(p))
        assert p.stat().st_size > 0
