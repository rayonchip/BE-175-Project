"""
TDD tests for aim3_plsr.py.
Run with: python -m pytest test_aim3.py -v
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.cross_decomposition import PLSRegression

from aim1_classification import load_data, split_data, scale_features
from aim3_plsr import fit_plsr, select_n_components, compute_vip

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
