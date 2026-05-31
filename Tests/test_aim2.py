import pathlib
"""
TDD tests for aim2_analysis.py.
Written before the implementation — tests define the expected API.

Run with: python -m pytest test_aim2.py -v
"""

import os
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from aim1_classification import load_data, split_data, scale_features
from aim2_analysis import (
    cross_validate_model,
    tune_logistic_regression,
    tune_decision_tree,
    get_feature_importance,
    plot_feature_importance,
    run_sensitivity_analysis,
    plot_sensitivity,
    plot_regularization_path,
)

DATA_PATH = pathlib.Path(__file__).parent.parent / "Data" / "ckd_cleaned.csv"
CV_METRIC_KEYS = {
    "Model",
    "mean_accuracy", "std_accuracy",
    "mean_train_accuracy", "std_train_accuracy",
    "mean_precision", "std_precision",
    "mean_recall", "std_recall",
    "mean_f1", "std_f1",
}
FEATURE_NAMES = [
    "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba",
    "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc",
    "htn", "dm", "cad", "appet", "pe", "ane",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_data():
    return load_data(DATA_PATH)


@pytest.fixture(scope="module")
def split(real_data):
    X, y = real_data
    return split_data(X, y)


@pytest.fixture(scope="module")
def scaled(split):
    X_train, X_test, _, _ = split
    return scale_features(X_train, X_test)


@pytest.fixture(scope="module")
def trained_lr(split, scaled):
    X_train_scaled, _, _ = scaled
    _, _, y_train, _ = split
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    return model


@pytest.fixture(scope="module")
def trained_dt(split):
    X_train, _, y_train, _ = split
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


@pytest.fixture(scope="module")
def full_X(real_data):
    return real_data[0]


@pytest.fixture(scope="module")
def full_y(real_data):
    return real_data[1]


# ---------------------------------------------------------------------------
# cross_validate_model
# ---------------------------------------------------------------------------

class TestCrossValidateModel:
    def test_returns_dict(self, trained_lr, full_X, full_y, scaled):
        X_train_scaled, _, scaler = scaled
        X_scaled = scaler.transform(full_X)
        result = cross_validate_model("LR", trained_lr, X_scaled, full_y)
        assert isinstance(result, dict)

    def test_has_expected_keys(self, trained_lr, full_X, full_y, scaled):
        X_train_scaled, _, scaler = scaled
        X_scaled = scaler.transform(full_X)
        result = cross_validate_model("LR", trained_lr, X_scaled, full_y)
        assert CV_METRIC_KEYS == set(result.keys())

    def test_model_name_stored(self, trained_lr, full_X, full_y, scaled):
        X_train_scaled, _, scaler = scaled
        X_scaled = scaler.transform(full_X)
        result = cross_validate_model("LR", trained_lr, X_scaled, full_y)
        assert result["Model"] == "LR"

    def test_mean_values_between_0_and_1(self, trained_lr, full_X, full_y, scaled):
        X_train_scaled, _, scaler = scaled
        X_scaled = scaler.transform(full_X)
        result = cross_validate_model("LR", trained_lr, X_scaled, full_y)
        for key in ("mean_accuracy", "mean_precision", "mean_recall", "mean_f1"):
            assert 0.0 <= result[key] <= 1.0, f"{key} out of range"

    def test_std_values_nonnegative(self, trained_lr, full_X, full_y, scaled):
        X_train_scaled, _, scaler = scaled
        X_scaled = scaler.transform(full_X)
        result = cross_validate_model("LR", trained_lr, X_scaled, full_y)
        for key in ("std_accuracy", "std_precision", "std_recall", "std_f1"):
            assert result[key] >= 0.0, f"{key} is negative"

    def test_lr_cv_accuracy_above_threshold(self, trained_lr, full_X, full_y, scaled):
        X_train_scaled, _, scaler = scaled
        X_scaled = scaler.transform(full_X)
        result = cross_validate_model("LR", trained_lr, X_scaled, full_y, cv=5)
        assert result["mean_accuracy"] >= 0.90

    def test_dt_cv_accuracy_above_threshold(self, trained_dt, full_X, full_y):
        result = cross_validate_model("DT", trained_dt, full_X, full_y, cv=5)
        assert result["mean_accuracy"] >= 0.90

    def test_custom_cv_folds(self, trained_dt, full_X, full_y):
        result_3 = cross_validate_model("DT", trained_dt, full_X, full_y, cv=3)
        result_5 = cross_validate_model("DT", trained_dt, full_X, full_y, cv=5)
        assert isinstance(result_3, dict)
        assert isinstance(result_5, dict)

    def test_train_accuracy_in_range(self, trained_dt, full_X, full_y):
        result = cross_validate_model("DT", trained_dt, full_X, full_y)
        assert 0.0 <= result["mean_train_accuracy"] <= 1.0

    def test_train_std_nonnegative(self, trained_dt, full_X, full_y):
        result = cross_validate_model("DT", trained_dt, full_X, full_y)
        assert result["std_train_accuracy"] >= 0.0

    def test_train_accuracy_gte_cv_accuracy(self, trained_dt, full_X, full_y):
        """Training accuracy should be >= CV accuracy — fits on full fold, tests on held-out."""
        result = cross_validate_model("DT", trained_dt, full_X, full_y)
        assert result["mean_train_accuracy"] >= result["mean_accuracy"]


# ---------------------------------------------------------------------------
# tune_logistic_regression
# ---------------------------------------------------------------------------

class TestTuneLogisticRegression:
    def test_returns_tuple(self, split, scaled):
        X_train_scaled, _, _ = scaled
        _, _, y_train, _ = split
        result = tune_logistic_regression(X_train_scaled, y_train)
        assert len(result) == 2

    def test_returns_lr_model(self, split, scaled):
        X_train_scaled, _, _ = scaled
        _, _, y_train, _ = split
        model, _ = tune_logistic_regression(X_train_scaled, y_train)
        assert isinstance(model, LogisticRegression)

    def test_model_is_fitted(self, split, scaled):
        X_train_scaled, _, _ = scaled
        _, _, y_train, _ = split
        model, _ = tune_logistic_regression(X_train_scaled, y_train)
        assert hasattr(model, "coef_")

    def test_best_params_is_dict(self, split, scaled):
        X_train_scaled, _, _ = scaled
        _, _, y_train, _ = split
        _, best_params = tune_logistic_regression(X_train_scaled, y_train)
        assert isinstance(best_params, dict)

    def test_best_params_contains_C(self, split, scaled):
        X_train_scaled, _, _ = scaled
        _, _, y_train, _ = split
        _, best_params = tune_logistic_regression(X_train_scaled, y_train)
        assert "C" in best_params

    def test_tuned_lr_accuracy_above_threshold(self, split, scaled):
        X_train_scaled, X_test_scaled, _ = scaled
        _, _, y_train, y_test = split
        model, _ = tune_logistic_regression(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        from sklearn.metrics import accuracy_score
        assert accuracy_score(y_test, preds) >= 0.90


# ---------------------------------------------------------------------------
# tune_decision_tree
# ---------------------------------------------------------------------------

class TestTuneDecisionTree:
    def test_returns_tuple(self, split):
        X_train, _, y_train, _ = split
        result = tune_decision_tree(X_train, y_train)
        assert len(result) == 2

    def test_returns_dt_model(self, split):
        X_train, _, y_train, _ = split
        model, _ = tune_decision_tree(X_train, y_train)
        assert isinstance(model, DecisionTreeClassifier)

    def test_model_is_fitted(self, split):
        X_train, _, y_train, _ = split
        model, _ = tune_decision_tree(X_train, y_train)
        assert hasattr(model, "feature_importances_")

    def test_best_params_is_dict(self, split):
        X_train, _, y_train, _ = split
        _, best_params = tune_decision_tree(X_train, y_train)
        assert isinstance(best_params, dict)

    def test_best_params_contains_max_depth(self, split):
        X_train, _, y_train, _ = split
        _, best_params = tune_decision_tree(X_train, y_train)
        assert "max_depth" in best_params

    def test_tuned_dt_accuracy_above_threshold(self, split):
        X_train, X_test, y_train, y_test = split
        model, _ = tune_decision_tree(X_train, y_train)
        preds = model.predict(X_test)
        from sklearn.metrics import accuracy_score
        assert accuracy_score(y_test, preds) >= 0.90


# ---------------------------------------------------------------------------
# get_feature_importance
# ---------------------------------------------------------------------------

class TestGetFeatureImportance:
    def test_returns_dataframe(self, trained_lr):
        result = get_feature_importance(trained_lr, FEATURE_NAMES)
        assert isinstance(result, pd.DataFrame)

    def test_has_correct_columns(self, trained_lr):
        result = get_feature_importance(trained_lr, FEATURE_NAMES)
        assert set(result.columns) == {"feature", "importance"}

    def test_length_equals_feature_count(self, trained_lr):
        result = get_feature_importance(trained_lr, FEATURE_NAMES)
        assert len(result) == len(FEATURE_NAMES)

    def test_all_importances_nonnegative(self, trained_lr):
        result = get_feature_importance(trained_lr, FEATURE_NAMES)
        assert (result["importance"] >= 0).all()

    def test_sorted_descending(self, trained_lr):
        result = get_feature_importance(trained_lr, FEATURE_NAMES)
        assert result["importance"].is_monotonic_decreasing

    def test_feature_names_preserved(self, trained_lr):
        result = get_feature_importance(trained_lr, FEATURE_NAMES)
        assert set(result["feature"]) == set(FEATURE_NAMES)

    def test_dt_importances_sum_to_one(self, trained_dt):
        result = get_feature_importance(trained_dt, FEATURE_NAMES)
        assert result["importance"].sum() == pytest.approx(1.0, abs=1e-6)

    def test_works_for_both_model_types(self, trained_lr, trained_dt):
        lr_imp = get_feature_importance(trained_lr, FEATURE_NAMES)
        dt_imp = get_feature_importance(trained_dt, FEATURE_NAMES)
        assert len(lr_imp) == len(dt_imp) == len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# plot_feature_importance
# ---------------------------------------------------------------------------

class TestPlotFeatureImportance:
    def test_output_file_created(self, trained_dt, tmp_path):
        imp = get_feature_importance(trained_dt, FEATURE_NAMES)
        out = str(tmp_path / "importance.png")
        plot_feature_importance(imp, title="DT Importance", save_path=out)
        assert os.path.isfile(out)

    def test_output_file_nonzero_size(self, trained_dt, tmp_path):
        imp = get_feature_importance(trained_dt, FEATURE_NAMES)
        out = str(tmp_path / "importance.png")
        plot_feature_importance(imp, title="DT Importance", save_path=out)
        assert os.path.getsize(out) > 0


# ---------------------------------------------------------------------------
# run_sensitivity_analysis
# ---------------------------------------------------------------------------

class TestRunSensitivityAnalysis:
    @pytest.fixture(scope="class")
    def reference_point(self, split):
        _, X_test, _, _ = split
        return X_test.median()

    def test_returns_dataframe(self, trained_dt, split):
        _, X_test, _, _ = split
        ref = X_test.median()
        result = run_sensitivity_analysis(trained_dt, ref, FEATURE_NAMES)
        assert isinstance(result, pd.DataFrame)

    def test_has_correct_columns(self, trained_dt, split):
        _, X_test, _, _ = split
        ref = X_test.median()
        result = run_sensitivity_analysis(trained_dt, ref, FEATURE_NAMES)
        assert set(result.columns) == {"feature", "delta_plus", "delta_minus"}

    def test_one_row_per_feature(self, trained_dt, split):
        _, X_test, _, _ = split
        ref = X_test.median()
        result = run_sensitivity_analysis(trained_dt, ref, FEATURE_NAMES)
        assert len(result) == len(FEATURE_NAMES)

    def test_all_feature_names_present(self, trained_dt, split):
        _, X_test, _, _ = split
        ref = X_test.median()
        result = run_sensitivity_analysis(trained_dt, ref, FEATURE_NAMES)
        assert set(result["feature"]) == set(FEATURE_NAMES)

    def test_deltas_are_floats(self, trained_dt, split):
        _, X_test, _, _ = split
        ref = X_test.median()
        result = run_sensitivity_analysis(trained_dt, ref, FEATURE_NAMES)
        assert result["delta_plus"].dtype in (np.float32, np.float64, float)
        assert result["delta_minus"].dtype in (np.float32, np.float64, float)

    def test_deltas_bounded(self, trained_dt, split):
        """Probability deltas must be in [-1, 1]."""
        _, X_test, _, _ = split
        ref = X_test.median()
        result = run_sensitivity_analysis(trained_dt, ref, FEATURE_NAMES)
        assert result["delta_plus"].between(-1, 1).all()
        assert result["delta_minus"].between(-1, 1).all()

    def test_works_with_scaler_for_lr(self, trained_lr, split, scaled):
        _, X_test, _, _ = split
        _, _, scaler = scaled
        ref = X_test.median()
        result = run_sensitivity_analysis(trained_lr, ref, FEATURE_NAMES, scaler=scaler)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# plot_sensitivity
# ---------------------------------------------------------------------------

class TestPlotSensitivity:
    def test_output_file_created(self, trained_dt, split, tmp_path):
        _, X_test, _, _ = split
        ref = X_test.median()
        sens = run_sensitivity_analysis(trained_dt, ref, FEATURE_NAMES)
        out = str(tmp_path / "sensitivity.png")
        plot_sensitivity(sens, title="DT Sensitivity", save_path=out)
        assert os.path.isfile(out)

    def test_output_file_nonzero_size(self, trained_dt, split, tmp_path):
        _, X_test, _, _ = split
        ref = X_test.median()
        sens = run_sensitivity_analysis(trained_dt, ref, FEATURE_NAMES)
        out = str(tmp_path / "sensitivity.png")
        plot_sensitivity(sens, title="DT Sensitivity", save_path=out)
        assert os.path.getsize(out) > 0


# ---------------------------------------------------------------------------
# plot_regularization_path
# ---------------------------------------------------------------------------

class TestPlotRegularizationPath:
    def test_output_file_created(self, split, scaled, tmp_path):
        X_train_scaled, X_test_scaled, _ = scaled
        _, _, y_train, y_test = split
        out = str(tmp_path / "reg_path.png")
        plot_regularization_path(X_train_scaled, y_train, X_test_scaled, y_test, save_path=out)
        assert os.path.isfile(out)

    def test_output_file_nonzero_size(self, split, scaled, tmp_path):
        X_train_scaled, X_test_scaled, _ = scaled
        _, _, y_train, y_test = split
        out = str(tmp_path / "reg_path.png")
        plot_regularization_path(X_train_scaled, y_train, X_test_scaled, y_test, save_path=out)
        assert os.path.getsize(out) > 0

    def test_accepts_custom_C_values(self, split, scaled, tmp_path):
        X_train_scaled, X_test_scaled, _ = scaled
        _, _, y_train, y_test = split
        out = str(tmp_path / "reg_path_custom.png")
        plot_regularization_path(
            X_train_scaled, y_train, X_test_scaled, y_test,
            C_values=[0.001, 0.1, 10], save_path=out
        )
        assert os.path.isfile(out)
