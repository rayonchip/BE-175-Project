"""
TDD tests for aim1_classification.py.
Written before refactoring the source — tests define the expected API.

Run with: python -m pytest test_aim1.py -v
"""

import os
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from aim1_classification import (
    load_data,
    split_data,
    scale_features,
    train_logistic_regression,
    train_decision_tree,
    evaluate_model,
    plot_confusion_matrices,
)

DATA_PATH = "Training Data/ckd_cleaned.csv"
EXPECTED_FEATURES = [
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


@pytest.fixture
def tiny_data():
    """Synthetic 20-row binary dataset — no file I/O needed."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((20, 5)), columns=list("abcde"))
    y = pd.Series([0] * 10 + [1] * 10)
    return X, y


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------

class TestLoadData:
    def test_returns_X_and_y(self, real_data):
        X, y = real_data
        assert X is not None and y is not None

    def test_row_count(self, real_data):
        X, y = real_data
        assert len(X) == 400

    def test_feature_count(self, real_data):
        X, _ = real_data
        assert X.shape[1] == 24

    def test_expected_columns_present(self, real_data):
        X, _ = real_data
        assert set(EXPECTED_FEATURES).issubset(set(X.columns))

    def test_class_not_in_X(self, real_data):
        X, _ = real_data
        assert "class" not in X.columns

    def test_class_binary(self, real_data):
        _, y = real_data
        assert set(y.unique()).issubset({0, 1})

    def test_no_missing_values(self, real_data):
        X, y = real_data
        assert X.isnull().sum().sum() == 0
        assert y.isnull().sum() == 0


# ---------------------------------------------------------------------------
# split_data
# ---------------------------------------------------------------------------

class TestSplitData:
    def test_returns_four_parts(self, split):
        assert len(split) == 4

    def test_test_size_is_20_percent(self, split):
        X_train, X_test, _, _ = split
        total = len(X_train) + len(X_test)
        assert len(X_test) == pytest.approx(total * 0.2, abs=1)

    def test_train_test_no_overlap(self, split):
        X_train, X_test, _, _ = split
        assert len(set(X_train.index) & set(X_test.index)) == 0

    def test_stratification_preserves_class_ratio(self, split):
        _, _, y_train, y_test = split
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        assert abs(train_ratio - test_ratio) < 0.05


# ---------------------------------------------------------------------------
# scale_features
# ---------------------------------------------------------------------------

class TestScaleFeatures:
    def test_returns_three_values(self, scaled):
        assert len(scaled) == 3

    def test_train_mean_near_zero(self, scaled):
        X_train_scaled, _, _ = scaled
        assert np.abs(X_train_scaled.mean(axis=0)).mean() < 1e-10

    def test_train_std_near_one(self, scaled):
        X_train_scaled, _, _ = scaled
        assert np.abs(X_train_scaled.std(axis=0) - 1).mean() < 1e-10

    def test_output_shapes_match_input(self, split, scaled):
        X_train, X_test, _, _ = split
        X_train_scaled, X_test_scaled, _ = scaled
        assert X_train_scaled.shape == X_train.shape
        assert X_test_scaled.shape == X_test.shape

    def test_scaler_not_refit_on_test(self, split, scaled):
        """Test set mean should differ from 0 (scaler fitted on train only)."""
        _, X_test, _, _ = split
        _, X_test_scaled, _ = scaled
        # If scaler were re-fit on test, its mean would be ~0; it should not be
        # (unless train/test happen to share the exact same distribution)
        test_mean = np.abs(X_test_scaled.mean(axis=0)).mean()
        assert test_mean > 1e-6


# ---------------------------------------------------------------------------
# train_logistic_regression
# ---------------------------------------------------------------------------

class TestTrainLogisticRegression:
    def test_returns_lr_model(self, scaled, split):
        X_train_scaled, _, _ = scaled
        _, _, y_train, _ = split
        model = train_logistic_regression(X_train_scaled, y_train)
        assert isinstance(model, LogisticRegression)

    def test_model_is_fitted(self, scaled, split):
        X_train_scaled, _, _ = scaled
        _, _, y_train, _ = split
        model = train_logistic_regression(X_train_scaled, y_train)
        assert hasattr(model, "coef_")

    def test_predict_returns_binary(self, scaled, split):
        X_train_scaled, X_test_scaled, _ = scaled
        _, _, y_train, _ = split
        model = train_logistic_regression(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        assert set(preds).issubset({0, 1})


# ---------------------------------------------------------------------------
# train_decision_tree
# ---------------------------------------------------------------------------

class TestTrainDecisionTree:
    def test_returns_dt_model(self, split):
        X_train, _, y_train, _ = split
        model = train_decision_tree(X_train, y_train)
        assert isinstance(model, DecisionTreeClassifier)

    def test_model_is_fitted(self, split):
        X_train, _, y_train, _ = split
        model = train_decision_tree(X_train, y_train)
        assert hasattr(model, "feature_importances_")

    def test_feature_importances_sum_to_one(self, split):
        X_train, _, y_train, _ = split
        model = train_decision_tree(X_train, y_train)
        assert model.feature_importances_.sum() == pytest.approx(1.0, abs=1e-6)

    def test_predict_returns_binary(self, split):
        X_train, X_test, y_train, _ = split
        model = train_decision_tree(X_train, y_train)
        preds = model.predict(X_test)
        assert set(preds).issubset({0, 1})


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------

class TestEvaluateModel:
    EXPECTED_KEYS = {"Model", "Accuracy", "Precision", "Recall", "F1 Score"}

    def _fitted_lr(self, tiny_data):
        X, y = tiny_data
        X_train_scaled, X_test_scaled, _ = scale_features(X, X)
        model = train_logistic_regression(X_train_scaled, y)
        return model, X_test_scaled, y

    def test_returns_dict(self, tiny_data):
        model, X_eval, y = self._fitted_lr(tiny_data)
        result = evaluate_model("LR", model, X_eval, y)
        assert isinstance(result, dict)

    def test_has_expected_keys(self, tiny_data):
        model, X_eval, y = self._fitted_lr(tiny_data)
        result = evaluate_model("LR", model, X_eval, y)
        assert self.EXPECTED_KEYS == set(result.keys())

    def test_model_name_stored(self, tiny_data):
        model, X_eval, y = self._fitted_lr(tiny_data)
        result = evaluate_model("LR", model, X_eval, y)
        assert result["Model"] == "LR"

    def test_metrics_between_0_and_1(self, tiny_data):
        model, X_eval, y = self._fitted_lr(tiny_data)
        result = evaluate_model("LR", model, X_eval, y)
        for key in ("Accuracy", "Precision", "Recall", "F1 Score"):
            assert 0.0 <= result[key] <= 1.0, f"{key} out of range"

    def test_perfect_predictions_give_all_ones(self):
        """A model that predicts perfectly should score 1.0 on all metrics."""
        X = pd.DataFrame({"a": [0.0, 1.0, 0.0, 1.0]})
        y = pd.Series([0, 1, 0, 1])
        model = train_decision_tree(X, y)
        result = evaluate_model("DT", model, X, y)
        for key in ("Accuracy", "Precision", "Recall", "F1 Score"):
            assert result[key] == pytest.approx(1.0), f"{key} should be 1.0"


# ---------------------------------------------------------------------------
# Integration: real-data accuracy thresholds
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_lr_accuracy_above_threshold(self, scaled, split):
        X_train_scaled, X_test_scaled, _ = scaled
        _, _, y_train, y_test = split
        model = train_logistic_regression(X_train_scaled, y_train)
        result = evaluate_model("LR", model, X_test_scaled, y_test)
        assert result["Accuracy"] >= 0.90, f"LR accuracy {result['Accuracy']} below 0.90"

    def test_dt_accuracy_above_threshold(self, split):
        X_train, X_test, y_train, y_test = split
        model = train_decision_tree(X_train, y_train)
        result = evaluate_model("DT", model, X_test, y_test)
        assert result["Accuracy"] >= 0.90, f"DT accuracy {result['Accuracy']} below 0.90"

    def test_lr_recall_above_threshold(self, scaled, split):
        """Missing CKD (false negatives) is clinically more costly."""
        X_train_scaled, X_test_scaled, _ = scaled
        _, _, y_train, y_test = split
        model = train_logistic_regression(X_train_scaled, y_train)
        result = evaluate_model("LR", model, X_test_scaled, y_test)
        assert result["Recall"] >= 0.90, f"LR recall {result['Recall']} below 0.90"

    def test_dt_recall_above_threshold(self, split):
        X_train, X_test, y_train, y_test = split
        model = train_decision_tree(X_train, y_train)
        result = evaluate_model("DT", model, X_test, y_test)
        assert result["Recall"] >= 0.90, f"DT recall {result['Recall']} below 0.90"


# ---------------------------------------------------------------------------
# plot_confusion_matrices
# ---------------------------------------------------------------------------

class TestPlotConfusionMatrices:
    def test_output_file_created(self, split, scaled, tmp_path):
        X_train, X_test, y_train, y_test = split
        X_train_scaled, X_test_scaled, _ = scaled
        lr = train_logistic_regression(X_train_scaled, y_train)
        dt = train_decision_tree(X_train, y_train)
        models = {
            "Logistic Regression": (lr, X_test_scaled),
            "Decision Tree": (dt, X_test),
        }
        out = str(tmp_path / "cm.png")
        plot_confusion_matrices(models, y_test, save_path=out)
        assert os.path.isfile(out)

    def test_output_file_nonzero_size(self, split, scaled, tmp_path):
        X_train, X_test, y_train, y_test = split
        X_train_scaled, X_test_scaled, _ = scaled
        lr = train_logistic_regression(X_train_scaled, y_train)
        dt = train_decision_tree(X_train, y_train)
        models = {
            "Logistic Regression": (lr, X_test_scaled),
            "Decision Tree": (dt, X_test),
        }
        out = str(tmp_path / "cm.png")
        plot_confusion_matrices(models, y_test, save_path=out)
        assert os.path.getsize(out) > 0
