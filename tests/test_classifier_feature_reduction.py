import numpy as np
import pytest
import torch

from tabicl import TabICLClassifier
from tabicl._sklearn.preprocessing import FeatureReducer


class _FakeModel:
    max_features = 4
    max_classes = 3

    def to(self, device):
        return self

    def __call__(self, X, y_train, **kwargs):
        n_test = X.shape[1] - y_train.shape[1]
        return torch.zeros((X.shape[0], n_test, 2), device=X.device)


def _fake_load_model(self):
    self.model_ = _FakeModel()
    self.model_config_ = {"max_features": 4}
    self.model_path_ = "fake"


def test_feature_reducer_pca_roundtrip():
    X = np.random.RandomState(0).randn(20, 5)
    reducer = FeatureReducer("pca", 3, random_state=0).fit(X)
    assert reducer.transform(X[:4]).shape == (4, 3)


def test_classifier_automatically_ensembles_wide_features(monkeypatch):
    monkeypatch.setattr(TabICLClassifier, "_load_model", _fake_load_model)
    X = np.random.RandomState(1).randn(20, 10)
    y = np.array([0, 1] * 10)

    classifier = TabICLClassifier(n_estimators=2, device="cpu").fit(X[:15], y[:15])

    assert classifier.feature_reduction_ == "ensemble"
    assert [len(subset) for subset in classifier.feature_subsets_] == [4, 4, 2]
    assert classifier.predict_proba(X[15:]).shape == (5, 2)


def test_classifier_explicitly_ensembles_wide_features(monkeypatch):
    monkeypatch.setattr(TabICLClassifier, "_load_model", _fake_load_model)
    X = np.random.RandomState(4).randn(20, 10)
    y = np.array([0, 1] * 10)

    classifier = TabICLClassifier(
        n_estimators=2, feature_reduction="ensemble", device="cpu"
    ).fit(X[:15], y[:15])

    assert [len(subset) for subset in classifier.feature_subsets_] == [4, 4, 2]
    assert classifier.predict_proba(X[15:]).shape == (5, 2)


def test_classifier_pca_reduces_features(monkeypatch):
    monkeypatch.setattr(TabICLClassifier, "_load_model", _fake_load_model)
    X = np.random.RandomState(2).randn(20, 10)
    y = np.array([0, 1] * 10)

    classifier = TabICLClassifier(
        n_estimators=2, feature_reduction="pca", n_components=3, device="cpu"
    ).fit(X[:15], y[:15])

    assert classifier.feature_reduction_ == "pca"
    assert classifier.n_features_model_ == 3
    assert classifier.predict_proba(X[15:]).shape == (5, 2)


def test_reducer_requires_components_for_explicit_reduction(monkeypatch):
    monkeypatch.setattr(TabICLClassifier, "_load_model", _fake_load_model)
    X = np.random.RandomState(3).randn(10, 5)
    y = np.array([0, 1] * 5)

    with pytest.raises(ValueError, match="n_components"):
        TabICLClassifier(feature_reduction="pca", device="cpu").fit(X, y)
