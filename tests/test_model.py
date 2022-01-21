from starter.ml.model import train_model, compute_model_metrics, inference
from sklearn.ensemble import RandomForestClassifier
import pytest
import numpy as np


@pytest.fixture(scope="session")
def data():
    X = np.array([[1.0, 2.0, 1.0], [3.0, 4.0, 3.0]])
    y = np.array([[1], [0]])
    return (X, y)


def test_model_type(data):
    clf = train_model(data[0], data[1])
    assert type(clf) == RandomForestClassifier


def test_model_metrics():
    y_true = [0, 1, 0, 0, 1, 1]
    y_pred = [0, 1, 0, 1, 1, 1]
    pr, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert pr == 0.75
    assert recall == 1
    assert np.round(fbeta, 4) == 0.8571


def test_inference(data):
    clf = train_model(data[0], data[1])
    test_X = np.array([[1.0, 1.0, 1.0]])
    assert inference(clf, test_X) == np.array([1])
    assert len(inference(clf, test_X)) == 1
