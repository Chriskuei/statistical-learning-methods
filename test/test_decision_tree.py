import numpy as np
import pytest

import methods


@pytest.fixture(scope='module')
def data():
    X = np.asarray([
        [0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1],
        [0, 1, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0],
        [1, 0, 0, 1], [1, 1, 1, 1], [1, 0, 1, 2],
        [1, 0, 1, 2], [2, 0, 1, 2], [2, 0, 1, 1],
        [2, 1, 0, 1], [2, 1, 0, 2], [2, 0, 0, 0]
    ])
    y = np.asarray(
        [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
    )
    return X, y

@pytest.fixture(scope='module', params=[1e-3, 1])
def epsilon(request):
    return request.param

@pytest.fixture(scope='module')
def model(epsilon):
    DT = methods.DecisionTree(
        epsilon=epsilon
    )
    return DT

def test_knn(data, model):
    X, y = data
    model.fit(X, y)

    X_test = np.asarray(
        [[2, 0, 1, 2], [0, 0, 0, 0], [0, 1, 0, 1]]
    )
    prediction = model.predict(X_test)
