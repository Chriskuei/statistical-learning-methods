import numpy as np
import pytest

import methods


@pytest.fixture(scope='module')
def data():
    X = np.asarray([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    y = np.asarray([-1, +1, +1, +1, -1, -1])
    return X, y

@pytest.fixture(scope='module', params=[1, 2, 3])
def n_neighbors(request):
    return request.param

@pytest.fixture(scope='module', params=[1, 2])
def p(request):
    return request.param

@pytest.fixture(scope='module', params=['minkowski'])
def metric(request):
    return request.param

@pytest.fixture(scope='module')
def model(n_neighbors, p, metric):
    kNN = methods.KNearestNeighbor(
        n_neighbors=n_neighbors,
        p=p,
        metric=metric
    )
    return kNN

def test_knn(data, model):
    X, y = data
    model.fix(X, y)
    prediction = kNN.predict(
        np.asarray([[3, 4.5], [8, 0]])
    )
    print(prediction)
