import numpy as np
import pytest

import methods


@pytest.fixture(scope='module')
def data():
    X = np.asarray([[3, 3], [4, 3], [1, 1]])
    y = np.asarray([1, 1, -1])
    return X, y

@pytest.fixture(scope='module', params=[0.5, 0.8, 1])
def learning_rate(request):
    return request.param

@pytest.fixture(scope='module', params=[10, 20, 50, 100])
def epoch(request):
    return request.param

@pytest.fixture(scope='module')
def model(learning_rate, epoch):
    perceptron = methods.Perceptron(
        learning_rate=learning_rate,
        epoch=epoch
    )
    return perceptron

def test_perceptron(data, model):
    X, y = data
    model.fit(X, y)
    prediction = model.predict(
        np.asarray([[0, 0]])
    )
    print(prediction)
