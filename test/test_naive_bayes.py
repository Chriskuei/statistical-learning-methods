import numpy as np
import pytest

import methods


@pytest.fixture(scope='module')
def data():
    X = np.array(
        [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]
    )
    y = np.array([1, 1, 1, 2, 2, 2])
    return X, y

@pytest.fixture(scope='module')
def model():
    nb = methods.NaiveBayes()
    return nb

def test_naive_bayes(data, model):
    X, y = data
    model.fit(X, y)
    prediction = model.predict(
        np.asarray([[-1, -1], [1, 1]])
    )
    print(prediction)
