import numpy as np
import pytest

import methods


@pytest.fixture(scope='module')
def data():
    X = np.asarray([[3, 3], [4, 3], [1, 1]])
    y = np.asarray([1, 1, -1])
    return X, y

@pytest.fixture(scope='module', params=[1.0, 2.0])
def C(request):
    return request.param

@pytest.fixture(scope='module', params=[
    'linear', 'poly'
])
def kernel(request):
    return request.param

@pytest.fixture(scope='module', params=[10, 20, 50, 100])
def epoch(request):
    return request.param

@pytest.fixture(scope='module')
def model(C, kernel, epoch):
    svm = methods.SVM(
        C=C,
        kernel=kernel,
        epoch=epoch
    )
    return svm

def test_svm(data, model):
    X, y = data
    model.fit(X, y)
    prediction = model.predict(
        np.asarray([[0, 0]])
    )

def test_svm_with_kernel_error(data):
    with pytest.raises(ValueError):
        svm = methods.SVM(kernel='undefined')
