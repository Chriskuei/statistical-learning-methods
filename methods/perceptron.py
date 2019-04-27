"""Implement of perceptron."""

import numpy as np


class Perceptron(object):
    r"""Perceptron is a linear model which is for binary classification.
    Perceptron can be defined by the following function:

        f(x) = sign(W*x + b)

    W is called weight or weight vector, and b is callled bias
    """
    def __init__(self, learning_rate=1, epoch=10):
        super(Perceptron, self).__init__()
        self._W = []
        self._b = 0.
        self._learning_rate = learning_rate
        self._epoch = epoch

    def _forward(self, X):
        r"""Defines the computation performed at every call.

        Args:
            X (ndarray): array-like, shape (n_samples, n_features).
                Training data.

        Returns:
            output (ndarray): array-like, shape (n_sample).
                Output of perceptron.
        """
        # Shape: (batch_size)
        output = np.zeros(shape=(X.shape[0]), dtype=int)

        # Calculate W * x_i + b
        for i, x in enumerate(X):
            out = np.sign(np.dot(self._W, x) + self._b)
            output[i] = out

        assert output.shape[0] == X.shape[0]
        return output

    def _loss(self, output, y):
        r"""Calculate the loss and find out misclassified samples

        Args:
            output (ndarray): array-like, shape (n_sample).
                Output of perceptron.

            y (ndarray): array-like, shape (n_samples,).
                Target values.

        Returns:
            mis_sample_indexes (ndarray): indexs of misclassified samples.
        """
        if output.shape[0] != y.shape[0]:
            raise ValueError('arrays must have same number of dimensions')
        mis_sample_indexes = []

        # Find out misclassified samples
        for index, (out, y_i) in enumerate(zip(output, y)):
            # Judge y_i * (W * x_i + b) <= 0
            if out * y_i <= 0:
                mis_sample_indexes.append(index)

        mis_sample_indexes = np.asarray(mis_sample_indexes)

        # Save current loss
        # self._loss = mis_sample_indexes.shape[0]
        return mis_sample_indexes

    def _backword(self, x, y):
        r"""Update W and b by using misclassified sample

        Args:
            x (ndarray): array-like, shape (n_features,).
                misclassified sample.
            y (float): array-like, label of misclassified value
        """
        # W <- w + `\eta` * y_i * x_i
        # b <- w + `\eta` * y_i
        self._W = self._W + self._learning_rate * y * x
        self._b = self._b + self._learning_rate * y

    def fit(self, X, y):
        r"""Fit the perceptron

        Args:
            X (ndarray): array-like, shape (n_samples, n_features).
                Training data.
            y (ndarray): array-like, shape (n_samples,).
                Target values.
        """
        # 1. Initialize W and b
        self._W = np.random.randn(X.shape[1])
        # 2. Train percetron
        for _ in range(self._epoch):
            output = self._forward(X)
            mis_sample_indexes = self._loss(output, y)

            if mis_sample_indexes.shape[0] == 0:
                break

            index = np.random.choice(mis_sample_indexes)
            self._backword(X[index], y[index])

    def predict(self, X):
        r"""Predict the target of new samples.

        Args:
            X (ndarray): array-like, shape (n_samples, n_features).
                The samples.

        Returns:
            ndarray: The predicted class.
        """
        return self._forward(X)

    @property
    def parameters(self):
        r"""Returns perceptron's parameters."""
        return self._W, self._b
