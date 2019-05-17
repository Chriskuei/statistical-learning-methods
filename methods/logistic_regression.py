"""Implement of logistic regression."""

import math
import numpy as np


class LogisticRegression(object):
    r"""Logistic Regression is a model which is for binary
    classification.
    """
    def __init__(self, learning_rate=0.1, epoch=100):
        super(LogisticRegression, self).__init__()
        self._W = []
        self._b = 0.
        self._learning_rate = learning_rate
        self._epoch = epoch

    def sigmoid(self, x):
        return 1. / (1. + math.exp(-x))

    def _forward(self, X):
        r"""Defines the computation performed at every call.

        Args:
            X (ndarray): array-like, shape (n_samples, n_features).
                Training data.

        Returns:
            output (ndarray): array-like, shape (n_sample).
                Output of LogisticRegression.
        """
        # Shape: (batch_size)
        output = np.zeros(shape=(X.shape[0]), dtype=float)

        # Calculate sigmoid(W * x_i + b)
        for i, x in enumerate(X):
            out = self.sigmoid(np.dot(self._W, x) + self._b)
            output[i] = out

        assert output.shape[0] == X.shape[0]
        return output

    def _loss(self, output, y):
        r"""Calculate the loss
        Args:
            output (ndarray): array-like, shape (n_sample).
                Output of LogisticRegression.

            y (ndarray): array-like, shape (n_samples,).
                Target values.

        Returns:
            mis_sample_indexes (ndarray): indexs of misclassified samples.
        """
        if output.shape[0] != y.shape[0]:
            raise ValueError('arrays must have same number of dimensions')

        return y - output

    def _backword(self, X, loss):
        r"""Update W and b by using misclassified sample

        Args:
            x (ndarray): array-like, shape (n_features,).
                misclassified sample.
            y (float): array-like, label of misclassified value
        """
        # W <- w + `\eta` * loss * x_i
        # b <- w + `\eta` * loss
        for x, e in zip(X, loss):
            self._W += self._learning_rate * e * x
            self._b += self._learning_rate * e

    def fit(self, X, y):
        r"""Fit the LogisticRegression

        Args:
            X (ndarray): array-like, shape (n_samples, n_features).
                Training data.
            y (ndarray): array-like, shape (n_samples,).
                Target values.
        """
        # 1. Initialize W
        self._W = np.zeros(shape=(X.shape[1]), dtype=float)
        # 2. Train LogisticRegression
        for _ in range(self._epoch):
            output = self._forward(X)
            loss = self._loss(output, y)
            self._backword(X, loss)

    def predict(self, X):
        r"""Predict the target of new samples.

        Args:
            X (ndarray): array-like, shape (n_samples, n_features).
                The samples.

        Returns:
            ndarray: The predicted class.
        """
        out = self._forward(X)
        out = np.array([1 if o >= 0.5 else 0 for o in out])
        return out

    @property
    def parameters(self):
        r"""Returns LogisticRegression's parameters."""
        return self._W, self._b
