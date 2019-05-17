"""Implement of support vector machine."""

import numpy as np


class SVM(object):
    r"""SVM is a linear model which is for binary classification.

    Args:
        C (float): penalty parameter.
        kernel (str): the type of kernel to use.
        epoch (int): training epochs.
    """

    def __init__(self, C=1.0, kernel='linear', epoch=10):
        super(SVM, self).__init__()
        self._C = C
        self._epoch = epoch
        self._kernel_type = kernel
        if kernel not in ['linear', 'poly']:
            raise ValueError(
                "Parameter `kernel` should be `linear` or 'poly'"
            )

    def _init_parameters(self, X, y):
        """Init parameters."""
        self._X = X
        self._y = y

        self._b = 0.
        self.sample_num = X.shape[0]
        self.feature_num = X.shape[1]
        self._alpha = np.ones(self.sample_num)
        self.E = np.array([self._E(i) for i in range(self.sample_num)])

    def _KKT(self, i):
        """Check if sample x satifies KKT."""
        y_g = self._g(i) * self._y[i]
        # alpha = 0 <-> y_g >= 1
        if self._alpha[i] == 0:
            return y_g >= 1
        # 0 < alpha < C <-> y_g = 1
        elif 0 < self._alpha[i] < self._C:
            return y_g == 1
        # alpha = C <-> y_g <= 1
        else:
            return y_g <= 1

    def _g(self, i):
        """Compute g(x)."""
        # g(x)=\sum_{j}^{M}{\alpha_j * y_j * kernel(x_i * x_j)}
        out = self._b
        for j in range(self.sample_num):
            out += self._alpha[j] * self._y[j] * \
                self._kernel(self._X[i], self._X[j])
        return out

    def _kernel(self, x_1, x_2):
        """Compute kernel."""
        if self._kernel_type == 'linear':
            return sum(
                [x_1[k] * x_2[k] for k in range(self.feature_num)])
        elif self._kernel_type == 'poly':
            return (
                sum([x_1[k] * x_2[k] for k in range(self.feature_num)])
                + 1) ** 2

    def _E(self, i):
        """Compute E."""
        return self._g(i) - self._y[i]

    def _choose_alpha(self):
        """Choose alpha_1 and alpha_2."""
        # First check samples whose alpha between 0 and C
        borders = [i for i in range(self.sample_num)
                   if 0 < self._alpha[i] < self._C]
        # Then check others
        others = [i for i in range(self.sample_num)
                  if i not in borders]

        for i in borders + others:
            if self._KKT(i):
                continue

            # Max |E1 - E2|
            if self.E[i] >= 0:
                j = np.argmin(self.E)
            else:
                j = np.argmax(self.E)
            return i, j

    def _compute_L_H(self, i, j):
        """Compute L and H."""
        if self._y[i] == self._y[j]:
            L = max(0, self._alpha[i] + self._alpha[j] - self._C)
            H = min(self._C, self._alpha[i] + self._alpha[j])
        else:
            L = max(0, self._alpha[j] - self._alpha[i])
            H = min(self._C, self._C + self._alpha[j] - self._alpha[i])
        return L, H

    def _compute_eta(self, i, j):
        """Compute eta = K11 + K22 -2 * K12."""
        return self._kernel(self._X[i], self._X[i]) + \
            self._kernel(self._X[j], self._X[j]) - \
            2 * self._kernel(self._X[i], self._X[j])

    def _compute_alpha_2_new(self, i, j):
        """
        Compute alpha_2_new_unc =
            alpha_2_old + y_2 * (E_1 - E_2) / eta.
        """
        eta = self._compute_eta(i, j)
        alpha_2_new_unc = self._alpha[j] + \
            self._y[j] * (self.E[j] - self.E[i]) / eta

        L, H = self._compute_L_H(i, j)
        if alpha_2_new_unc > H:
            return H
        elif alpha_2_new_unc < L:
            return L
        else:
            return alpha_2_new_unc

    def _compute_alpha_1_new(self, i, j, alpha_2_new):
        """Compute alpha_1_new."""
        return self._alpha[i] + self._y[i] * self._y[j] * \
            (self._alpha[j] - alpha_2_new)

    def _compute_b_1_new(self, i, j, alpha_1_new, alpha_2_new):
        """Compute b_1_new."""
        return -self.E[i] - self._y[i] * \
            self._kernel(self._X[i], self._X[i]) * \
            (alpha_1_new - self._alpha[i]) - \
            self._y[j] * self._kernel(self._X[j], self._X[i]) * \
            (alpha_2_new - self._alpha[j]) + self._b

    def _compute_b_2_new(self, i, j, alpha_1_new, alpha_2_new):
        """Compute b_2_new."""
        return -self.E[j] - self._y[i] * \
            self._kernel(self._X[i], self._X[j]) * \
            (alpha_1_new - self._alpha[i]) - \
            self._y[j] * self._kernel(self._X[j], self._X[j]) * \
            (alpha_2_new - self._alpha[j]) + self._b

    def _compute_b_new(self, i, j, alpha_1_new, alpha_2_new):
        """Compute b_new."""
        b_1_new = self._compute_b_1_new(
            i, j, alpha_1_new, alpha_2_new
        )
        b_2_new = self._compute_b_2_new(
            i, j, alpha_1_new, alpha_2_new
        )
        if 0 < alpha_1_new < self._C:
            return b_1_new
        elif 0 < alpha_2_new < self._C:
            return b_2_new
        else:
            return (b_1_new + b_2_new) / 2

    def fit(self, X, y):
        r"""Fit SVM

        Args:
            X (ndarray): array-like, shape (n_samples, n_features).
                Training data.
            y (ndarray): array-like, shape (n_samples,).
                Target values.
        """
        # 1. Initialize parameters
        self._init_parameters(X, y)

        # 2. Train SVM
        for _ in range(self._epoch):
            i, j = self._choose_alpha()

            self._alpha[j] = self._compute_alpha_2_new(i, j)
            self._alpha[i] = self._compute_alpha_1_new(i, j, self._alpha[j])

            self._b = self._compute_b_new(
                i, j, self._alpha[j], self._alpha[i]
            )

            self.E[i] = self._E(i)
            self.E[j] = self._E(j)

    def predict(self, X):
        r"""Predict the target of new samples.

        Args:
            X (ndarray): array-like, shape (n_samples, n_features).
                The samples.

        Returns:
            ndarray: The predicted class.
        """
        output = np.zeros(shape=(X.shape[0]), dtype=int)

        # Calculate W * x_i + b
        for i, x in enumerate(X):
            out = self._b
            for j in range(self.sample_num):
                out += self._alpha[j] * self._y[j] * self._kernel(
                    x, self._X[j])
            out = 1 if out > 0 else -1
            output[i] = out

        assert output.shape[0] == X.shape[0]
        return output
