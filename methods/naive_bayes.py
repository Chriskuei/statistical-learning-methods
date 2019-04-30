"""Implement of Naive Bayes."""

import numpy as np
from collections import defaultdict


class NaiveBayes(object):
    r"""Naive Bayes is supervised learning methods based on applying
    Bayes' theorem with strong (naive) feature independence assumptions
    """
    def __init__(self, var_smoothing=1e-9):
        super(NaiveBayes, self).__init__()
        self.var_smoothing = var_smoothing
        self.prior_proba = None
        self.conditional_proba = None

    def fit(self, X, y):
        r"""Fit the NaiveBayes

        Args:
            X (ndarray): array-like, shape (n_samples, n_features).
                Training data.
            y (ndarray): array-like, shape (n_samples,).
                Target values.
        """
        # N is the number of samples
        N = X.shape[0]

        # Get the class
        cls = list(set(y))
        cls_num = len(cls)

        # Rearrange samples by class
        data = defaultdict(list)
        for x, c in zip(X, y):
            data[c].append(x)
        # Number of specific class's data
        cls_data_num = {c: len(data[c]) for c in data}

        # Features map
        features = []
        for i in range(X.shape[1]):
            features.append(list((set(X[:, i]))))
        # Number of every feature's instance
        features_num = [len(f) for f in features]

        # Prior probability
        self.prior_proba = {}
        # Count data number: class -> feature -> value count
        count = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(int)
            )
        )
        # Conditional_probability
        self.conditional_proba = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(float)
            )
        )
        # Calculate prior probability
        for c, samples in data.items():
            self.prior_proba[c] = (
                (len(samples) + self.var_smoothing) / (
                    N + cls_num * self.var_smoothing)
            )
            for sample in samples:
                for f, value in enumerate(sample):
                    count[c][f][value] = count[c][f][value] + 1

        # Smoothing
        for c, feats in count.items():
            for f, cnts in feats.items():
                for value in features[f]:
                    if value not in cnts:
                        count[c][f][value] = 0

        # Calculate conditional probability
        for c, feats in count.items():
            for f, cnts in feats.items():
                for value, cnt in cnts.items():
                    self.conditional_proba[c][f][value] = (
                        (cnt + self.var_smoothing) / (
                            cls_data_num[c] + (
                                features_num[f] * self.var_smoothing)))

    def _compute_probability(self, c, x):
        r"""Compute probability

        Args:
            c (int): class.
            x (ndarray): array-like, shape (n_features).

        Returns:
            ndarray: The probability.
        """
        output = self.prior_proba[c]
        for f, value in enumerate(x):
            output = output * self.conditional_proba[c][f][value]
        return output

    def predict(self, X):
        r"""Predict the target of new samples.

        Args:
            X (ndarray): array-like, shape (n_samples, n_features).
                The samples.

        Returns:
            ndarray: The predicted class.
        """
        # Shape: (batch_size)
        output = np.zeros(shape=(X.shape[0]), dtype=int)

        # Classification
        for i, x in enumerate(X):
            out = None
            max_probabitlity = 0
            for c in self.prior_proba:
                probability = self._compute_probability(c, x)
                if probability > max_probabitlity:
                    max_probabitlity = probability
                    out = c
            output[i] = out

        return output

if __name__ == "__main__":
    
    nb = NaiveBayes()
    nb.fit(X, y)
    pre = nb.predict(np.array([[-1, -1], [1, 1]]))
    print(pre)
