"""Implement of Decison tree."""

import numpy as np
import pandas as pd


class DTreeNode(object):
    r"""DTree node"""
    def __init__(self, leaf=True, label=None, feature=None):
        super(DTreeNode, self).__init__()
        self.leaf = leaf
        self.label = label
        self.feature = feature
        self.children = {}

    def add_child(self, value, node):
        r"""Add child by value."""
        self.children[value] = node

    def predict(self, x):
        r"""Predict result."""
        # If the node is leaf, return the result
        if self.leaf is True:
            return self.label
        return self.children[x[self.feature]].predict(x)


class DTree(object):
    r"""Decision tree.

    Args:
        data (DataFrame): training data, including X and y
        epsilon (float, optional): default = 1e-3. Max information
            gain should be larger than epsilon
    """
    def __init__(self, data, epsilon=1e-3):
        super(DTree, self).__init__()
        self._data = data
        self._epsilon = epsilon
        self._root = self._create(self._data)

    @staticmethod
    def _entropy(data):
        r"""Calculate entropy.
        Args:
            data (DataFrame): training data, including X and y

        Returns:
            float: the value of entropy
        """
        # Get data length
        data_counts = len(data)
        y = data.iloc[:, -1]
        # Calculate y probability
        probas = y.value_counts().values / data_counts
        # Sum up every probability
        entropy = sum([-p * np.log2(p) for p in probas])

        return entropy

    def _conditional_entropy(self, data, feature):
        r"""Calculate conditional entropy.
        Args:
            data (DataFrame): training data, including X and y
            feature (int): feature axis

        Returns:
            float: the value of conditional entropy
        """
        # Get data length
        data_counts = len(data)
        # Get all the values of current feature
        values = data[feature].unique()
        # Calculate value probability
        probas = data[feature].value_counts().values / data_counts

        # Calculate conditional entropy
        conditional_entropy = 0
        for proba, value in zip(probas, values):
            _data = data.loc[data[feature] == value].drop(
                [feature], axis=1)
            conditional_entropy += proba * self._entropy(_data)
        return conditional_entropy

    def _get_max_info_gain(self, data):
        r"""Get max information entropy.
        Args:
            data (DataFrame): training data, including X and y

        Returns:
            (int, float): max feature and max information gain
        """
        # Get entropy
        entropy = self._entropy(data)
        # Get all the features
        features = data.columns[:-1].values

        # Calculate information gain for each feature
        gain = []
        for feature in features:
            conditional_entropy = self._conditional_entropy(
                data, feature=feature)
            info_gain = entropy - conditional_entropy
            gain.append((feature, info_gain))
        # Get max information gain
        max_info_gain = max(gain, key=lambda x: x[1])
        return max_info_gain

    def _create(self, data):
        r"""Create DTree

        Args:
            data (DataFrame): training data, including X and y

        Returns:
            root (DTreeNode): the root node of DTree
        """
        # Get the label and features
        y, features = data.iloc[:, -1], data.columns[:-1]

        # If all the datas are in the same class, then return tree
        if len(y.value_counts()) == 1:
            return DTreeNode(leaf=True, label=int(y.iloc[0]))

        # If there is no features, then construct a node whose feature
        # has the most counts
        if len(features) == 0:
            label = y.value_counts().sort_values(
                ascending=False).index[0]
            return DTreeNode(leaf=True, label=label)

        # Then calculate maximum information gain
        feature, max_info_gain = self._get_max_info_gain(data)
        # It works only if maximun information gain is more then epsilon
        if max_info_gain < self._epsilon:
            label = y.value_counts().sort_values(
                ascending=False).index[0]
            return DTreeNode(leaf=True, label=label)

        # Construct a new node with max feature
        node = DTreeNode(leaf=False, feature=feature)

        # Get values of max feature
        values = data[feature].unique()
        for value in values:
            # Get child node
            _data = data.loc[data[feature] == value].drop(
                [feature], axis=1)
            child_node = DTree(data=_data, epsilon=self._epsilon).root
            node.add_child(value, child_node)

        return node

    def predict(self, x):
        r"""Predict the target of new sample.

        Args:
            x (ndarray): array-like, a sample.

        Returns:
            ndarray: The predicted class.
        """
        out = self._root.predict(x)

        return out

    @property
    def root(self):
        return self._root


class DecisionTree(object):
    """Decision Tree is a basic classification and
    regression method.

    Args:
        tree (DTree): `DTree` instance
        epsilon (float): control maximun information gain
    """

    def __init__(self, epsilon=1e-3):
        super(DecisionTree, self).__init__()
        self._tree = None
        self._epsilon = epsilon

    def fit(self, X, y):
        """Fit the DecisionTree

        Args:
            X (ndarray): array-like, shape (n_samples, n_features).
                Training data.
            y (ndarray): array-like, shape (n_samples,).
                Target values.
        """
        data = pd.DataFrame(data=X)
        data['y'] = y
        # Create DTree
        self._tree = DTree(data, epsilon=self._epsilon)

    def predict(self, X):
        """Predict the target of new samples

        Args:
            X (ndarray): array-like, shape (n_samples, n_features).
                The samples.

        Returns:
            ndarray: The predicted class.
        """
        output = np.zeros(shape=(X.shape[0]), dtype=int)

        for i, x in enumerate(X):
            out = self._tree.predict(x)
            output[i] = out
        print(output)
        return output
