"""k-Nearest Neighbor."""

import numpy as np


class KDTreeNode(object):
    r"""KDTree node"""
    def __init__(self, x=None, dim=0, father=None, left=None,
                 right=None, sibling=None):
        super(KDTreeNode, self).__init__()
        assert isinstance(father, (KDTreeNode, type(None)))
        assert isinstance(left, (KDTreeNode, type(None)))
        assert isinstance(left, (KDTreeNode, type(None)))

        self.x = x
        self.dim = dim
        self.father = father
        self.left = left
        self.right = right
        self.sibling = sibling

    def desc(self):
        r"""print basic information of KDTreeNode"""
        print('x: {}, dim: {}'.format(self.x, self.dim))


class KDTree(object):
    r"""KDTree for fast generalized N-point problems

    Args:
        X (ndarray): array-like, shape = [n_samples, n_features]
        p (int, optional): default = 2. Power parameter for the
            Minkowski metric
        metric (str): the distance metric to use for the tree
    """
    def __init__(self, X, p=2, metric='minkowski'):
        super(KDTree, self).__init__()
        self._data = X
        self._p = p
        self._metrix = metric
        self._root = self._create(X, dim=self._max_var_dim(X))

    def _max_var_dim(self, data):
        r"""Find dimension that has the biggest variance

        Args:
            data (ndarray): array-like, shape = [n_samples, n_features]

        Returns:
            max_var_dim: an integer, the max variance dimension
        """
        if data.shape[0] == 0:
            return 0

        # Compute the variance for every dimension
        var = np.var(data, axis=0)

        # Returns the indice of the maximum values along an axis.
        max_var_dim = np.argmax(var)
        return max_var_dim

    def _create(self, data, dim, father=None):
        r"""Create KDTree

        Args:
            data (ndarray): array-like, sample of training data
            dim (int): an integer, the dimension needed to split
            father (KDTreeNode, optional): defaults to None. Father
                of the current node

        Returns:
            root (KDTreeNode): the root node of KDTree
        """
        root = None
        # Leaf node
        if data.shape[0] == 0:
            return None

        # Find the median data
        data = sorted(data, key=lambda x: x[dim])
        data = np.asarray(data)

        mid = data.shape[0] // 2

        # Generate root node
        root = KDTreeNode(data[mid], dim=dim, father=father)

        # Generate left and right child node
        root.left = self._create(
            data=data[:mid],
            dim=self._max_var_dim(data[:mid]),
            father=root
        )
        root.right = self._create(
            data=data[mid + 1:],
            dim=self._max_var_dim(data[mid + 1:]),
            father=root
        )
        if root.left and root.right:
            root.left.sibling = root.right
            root.right.sibling = root.left
        return root

    def _distance(self, x_1, x_2):
        r"""Compute the distance between x_1 and x_2

        Args:
            x_1 (ndarray): array-like, shape = [n_features]
            x_2 (ndarray): array-like, shape = [n_features]

        Returns:
            [ndarray]: the the distance between x_1 and x_2
        """
        # x_1 and x_2 should have same shape
        assert x_1.shape == x_2.shape

        # Compute p-norm between x_1 and x_2
        p = self._p
        _sum = np.sum(np.power(np.abs(x_1 - x_2), p))

        return np.power(_sum, 1 / p)

    def query(self, x, k=1, return_distance=True):
        r"""Query the tree for the k nearest neighbors

        Args:
            x (ndarray): array-like, shape = [n_features]
            k (int, optional): defaults to 1. Number of neighbors
            return_distance (bool, optional): defaults to True.
                If True, return (dist, index); If False, return index

        Returns:
            dist (list): list of k-nearest distance
            index (list): list of k-nearest neighbors index
        """

        node = self._root

        # Find current closest node
        # Stop when node is leaf
        while node:
            dim = node.dim

            # If the feature in specific dimension of x is
            # smaller than that of current node, then shift
            # to left child, else shift to right child
            if x[dim] < node.x[dim]:
                if node.left is None:
                    break
                node = node.left
            else:
                if node.right is None:
                    break
                node = node.right

        neighbors = []

        # Stop when node is root
        while node:
            # Check current node
            distance = self._distance(x, node.x)
            neighbors.append((distance, node.x))

            # Check sibling of current node
            if node.sibling:
                node = node.sibling
                distance = self._distance(x, node.x)
                neighbors.append((distance, node.x))

            # Back to father
            node = node.father

        import heapq
        heapq.heapify(neighbors)
        k_neighbors = heapq.nsmallest(k, neighbors)

        # Generate distance list and index list
        dist = []
        index = []
        for distance, _x in k_neighbors:
            dist.append(distance)
            index.append(np.argwhere(self._data == _x)[0][0])

        if return_distance:
            return (dist, index)
        else:
            return index

    def _print(self, root, depth=0):
        r"""print KDTreeNode, indent by depth"""
        if isinstance(root, type(None)):
            return
        # Indent control
        print('  ' * depth, end='')
        # Print root
        root.desc()
        # Print child
        self._print(root.left, depth=depth + 1)
        self._print(root.right, depth=depth + 1)

    def desc(self):
        r"""print basic information of KDTreeNode"""
        return self._print(self._root, depth=0)

    @property
    def data(self):
        r"""Returns KDTree's data"""
        return self._data


class KNearestNeighbor(object):
    """k-Nearest Neighbor algorithm is a basic classification and
    regression method.

    Args:
        n_neighbors (int): defaults to 1. Number of neighbors
        p (int, optional): default = 2. Power parameter for the
            Minkowski metric
        metric (str): the distance metric to use for the tree
    """

    def __init__(self, n_neighbors=5, p=2, metric='minkowski'):
        super(KNearestNeighbor, self).__init__()
        self.kdtree = None
        self._n_neighbors = n_neighbors
        self._p = p
        self._metrix = metric

    def fit(self, X, y):
        """Fit the KNearestNeighbor

        Args:
            X (ndarray): array-like, shape (n_samples, n_features).
                Training data.
            y (ndarray): array-like, shape (n_samples,).
                Target values.
        """
        # Create KDTree
        self.kdtree = KDTree(X, p=self._p, metric='minkowski')
        # Save labels
        self._y = y

    def predict(self, X):
        """Predict the target of new samples

        Args:
            X (ndarray): array-like, shape (n_samples, n_features).
                The samples.

        Returns:
            ndarray: The predicted class.
        """
        predictions = []
        for x in X:
            # Get k-nearest neighbors index
            _, index = self.kdtree.query(x, k=self._n_neighbors,
                                         return_distance=True)
            # Transform index to label
            y = [self._y[i] for i in index]

            # Vote for prediction
            from collections import Counter
            y = sorted(Counter(y), key=lambda x: x)[-1]

            predictions.append(y)
        return np.asarray(predictions)

    def __str__(self):
        return ('KNearestNeighbor(n_neighors={}, p={}, metric=\'{}\')'
                .format(self._n_neighbors, self._p, self._metrix))
