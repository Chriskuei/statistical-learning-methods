# -*- coding:utf-8 -*- 
"""
 * @Author: Jiangui Chen 
 * @Date: 2018-11-25 19:00:32 
 * @Last Modified by: Jiangui Chen 
 * @Last Modified time: 2018-11-25 19:00:32 
 * @Desc: Example for using KNearestNeighbor
"""

import numpy as np
from knn import KNearestNeighbor

def work():
    r"""Example for using KNearestNeighbor"""
    # Create example X and y
    X = np.asarray([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    y = np.asarray([-1, +1, +1, +1, -1, -1])
    
    # Construct KNearestNeighbor
    kNN = KNearestNeighbor(n_neighbors=2, p=2, metric='minkowski')
    
    # Train KNearestNeighbor
    kNN.fit(X, y)

    # Predict
    prediction = kNN.predict(np.asarray([[3, 4.5], [8, 0]]))
    print(prediction)


if __name__ == "__main__":
    work()