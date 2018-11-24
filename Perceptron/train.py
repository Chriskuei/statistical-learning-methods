# -*- coding:utf-8 -*- 
"""
 * @Author: Jiangui Chen 
 * @Date: 2018-11-24 18:12:38 
 * @Last Modified by:   Jiangui Chen 
 * @Last Modified time: 2018-11-24 18:12:38 
 * @Desc: 
"""

import numpy as np
from perceptron import Perceptron


def work():
    r"""Example for using Perceptron"""
    # Create example X and y
    X = np.asarray([[3, 3], [4, 3], [1, 1]])
    y = np.asarray([1, 1, -1])
    
    # Construct perceptron
    perceptron = Perceptron(learning_rate=0.5, epoch=100)
    
    # Train perception
    perceptron.fit(X, y)

    # Predict
    prediction = perceptron.predict(np.asarray([[0, 0]]))
    print(prediction)

if __name__ == "__main__":
    work()