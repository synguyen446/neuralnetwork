import numpy as np


class Function:
    def __init__(self):
        pass

    def foward(self, x: np.ndarray):
        pass

    def backward(self, a_current, y_true=None):
        pass


class ReLu(Function):

    def forward(self, x: np.ndarray):
        return np.maximum(0, x)

    def backward(self, a_current, y_true=None):
        return np.where(a_current > 0, 1, 0)


class Sigmoid(Function):

    def forward(self, x: np.ndarray):
        return 1 / (1 + np.exp(-x))

    def backward(self, a_current, y_true=None):
        return a_current * (1 - a_current)


class Softmax(Function):

    def forward(self, x):
        exp_array = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_array / np.sum(exp_array, axis=-1, keepdims=True)

    def backward(self, a_current, y_true):
        return y_true - a_current
