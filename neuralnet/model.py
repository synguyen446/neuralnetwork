import numpy as np
from neuralnet.utils import xavier_uniform_init


class Model:
    def __init__(self, *arg):
        self.dag = arg
        self.loss_fn = None
        self.y_true = None
        self.gradients = []

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.y_true = y
        if not self.loss_fn:
            raise RuntimeError(
                "Please define the loss function before calling forward passing!"
            )
        for layer in self.dag:
            x = layer(x)
        return x, self.loss_fn(x, y)

    def backward(self):
        self.gradients = []
        dout = 1
        for i in range(len(self.dag) - 1, -1, -1):
            dw, dout = self.dag[i].backward(dout, self.y_true)
            self.gradients.append(dw)
        return self.gradients

    def step(self, alpha):
        self.gradients.reverse()
        for i in range(len(self.dag) - 1, -1, -1):
            self.dag[i].weight -= alpha * self.gradients[i]

    def get_weights(self):
        weights = []
        for i in range(len(self.dag)):
            weights.append(self.dag[i].weight)
        return weights

    def compile(self, loss_fn):
        self.loss_fn = loss_fn


class Layer:
    def __init__(self, in_feature, out_feature, activation):
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.activation = activation
        self.weight = xavier_uniform_init(in_feature, out_feature)
        self.a_previous = None
        self.z_current = None
        self.a_current = None

    def backward(
        self,
        dout,
        y_true=None,
    ):
        delta = dout * self.activation.backward(self.a_current, y_true)
        dw = self.a_previous.T @ delta
        dout = delta @ self.weight.T
        return dw, dout

    def __call__(self, x):
        self.a_previous = x
        self.z_current = x @ self.weight
        self.a_current = self.activation.forward(self.z_current)
        return self.a_current
