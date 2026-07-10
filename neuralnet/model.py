import numpy as np
from neuralnet.utils import xavier_uniform_init


class Model:
    def __init__(self, *arg):
        self.dag = arg
        self.loss_fn = None
        self.y_true = None
        self.gradients = []
        self.bias_gradient = []

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
        self.bias_gradient = []
        dout = 1
        for i in range(len(self.dag) - 1, -1, -1):
            dw, db, dout = self.dag[i].backward(dout, self.y_true)
            self.gradients.append(dw)
            self.bias_gradient.append(db)
        return self.gradients, self.bias_gradient

    def step(self, alpha):
        self.gradients.reverse()
        self.bias_gradient.reverse()
        for i in range(len(self.dag) - 1, -1, -1):
            self.dag[i].weight -= alpha * self.gradients[i]
            self.dag[i].bias -= alpha * self.bias_gradient[i]

    def get_weights(self):
        weights = []
        for i in range(len(self.dag)):
            weights.append(self.dag[i].weight)
        return weights

    def fit(self, epochs, batch_size, X_train, y_train):
        for i in range(epochs):
            indices = np.random.permutation(len(X_train))
            epoch_loss = 0
            correct_count = 0

            X_shuffle = X_train[indices]
            y_shuffle = y_train[indices]
            for start in range(0, len(X_train), batch_size):
                X_batch = X_shuffle[start : start + batch_size]
                y_batch = y_shuffle[start : start + batch_size]
                y_pred_prob, loss = self.forward(X_batch, y_batch)
                y_pred = y_pred_prob.argmax(axis=1)
                correct_count += np.sum(y_pred == y_batch.argmax(axis=-1))
                epoch_loss += loss * len(X_batch)
                self.backward()
                self.step(0.003)
            print(
                f"STEP {i + 1}: Loss - {epoch_loss/len(X_train):.4f} | Accuracy - {correct_count/len(X_train):.4f}"
            )

    def compile(self, loss_fn):
        self.loss_fn = loss_fn


class Layer:
    def __init__(self, in_feature, out_feature, activation):
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.activation = activation
        self.weight = xavier_uniform_init(in_feature, out_feature)
        self.bias = np.zeros((1, out_feature))
        self.a_previous = None
        self.z_current = None
        self.a_current = None

    def backward(
        self,
        dout,
        y_true=None,
    ):
        delta = dout * self.activation.backward(self.a_current, y_true)
        dw = (self.a_previous.T @ delta) / self.a_current.shape[0]
        db = np.sum(delta, axis=0, keepdims=True) / self.a_current.shape[0]
        dout = delta @ self.weight.T
        return dw, db, dout

    def __call__(self, x):
        self.a_previous = x
        self.z_current = x @ self.weight + self.bias
        self.a_current = self.activation.forward(self.z_current)
        return self.a_current
