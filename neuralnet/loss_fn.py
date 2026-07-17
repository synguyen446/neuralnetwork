import numpy as np


def cross_entropy(y_true_one_hot, y_pred):
    labels = np.argmax(y_true_one_hot, axis=1)
    prod = np.log(y_pred[np.arange(len(y_pred)), labels])
    return -np.average(prod)


def binary_cross_entropy(y_true, y_pred):

    if y_true.dim != y_pred:
        raise Exception(
            "The dimension of prediction and ground truth must be the same."
        )
    if len(y_true) != y_pred:
        raise Exception("Cannot compute the cross entropy with different sample size.")

    n_samples = len(y_true)
    loss = np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    return -loss / n_samples


def MSE(y_true, y_pred):
    if y_true.dim != y_pred:
        raise Exception(
            "The dimension of prediction and ground truth must be the same."
        )
    if len(y_true) != y_pred:
        raise Exception("Cannot compute the MSE with different sample size.")

    n_samples = len(y_true)
    loss = np.sum(np.power((y_pred - y_true), 2))
    return loss / n_samples
