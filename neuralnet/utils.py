import numpy as np


def xavier_uniform_init(in_dim, out_dim):
    limit = np.sqrt(6 / (in_dim + out_dim))
    return np.random.uniform(low=-limit, high=limit, size=(in_dim, out_dim))


def normalize(input_array):
    array_norms = np.linalg.norm(input_array)
    return input_array / array_norms


def one_hot_encoding(y, num_classes):
    array = np.zeros(
        (
            len(
                y,
            ),
            num_classes,
        )
    )
    array[np.arange(len(y)), y] = 1
    return array
