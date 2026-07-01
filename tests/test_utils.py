from neuralnet.utils import *
import torch
import torch.nn.functional as F


def test_one_hot_encoding():
    # 4 classes
    input_array = np.array([0, 2, 3, 1])
    result = one_hot_encoding(input_array, 4)
    expected = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0]])

    assert np.allclose(result, expected)

    # 3 classes
    input_array = np.array([0, 2, 1])
    result = one_hot_encoding(input_array, 3)
    expected = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    assert np.allclose(result, expected)


def test_normalize():
    input_array = np.array([0, 2, 3, 1])

    result = normalize(input_array)
    expected = F.normalize(
        torch.tensor(input_array, dtype=torch.float32), p=2, dim=-1
    ).tolist()

    assert np.allclose(result, expected)
