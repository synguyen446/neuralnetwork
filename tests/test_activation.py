from neuralnet.activation import *
from neuralnet.model import *
import torch
import pytest

INPUT_ARRAYS = [
    np.array([1, 2, 3]),
    np.array([-1, 2, -10]),
    np.array([[-1123, 201230, -10], [12312, 123, -12]]),
]


def test_relu():

    for input_array in INPUT_ARRAYS:
        custom_layer = ReLu()
        torch_layer = torch.nn.ReLU()

        result = custom_layer.forward(input_array)
        excepted = torch_layer(torch.tensor(input_array))

        assert np.allclose(result, excepted)


def test_sigmoid():
    for input_array in INPUT_ARRAYS:
        custom_layer = Sigmoid()
        torch_layer = torch.nn.Sigmoid()

        result = custom_layer.forward(input_array)
        excepted = torch_layer(torch.tensor(input_array))

        assert np.allclose(result, excepted)


def test_softmax():
    for input_array in INPUT_ARRAYS:
        custom_layer = Softmax()
        torch_layer = torch.nn.Softmax()

        result = custom_layer.forward(input_array)
        excepted = torch_layer(
            torch.tensor(
                input_array,
                dtype=torch.float32,
            )
        )

        assert np.allclose(result, excepted)
