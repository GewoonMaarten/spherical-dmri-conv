import pytest
from autoencoder.concrete_select import Decoder

test_data = [
    (500, 1344, 2, ["linear_0", "relu_0", "linear_1", "sigmoid_1"]),
    (3000, 1344, 2, ["linear_0", "relu_0", "linear_1", "sigmoid_1"]),
]


@pytest.mark.parametrize(
    "input_size, output_size, n_hidden_layers, expected_layers", test_data
)
def test_decoder_network(input_size, output_size, n_hidden_layers, expected_layers):
    for i, module in enumerate(
        Decoder(input_size, output_size, n_hidden_layers).decoder.named_children()
    ):
        assert module[0] == expected_layers[i]
        if i == len(expected_layers):
            assert module[1].out_features == output_size
