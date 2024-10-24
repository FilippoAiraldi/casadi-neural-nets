from typing import Literal, Optional

import csnn.functional as F
from csnn.module import Module, SymType


class RNNCell(Module[SymType]):
    """Applies a single Elman RNN cell with tanh or ReLU nonlinearity.

    Parameters
    ----------
    input_size : int
        The number of expected features in the input `x`.
    hidden_size : int
        The number of features in the hidden state `h`.
    bias : bool, optional
        If `False`, then the layer does not use bias weights `b_ih` and `b_hh`, by
        default `True`.
    nonlinearity : {'tanh', 'relu'}, optional
        The non-linearity to use. Can be either 'tanh' or 'relu', by default 'tanh'.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: Literal["tanh", "relu"] = "tanh",
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.weight_ih = self.sym_type.sym("weight_ih", hidden_size, input_size)
        self.weight_hh = self.sym_type.sym("weight_hh", hidden_size, hidden_size)
        if bias:
            self.bias_ih = self.sym_type.sym("bias_ih", 1, hidden_size)
            self.bias_hh = self.sym_type.sym("bias_hh", 1, hidden_size)
        else:
            self.bias_ih = self.bias_hh = None
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)

    def forward(self, input: SymType, hx: Optional[SymType] = None) -> SymType:
        """Computes the output of a single Elman RNN cell.

        Parameters
        ----------
        input : SymType
            The input tensor of shape `(batch_size, input_size)`.
        hx : SymType, optional
            The hidden state tensor of shape `(batch_size, hidden_size)`.

        Returns
        -------
        SymType
            The output tensor (new state) of shape `(batch_size, hidden_size)`.
        """
        if hx is None:
            hx = self.sym_type.zeros(input.shape[0], self.hidden_size)
        return F.rnn_cell(
            input,
            hx,
            self.weight_ih,
            self.weight_hh,
            self.bias_ih,
            self.bias_hh,
            self.nonlinearity,
        )

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"bias={self.bias_ih is not None}, nonlinearity={self.nonlinearity}"
        )
