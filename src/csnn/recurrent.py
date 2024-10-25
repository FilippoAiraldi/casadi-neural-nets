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


class RNN(Module[SymType]):
    """Applies a multi-layer Elman RNN cell with tanh or ReLU nonlinearity.

    Parameters
    ----------
    input_size : int
        The number of expected features in the input `x`.
    hidden_size : int
        The number of features in the hidden state `h`.
    num_layers : int, optional
        Number of recurrent layers, by default `1`.
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
        num_layers: int = 1,
        bias: bool = True,
        nonlinearity: Literal["tanh", "relu"] = "tanh",
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hs = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity

        f = self.sym_type.sym
        self.weight_ih_l0 = f("weight_ih_l0", hs, input_size)
        self.weight_hh_l0 = f("weight_hh_l0", hs, hs)
        if bias:
            self.bias_ih_l0 = f("bias_ih_l0", 1, hs)
            self.bias_hh_l0 = f("bias_hh_l0", 1, hs)
        for i in range(1, num_layers):
            w_ih = f"weight_ih_l{i}"
            w_hh = f"weight_hh_l{i}"
            b_ih = f"bias_ih_l{i}"
            b_hh = f"bias_hh_l{i}"
            setattr(self, w_ih, f(w_ih, hs, hs))
            setattr(self, w_hh, f(w_hh, hs, hs))
            if bias:
                setattr(self, b_ih, f(b_ih, 1, hs))
                setattr(self, b_hh, f(b_hh, 1, hs))
            else:
                setattr(self, b_ih, None)
                setattr(self, b_hh, None)
                self.register_parameter(b_ih, None)
                self.register_parameter(b_hh, None)

        self._weights_ih = []
        self._weights_hh = []
        for i in range(num_layers):
            self._weights_ih.append(getattr(self, f"weight_ih_l{i}"))
            self._weights_hh.append(getattr(self, f"weight_hh_l{i}"))
        if bias:
            self._biases_ih = []
            self._biases_hh = []
            for i in range(num_layers):
                self._biases_ih.append(getattr(self, f"bias_ih_l{i}"))
                self._biases_hh.append(getattr(self, f"bias_hh_l{i}"))
        else:
            self._biases_ih = self._biases_hh = None

    def forward(
        self, input: SymType, hx: Optional[SymType] = None
    ) -> tuple[SymType, SymType]:
        """Computes the output of a multi-layer Elman RNN cell.

        Parameters
        ----------
        input : SymType
            The input tensor of shape `(L, input_size)`, where `L` is the sequence
            length.
        hx : SymType, optional
            The hidden state tensor of shape `(num_layers, hidden_size)`.

        Returns
        -------
        tuple of 2 SymTypes
            The output tensor of shape `(L, hidden_size)` and the hidden state tensor of
            shape `(num_layers, hidden_size)`.
        """
        if hx is None:
            hx = self.sym_type.zeros(self.num_layers, self.hidden_size)
        return F.rnn(
            input,
            hx,
            self._weights_ih,
            self._weights_hh,
            self._biases_ih,
            self._biases_hh,
            self.nonlinearity,
        )

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, bias={self._biases_ih is not None}, "
            f"nonlinearity={self.nonlinearity}"
        )
