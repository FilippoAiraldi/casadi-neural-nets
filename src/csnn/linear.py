import csnn.functional as F
from csnn.module import Module, SymType


class Linear(Module[SymType]):
    """Applies a linear transformation to the incoming data: `y = xA^T + b`, where `x`
    has shape `(*, in_features)` and `y` has shape `(*, out_features)`.

    Parameters
    ----------
    in_features : int
        Size of each input sample
    out_features : int
        Size of each output sample
    bias : bool, optional
        If set to `False`, the layer will not learn an additive bias. Defaults to
        `True`.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = self.sym_type.sym("weight", out_features, in_features)
        if bias:
            self.bias = self.sym_type.sym("bias", 1, out_features)
        else:
            self.bias = None
            self.register_parameter("bias", None)

    def forward(self, input: SymType) -> SymType:
        """Computes the output of a linear layer.

        Parameters
        ----------
        input : SymType
            The input tensor of shape `(batch_size, in_features)`.

        Returns
        -------
        SymType
            The output tensor of shape `(batch_size, out_features)`.
        """
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )
