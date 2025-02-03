import csnn.functional as F
from csnn.module import Module, SymType


class BatchNorm1d(Module[SymType]):
    """Applies batch normalization over a 2D input. The mean and standard deviation are
    symbolical, and are not learned.

    Parameters
    ----------
    num_features : int
        Number of features or channels in the input tensor
    affine : bool, optional
        If set to `True`, this module has affine parameters. Defaults to `True`.

    Notes
    -----
    This layer is implemented for compatibility with PyTorch, but it is not meant to
    compute mean and standard deviation of the input and keep a running estimate of
    these. Its goal is to mimic the same operations as PyTorch's `BatchNorm1d`.
    """

    def __init__(self, num_features: int, affine: bool = True) -> None:
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.running_mean = self.sym_type.sym("running_mean", num_features, 1)
        self.running_std = self.sym_type.sym("running_std", num_features, 1)
        if affine:
            self.weight = self.sym_type.sym("weight", num_features, 1)
            self.bias = self.sym_type.sym("bias", num_features, 1)
        else:
            self.weight = self.bias = None
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, input: SymType) -> SymType:
        """Computes the batch-normalized output.

        Parameters
        ----------
        input : SymType
            The input tensor of shape `(batch_size, num_features)`.

        Returns
        -------
        SymType
            The output tensor of shape `(batch_size, num_features)`.
        """
        return F.batch_norm1d(
            input, self.running_mean, self.running_std, self.weight, self.bias
        )

    def extra_repr(self) -> str:
        return f"{self.num_features}, affine={self.affine}"
