import csnn.functional as F
from csnn.module import Module, SymType


class ReLU(Module[SymType]):
    """Applies the rectified linear unit function element-wise as
    `ReLU(x) = (x)^+ = max(0, x)`."""

    def forward(self, input: SymType) -> SymType:
        return F.relu(input)
