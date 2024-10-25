import sys
from collections.abc import Generator, Iterator, Sequence
from math import prod, sqrt
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from csnn import Module

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

RngType: TypeAlias = Union[
    None,
    int,
    Sequence[int],
    np.random.SeedSequence,
    np.random.BitGenerator,
    np.random.Generator,
]


def _calculate_fan_in_and_fan_out(tensor: np.ndarray) -> tuple[int, int]:
    dimensions = tensor.ndim
    if dimensions < 2:
        raise ValueError(
            "Fan in, fan out cannot be computed for tensor with fewer than 2 dimensions"
        )
    num_output_fmaps, num_input_fmaps = tensor.shape[:2]
    receptive_field_size = prod(tensor.shape[2:])
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def _calculate_correct_fan(
    tensor: np.ndarray, mode: Literal["fan_in", "fan_out"]
) -> int:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == "fan_in" else fan_out


def _calculate_gain(
    nonlinearity: Literal["linear", "sigmoid", "tanh", "relu", "leaky_relu", "selu"],
    param: Any = None,
) -> float:
    """See https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.calculate_gain."""
    if nonlinearity == "linear" or nonlinearity == "sigmoid":
        return 1
    if nonlinearity == "tanh":
        return 5.0 / 3.0
    if nonlinearity == "relu":
        return sqrt(2.0)
    if nonlinearity == "leaky_relu":
        if param is None:
            negative_slope = 0.01
        elif (
            not isinstance(param, bool)
            and isinstance(param, int)
            or isinstance(param, float)
        ):
            negative_slope = param
        else:
            raise ValueError(f"negative_slope {param} not a valid number")
        return sqrt(2.0 / (1 + negative_slope**2))
    if nonlinearity == "selu":
        return 0.75
    raise ValueError(f"Unsupported nonlinearity {nonlinearity}")


def uniform_(
    tensor: np.ndarray, a: float = 0.0, b: float = 1.0, seed: RngType = None
) -> np.ndarray:
    """See https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.uniform_."""
    np_random = np.random.default_rng(seed)
    np.copyto(tensor, np_random.uniform(a, b, tensor.shape))
    return tensor


def kaiming_uniform_(
    tensor: np.ndarray,
    a: float = 0.0,
    mode: Literal["fan_in", "fan_out"] = "fan_in",
    nonlinearity: Literal[
        "linear", "sigmoid", "tanh", "relu", "leaky_relu", "selu"
    ] = "leaky_relu",
    seed: RngType = None,
) -> np.ndarray:
    """See
    https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_."""
    fan = _calculate_correct_fan(tensor, mode)
    gain = _calculate_gain(nonlinearity, a)
    std = gain / sqrt(fan)
    bound = sqrt(3.0) * std
    return uniform_(tensor, -bound, bound, seed)


def _init(
    module: "Module", skip_none: bool, seed: RngType = None
) -> Generator[tuple[str, Optional[npt.NDArray[np.floating]]], None, None]:
    """Internal function to initialize a module's parameters based on its class."""
    from csnn import RNN, Linear, RNNCell

    np_random = np.random.default_rng(seed)

    if isinstance(module, Linear):
        weight = np.empty(module.weight.shape)
        kaiming_uniform_(weight, a=sqrt(5), seed=np_random)
        yield "weight", weight

        if module.bias is not None:
            bias = np.empty(module.bias.shape)
            fan_in, _ = _calculate_fan_in_and_fan_out(weight)
            bound = 1 / sqrt(fan_in) if fan_in > 0 else 0
            uniform_(bias, -bound, bound, seed=np_random)
            yield "bias", bias
        elif not skip_none:
            yield "bias", None

    elif isinstance(module, (RNNCell, RNN)):
        stdv = 1.0 / sqrt(module.hidden_size) if module.hidden_size > 0 else 0
        for name, weight_ in module.parameters(skip_none=skip_none):
            if weight_ is not None:
                weight = np.empty(weight_.shape)
                uniform_(weight, -stdv, stdv)
                yield name, weight
            elif not skip_none:
                yield name, None


def init_parameters(
    module: "Module",
    recurse: bool = True,
    prefix: str = "",
    skip_none: bool = True,
    seed: RngType = None,
) -> Iterator[tuple[str, Optional[npt.NDArray[np.floating]]]]:
    """Initializes the parameters (i.e., weights) of a module.

    Parameters
    ----------
    module : Module
        Module whose parameters are to be initialized.
    recurse : bool, optional
        Whether to recursively initialize the parameters of the module's children,
        by default `True`.
    prefix : str, optional
        Prefix to prepend to the names of the parameters, by default "".
    skip_none : bool, optional
        Whether to skip parameters whose values are `None`, by default `True`.
    seed : int, sequence of ints, or rng engine, optional
        Seed for the random number generator, or an engine itself, by default `None`.

    Yields
    ------
    Iterator of 2-tuples of str and arrays
        Yields the name and value of each parameter of the module.
    """
    seed = np.random.default_rng(seed)
    if prefix != "":
        prefix += "."
    for name, par_value in _init(module, skip_none, seed):
        yield prefix + name, par_value
    if recurse:
        for name, module in module.children():
            yield from init_parameters(module, True, prefix + name, skip_none, seed)
