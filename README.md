# Neural Networks with CasADi

**csnn** is a package for creating symbolic neural networks in [CasADi](https://web.casadi.org) in a [PyTorch](https://pytorch.org/)-like API style.

[![PyPI version](https://badge.fury.io/py/csnn.svg)](https://badge.fury.io/py/csnn)
[![Source Code License](https://img.shields.io/badge/license-MIT-blueviolet)](https://github.com/FilippoAiraldi/casadi-neural-nets/blob/master/LICENSE)
![Python 3.9](https://img.shields.io/badge/python->=3.9-green.svg)

[![Tests](https://github.com/FilippoAiraldi/casadi-neural-nets/actions/workflows/ci.yml/badge.svg)](https://github.com/FilippoAiraldi/casadi-neural-nets/actions/workflows/ci.yml)
[![Downloads](https://static.pepy.tech/badge/csnn)](https://www.pepy.tech/projects/csnn)
[![Maintainability](https://api.codeclimate.com/v1/badges/6006c41542cd8e902125/maintainability)](https://codeclimate.com/github/FilippoAiraldi/casadi-neural-nets/maintainability)
[![Test Coverage](https://api.codeclimate.com/v1/badges/6006c41542cd8e902125/test_coverage)](https://codeclimate.com/github/FilippoAiraldi/casadi-neural-nets/test_coverage)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Introduction

The package allows the creation of neural networks with the symbolic language offered by [CasADi](https://web.casadi.org). This is done in a similar way to [PyTorch](https://pytorch.org/). For example, the following code allows us to create an MLP with a hidden layer:

```python
import casadi as cs
from csnn import set_sym_type, Linear, Sequential, ReLU

set_sym_type("SX")  # can set either MX or SX

net = Sequential[cs.SX]((
    Linear(4, 32),
    ReLU(),
    Linear(32, 1),
    ReLU()
))

batch = 2
input = cs.SX.sym("in", batch, 4)
output = net(input)
assert output.shape == (batch, 1)
```

---

## Implemented Modules

So far, the following modules that are available in PyTorch have been implemented:

- Containers
  * Module
  * Sequential
- Activation functions
  * GELU
  * SELU
  * LeakyReLU
  * ReLU
  * Sigmoid
  * Softplus
  * Tanh
- Linear layers
  * Linear
- Recurrent layers
  * RNNCell
  * RNN
- Dropout layers
  * Dropout
  * Dropout1d

Additionally, the library provides the implementation for the following convex neural networks (see `csnn.convex`):

- FicNN
- PwqNN
- PsdNN

---

## Installation

To install the package, run

```bash
pip install csnn
```

**csnn** has the following dependencies

- [CasADi](https://web.casadi.org)

For playing around with the source code instead, run

```bash
git clone https://github.com/FilippoAiraldi/casadi-neural-nets.git
```

---

## License

The repository is provided under the MIT License. See the LICENSE file included with this repository.

---

## Author

[Filippo Airaldi](https://www.tudelft.nl/staff/f.airaldi/), PhD Candidate [f.airaldi@tudelft.nl | filippoairaldi@gmail.com]

> [Delft Center for Systems and Control](https://www.tudelft.nl/en/3me/about/departments/delft-center-for-systems-and-control/) in [Delft University of Technology](https://www.tudelft.nl/en/)

Copyright (c) 2023 Filippo Airaldi.

Copyright notice: Technische Universiteit Delft hereby disclaims all copyright interest in the program “csnn” (Nueral Networks with CasADi) written by the Author(s). Prof. Dr. Ir. Fred van Keulen, Dean of 3mE.
