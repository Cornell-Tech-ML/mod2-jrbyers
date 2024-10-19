from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply method for ScalarFunction"""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass of addition."""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward pass of addition."""
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass of logarithmic function."""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass of log function."""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.


# TODO: Implement for Task 1.2.


class Mul(ScalarFunction):
    """Multiplication of two variables."""

    @staticmethod
    def forward(ctx: Context, x: float, y: float) -> float:
        """Forward pass multiply"""
        ctx.save_for_backward(x, y)
        return x * y

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass of multiplication."""
        (x, y) = ctx.saved_values
        grad_x = d_output * y
        grad_y = d_output * x

        return grad_x, grad_y


class Inv(ScalarFunction):
    """Calculates the reciprocal of a given scalar."""

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        """Forward pass inverse"""
        ctx.save_for_backward(x)
        return operators.inv(x)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass of inverse."""
        (x,) = ctx.saved_values
        return operators.inv_back(x, d_output)


class Neg(ScalarFunction):
    """Negates a given scalar."""

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        """Forward pass negation"""
        return operators.neg(x)

    @staticmethod
    def backward(ctx: Context, d: float) -> float:
        """Backward pass derivative for negation"""
        f_prime = -1.0
        return f_prime * d


class Sigmoid(ScalarFunction):
    """Calculates the sigmoid function."""

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        """Forward pass sigmoid"""
        sig_output = operators.sigmoid(x)
        ctx.save_for_backward(sig_output)
        return sig_output

    @staticmethod
    def backward(ctx: Context, d: float) -> float:
        """Backward pass derivative for sigmoid"""
        sig_output: float = ctx.saved_values[0]
        return sig_output * (1.0 - sig_output) * d


class ReLU(ScalarFunction):
    """Applies the ReLU activation function."""

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        """Forward pass ReLU"""
        ctx.save_for_backward(x)
        return operators.relu(x)

    @staticmethod
    def backward(ctx: Context, d: float) -> float:
        """Backward pass derivative for ReLu"""
        (x,) = ctx.saved_values
        return operators.relu_back(x, d)


class Exp(ScalarFunction):
    """Calculates the exponential function."""

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        """Forward pass exponential function"""
        exponential_output = operators.exp(x)
        ctx.save_for_backward(exponential_output)
        return exponential_output

    @staticmethod
    def backward(ctx: Context, d: float) -> float:
        """Backward pass derivative for exponential"""
        exponential_output: float = ctx.saved_values[0]
        return exponential_output * d


class LT(ScalarFunction):
    """Checks if one number is less than another."""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass less than operation"""
        return 1.0 if a < b else 0.0

    @staticmethod
    def backward(ctx: Context, d: float) -> Tuple[float, float]:
        """Backward pass derivative for less than"""
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Checks if two numbers are equal."""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass equal operator"""
        return 1.0 if a == b else 0.0

    @staticmethod
    def backward(ctx: Context, d: float) -> Tuple[float, float]:
        """Backward pass derivative for equivalence"""
        return 0.0, 0.0
