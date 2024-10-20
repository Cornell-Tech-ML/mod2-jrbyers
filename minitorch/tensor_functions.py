"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Optional

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Call the forward function and track history"""
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass of negation function"""
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass of negation function."""
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass of inverse."""
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass of inverse."""
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass of adding two tensors"""
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass of adding tensors"""
        return grad_output, grad_output


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Optional[Tensor] = None) -> Tensor:
        """Return 1 if all are true"""
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(
                a.contiguous().view(int(operators.prod(list(a.shape)))), 0
            )


# TODO: Implement for Task 2.3.


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass multiply"""
        ctx.save_for_backward(t1, t2)
        return t1.f.mul_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass of multiplication."""
        (t1, t2) = ctx.saved_values
        grad_t1 = grad_output.f.mul_zip(grad_output, t2)
        grad_t2 = grad_output.f.mul_zip(grad_output, t1)
        return grad_t1, grad_t2


class Sigmoid(Function):
    """Calculates the sigmoid function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass sigmoid"""
        sig_output = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(sig_output)
        return sig_output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass derivative for sigmoid"""
        (sig_output,) = ctx.saved_tensors
        grad_input = sig_output * (1.0 - sig_output) * grad_output
        return grad_input


class ReLU(Function):
    """Applies the ReLU activation function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass ReLU"""
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass derivative for ReLu"""
        (t1,) = ctx.saved_values
        return grad_output.f.relu_back_zip(t1, grad_output)


class Log(Function):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass of logarithmic function."""
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass of log function."""
        (t1,) = ctx.saved_values
        return grad_output.f.log_back_zip(t1, grad_output)


class Exp(Function):
    """Calculates the exponential function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass exponential function"""
        exponential_output = t1.f.exp_map(t1)
        ctx.save_for_backward(exponential_output)
        return exponential_output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass derivative for exponential"""
        exponential_output: Tensor = ctx.saved_values[0]
        return exponential_output * grad_output


class Sum(Function):
    """Calculates the sum function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim: Optional[Tensor] = None) -> Tensor:
        """Forward pass sum function"""
        ctx.save_for_backward(t1, dim)
        if dim is not None:
            return t1.f.add_reduce(t1, int(dim.item()))
        else:
            return t1.f.add_reduce(
                t1.contiguous().view(int(operators.prod(list(t1.shape)))), 0
            )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass derivative for sum"""
        t1, dim = ctx.saved_tensors  # Retrieve the original tensor and dim
        gradient_tensor = zeros(t1._tensor.shape)

        # Expand grad_output to match the shape of the original tensor
        expanded_grad_output = t1.expand(grad_output)
        gradient_tensor += expanded_grad_output

        return gradient_tensor


class Mean(Function):
    """Calculates the mean function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim: Optional[Tensor] = None) -> Tensor:
        """Forward pass mean function"""
        ctx.save_for_backward(t1, dim)
        if dim is not None:
            sum_result = t1.f.add_reduce(t1, int(dim.item()))
            return sum_result / t1.shape[int(dim.item())]
        else:
            total_elements = t1.size
            sum_result = t1.f.add_reduce(
                t1.contiguous().view(int(operators.prod(list(t1.shape)))), 0
            )
            return sum_result / total_elements

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass derivative for sum"""
        t1, dim = ctx.saved_tensors  # Retrieve the original tensor and dim
        gradient_tensor = zeros(t1._tensor.shape)  # Initialize the gradient tensor

        if dim is not None:
            # Expand grad_output and scale by the number of elements in the dimension
            expanded_grad_output = t1.expand(
                grad_output
            )  # Expand the gradient to match the original tensor
            scale_factor = t1.shape[
                int(dim.item())
            ]  # Number of elements along the reduced dimension
            gradient_tensor += expanded_grad_output / scale_factor  # Scale by 1/N
        else:
            # If no dimension is specified, scale by the total number of elements
            total_elements = int(
                operators.prod(list(t1.shape))
            )  # Total number of elements
            expanded_grad_output = t1.expand(
                grad_output
            )  # Expand to match the original tensor
            gradient_tensor += (
                expanded_grad_output / total_elements
            )  # Scale by 1/N for the entire tensor

        return gradient_tensor


class LT(Function):
    """Checks if one tensor is less than another at each index."""

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Forward pass less than operation"""
        return a.f.lt_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass derivative for less than"""
        return grad_output.zeros(), grad_output.zeros()


class EQ(Function):
    """Checks if two tensors are equal."""

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Forward pass equal operator"""
        return a.f.eq_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass derivative for equivalence"""
        return grad_output.zeros(), grad_output.zeros()


class IsClose(Function):
    """Checks if two tensors are close in value."""

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Forward pass is close operator"""
        return a.f.is_close_zip(a, b)


class Permute(Function):
    """Modifies dimensions of tensor."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim: Tensor) -> Tensor:
        """Forward pass permute"""
        dim_tuple = [int(i) for i in dim]

        ctx.save_for_backward(dim_tuple)
        t1._tensor.permute(*dim_tuple)
        return t1

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for permute"""
        dim_tuple = ctx.saved_tensors[0]

        # Get the inverse of the permutation indices
        inv_dim_tuple = [0] * len(dim_tuple)
        for i, d in enumerate(dim_tuple):
            inv_dim_tuple[d] = i

        # Initialize the gradient for the input tensor
        grad_output._tensor.permute(*inv_dim_tuple)  # Permute the gradient back

        return grad_output


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """Forward pass of view function"""
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Matrix Multiply backward (module 3)"""
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(list(shape))), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(list(shape))))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Compute gradient with central difference method."""
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
