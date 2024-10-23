"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, List

# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(a: float, b: float) -> float:
    """Multipy two floats together."""
    return a * b


def id(input: float) -> float:
    """Returns input unchanged."""
    return input


def add(a: float, b: float) -> float:
    """Adds two numbers."""
    return a + b


def neg(a: float) -> float:
    """Negates a number."""
    return -a


def lt(a: float, b: float) -> float:
    """Checks if one number is less than another."""
    return 1.0 if a < b else 0.0


def eq(a: float, b: float) -> float:
    """Checks if two numbers are equal."""
    return 1.0 if a == b else 0.0


def max(a: float, b: float) -> float:
    """Returns the larger to two numbers."""
    return a if a > b else b


def is_close(a: float, b: float) -> float:
    """Checks if two numbers are close in value."""
    return (a - b < 1e-2) and (b - a < 1e-2)


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Applies the ReLU activation function."""
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Calculates the natural logarithm."""
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Calculates the exponential function."""
    return math.exp(x)


def inv(x: float) -> float:
    """Calculates the reciprocal."""
    return 1.0 / x


def log_back(x: float, d: float) -> float:
    """Computes the derivative of log times a second arg."""
    # derivative of log(x) is 1/x
    # 1/x * d = d/x
    return d / (x + EPS)


def inv_back(x: float, d: float) -> float:
    """Computes the derivative of reciprocal times a second arg."""
    # derivative of 1/x is -1/(x ** 2)
    return -(1.0 / x**2) * d


def relu_back(x: float, d: float) -> float:
    """Computes the derivative of ReLU times a second arg."""
    return d if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(func: Callable[[float], float], lst: list) -> list:
    """Higher-order function that applies a given function to each element of an iterable."""
    return [func(x) for x in lst]


def zipWith(func: Callable[[float, float], float], lst1: list, lst2: list) -> list:
    """Higher-order function that combines elements from two iterables using a given function."""
    result = []
    for i in range(len(lst1)):
        result.append(func(lst1[i], lst2[i]))
    return result


def reduce(func: Callable[[float, float], float], lst: list, initial: float) -> float:
    """Higher-order function that reduces an iterable to a single value using a given function."""
    result = initial
    for item in lst:
        result = func(result, item)
    return result


def negList(lst: List[float]) -> List[float]:
    """Negate all elements in a list using map."""
    return map(lambda x: -x, lst)


def addLists(lst1: List[float], lst2: List[float]) -> List[float]:
    """Add corresponding elements from two lists using zipWith."""
    return zipWith(lambda x, y: x + y, lst1, lst2)


def sum(lst: List[float]) -> float:
    """Sum all elements in a list using reduce."""
    return reduce(lambda x, y: x + y, lst, 0.0)


def prod(lst: List[float]) -> float:
    """Calculate the product of all elements in a list using reduce."""
    return reduce(lambda x, y: x * y, lst, 1.0)
