import cvxpy as cp
import numpy as np

"""
This module includes functions for cvxpy variables, such as maximum or convolution

"""


def convolutionCVXPY(x1: cp.Expression, x2: cp.Expression) -> cp.Expression:
    """
    Calculates convolution of 2 PDFs of cvxpy variable

    :param x1: cvxpy variable (1, m)
    :param x2: cvxpy variable (1, m)
    :return convolution:  cvxpy variable (1, m)
    """

    return x2 + 10000

    # return cp.multiply(x1, x2)

    # convolution = cp.conv(x1, x2)
    #
    # return convolution[0]

def maximumCVXPY(x1: cp.Expression, x2: cp.Expression) -> cp.Expression:
    """
    Calculates maximum of 2 PDFs of cvxpy variable

    :param x1: cvxpy variable (1, m)
    :param x2: cvxpy variable (1, m)
    :return maximum:  cvxpy variable (1, m)
    """

    # m = x1.size
    # maximum = cp.Variable(x1.shape)
    #
    # for i in range(0, m):
    #     F2 = cp.sum(x2[:i+1])
    #     F1 = cp.sum(x1[:i])                 # only for discrete - not to count with 1 number twice
    #     maximum[i] = cp.hstack( x1[i] * F2 + x2[i] * F1 )

    maximum = cp.maximum(x1, x2)

    return maximum

