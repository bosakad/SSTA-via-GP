import cvxpy as cp
import numpy as np

"""
This module includes functions for cvxpy variables, such as maximum or convolution

"""


# def convolutionCVXPY(x1: cp.Expression, x2: cp.Expression) -> cp.Expression:
def convolutionCVXPY(x1: {cp.Expression}, x2: {cp.Expression}) -> {cp.Expression}:
    """
    Calculates convolution of 2 PDFs of cvxpy variable

    :param x1: cvxpy variable (1, m)
    :param x2: cvxpy variable (1, m)
    :return convolution:  cvxpy variable (1, m)
    """

    size = len(x1.values())

    convolution = {}
    for z in range(0, size):
        convolution[z] = 0

    for z in range(0, size):
        for k in range(0, z + 1):
            convolution[z] += x1[k] * x2[z - k]
            # convolution[z] += x1[k] + x2[z - k]

    return convolution


        # non dict
    # size = x1.size
    # convolution = [None] * size
    #
    # for z in range(0, size):
    #     for k in range(0, z + 1):
    #         convolution[z] += x1[k] - x2[z - k]

    # return cp.hstack(convolution)

def maximumCVXPY(x1: {cp.Expression}, x2: {cp.Expression}) -> {cp.Expression}:
    """
    Calculates maximum of 2 PDFs of cvxpy variable. Works only for 2 identical edges.

    :param x1: cvxpy variable (1, m)
    :param x2: cvxpy variable (1, m)
    :return maximum:  cvxpy variable (1, m)
    """


    size = len(x1.values())
    maximum = {}

    for i in range(0, size):
        maximum[i] = cp.maximum(x1[i], x2[i])

    return maximum

    # n = x1.size
    # maximum = [None] * n
    #
    # for i in range(0, n):
    #     for j in range(0, n):
    #
    #         if i >= j:
    #             maximum[i] += x1[i] * x2[j]
    #         elif i < j:
    #             maximum[j] += x1[i] * x2[j]

    # maximum = cp.maximum(x1, x2)

    return cp.hstack(maximum)

