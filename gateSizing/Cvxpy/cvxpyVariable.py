import cvxpy as cp
import numpy as np

"""
This module includes functions for cvxpy variables, such as maximum or convolution

"""


def convolutionCVXPY(x1: {cp.Expression}, x2: {cp.Expression}) -> {cp.Expression}:
    """
    Calculates convolution of 2 PDFs of cvxpy variable

    :param x1: dictionary with cvxpy variables (1, 1)
    :param x2: dictionary with cvxpy variables (1, 1)
    :return convolution:  dictionary with cvxpy variables (1, 1)
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

    # self.cutBins(self.edges, convolution)     # todo: cut bins when edges interval does not start with 0


def maximumCVXPY(x1: {cp.Expression}, x2: {cp.Expression}) -> {cp.Expression}:
    """
    Calculates maximum of 2 PDFs of cvxpy variable. Works only for 2 identical edges.

    :param x1: dictionary with cvxpy variables (1, 1)
    :param x2: dictionary with cvxpy variables (1, 1)
    :return maximum:  dictionary with cvxpy slack variables (1, 1)
    :return MaxConstraints: python array with inequalities - for computing the maximum
    """

    size = len(x1.values())
    maximum = {}
    MaxConstraints = [0 <= 0] * 2 *  size   # allocation

    for i in range(0, size):
        # maximum[i] = cp.maximum(x1[i], x2[i])     # old version
        slackMax = cp.Variable(nonneg=True)
        maximum[i] = slackMax
        MaxConstraints[2*i] = x1[i] <= slackMax
        MaxConstraints[2*i + 1] = x2[i] <= slackMax

    return maximum, MaxConstraints


