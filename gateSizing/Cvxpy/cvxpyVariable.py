import cvxpy as cp
import numpy as np
import sys

sys.path.append('../Numpy')
import histogramGenerator

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

    return convolution

    # self.cutBins(self.edges, convolution)     # todo: cut bins when edges interval does not start with 0


def convolutionCVXPY_CONSTANT(x1: {cp.Expression}, x2: {cp.Expression}) -> {cp.Expression}:
    """
    Calculates convolution of 2 PDFs of cvxpy variable

    :param x1: dictionary with cvxpy variables (1, 1)
    :param x2: dictionary with cvxpy variables (1, 1)
    :return convolution:  dictionary with cvxpy variables (1, 1)
    """

    size = len(x1.values())

    const = histogramGenerator.get_gauss_bins(8, 0.45, 5, 1000000, (0, 20)).bins

    convolution = {}
    for z in range(0, size):
        convolution[z] = 0

    for z in range(0, size):
        for k in range(0, z + 1):
            convolution[z] += x1[k] * const[z - k]

    return convolution



def maximumCVXPY_QUAD_UNIFIED(x1: {cp.Expression}, x2: {cp.Expression}) -> {cp.Expression}:
    """
    Calculates maximum of 2 PDFs of cvxpy variable. Works only for 2 identical edges. Is computed
    using the 'quadratic' algorithm and unified representation of bins - M 0/1-bins for each bin.

    :param x1: dictionary of dictionary with cvxpy variables (1, 1)
    :param x2: dictionary of dictionary with cvxpy variables (1, 1)
    :return maximum:  dictionary of dictionary with cvxpy slack variables (1, 1)
    :return MaxConstraints: python array with inequalities - for computing the maximum
    """

    numberOfBins = len(x1.values())
    numberOfUnions = len(x1[0].values())

    maximum = {}
    # MaxConstraints = [True] * numberOfBins * numberOfBins * 3 *  numberOfUnions # allocation
    MaxConstraints = []

        # allocation of maximum
    for i in range(0, numberOfBins):
        maximum[i] = {}
        for union in range(0, numberOfUnions):
            (maximum[i])[union] = 0


    # i >= j
    for i in range(0, numberOfBins):
        for j in range(0, i+1):

            for union in range(0, numberOfUnions):

                # new variable - multiplication of x*y
                slackMult = cp.Variable(boolean=True)
                (maximum[i])[union] += slackMult

                # help constraints
                x = (x1[i])[union]
                y = (x2[j])[union]

                MaxConstraints.append(  slackMult <= x          )
                MaxConstraints.append(  slackMult <= y          )
                MaxConstraints.append(  slackMult >= x + y - 1  )      # driving constr.


    # i < j
    for i in range(0, numberOfBins):
        for j in range(i+1, numberOfBins):

            for union in range(0, numberOfUnions):

                # new variable - multiplication of x*y
                slackMult = cp.Variable(boolean=True)
                (maximum[j])[union] += slackMult

                # help constraints
                x = (x1[i])[union]
                y = (x2[j])[union]

                MaxConstraints.append(  slackMult <= x          )
                MaxConstraints.append(  slackMult <= y          )
                MaxConstraints.append(  slackMult >= x + y - 1  )  # driving constr.


    return maximum, MaxConstraints



def maximumCVXPY_ELEMENTWISE(x1: {cp.Expression}, x2: {cp.Expression}) -> {cp.Expression}:
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


