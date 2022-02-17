import cvxpy.atoms.affine.affine_atom
import numpy as np
import cvxpy as cp

import sys
# setting path
from cvxpyVariable import RandomVariableCVXPY

sys.path.append('../Numpy')

from node import Node
import histogramGenerator
import SSTA
import networkx as nx
from randomVariableHist import RandomVariable
import matplotlib.pyplot as plt

sys.path.append('../../montecarlo')

from montecarlo import get_inputs, get_unknown_nodes, simulation, preprocess
from queue import Queue




def SSTA_CVXPY_UNIFIED_AS_MIN(dec: int):
    """
    Formulated as minimization problem
    :param dec:
    :return:
    """

    numberOfSamples = 1000000

    numberOfGates = 5

    numberOfBins = 12
    numberOfUnions = 10

    binsInterval = (-5, 40)

    # create cvxpy variable

    xs = {}

    # create a variable as a dict.
    for gate in range(0, numberOfGates):
        xs[gate] = {}
        for bin in range(0, numberOfBins):
            (xs[gate])[bin] = {}
            for union in range(0, numberOfUnions):
                ((xs[gate])[bin])[union] = cp.Variable(boolean=True)

        # ACTUAL - ssta


    g1 = histogramGenerator.get_gauss_bins(10, 0.45, numberOfBins, numberOfSamples, binsInterval)  # g1, g2 INPUT gates, g3 middle
    g2 = histogramGenerator.get_gauss_bins(12, 0.3, numberOfBins, numberOfSamples, binsInterval)  # g4 output - inputs: g3 g1
    g3 = histogramGenerator.get_gauss_bins(5, 3, numberOfBins, numberOfSamples, binsInterval)  # g5 output - inputs: g3, g2
    g4 = histogramGenerator.get_gauss_bins(5, 2, numberOfBins, numberOfSamples, binsInterval)
    g5 = histogramGenerator.get_gauss_bins(5, 0.5, numberOfBins, numberOfSamples, binsInterval)

    g1U = histogramGenerator.get_gauss_bins_UNIFIED(10, 0.45, numberOfBins, numberOfSamples,binsInterval, numberOfUnions)
    g2U = histogramGenerator.get_gauss_bins_UNIFIED(12, 0.3, numberOfBins, numberOfSamples,binsInterval, numberOfUnions)
    g3U = histogramGenerator.get_gauss_bins_UNIFIED(5, 3, numberOfBins, numberOfSamples,binsInterval, numberOfUnions)
    g4U = histogramGenerator.get_gauss_bins_UNIFIED(5, 2, numberOfBins, numberOfSamples, binsInterval, numberOfUnions)
    g5U = histogramGenerator.get_gauss_bins_UNIFIED(5, 0.5, numberOfBins, numberOfSamples, binsInterval, numberOfUnions)
    generatedDistros = [g1U, g2U, g3U, g4U, g5U]


    n1 = Node( RandomVariableCVXPY(xs[0], g1.edges) )
    n2 = Node( RandomVariableCVXPY(xs[1], g1.edges) )
    n3 = Node( RandomVariableCVXPY(xs[2], g1.edges) )
    n4 = Node( RandomVariableCVXPY(xs[3], g1.edges) )
    n5 = Node( RandomVariableCVXPY(xs[4], g1.edges) )

    # set circuit design
    n1.setNextNodes([n3, n4])
    n2.setNextNodes([n3, n5])
    n3.setNextNodes([n4, n5])

    # calculate delay with ssta
    delays, constraints = SSTA.calculateCircuitDelay([n1, n2], cvxpy=True, unified=True)

    # set constraints

    for gate in range(0, numberOfGates):
        for bin in range(0, numberOfBins):
            for union in range(0, numberOfUnions):
                constraints.append( ((xs[gate])[bin])[union] >= (generatedDistros[gate].bins[bin])[union] )    # set lower constr.

        # old constr.
    # constraints = [x[0, :] >= g1.bins, x[1, :] >= g2.bins, x[2, :] >= g3.bins, x[3, :] >= g4.bins, x[4, :] >= g5.bins]

    # set objective

    sum = 0
    for gate in range(0, numberOfGates):
        for bin in range(0, numberOfBins):
            for union in range(0, numberOfUnions):
               sum += ((delays[gate].bins)[bin])[union]

    # solve
    objective = cp.Minimize( sum )
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=True, solver=cp.MOSEK)



    # print out the values
    print("PROBLEM VALUE: ", prob.value)

    rvs = []

    for gate in range(0, numberOfGates):    # construct RVs

        finalBins = np.zeros((numberOfBins, numberOfUnions))
        for bin in range(0, numberOfBins):
            for union in range(0, numberOfUnions):
                finalBins[bin, union] = ((delays[gate].bins)[bin])[union].value

        rvs.append( RandomVariable(finalBins, generatedDistros[gate].edges, unified=True) )


    print("\n APRX. VALUES: \n")
    for i in range(0, numberOfGates):
        print( rvs[i].mean, rvs[i].std )


    # calculate with numpy

    n1 = Node(g1)
    n2 = Node(g2)
    n3 = Node(g3)
    n4 = Node(g4)
    n5 = Node(g5)

    # set circuit design
    n1.setNextNodes([n3, n4])
    n2.setNextNodes([n3, n5])
    n3.setNextNodes([n4, n5])

    delays = SSTA.calculateCircuitDelay([n1, n2])

    desired = np.zeros((len(delays), 2))
    size = len(delays)
    for i in range(0, size):
        delay = delays[i]
        desired[i, 0] = delay.mean
        desired[i, 1] = delay.std


    print("\n REAL VALUES: \n")
    for i in range(0, numberOfGates):
        print( desired[i, 0], desired[i, 1] )


        # test whole
    # np.testing.assert_almost_equal(desired, actual, decimal=dec, err_msg= "Monte Carlo: \n" + str(desired) + '\n\n' + "SSTA: \n" + str(actual))


    return None


def SSTA_CVXPY_UNIFIED_AS_MAX(dec: int):
    """
    Formulated as minimization problem
    :param dec:
    :return:
    """

    numberOfSamples = 1000000

    numberOfGates = 5

    numberOfBins = 16
    numberOfUnions = 20

    binsInterval = (-5, 40)

    # create cvxpy variable

    xs = {}

    # create a variable as a dict.
    for gate in range(0, numberOfGates):
        xs[gate] = {}
        for bin in range(0, numberOfBins):
            (xs[gate])[bin] = {}
            for union in range(0, numberOfUnions):
                ((xs[gate])[bin])[union] = cp.Variable(boolean=True)

        # ACTUAL - ssta


    g1 = histogramGenerator.get_gauss_bins(10, 0.45, numberOfBins, numberOfSamples, binsInterval)  # g1, g2 INPUT gates, g3 middle
    g2 = histogramGenerator.get_gauss_bins(12, 0.3, numberOfBins, numberOfSamples, binsInterval)  # g4 output - inputs: g3 g1
    g3 = histogramGenerator.get_gauss_bins(5, 2, numberOfBins, numberOfSamples, binsInterval)  # g5 output - inputs: g3, g2
    g4 = histogramGenerator.get_gauss_bins(5, 0.5, numberOfBins, numberOfSamples, binsInterval)
    g5 = histogramGenerator.get_gauss_bins(5, 3, numberOfBins, numberOfSamples, binsInterval)

    g1U = histogramGenerator.get_gauss_bins_UNIFIED(10, 0.45, numberOfBins, numberOfSamples,binsInterval, numberOfUnions)
    g2U = histogramGenerator.get_gauss_bins_UNIFIED(12, 0.3, numberOfBins, numberOfSamples,binsInterval, numberOfUnions)
    g3U = histogramGenerator.get_gauss_bins_UNIFIED(5, 2, numberOfBins, numberOfSamples,binsInterval, numberOfUnions)
    g4U = histogramGenerator.get_gauss_bins_UNIFIED(5, 0.5, numberOfBins, numberOfSamples, binsInterval, numberOfUnions)
    g5U = histogramGenerator.get_gauss_bins_UNIFIED(5, 3, numberOfBins, numberOfSamples, binsInterval, numberOfUnions)
    generatedDistros = [g1U, g2U, g3U, g4U, g5U]


    n1 = Node( RandomVariableCVXPY(xs[0], g1.edges) )
    n2 = Node( RandomVariableCVXPY(xs[1], g1.edges) )
    n3 = Node( RandomVariableCVXPY(xs[2], g1.edges) )
    n4 = Node( RandomVariableCVXPY(xs[3], g1.edges) )
    n5 = Node( RandomVariableCVXPY(xs[4], g1.edges) )

    # set circuit design
    n1.setNextNodes([n3, n4])
    n2.setNextNodes([n3, n5])
    n3.setNextNodes([n4, n5])

    # calculate delay with ssta
    delays, constraints = SSTA.calculateCircuitDelay([n1, n2], cvxpy=True, unified=True)

    # set constraints

    for gate in range(0, numberOfGates):
        for bin in range(0, numberOfBins):
            for union in range(0, numberOfUnions):
                constraints.append( ((xs[gate])[bin])[union] <= (generatedDistros[gate].bins[bin])[union] )    # set lower constr.


    # set objective

    sum = 0
    for gate in range(0, numberOfGates):
        for bin in range(0, numberOfBins):
            for union in range(0, numberOfUnions):
               sum += ((delays[gate].bins)[bin])[union]

    # solve
    objective = cp.Maximize( sum )
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=True, solver=cp.MOSEK)



    # print out the values
    print("PROBLEM VALUE: ", prob.value)

    rvs = []

    for gate in range(0, numberOfGates):    # construct RVs

        finalBins = np.zeros((numberOfBins, numberOfUnions))
        for bin in range(0, numberOfBins):
            for union in range(0, numberOfUnions):
                finalBins[bin, union] = ((delays[gate].bins)[bin])[union].value

        rvs.append( RandomVariable(finalBins, generatedDistros[gate].edges, unified=True) )


    print("\n APRX. VALUES: \n")
    for i in range(0, numberOfGates):
        print( rvs[i].mean, rvs[i].std )


    # calculate with numpy

    n1 = Node(g1)
    n2 = Node(g2)
    n3 = Node(g3)
    n4 = Node(g4)
    n5 = Node(g5)

    # set circuit design
    n1.setNextNodes([n3, n4])
    n2.setNextNodes([n3, n5])
    n3.setNextNodes([n4, n5])

    delays = SSTA.calculateCircuitDelay([n1, n2])

    desired = np.zeros((len(delays), 2))
    size = len(delays)
    for i in range(0, size):
        delay = delays[i]
        desired[i, 0] = delay.mean
        desired[i, 1] = delay.std


    print("\n REAL VALUES: \n")
    for i in range(0, numberOfGates):
        print( desired[i, 0], desired[i, 1] )


        # test whole
    # np.testing.assert_almost_equal(desired, actual, decimal=dec, err_msg= "Monte Carlo: \n" + str(desired) + '\n\n' + "SSTA: \n" + str(actual))


    return None



if __name__ == "__main__":

        # dec param is Desired precision

    # SSTA_CVXPY_UNIFIED_AS_MIN(dec=5)

        # one has to change functions in SSTA to change MAX and MIN
    SSTA_CVXPY_UNIFIED_AS_MAX(dec=5)
