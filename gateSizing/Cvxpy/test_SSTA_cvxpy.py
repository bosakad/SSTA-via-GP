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



def putTuplesIntoArray(rvs: [RandomVariable] = None, numbers: [float] = None):

    if rvs != None:
        actual = np.zeros( (len(rvs), 2))

        size = len(rvs)
        for i in range(0, size):
            delay = rvs[i]
            actual[i, 0] = delay.mean
            actual[i, 1] = delay.std

    else:
        actual = np.zeros( (len(numbers) - 1, 2) )

        size = len(numbers) - 1
        for i in range(0, size):
            delay = numbers[i + 1]
            actual[i, 0] = np.mean(delay)
            actual[i, 1] = np.std(delay)

    return actual

def SSTA_CVXPY_UNARY_AS_MIN(dec: int):
    """
    Formulated as minimization problem
    :param dec:
    :return:
    """

    numberOfSamples = 1000000

    numberOfGates = 5

    numberOfBins = 12
    numberOfUnaries = 10

    binsInterval = (-5, 40)

    # create cvxpy variable

    xs = {}

    # create a variable as a dict.
    for gate in range(0, numberOfGates):
        xs[gate] = {}
        for bin in range(0, numberOfBins):
            (xs[gate])[bin] = {}
            for unary in range(0, numberOfUnaries):
                ((xs[gate])[bin])[unary] = cp.Variable(boolean=True)

        # ACTUAL - ssta


    g1 = histogramGenerator.get_gauss_bins(10, 0.45, numberOfBins, numberOfSamples, binsInterval)  # g1, g2 INPUT gates, g3 middle
    g2 = histogramGenerator.get_gauss_bins(12, 0.3, numberOfBins, numberOfSamples, binsInterval)  # g4 output - inputs: g3 g1
    g3 = histogramGenerator.get_gauss_bins(5, 3, numberOfBins, numberOfSamples, binsInterval)  # g5 output - inputs: g3, g2
    g4 = histogramGenerator.get_gauss_bins(5, 2, numberOfBins, numberOfSamples, binsInterval)
    g5 = histogramGenerator.get_gauss_bins(5, 0.5, numberOfBins, numberOfSamples, binsInterval)

    g1U = histogramGenerator.get_gauss_bins_UNARY(10, 0.45, numberOfBins, numberOfSamples, binsInterval, numberOfUnaries)
    g2U = histogramGenerator.get_gauss_bins_UNARY(12, 0.3, numberOfBins, numberOfSamples, binsInterval, numberOfUnaries)
    g3U = histogramGenerator.get_gauss_bins_UNARY(5, 3, numberOfBins, numberOfSamples, binsInterval, numberOfUnaries)
    g4U = histogramGenerator.get_gauss_bins_UNARY(5, 2, numberOfBins, numberOfSamples, binsInterval, numberOfUnaries)
    g5U = histogramGenerator.get_gauss_bins_UNARY(5, 0.5, numberOfBins, numberOfSamples, binsInterval, numberOfUnaries)
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
    delays, constraints = SSTA.calculateCircuitDelay([n1, n2], cvxpy=True, unary=True)

    # set constraints

    for gate in range(0, numberOfGates):
        for bin in range(0, numberOfBins):
            for unary in range(0, numberOfUnaries):
                constraints.append( ((xs[gate])[bin])[unary] >= (generatedDistros[gate].bins[bin])[unary] )    # set lower constr.

        # old constr.
    # constraints = [x[0, :] >= g1.bins, x[1, :] >= g2.bins, x[2, :] >= g3.bins, x[3, :] >= g4.bins, x[4, :] >= g5.bins]

    # set objective

    sum = 0
    for gate in range(0, numberOfGates):
        for bin in range(0, numberOfBins):
            for unary in range(0, numberOfUnaries):
               sum += ((delays[gate].bins)[bin])[unary]

    # solve
    objective = cp.Minimize( sum )
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=True, solver=cp.MOSEK)



    # print out the values
    print("PROBLEM VALUE: ", prob.value)

    rvs = []

    for gate in range(0, numberOfGates):    # construct RVs

        finalBins = np.zeros((numberOfBins, numberOfUnaries))
        for bin in range(0, numberOfBins):
            for unary in range(0, numberOfUnaries):
                finalBins[bin, unary] = ((delays[gate].bins)[bin])[unary].value

        rvs.append( RandomVariable(finalBins, generatedDistros[gate].edges, unary=True) )


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


def SSTA_CVXPY_UNARY_AS_MAX(dec: int):
    """
    Formulated as minimization problem
    :param dec:
    :return:
    """

    numberOfSamples = 2000000

    numberOfGates = 5

    numberOfBins = 10
    numberOfUnaries = 10

    binsInterval = (-5, 40)

    # create cvxpy variable

    xs = {}

    # create a variable as a dict.
    for gate in range(0, numberOfGates):
        xs[gate] = {}
        for bin in range(0, numberOfBins):
            (xs[gate])[bin] = {}
            for unary in range(0, numberOfUnaries):
                ((xs[gate])[bin])[unary] = cp.Variable(boolean=True)

        # ACTUAL - ssta


    g1 = histogramGenerator.get_gauss_bins(10, 0.45, numberOfBins, numberOfSamples, binsInterval)  # g1, g2 INPUT gates, g3 middle
    g2 = histogramGenerator.get_gauss_bins(12, 0.3, numberOfBins, numberOfSamples, binsInterval)  # g4 output - inputs: g3 g1
    g3 = histogramGenerator.get_gauss_bins(5, 2, numberOfBins, numberOfSamples, binsInterval)  # g5 output - inputs: g3, g2
    g4 = histogramGenerator.get_gauss_bins(5, 0.5, numberOfBins, numberOfSamples, binsInterval)
    g5 = histogramGenerator.get_gauss_bins(5, 3, numberOfBins, numberOfSamples, binsInterval)

    g1U = histogramGenerator.get_gauss_bins_UNARY(10, 0.45, numberOfBins, numberOfSamples, binsInterval, numberOfUnaries)
    g2U = histogramGenerator.get_gauss_bins_UNARY(12, 0.3, numberOfBins, numberOfSamples, binsInterval, numberOfUnaries)
    g3U = histogramGenerator.get_gauss_bins_UNARY(5, 2, numberOfBins, numberOfSamples, binsInterval, numberOfUnaries)
    g4U = histogramGenerator.get_gauss_bins_UNARY(5, 0.5, numberOfBins, numberOfSamples, binsInterval, numberOfUnaries)
    g5U = histogramGenerator.get_gauss_bins_UNARY(5, 3, numberOfBins, numberOfSamples, binsInterval, numberOfUnaries)
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
    delays, constraints = SSTA.calculateCircuitDelay([n1, n2], cvxpy=True, unary=True)

    # set constraints

    for gate in range(0, numberOfGates):
        for bin in range(0, numberOfBins):
            for unary in range(0, numberOfUnaries):
                constraints.append( ((xs[gate])[bin])[unary] <= (generatedDistros[gate].bins[bin])[unary] )    # set lower constr.


    # set objective

    sum = 0
    for gate in range(0, numberOfGates):
        for bin in range(0, numberOfBins):
            for unary in range(0, numberOfUnaries):
               sum += ((delays[gate].bins)[bin])[unary]

    # solve
    objective = cp.Maximize( sum )
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=True, solver=cp.MOSEK)



    # print out the values
    print("PROBLEM VALUE: ", prob.value)

    rvs = []

    for gate in range(0, numberOfGates):    # construct RVs

        finalBins = np.zeros((numberOfBins, numberOfUnaries))
        for bin in range(0, numberOfBins):
            for unary in range(0, numberOfUnaries):
                finalBins[bin, unary] = ((delays[gate].bins)[bin])[unary].value

        rvs.append( RandomVariable(finalBins, generatedDistros[gate].edges, unary=True) )


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

def SSTA_CVXPY_McCormick(dec: int):
    """
    Formulated as minimization problem
    :param dec:
    :return:
    """

    numberOfSamples = 2000000
    numberOfGates = 5

    numberOfBins = 18

    binsInterval = (-2, 30)

    # create cvxpy variable

    xs = {}

    # create a variable as a dict.
    for gate in range(0, numberOfGates):
        xs[gate] = {}
        for bin in range(0, numberOfBins):
            (xs[gate])[bin] = cp.Variable(nonneg=True)

        # ACTUAL - ssta


    g1 = histogramGenerator.get_gauss_bins(10, 0.45, numberOfBins, numberOfSamples, binsInterval)  # g1, g2 INPUT gates, g3 middle
    g2 = histogramGenerator.get_gauss_bins(12, 0.3, numberOfBins, numberOfSamples, binsInterval)  # g4 output - inputs: g3 g1
    g3 = histogramGenerator.get_gauss_bins(5, 0.5, numberOfBins, numberOfSamples, binsInterval)  # g5 output - inputs: g3, g2
    g4 = histogramGenerator.get_gauss_bins(5, 0.5, numberOfBins, numberOfSamples, binsInterval)
    g5 = histogramGenerator.get_gauss_bins(5, 0.5, numberOfBins, numberOfSamples, binsInterval)
    generatedDistros = [g1, g2, g3, g4, g5]

    n1 = Node( RandomVariableCVXPY(xs[0], g1.edges, g1.bins) )
    n2 = Node( RandomVariableCVXPY(xs[1], g1.edges, g2.bins) )
    n3 = Node( RandomVariableCVXPY(xs[2], g1.edges, g3.bins) )
    n4 = Node( RandomVariableCVXPY(xs[3], g1.edges, g4.bins) )
    n5 = Node( RandomVariableCVXPY(xs[4], g1.edges, g5.bins) )


    # n1 = Node( RandomVariableCVXPY(xs[0], g1.edges) )
    # n2 = Node( RandomVariableCVXPY(xs[1], g1.edges) )
    # n3 = Node( RandomVariableCVXPY(xs[2], g1.edges) )
    # n4 = Node( RandomVariableCVXPY(xs[3], g1.edges) )
    # n5 = Node( RandomVariableCVXPY(xs[4], g1.edges) )


    # set circuit design
    n1.setNextNodes([n3, n4])
    n2.setNextNodes([n3, n5])
    n3.setNextNodes([n4, n5])

    # calculate delay with ssta
    delays, constraints = SSTA.calculateCircuitDelay([n1, n2], cvxpy=True, unary=False)

    # set constraints

    # for gate in range(0, numberOfGates):
    #     for bin in range(0, numberOfBins):
    #         constraints.append( (xs[gate])[bin] >= generatedDistros[gate].bins[bin] )    # set lower constr.


    # set objective

    sum = 0
    for gate in range(0, numberOfGates + 1):
        for bin in range(0, numberOfBins):
               sum += (delays[gate].bins)[bin]

    # solve
    objective = cp.Minimize( sum )
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False, solver=cp.MOSEK)



    # print out the values
    print("PROBLEM VALUE: ", prob.value)

    rvs = []

    for gate in range(0, numberOfGates + 1):    # construct RVs

        finalBins = np.zeros(numberOfBins)
        for bin in range(0, numberOfBins):
                finalBins[bin] = (delays[gate].bins)[bin].value

        print(finalBins)
        rvs.append( RandomVariable(finalBins, g1.edges, unary=False) )


    print("\n APRX. VALUES: \n")
    for i in range(0, numberOfGates + 1):
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

    delays = SSTA.calculateCircuitDelay([n1, n2], unary=False, cvxpy=False)
    desired = putTuplesIntoArray(rvs=delays)
    #
    # desired = np.zeros((len(delays), 2))
    # size = len(delays)
    # for i in range(0, size):
    #     delay = delays[i]
    #     desired[i, 0] = delay.mean
    #     desired[i, 1] = delay.std


    print("\n NUMPY VALUES: \n")
    for i in range(0, numberOfGates + 1):
        print( desired[i, 0], desired[i, 1] )

    # monte carlo

    adjacency = np.array([[0, 1, 1, 0, 0, 0, 0],
                          [0, 0, 0, 1, 1, 0, 0],
                          [0, 0, 0, 1, 0, 1, 0],
                          [0, 0, 0, 0, 1, 1, 0],
                          [0, 0, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0, 0]])

    G = nx.from_numpy_matrix(adjacency, create_using=nx.DiGraph())

    list_of_inputs = get_inputs(adjacency)

    unknown_nodes = get_unknown_nodes(G, list_of_inputs)
    gate = [5, 0.5]

    input_means = [0, 10, 12]
    input_stds = [0, 0.45, 0.3]

    inputs_simulation = preprocess(list_of_inputs, input_means, input_stds, unknown_nodes, gate, numberOfSamples,
                                   'Normal')

    mc = simulation(G, inputs_simulation, unknown_nodes, gate, numberOfSamples)

    desired2 = putTuplesIntoArray(numbers=mc)

    print("\nMONTE CARLO VALUES (GROUND-TRUTH):")
    print(desired2)

        # test whole
    # np.testing.assert_almost_equal(desired, actual, decimal=dec, err_msg= "Monte Carlo: \n" + str(desired) + '\n\n' + "SSTA: \n" + str(actual))

    return None

if __name__ == "__main__":

        # dec param is Desired precision

    # SSTA_CVXPY_UNARY_AS_MIN(dec=5)

        # one has to change functions in SSTA to change MAX and MIN
    # SSTA_CVXPY_UNARY_AS_MAX(dec=5)

    SSTA_CVXPY_McCormick(dec=5)