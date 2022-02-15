import cvxpy.atoms.affine.affine_atom
import numpy as np
import cvxpy as cp

import sys
# setting path
import cvxpyVariable

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

'''
    Puts means and stds of either array of rvs or of array of numbers from monte carlo into 
        [n, 2] numpy array

'''


def putTuplesIntoArray(rvs: [RandomVariable] = None, numbers: [float] = None):
    if rvs != None:
        actual = np.zeros((len(rvs), 2))

        size = len(rvs)
        for i in range(0, size):
            delay = rvs[i]
            actual[i, 0] = delay.mean
            actual[i, 1] = delay.std

    else:
        actual = np.zeros((len(numbers) - 1, 2))

        size = len(numbers) - 1
        for i in range(0, size):
            delay = numbers[i + 1]
            actual[i, 0] = np.mean(delay)
            actual[i, 1] = np.std(delay)

    return actual


def SSTA_CVXPY(dec: int):

    numberOfSamples = 1000000
    numberOfBins = 5
    numberOfGates = 5
    binsInterval = (0, 20)

    # create cvxpy variable

    xs = {}

    # create a variable as a dict.
    for gate in range(0, numberOfGates):
        xs[gate] = {}
        for bin in range(0, numberOfBins):
            (xs[gate])[bin] = cp.Variable(nonneg=True)

        # ACTUAL - ssta


    g1 = histogramGenerator.get_gauss_bins(8, 0.45, numberOfBins, numberOfSamples, binsInterval)  # g1, g2 INPUT gates, g3 middle
    g2 = histogramGenerator.get_gauss_bins(12, 0.3, numberOfBins, numberOfSamples, binsInterval)  # g4 output - inputs: g3 g1
    g3 = histogramGenerator.get_gauss_bins(5, 0.5, numberOfBins, numberOfSamples, binsInterval)  # g5 output - inputs: g3, g2
    g4 = histogramGenerator.get_gauss_bins(5, 0.5, numberOfBins, numberOfSamples, binsInterval)
    g5 = histogramGenerator.get_gauss_bins(5, 0.5, numberOfBins, numberOfSamples, binsInterval)
    generatedDistros = [g1, g2, g3, g4, g5]


    n1 = Node(xs[0])
    n2 = Node(xs[1])
    n3 = Node(xs[2])
    n4 = Node(xs[3])
    n5 = Node(xs[4])

    # set circuit design
    n1.setNextNodes([n3, n4])
    n2.setNextNodes([n3, n5])
    n3.setNextNodes([n4, n5])

    # calculate delay with ssta
    delays, constraints = SSTA.calculateCircuitDelay([n1, n2], cvxpy=True)

    # set constraints

    for gate in range(0, numberOfGates):
        for bin in range(0, numberOfBins):
            constraints.append( (xs[gate])[bin] >= generatedDistros[gate].bins[bin] )    # set lower constr.

        # old constr.
    # constraints = [x[0, :] >= g1.bins, x[1, :] >= g2.bins, x[2, :] >= g3.bins, x[3, :] >= g4.bins, x[4, :] >= g5.bins]

    # set objective

    sum = 0
    for gate in range(0, numberOfGates):
        for bin in range(0, numberOfBins):
           sum += (delays[gate])[bin]

    # solve
    objective = cp.Minimize( sum )
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=True, solver=cp.MOSEK)



    # print out the values
    print("PROBLEM VALUE: ", prob.value)

    rvs = []

    for gate in range(0, numberOfGates):    # construct RVs

        bins = [0] * numberOfBins

        for bin in range(0, numberOfBins):
            bins[bin] = (xs[gate])[bin].value

        rvs.append( RandomVariable(bins, generatedDistros[gate].edges) )


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

    actual = np.zeros((len(delays), 2))
    size = len(delays)
    for i in range(0, size):
        delay = delays[i]
        actual[i, 0] = delay.mean
        actual[i, 1] = delay.std


    print("\n REAL VALUES: \n")
    for i in range(0, numberOfGates):
        print( actual[i, 0], actual[i, 1] )


    return None



def maxOfDistributionsCVXPY(delays: [cp.Expression]) -> cp.Variable:
    """
    Calculates maximum of an array of PDFs of cvxpy variable

    :param delays: array of cvxpy variables (n, m), n gates, m bins
    :return maximum:  cvxpy variable (1, m)
    """

    size = len(delays)
    for i in range(0, size - 1):
        newRV = cvxpyVariable.maximumCVXPY(delays[i], delays[i + 1])
        delays[i + 1] = newRV

    maximum = delays[-1]

    return maximum




def maxOfDistributionsELEMENTWISE(delays: [RandomVariable]) -> RandomVariable:
    """
    Calculates maximum of an array of PDFs of cvxpy variable
    Using elementwise maximum - np.maximum

    :param delays: array of RandomVariables
    :return maximum: RandomVariable - maximum delay
    """

    size = len(delays)
    for i in range(0, size - 1):
        newRV = delays[i].maxOfDistributionsELEMENTWISE(delays[i + 1])
        delays[i + 1] = newRV

    maximum = delays[-1]

    return maximum



def maxOfDistributionsFORM(delays: [RandomVariable]) -> RandomVariable:
    """
    Calculates maximum of an array of PDFs of cvxpy variable
    Using formula - look up function maxOfDistributionsFORM

    :param delays: array of RandomVariables
    :return maximum: RandomVariable - maximum delay
    """


    size = len(delays)

    for i in range(0, size - 1):
        newRV = delays[i].maxOfDistributionsFORM(delays[i + 1])
        delays[i + 1] = newRV

    maximum = delays[-1]

    return maximum


def maxOfDistributionsQUAD(delays: [RandomVariable]) -> RandomVariable:
    """
    Calculates maximum of an array of PDFs of cvxpy variable
    Using maxOfDistributionsQUAD, quadratic algorithm

    :param delays: array of RandomVariables
    :return maximum: RandomVariable - maximum delay
    """

    size = len(delays)

    for i in range(0, size - 1):

        newRV = delays[i].maxOfDistributionsQUAD(delays[i + 1])
        delays[i + 1] = newRV

    max = delays[-1]

    return max


def putIntoQueue(queue: Queue, list: [Node]) -> None:
    """
    Function puts list into queue.

    :param queue: Queue
    :return list: array of Node class
    """

    for item in list:
        queue.put(item)



if __name__ == "__main__":

        # dec param is Desired precision

    # testSSTA_1(dec=1)
    SSTA_CVXPY(dec=5)

