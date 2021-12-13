import numpy as np
import cvxpy as cp

import sys
# setting path

sys.path.append('../Numpy')

from node import Node
import histogramGenerator
import SSTA
import networkx as nx
from randomVariableHist import RandomVariable
import matplotlib.pyplot as plt

sys.path.append('../../montecarlo')

from montecarlo import get_inputs, get_unknown_nodes, simulation, preprocess

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
    numberOfBins = 1000
    distribution = 'Normal'
    binsInterval = (-20, 100)

    # create cvxpy variable

    x = cp.Variable((5, numberOfBins-1))

        # ACTUAL - ssta

    g1 = histogramGenerator.get_gauss_bins(10, 0.45, numberOfBins, numberOfSamples, binsInterval)  # g1, g2 INPUT gates, g3 middle
    g2 = histogramGenerator.get_gauss_bins(12, 0.3, numberOfBins, numberOfSamples, binsInterval)  # g4 output - inputs: g3 g1
    g3 = histogramGenerator.get_gauss_bins(5, 0.5, numberOfBins, numberOfSamples, binsInterval)  # g5 output - inputs: g3, g2
    g4 = histogramGenerator.get_gauss_bins(5, 0.5, numberOfBins, numberOfSamples, binsInterval)
    g5 = histogramGenerator.get_gauss_bins(5, 0.5, numberOfBins, numberOfSamples, binsInterval)

    n1 = Node(x[0, :])
    n2 = Node(x[1, :])
    n3 = Node(x[2, :])
    n4 = Node(x[3, :])
    n5 = Node(x[4, :])

    # set circuit design
    n1.setNextNodes([n3, n4])
    n2.setNextNodes([n3, n5])
    n3.setNextNodes([n4, n5])

    # calculate delay with ssta
    delays = SSTA.calculateCircuitDelay([n1, n2], cvxpy=True)

    print(delays)

    # solve
    constraints = [x[0, :] >= g1.bins, x[1, :] >= g2.bins, x[2, :] >= g3.bins, x[3, :] >= g4.bins, x[4, :] >= g5.bins]
    objective = cp.Minimize( cp.sum( cp.sum(delays) ))
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=True, solver=cp.MOSEK)

    print("prob value: ", prob.value)
    print("x: ", x.value)

    # get means and stds
    rv1 = RandomVariable(x.value[0, :], g1.edges)
    rv2 = RandomVariable(x.value[1, :], g2.edges)
    rv3 = RandomVariable(x.value[2, :], g3.edges)
    rv4 = RandomVariable(x.value[3, :], g4.edges)
    rv5 = RandomVariable(x.value[4, :], g5.edges)

    # print out the results

    print(rv1.mean, rv1.std)
    print(rv2.mean, rv2.std)
    print(rv3.mean, rv3.std)
    print(rv4.mean, rv4.std)
    print(rv5.mean, rv5.std)

    return None


if __name__ == "__main__":

        # dec param is Desired precision

    # testSSTA_1(dec=1)
    SSTA_CVXPY(dec=5)

