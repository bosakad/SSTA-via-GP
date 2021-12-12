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

    delays = SSTA.calculateCircuitDelay([n1, n2])


    actual = putTuplesIntoArray(rvs=delays)

    print(actual)

    # testMCMax(mc, binsInterval, numberOfBins)

        # test whole
    # np.testing.assert_almost_equal(desired, actual, decimal=dec, err_msg= "Monte Carlo: \n" + str(desired) + '\n\n' + "SSTA: \n" + str(actual))

    return None


if __name__ == "__main__":

        # dec param is Desired precision

    # testSSTA_1(dec=1)
    SSTA_CVXPY(dec=5)

