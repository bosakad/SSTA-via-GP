import numpy as np

from node import Node
import histogramGenerator
import SSTA
import networkx as nx
from randomVariableHist import RandomVariable

import sys
# setting path
sys.path.append('../montecarlo')

from montecarlo import get_inputs, get_unknown_nodes, simulation, preprocess


'''
    Puts means and stds of either array of rvs or of array of numbers from monte carlo into 
        [n, 2] numpy array
    
'''
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


def testSSTA_1(dec: int):

    numberOfSamples = int(1000000)
    numberOfBins = 500
    distribution = 'Normal'
    binsInterval = (-1, 8)

        # DESIRED - monte carlo

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
    gate = [0.5, 0.5]

    input_means = [0, 1, 0.5]
    input_stds = [0, 0.45, 0.3]

    inputs_simulation = preprocess(list_of_inputs, input_means, input_stds, unknown_nodes, gate, numberOfSamples,
                                   distribution)

    mc = simulation(G, inputs_simulation, unknown_nodes, gate, numberOfSamples)

    desired = putTuplesIntoArray(numbers=mc)


        # ACTUAL - ssta

    g1 = histogramGenerator.get_gauss_bins(1, 0.45, numberOfBins, numberOfSamples, binsInterval)  # g1, g2 INPUT gates, g3 middle
    g2 = histogramGenerator.get_gauss_bins(0.5, 0.3, numberOfBins, numberOfSamples, binsInterval)  # g4 output - inputs: g3 g1
    g3 = histogramGenerator.get_gauss_bins(0.5, 0.5, numberOfBins, numberOfSamples, binsInterval)  # g5 output - inputs: g3, g2
    g4 = histogramGenerator.get_gauss_bins(0.5, 0.5, numberOfBins, numberOfSamples, binsInterval)
    g5 = histogramGenerator.get_gauss_bins(0.5, 0.5, numberOfBins, numberOfSamples, binsInterval)

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

    actual = putTuplesIntoArray(rvs=delays)

        # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec, err_msg= "Monte Carlo: \n" + str(desired) + '\n\n' + "SSTA: \n" + str(actual))

    return None

def testSSTA_2(dec: int):

    numberOfSamples = int(2000000)
    numberOfBins = 2000
    distribution = 'Normal'
    binsInterval = (-20, 60)

        # DESIRED - monte carlo

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
                                   distribution)

    mc = simulation(G, inputs_simulation, unknown_nodes, gate, numberOfSamples)

    desired = putTuplesIntoArray(numbers=mc)
    print()

        # ACTUAL - ssta

    g1 = histogramGenerator.get_gauss_bins(10, 0.45, numberOfBins, numberOfSamples, binsInterval)  # g1, g2 INPUT gates, g3 middle
    g2 = histogramGenerator.get_gauss_bins(12, 0.3, numberOfBins, numberOfSamples, binsInterval)  # g4 output - inputs: g3 g1
    g3 = histogramGenerator.get_gauss_bins(5, 0.5, numberOfBins, numberOfSamples, binsInterval)  # g5 output - inputs: g3, g2
    g4 = histogramGenerator.get_gauss_bins(5, 0.5, numberOfBins, numberOfSamples, binsInterval)
    g5 = histogramGenerator.get_gauss_bins(5, 0.5, numberOfBins, numberOfSamples, binsInterval)

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

    actual = putTuplesIntoArray(rvs=delays)

        # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec, err_msg= "Monte Carlo: \n" + str(desired) + '\n\n' + "SSTA: \n" + str(actual))

    return None


if __name__ == "__main__":

        # dec param is Desired precision

    # testSSTA_1(dec=1)
    testSSTA_2(dec=2)


    print("All tests passed!")