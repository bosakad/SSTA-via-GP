import numpy as np

from gateSizing.node import Node
from gateSizing import histogramGenerator
from gateSizing import SSTA
import networkx as nx
from randomVariableHist import RandomVariable
import matplotlib.pyplot as plt
import scipy.stats as stats

from examples_monteCarlo.montecarlo import get_inputs, get_unknown_nodes, simulation, preprocess



"""
Plots error as function of depth of the gate

Params:
    mc: (n, 2) matrix
    ssta: (n, 2) matrix

"""
def plotError(mc, ssta):

    error = mc - ssta
    errors = np.sum( np.abs(error), axis=1 )

    n = len(errors)

    edges = np.arange(0, n+1, 1)

    plt.hist(edges[:-1], edges, weights=errors)
    plt.ylabel('Error', size=14)
    plt.xlabel('Gate', size=14)
    # plt.title('Function of an error.', size=16)
    plt.show()



"""
 Test maximum on data from mc - compare it to ssta data
"""
def testMCMax(mc, binsInterval, numberOfBins):

    rv1 = mc[-3]
    rv2 = mc[-2]

    STATIC_BINS = np.linspace(-100, 500, 15000)

    data, edges = np.histogram(rv1, bins=STATIC_BINS)
    dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))
    h1 = RandomVariable(dataNorm, edges)

    data, edges = np.histogram(rv2, bins=STATIC_BINS)
    dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))
    h2 = RandomVariable(dataNorm, edges)

    # plt.hist(h2.edges[:-1], h2.edges, weights=h2.bins, density="PDF")
    #
    # plt.hist(h1.edges[:-1], h1.edges, weights=h1.bins, density="PDF")
    # plt.show()

    print(h1.mean, h2.mean)

        # compute max from mc data
    # h3 = h1.maxOfDistributionsFORM(h2)


    _ = plt.hist(mc[-1], bins=2000, density='PDF', alpha=0.7)

    # mu1 = 21.98553396
    # sigma1 = 0.76804456
    # rv1 = np.random.normal(mu1, sigma1, 2000000)
    # _ = plt.hist(rv1, bins=2000, density='PDF', alpha=0.7)
    # _ = plt.hist(rv2, bins=2000, density='PDF', alpha=0.7)
    # _ = plt.hist(np.maximum(mc[-2], mc[-3]), bins=2000, density='PDF', alpha=0.7)
    # plt.hist(h3.edges[:-1], h3.edges, weights=h3.bins, density="PDF")

    # plt.show()


    # plt.hist(h1.edges[:-1], h1.edges, weights=h1.bins, density="PDF")
    # plt.show()

    # actual = [h3.mean, h3.std]
    # desired = [np.mean(mc[-1]), np.std(mc[-1])]
    #
    # np.testing.assert_almost_equal(desired, actual, decimal=3)








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

    g1 = histogramGenerator.get_gauss_bins(8, 0.45, numberOfBins, numberOfSamples, binsInterval)  # g1, g2 INPUT gates, g3 middle
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

def testSSTA_2(dec: int):

    numberOfSamples = 5000000
    numberOfBins = 30
    distribution = 'Normal'
    binsInterval = (-2, 30)

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
    # print()
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

    # density = stats.gaussian_kde(mc[-1])
    # plt.hist(mc[-1], bins=numberOfBins, density='PDF', alpha=0.7)
    # plt.plot(delays[-1].edges[450:-445], density(delays[-1].edges[450:-445]))

    # plt.hist(delays[-1].edges[450:-446], delays[-1].edges[450:-445], weights=delays[-1].bins[450:-445], density="PDF")
    # plt.legend(['Monte Carlo', 'Histogram approximation'])

    # plt.ylabel('PDF', size=14)
    # plt.xlabel('Time', size=14)

    # plt.show()


    # plotError(desired, actual)

    # testMCMax(mc, binsInterval, numberOfBins)

        # test whole
    np.testing.assert_almost_equal(desired, actual, decimal=dec, err_msg= "Monte Carlo: \n" + str(desired) + '\n\n' + "SSTA: \n" + str(actual))

    return None


if __name__ == "__main__":

        # dec param is Desired precision

    # testSSTA_1(dec=1)
    testSSTA_2(dec=5)


    print("All tests passed!")