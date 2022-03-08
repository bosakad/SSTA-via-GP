import histogramGenerator
import test_infiniteLadder
import numpy as np
from test_algorithms import computeMAPE
import matplotlib.pyplot as plt
import SSTA
from test_SSTA_Numpy import putTuplesIntoArray

from randomVariableHist_Numpy import RandomVariable

from examples_monteCarlo.infinite_ladder_montecarlo import MonteCarlo_inputs, MonteCarlo_nodes, get_moments_from_simulations


from Main.node import Node
from tabulate import tabulate


# test max computation

def testMax(numberOfBins, numberOfUnaries):
    mu1 = 12
    sigma1 = 2

    mu2 = 6
    sigma2 = 3

    numberOfSamples = 20000000
    #     numberOfBins = 100
    #     numberOfUnaries = 10000

    interval = (-5, 20)

    # DESIRED

    rv1 = np.random.normal(mu1, sigma1, numberOfSamples)
    rv2 = np.random.normal(mu2, sigma2, numberOfSamples)

    max1 = np.maximum(rv1, rv2)
    desired = np.array([np.mean(max1), np.std(max1)])

    # ACTUAL

    # histogram1
    h1 = histogramGenerator.get_gauss_bins_UNARY(mu1, sigma1, numberOfBins, numberOfSamples, interval, numberOfUnaries)
    h2 = histogramGenerator.get_gauss_bins_UNARY(mu2, sigma2, numberOfBins, numberOfSamples, interval, numberOfUnaries)

    max2 = h1.maxOfDistributionsQUAD_FORMULA_UNARY(h2)

    actual = np.array([max2.mean, max2.std])

    return (actual, desired)

    # TESTING

#     np.testing.assert_almost_equal(desired, actual, decimal=5)

# test mean computation

def testConvolution(numberOfBins, numberOfUnaries):
    mu1 = 12
    sigma1 = 2

    mu2 = 8
    sigma2 = 1.3

    numberOfSamples = 20000000
    #     numberOfBins = 100
    #     numberOfUnaries = 10000

    interval = (-5, 40)

    # DESIRED

    rv1 = np.random.normal(mu1, sigma1, numberOfSamples)
    rv2 = np.random.normal(mu2, sigma2, numberOfSamples)

    max1 = rv1 + rv2
    desired = np.array([np.mean(max1), np.std(max1)])

    # ACTUAL

    # histogram1
    h1 = histogramGenerator.get_gauss_bins_UNARY(mu1, sigma1, numberOfBins, numberOfSamples, interval, numberOfUnaries)
    h2 = histogramGenerator.get_gauss_bins_UNARY(mu2, sigma2, numberOfBins, numberOfSamples, interval, numberOfUnaries)

    rv1 = histogramGenerator.get_gauss_bins(mu1, sigma1, numberOfBins, numberOfSamples, interval)
    rv2 = histogramGenerator.get_gauss_bins(mu2, sigma2, numberOfBins, numberOfSamples, interval)

    max2 = h1.convolutionOfTwoVarsNaiveSAME_UNARY(h2)
    #     max2 = rv1.convolutionOfTwoVarsShift(rv2)

    actual = np.array([max2.mean, max2.std])

    return (actual, desired)

    # TESTING

#     np.testing.assert_almost_equal(desired, actual, decimal=5)

# test infinite ladder function

def LadderNumpy(numberOfBins=100, numberOfUnaries=100, number_of_nodes=1, interval=(-8, 20)):
    n_samples = 2000000
    seed = 0

    gateParams = [0.0, 1.0]

    # fix a random seed seed exists
    if seed != None:
        seed = seed
        np.random.seed(seed)

    ####################################
    ####### Generate Input data ########
    ####################################

    # list with inputs' mean values
    input_means = [np.random.randint(20, 70) / 10 for _ in range(number_of_nodes + 1)]
    # list with inputs' stds
    input_stds = [np.random.randint(20, 130) / 100 for _ in range(number_of_nodes + 1)]

    # CVXPY

    constraints = []

    # generate inputs
    startingNodes = []
    for i in range(0, number_of_nodes + 1):
        g = histogramGenerator.get_gauss_bins_UNARY(input_means[i], input_stds[i], numberOfBins, n_samples,
                                                    interval, numberOfUnaries)

        node = Node(g)
        startingNodes.append(node)

        # generetate nodes
    generatedNodes = []
    for i in range(0, number_of_nodes):
        g = histogramGenerator.get_gauss_bins_UNARY(gateParams[0], gateParams[1], numberOfBins, n_samples, interval,
                                                    numberOfUnaries)

        node = Node(g)
        generatedNodes.append(node)

    # set circuit design

    # start
    startingNodes[0].setNextNodes([generatedNodes[0]])

    # upper part
    for i in range(1, number_of_nodes + 1):
        start = startingNodes[i]
        start.setNextNodes([generatedNodes[i - 1]])

        # lower part
    for i in range(0, number_of_nodes - 1):
        node = generatedNodes[i]
        node.setNextNodes([generatedNodes[i + 1]])

    delays = SSTA.calculateCircuitDelay(startingNodes, cvxpy=False, unary=True)

    delays = delays[number_of_nodes + 1:]

    rvs = []

    for gate in range(0, number_of_nodes):  # construct RVs

        finalBins = np.zeros((numberOfBins, numberOfUnaries))
        for bin in range(0, numberOfBins):
            for unary in range(0, numberOfUnaries):
                finalBins[bin, unary] = ((delays[gate].bins)[bin])[unary]

        rvs.append(RandomVariable(finalBins, generatedNodes[0].randVar.edges, unary=True))

    # simulate inputs
    nodes_simulation = [0 for _ in range(number_of_nodes)]
    inputs_simulation = MonteCarlo_inputs(input_means, input_stds, n_samples, 'Normal')

    # traverse the circuit
    nodes_simulation[0] = MonteCarlo_nodes(inputs_simulation[0], inputs_simulation[1], gateParams, n_samples)
    for i in range(1, number_of_nodes):
        nodes_simulation[i] = MonteCarlo_nodes(nodes_simulation[i - 1], inputs_simulation[i + 1], gateParams,
                                               n_samples)

    desired = get_moments_from_simulations(nodes_simulation)

    return np.array([rvs[-1].mean, rvs[-1].std]), np.array(desired[0])










# compute MAPE heatmap for Convolution - takes a long time to compute

bins = 2
unaries = 2
start = 10

meanMAPE, stdMAPE = computeMAPE(bins, unaries, start, testConvolution)

meanMAPE = np.flip(meanMAPE, 0)
meanMAPE = np.around(meanMAPE, 2)

stdMAPE = np.around(stdMAPE, 2)
stdMAPE = np.flip(stdMAPE, 0)

# plot heatmap

fig, ax = plt.subplots(figsize=(13, 13), dpi=80)

im = ax.imshow(meanMAPE)

for i in range(0, meanMAPE.shape[0], ):
    for j in range(meanMAPE.shape[1]):
        text = ax.text(j, i, meanMAPE[i, j],
                       ha="center", va="center", color="w")

locs, labels = plt.xticks()  # Get the current locations and labels.
plt.xticks(np.arange(0, unaries), np.arange(start, start+unaries*2, step=2))
plt.yticks(np.arange(0, bins), np.flip(np.arange(start, start+bins*2, step=2)))

plt.xlabel('Number of unary variables')
plt.ylabel('Number of bins')
plt.title('Convolution, MAPE of mean')

fig.tight_layout()

plt.savefig("Inputs/testConvolution/meanHeatMap.jpeg", dpi=300)
#plt.show()



fig, ax = plt.subplots(figsize=(13, 13), dpi=80)
im = ax.imshow(stdMAPE)

for i in range(0, stdMAPE.shape[0], ):
    for j in range(stdMAPE.shape[1]):
        text = ax.text(j, i, stdMAPE[i, j],
                       ha="center", va="center", color="w")

locs, labels = plt.xticks()  # Get the current locations and labels.
plt.xticks(np.arange(0, unaries), np.arange(start, start+unaries*2, step=2))
plt.yticks(np.arange(0, bins), np.flip(np.arange(start, start+bins*2, step=2)))



plt.xlabel('Number of unary variables')
plt.ylabel('Number of bins')
plt.title('Convolution, MAPE of std')


fig.tight_layout()

plt.savefig("Inputs/testConvolution/stdHeatMap.jpeg", dpi=300)
#plt.show()


# compute MAPE heatmap for MAXIMUM - takes a long time to compute


bins = 2
unaries = 2
start = 10

meanMAPE, stdMAPE = computeMAPE(bins, unaries, start, testMax)

meanMAPE = np.around(meanMAPE, 2)
meanMAPE = np.flip(meanMAPE, 0)

stdMAPE = np.around(stdMAPE, 2)
stdMAPE = np.flip(stdMAPE, 0)

# plot heatmap

fig, ax = plt.subplots(figsize=(13, 13), dpi=300)

im = ax.imshow(meanMAPE)

for i in range(0, meanMAPE.shape[0], ):
    for j in range(meanMAPE.shape[1]):
        text = ax.text(j, i, meanMAPE[i, j],
                       ha="center", va="center", color="w")

locs, labels = plt.xticks()  # Get the current locations and labels.
plt.xticks(np.arange(0, unaries), np.arange(start, start+unaries*2, step=2))
plt.yticks(np.arange(0, bins), np.flip(np.arange(start, start+bins*2, step=2)))


fig.tight_layout()

plt.savefig("Inputs/testMAX/meanHeatMap.jpeg", dpi=300)
#plt.show()



fig, ax = plt.subplots(figsize=(13, 13), dpi=80)
im = ax.imshow(stdMAPE)

for i in range(0, stdMAPE.shape[0], ):
    for j in range(stdMAPE.shape[1]):
        text = ax.text(j, i, stdMAPE[i, j],
                       ha="center", va="center", color="w")

locs, labels = plt.xticks()  # Get the current locations and labels.
plt.xticks(np.arange(0, unaries), np.arange(start, start+unaries*2, step=2))
plt.yticks(np.arange(0, bins), np.flip(np.arange(start, start+bins*2, step=2)))


fig.tight_layout()

plt.savefig("Inputs/testMAX/stdHeatMap.jpeg", dpi=100)
#plt.show()

# compute MAPE heatmap for infinite ladder - takes a long time to compute

bins = 2
unaries = 2
start = 10

meanMAPE, stdMAPE = computeMAPE(bins, unaries, start, LadderNumpy)

meanMAPE = np.around(meanMAPE, 2)
meanMAPE = np.flip(meanMAPE, 0)

stdMAPE = np.around(stdMAPE, 2)
stdMAPE = np.flip(stdMAPE, 0)

# plot heatmap

fig, ax = plt.subplots(figsize=(13, 13), dpi=80)

im = ax.imshow(meanMAPE)

for i in range(0, meanMAPE.shape[0], ):
    for j in range(meanMAPE.shape[1]):
        text = ax.text(j, i, meanMAPE[i, j],
                       ha="center", va="center", color="w")

locs, labels = plt.xticks()  # Get the current locations and labels.
plt.xticks(np.arange(0, unaries), np.arange(start, start+unaries*2, step=2))
plt.yticks(np.arange(0, bins), np.flip(np.arange(start, start+bins*2, step=2)))


fig.tight_layout()

plt.savefig("Inputs/testInfiniteLadder/meanHeatMap.jpeg", dpi=300)
#plt.show()



fig, ax = plt.subplots(figsize=(13, 13), dpi=80)
im = ax.imshow(stdMAPE)

for i in range(0, stdMAPE.shape[0], ):
    for j in range(stdMAPE.shape[1]):
        text = ax.text(j, i, stdMAPE[i, j],
                       ha="center", va="center", color="w")

locs, labels = plt.xticks()  # Get the current locations and labels.
plt.xticks(np.arange(0, unaries), np.arange(start, start+unaries*2, step=2))
plt.yticks(np.arange(0, bins), np.flip(np.arange(start, start+bins*2, step=2)))


fig.tight_layout()

plt.savefig("Inputs/testInfiniteLadder/stdHeatMap.jpeg", dpi=300)
#plt.show()