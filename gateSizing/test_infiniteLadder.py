import numpy as np

from cvxpyVariable import RandomVariableCVXPY
import cvxpy as cp

from randomVariableHist_Numpy import RandomVariable
import histogramGenerator
from Main.node import Node
from tabulate import tabulate
import SSTA
from test_SSTA_Numpy import putTuplesIntoArray

from examples_monteCarlo.infinite_ladder_montecarlo import MonteCarlo_inputs, MonteCarlo_nodes, get_moments_from_simulations


def main():

    # parse command line arguments
    number_of_nodes = 10
    n_samples = 1000000
    seed = None
    numberOfBins = 2000
    interval = (-20, 120)

    gate = [0.0, 1.0]

    # fix a random seed seed exists
    if seed:
        seed = seed
        np.random.seed(seed)

    ####################################
    ####### Generate Input data ########
    ####################################

    # list with inputs' mean values
    input_means = [np.random.randint(20, 70) / 10 for _ in range(number_of_nodes + 1)]
    # list with inputs' stds
    input_stds = [np.random.randint(20, 130) / 100 for _ in range(number_of_nodes + 1)]

    ####################################
    ######## Perform Simulation ########
    ####################################

    # simulate inputs
    nodes_simulation = [0 for _ in range(number_of_nodes)]
    inputs_simulation = MonteCarlo_inputs(input_means, input_stds, n_samples, 'Normal')

    # traverse the circuit
    nodes_simulation[0] = MonteCarlo_nodes(inputs_simulation[0], inputs_simulation[1], gate, n_samples)
    for i in range(1, number_of_nodes):
        nodes_simulation[i] = MonteCarlo_nodes(nodes_simulation[i - 1], inputs_simulation[i + 1], gate, n_samples)

    desired = get_moments_from_simulations(nodes_simulation)

    print(
        tabulate(desired, headers=["Mean", "std"]
                 )
    )

    # NUMPY

        # generate inputs
    startingNodes = []
    for i in range(0, number_of_nodes + 1):
        g = histogramGenerator.get_gauss_bins(input_means[i], input_stds[i], numberOfBins, n_samples, interval)
        node = Node(g)
        startingNodes.append( node )

        # generetate nodes
    generatedNodes = []
    for i in range(0, number_of_nodes):
        g = histogramGenerator.get_gauss_bins(gate[0], gate[1], numberOfBins, n_samples, interval)
        node = Node(g)
        generatedNodes.append(node)

    # set circuit design

        # start
    startingNodes[0].setNextNodes([generatedNodes[0]])

        # upper part
    for i in range(1, number_of_nodes + 1):
        start = startingNodes[i]
        start.setNextNodes([generatedNodes[i-1]])

        # lower part
    for i in range(0, number_of_nodes-1):
        node = generatedNodes[i]
        node.setNextNodes([generatedNodes[i+1]])


    delays = SSTA.calculateCircuitDelay(startingNodes)



    actual = putTuplesIntoArray(rvs=delays)

    print(actual)

    # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=3)


if __name__ == "__main__":
        # dec param is Desired precision

    main()

    print("All tests passed!")