import numpy as np

from cvxpyVariable import RandomVariableCVXPY
import cvxpy as cp

from randomVariableHist_Numpy import RandomVariable
import histogramGenerator
from node import Node
from tabulate import tabulate
import SSTA
from test_SSTA_Numpy import putTuplesIntoArray

from examples_monteCarlo.infinite_ladder_montecarlo import MonteCarlo_inputs, MonteCarlo_nodes, get_moments_from_simulations


def main(numberOfGates=10, numberOfUnaries=20, numberOfBins=20, interval=(-8, 35), withSymmetryConstr=False):

    """
        Computes SSTA using unary encoding, can be computed in a 'precise' or non-precise way
    """

    n_samples = 2000000
    seed = 0

    # numberOfBins = 10
    # numberOfUnaries = 8
    # interval = (-8, 35)

    gateParams = [0.0, 1.0]

    # fix a random seed seed exists
    if seed != None:
        seed = seed
        np.random.seed(seed)


    ####################################
    ####### Generate Input data ########
    ####################################

    # list with inputs' mean values
    input_means = [np.random.randint(20, 70) / 10 for _ in range(numberOfGates + 1)]
    # list with inputs' stds
    input_stds = [np.random.randint(20, 130) / 100 for _ in range(numberOfGates + 1)]

    # CVXPY

    constraints = []

    # generate inputs
    startingNodes = []
    xs_starting = {}
    for i in range(0, numberOfGates + 1):
        g = histogramGenerator.get_gauss_bins_UNARY(input_means[i], input_stds[i], numberOfBins, n_samples,
                                                    interval, numberOfUnaries)
        xs_starting[i] = {}

        for bin in range(0, numberOfBins):
            xs_starting[i][bin] = {}
            for unary in range(0, numberOfUnaries):
                xs_starting[i][bin][unary] = cp.Variable(boolean=True)

                constraints.append(xs_starting[i][bin][unary] <= g.bins[bin][unary])

        node = Node(RandomVariableCVXPY(xs_starting[i], g.edges))
        startingNodes.append(node)

        # generetate nodes
    generatedNodes = []
    xs_generated = {}
    for i in range(0, numberOfGates):
        g = histogramGenerator.get_gauss_bins_UNARY(gateParams[0], gateParams[1], numberOfBins, n_samples, interval,
                                                    numberOfUnaries)
        xs_generated[i] = {}

        for bin in range(0, numberOfBins):
            xs_generated[i][bin] = {}
            for unary in range(0, numberOfUnaries):
                xs_generated[i][bin][unary] = cp.Variable(boolean=True)

                constraints.append(xs_generated[i][bin][unary] <= g.bins[bin][unary])

        node = Node(RandomVariableCVXPY(xs_generated[i], g.edges))
        generatedNodes.append(node)

    # set circuit design

    # start
    startingNodes[0].setNextNodes([generatedNodes[0]])

    # upper part
    for i in range(1, numberOfGates + 1):
        start = startingNodes[i]
        start.setNextNodes([generatedNodes[i - 1]])

        # lower part
    for i in range(0, numberOfGates - 1):
        node = generatedNodes[i]
        node.setNextNodes([generatedNodes[i + 1]])

    delays, newConstr = SSTA.calculateCircuitDelay(startingNodes, cvxpy=True, unary=True,
                                                withSymmetryConstr=withSymmetryConstr)
    delays = delays[numberOfGates + 1:]

    constraints.extend(newConstr)

    # setting objective
    # startingIndex = numberOfGates + 2
    sum = 0
    for gate in range(0, numberOfGates):
        for bin in range(0, numberOfBins):
            for unary in range(0, numberOfUnaries):
                sum += delays[gate].bins[bin][unary]

    # solve

    objective = cp.Maximize(sum)
    prob = cp.Problem(objective, constraints)

    prob.solve(verbose=True, solver=cp.GUROBI,
               MIPGAP=0.01,  # relative gap
               TimeLimit=1200,  # 'MSK_DPAR_OPTIMIZER_MAX_TIME': 1200}  # max time
               )

    num_nonZeros = prob.solver_stats.extra_stats.getAttr("NumNZs")
    ObjVal = prob.solver_stats.extra_stats.getAttr("ObjVal")
    time = prob.solver_stats.extra_stats.getAttr("Runtime")


    # print out the values
    # print("PROBLEM VALUE: ", prob.value)

    rvs = []

    for gate in range(0, numberOfGates):  # construct RVs

        finalBins = np.zeros((numberOfBins, numberOfUnaries))
        for bin in range(0, numberOfBins):
            for unary in range(0, numberOfUnaries):
                finalBins[bin, unary] = ((delays[gate].bins)[bin])[unary].value

        rvs.append(RandomVariable(finalBins, generatedNodes[0].randVar.edges, unary=True))

    print("\n APRX. VALUES: \n")
    for i in range(0, numberOfGates):
        print(rvs[i].mean, rvs[i].std)

    lastGate = (rvs[-1].mean, rvs[-1].std)

    return (num_nonZeros, ObjVal, lastGate, time)

    # # NUMPY
    #
    #     # generate inputs
    # startingNodes = []
    # for i in range(0, numberOfGates + 1):
    #     g = histogramGenerator.get_gauss_bins(input_means[i], input_stds[i], numberOfBins, n_samples, interval)
    #     node = Node(g)
    #     startingNodes.append( node )
    #
    # # generate inputs
    # startingNodes = []
    # for i in range(0, numberOfGates + 1):
    #     g = histogramGenerator.get_gauss_bins(input_means[i], input_stds[i], numberOfBins, n_samples, interval)
    #     node = Node(g)
    #     startingNodes.append(node)
    #
    #     # generetate nodes
    # generatedNodes = []
    # for i in range(0, numberOfGates):
    #     g = histogramGenerator.get_gauss_bins(gateParams[0], gateParams[1], numberOfBins, n_samples, interval)
    #     node = Node(g)
    #     generatedNodes.append(node)
    #
    # # set circuit design cvxpy
    #
    # # start
    # startingNodes[0].setNextNodes([generatedNodes[0]])
    #
    # # upper part
    # for i in range(1, numberOfGates + 1):
    #     start = startingNodes[i]
    #     start.setNextNodes([generatedNodes[i - 1]])
    #
    #     # lower part
    # for i in range(0, numberOfGates - 1):
    #     node = generatedNodes[i]
    #     node.setNextNodes([generatedNodes[i + 1]])
    #
    # delays = SSTA.calculateCircuitDelay(startingNodes)
    #
    # delays = delays[numberOfGates + 1:-1]
    #
    # actual = putTuplesIntoArray(rvs=delays)
    #
    # print("NUMPY VALUES")
    # print(actual)
    #
    #

def MonteCarlo(numberOfGates=10):

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
    input_means = [np.random.randint(20, 70) / 10 for _ in range(numberOfGates + 1)]
    # list with inputs' stds
    input_stds = [np.random.randint(20, 130) / 100 for _ in range(numberOfGates + 1)]

    # ####################################
    # ######## Perform Simulation ########
    # ####################################
    #
    # simulate inputs
    nodes_simulation = [0 for _ in range(numberOfGates)]
    inputs_simulation = MonteCarlo_inputs(input_means, input_stds, n_samples, 'Normal')

    # traverse the circuit
    nodes_simulation[0] = MonteCarlo_nodes(inputs_simulation[0], inputs_simulation[1], gateParams, n_samples)
    for i in range(1, numberOfGates):
        nodes_simulation[i] = MonteCarlo_nodes(nodes_simulation[i - 1], inputs_simulation[i + 1], gateParams, n_samples)

    desired = get_moments_from_simulations(nodes_simulation)

    return desired[-1]



def LadderNumpy(number_of_nodes=1, numberOfBins=10, numberOfUnaries=10, interval=(-8, 20)):


    # parse command line arguments
    # number_of_nodes = 1
    n_samples = 2000000
    seed = 0

    # numberOfBins = 1000
    # numberOfUnaries = numberOfBins*10
    # interval = (-8, 20)

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

    print("\n APRX. VALUES: \n")
    for i in range(0, number_of_nodes):
        print(rvs[i].mean, rvs[i].std)

    # generate inputs
    startingNodes = []
    for i in range(0, number_of_nodes + 1):
        g = histogramGenerator.get_gauss_bins(input_means[i], input_stds[i], numberOfBins, n_samples, interval)
        node = Node(g)
        startingNodes.append( node )

        # generetate nodes
    generatedNodes = []
    for i in range(0, number_of_nodes):
        g = histogramGenerator.get_gauss_bins(gateParams[0], gateParams[1], numberOfBins, n_samples, interval)
        node = Node(g)
        generatedNodes.append(node)

    # set circuit design cvxpy

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

    delays = delays[number_of_nodes + 1:-1]

    actual = putTuplesIntoArray(rvs=delays)

    print("NUMPY VALUES")
    print(actual)

    # simulate inputs
    nodes_simulation = [0 for _ in range(number_of_nodes)]
    inputs_simulation = MonteCarlo_inputs(input_means, input_stds, n_samples, 'Normal')

    # traverse the circuit
    nodes_simulation[0] = MonteCarlo_nodes(inputs_simulation[0], inputs_simulation[1], gateParams, n_samples)
    for i in range(1, number_of_nodes):
        nodes_simulation[i] = MonteCarlo_nodes(nodes_simulation[i - 1], inputs_simulation[i + 1], gateParams,
                                               n_samples)

    desired = get_moments_from_simulations(nodes_simulation)

    print('MONTE CARLO - GROUND TRUTH')
    print(
        tabulate(desired, headers=["Mean", "std"]
                 )
    )

    return desired


if __name__ == "__main__":
        # dec param is Desired precision

    # main()
    LadderNumpy()


    # print("All tests passed!")
