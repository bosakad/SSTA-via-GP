import numpy as np

from cvxpyVariable import RandomVariableCVXPY
import cvxpy as cp

from randomVariableHist_Numpy import RandomVariable
import histogramGenerator
from node import Node
from tabulate import tabulate
import sys
import SSTA
from test_SSTA_Numpy import putTuplesIntoArray
from mosekVariable import RandomVariableMOSEK

from examples_monteCarlo.infinite_ladder_montecarlo import MonteCarlo_inputs, MonteCarlo_nodes, get_moments_from_simulations

import matplotlib.pyplot as plt

import mosek
from mosek import *

def mainCVXPY(numberOfGates=10, numberOfUnaries=20, numberOfBins=20, interval=(-8, 35), withSymmetryConstr=False):

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

        # g = histogramGenerator.get_Histogram_from_UNARY(g)
        # print(g.bins)

        # exit(-1)

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

    prob.solve(verbose=True, solver=cp.MOSEK,
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

def mainCVXPY_GP(numberOfGates=8, numberOfBins=25, interval=(-4, 25)):

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
        g = histogramGenerator.get_gauss_bins(input_means[i], input_stds[i], numberOfBins, n_samples,
                                                    interval, forGP=True)

        xs_starting[i] = {}

        for bin in range(0, numberOfBins):
            xs_starting[i][bin] = cp.Variable(pos=True)

            constraints.append(xs_starting[i][bin] >= g.bins[bin])

        node = Node(RandomVariableCVXPY(xs_starting[i], g.edges))
        startingNodes.append(node)

        # generetate nodes
    generatedNodes = []
    xs_generated = {}
    for i in range(0, numberOfGates):
        g = histogramGenerator.get_gauss_bins(gateParams[0], gateParams[1], numberOfBins, n_samples, interval,
                                                    forGP=True)
        xs_generated[i] = {}

        for bin in range(0, numberOfBins):
            xs_generated[i][bin] = cp.Variable(pos=True)

            constraints.append(xs_generated[i][bin] >= g.bins[bin])

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

    delays, newConstr = SSTA.calculateCircuitDelay(startingNodes, cvxpy=True, GP=True)
    delays = delays[numberOfGates + 1:]

    constraints.extend(newConstr)

    # setting objective
    # startingIndex = numberOfGates + 2
    sum = 0
    midPoints = 0.5 * (delays[-1].edges[1:] + delays[-1].edges[:-1])  # midpoints of the edges of hist.
    # for gate in range(0, len(delays)):
    # print(midPoints)
    for bin in range(0, numberOfBins):
            sum += delays[-1].bins[bin]

    # solve

    objective = cp.Minimize(sum)
    prob = cp.Problem(objective, constraints)

    prob.solve(verbose=True, solver=cp.MOSEK, gp=True,
               mosek_params={  'MSK_DPAR_INTPNT_CO_TOL_MU_RED': 0.1,
               'MSK_DPAR_OPTIMIZER_MAX_TIME': 1200}
                               # , mosek.iparam.intpnt_solve_form: mosek.solveform.dual}  # max time
                )

    # num_nonZeros = prob.solver_stats.extra_stats.getAttr("NumNZs")
    # ObjVal = prob.solver_stats.extra_stats.getAttr("ObjVal")
    # time = prob.solver_stats.extra_stats.getAttr("Runtime")
    # time = prob.solver_stats.extra_stats.getAttr("dinfitem.optimizer_time")
    time = prob.solver_stats.solve_time


    # print out the values
    # print("PROBLEM VALUE: ", prob.value)

    rvs = []

    for gate in range(0, len(delays)):  # construct RVs

        finalBins = np.zeros(numberOfBins)
        for bin in range(0, numberOfBins):
            # if ((delays[gate].bins)[bin]) != 0:
            finalBins[bin] = ((delays[gate].bins)[bin]).value
            # else:
            #     finalBins[bin] = 0

        rvs.append(RandomVariable(finalBins, generatedNodes[0].randVar.edges))

    print("\n APRX. VALUES: \n")
    for i in range(0, len(delays)):
        print(rvs[i].mean, rvs[i].std)

    lastGate = (rvs[-1].mean, rvs[-1].std)

    return (lastGate, time)
    # return (num_nonZeros, ObjVal, lastGate, time)


    # NUMPY

        # generate inputs
    # startingNodes = []
    # for i in range(0, numberOfGates + 1):
    #     g = histogramGenerator.get_gauss_bins(input_means[i], input_stds[i], numberOfBins, n_samples, interval, forGP=False)
    #     node = Node(g)
    #     startingNodes.append( node )
    #
    # # generate inputs
    # startingNodes = []
    # for i in range(0, numberOfGates + 1):
    #     g = histogramGenerator.get_gauss_bins(input_means[i], input_stds[i], numberOfBins, n_samples, interval, forGP=False)
    #     node = Node(g)
    #     startingNodes.append(node)
    #
    #     # generetate nodes
    # generatedNodes = []
    # for i in range(0, numberOfGates):
    #     g = histogramGenerator.get_gauss_bins(gateParams[0], gateParams[1], numberOfBins, n_samples, interval, forGP=False)
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
    # delays = delays[numberOfGates + 1:]
    #
    # print(delays[-1].bins)
    #
    # actual = putTuplesIntoArray(rvs=delays)

        # monte carlo
    lastGate, values = MonteCarlo(numberOfGates, True)
    print(lastGate)

    # plt.hist(delays[-1].edges[:-1], delays[-1].edges, weights=delays[-1].bins,alpha=0.2, color='orange')

    plt.hist(rvs[-1].edges[:-1], rvs[-1].edges, weights=rvs[-1].bins, density="PDF", color='blue')
    _ = plt.hist(values, bins=numberOfBins, density='PDF', alpha=0.7)
    plt.show()


    # print("NUMPY VALUES")
    # print(actual)



def mainCVXPYMcCormick(numberOfGates=1, numberOfUnaries=10, numberOfBins=20, interval=(-5, 18)):

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
        g = histogramGenerator.get_gauss_bins(input_means[i], input_stds[i], numberOfBins, n_samples,
                                                    interval)
        xs_starting[i] = {}

        for bin in range(0, numberOfBins):
            xs_starting[i][bin] = cp.Variable(nonneg=True)
            # constraints.append(xs_starting[i][bin] <= g.bins[bin])

        node = Node(RandomVariableCVXPY(xs_starting[i], g.edges, g.bins))
        startingNodes.append(node)

        # generetate nodes
    generatedNodes = []
    xs_generated = {}
    for i in range(0, numberOfGates):
        g = histogramGenerator.get_gauss_bins(gateParams[0], gateParams[1], numberOfBins, n_samples, interval)
        xs_generated[i] = {}

        for bin in range(0, numberOfBins):
            xs_generated[i][bin] = cp.Variable(nonneg=True)

        node = Node(RandomVariableCVXPY(xs_generated[i], g.edges, g.bins))
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

    delays, newConstr = SSTA.calculateCircuitDelay(startingNodes, cvxpy=True, unary=False,
                                                withSymmetryConstr=False)
    delays = delays[numberOfGates + 1:]

    constraints.extend(newConstr)

    # setting objective
    # startingIndex = numberOfGates + 2
    sum = 0
    for gate in range(0, numberOfGates):
        for bin in range(0, numberOfBins):
            sum += delays[gate].bins[bin]

    # solve

    objective = cp.Minimize(sum)
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

        finalBins = np.zeros(numberOfBins)
        for bin in range(0, numberOfBins):
            finalBins[bin] = ((delays[gate].bins)[bin]).value

        rvs.append(RandomVariable(finalBins, generatedNodes[0].randVar.edges))

    print("\n APRX. VALUES: \n")
    for i in range(0, numberOfGates):
        print(rvs[i].mean, rvs[i].std)

    lastGate = (rvs[-1].mean, rvs[-1].std)

    # monte carlo

    #
    # simulate inputs
    nodes_simulation = [0 for _ in range(numberOfGates)]
    inputs_simulation = MonteCarlo_inputs(input_means, input_stds, n_samples, 'Normal')

    # traverse the circuit
    nodes_simulation[0] = MonteCarlo_nodes(inputs_simulation[0], inputs_simulation[1], gateParams, n_samples)
    for i in range(1, numberOfGates):
        nodes_simulation[i] = MonteCarlo_nodes(nodes_simulation[i - 1], inputs_simulation[i + 1], gateParams, n_samples)

    desired = get_moments_from_simulations(nodes_simulation)

    print('MONTE CARLO - GROUND TRUTH')
    print(
        tabulate(desired, headers=["Mean", "std"]
                 )
    )

    return (num_nonZeros, ObjVal, lastGate, time)


def MonteCarlo(numberOfGates=10, returnGate=False):

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

    if returnGate:
        return desired[-1], nodes_simulation[-1]
    else:
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



def mainMOSEK(number_of_nodes=10, numberOfUnaries=20, numberOfBins=20, interval=(-2, 18), withSymmetryConstr=True, TRI=False):

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



        # --------------------------------------- MOSEK ----------------------------

    # Make a MOSEK environment
    with mosek.Env() as env:
        # Attach a printer to the environment
        env.set_Stream(mosek.streamtype.log, streamprinter)

        # Create a task
        with env.Task(0, 0) as task:
            # Attach a printer to the task
            task.set_Stream(mosek.streamtype.log, streamprinter)

            numberVariablesRVs = (2 * number_of_nodes + 1) * numberOfBins * numberOfUnaries

            # The variables will initially be fixed at zero (x=0).
            task.appendvars(numberVariablesRVs)

            # set variables to be boolean
            rvIndices = np.array(range(0, numberVariablesRVs))
            task.putvartypelist(rvIndices,
                                [mosek.variabletype.type_int] * numberVariablesRVs)

            startingNodes = []
            curNofConstr = 0


            # set objective function
            for gate in range(0, number_of_nodes + 1):
                g = histogramGenerator.get_gauss_bins_UNARY(input_means[gate], input_stds[gate], numberOfBins,
                                                            n_samples,
                                                            interval, numberOfUnaries)
                bins = np.zeros((numberOfBins, numberOfUnaries)).astype(int)

                for bin in range(0, numberOfBins):
                    for unary in range(0, numberOfUnaries):
                        variableIndex = gate * numberOfBins * numberOfUnaries + bin * numberOfUnaries + unary

                        # Set the bounds on variable
                        # 0 <= x_j <= 1

                        task.putvarbound(variableIndex, mosek.boundkey.ra, 0.0, g.bins[bin, unary])

                        # save index to the bins
                        bins[bin, unary] = variableIndex


                node = Node(RandomVariableMOSEK(bins, g.edges, task))
                startingNodes.append(node)

            offset = (number_of_nodes + 1) * numberOfBins * numberOfUnaries

            # generetate nodes
            generatedNodes = []
            for gate in range(0, number_of_nodes):
                g = histogramGenerator.get_gauss_bins_UNARY(gateParams[0], gateParams[1], numberOfBins, n_samples,
                                                            interval,
                                                            numberOfUnaries)
                bins = np.zeros((numberOfBins, numberOfUnaries)).astype(int)

                for bin in range(0, numberOfBins):
                    for unary in range(0, numberOfUnaries):
                        variableIndex = offset + gate * numberOfBins * numberOfUnaries + bin * numberOfUnaries + unary

                        # Set the bounds on variable
                        # 0 <= x_j <= 1

                        task.putvarbound(variableIndex, mosek.boundkey.ra, 0.0, g.bins[bin, unary])

                        # save index to the bins
                        bins[bin, unary] = variableIndex

                node = Node(RandomVariableMOSEK(bins, g.edges, task))
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

            if TRI:
                delays, newNofVariables, newNofConstr = SSTA.calculateCircuitDelay(startingNodes, cvxpy=False, unary=True,
                                                                     mosekStatus=(numberVariablesRVs, curNofConstr), mosekTRI=True)
            else:
                delays, newNofVariables, newNofConstr = SSTA.calculateCircuitDelay(startingNodes, cvxpy=False, unary=True,
                                                                 mosekStatus=(numberVariablesRVs, curNofConstr),
                                                                 withSymmetryConstr=withSymmetryConstr)
            delays = delays[number_of_nodes + 1:]

            # setting objective
            # startingIndex = numberOfGates + 2
            sum = np.array([]).astype(int)
            for gate in range(0, number_of_nodes):
                sum = np.append(sum, np.concatenate(delays[gate].bins[:, :]))

            task.putclist(sum, [1] * sum.shape[0])

            # Input the objective sense (minimize/maximize)
            task.putobjsense(mosek.objsense.maximize)
            # set mip gap to 1%
            task.putdouparam(dparam.mio_tol_rel_gap, 1.0e-2)
            # task.putdouparam(dparam.mio_max_time, 1200)



            # task.putintparam(iparam.presolve_max_num_reductions, 1)
            # task.putintparam(iparam.presolve_eliminator_max_fill, 0)


            task.putintparam(iparam.presolve_use, presolvemode.on)
            # task.putintparam(iparam.presolve_max_num_pass, 0)
            # task.putparam("MSK_IPAR_LOG_PRESOLVE", "10")
            # task.putintparam(iparam.log_presolve, 100000000)


            usercallback = makeUserCallback(maxtime=0.01, task=task)
            task.set_InfoCallback(usercallback)


            # Solve the prolem
            task.optimize()

            # Print a summary containing information
            # about the solution for debugging purposes
            task.solutionsummary(mosek.streamtype.msg)


            # prosta = task.getprosta(mosek.soltype.itg)
            # solsta = task.getsolsta(mosek.soltype.itg)

            # Output a solution
            xx = np.array([0.] * newNofVariables)
            task.getxx(mosek.soltype.itg, xx)

            # delays[-1].bins = xx[delays[-1].bins]

            # rv = RandomVariable(delays[-1].bins, edges=delays[-1].edges, unary=True)
            # actual = [rv.mean, rv.std]
            #
            rvs = []


            for gate in range(0, len(delays)):  # construct RVs

                finalBins = xx[delays[gate].bins[:, :]]

                rvs.append(RandomVariable(finalBins, generatedNodes[0].randVar.edges, unary=True))


            print("\n MOSEK UNARY VALUES: \n")
            for i in range(0, number_of_nodes):
                print(rvs[i].mean, rvs[i].std)

            lastGate = (rvs[-1].mean, rvs[-1].std)

            time = task.getdouinf(mosek.dinfitem.optimizer_time)
            ObjVal = task.getdouinf(mosek.dinfitem.mio_obj_int)
            num_nonZeros = task.getlintinf(mosek.liinfitem.mio_presolved_anz)

            numConstrPriorPresolve = task.getintinf(mosek.iinfitem.mio_numcon)
            numVarsPriorPresolve = task.getintinf(iinfitem.mio_numbin)

            task.analyzeproblem(mosek.streamtype.log)
            numConstr = task.getintinf(iinfitem.ana_pro_num_con)
            numVariables = task.getintinf(iinfitem.ana_pro_num_var)

    return (num_nonZeros, ObjVal, lastGate, time, numVariables, numConstr, MIPgap, numVarsPriorPresolve, numConstrPriorPresolve)


# Define a stream printer to grab output from MOSEK
def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def makeUserCallback(maxtime, task):
    xx = np.zeros(task.getnumvar())    # Space for integer solutions

    def userCallback(caller,
                     douinf,
                     intinf,
                     lintinf):
        opttime = 0.0

        global MIPgap

        curRelax = intinf[iinfitem.mio_num_relax]
        if curRelax == 1:   # root node hit
            MIPgap = douinf[dinfitem.mio_obj_rel_gap]

        # if caller == callbackcode.begin_intpnt:
        #     print("Starting interior-point optimizer")
        # elif caller == callbackcode.intpnt:
        #     itrn = intinf[iinfitem.intpnt_iter]
        #     pobj = douinf[dinfitem.intpnt_primal_obj]
        #     dobj = douinf[dinfitem.intpnt_dual_obj]
        #     stime = douinf[dinfitem.intpnt_time]
        #     opttime = douinf[dinfitem.optimizer_time]
        #
        #     print("Iterations: %-3d" % itrn)
        #     print("  Elapsed time: %6.2f(%.2f) " % (opttime, stime))
        #     print("  Primal obj.: %-18.6e  Dual obj.: %-18.6e" % (pobj, dobj))
        # elif caller == callbackcode.end_intpnt:
        #     print("Interior-point optimizer finished.")
        # elif caller == callbackcode.begin_primal_simplex:
        #     print("Primal simplex optimizer started.")
        # elif caller == callbackcode.update_primal_simplex:
        #     itrn = intinf[iinfitem.sim_primal_iter]
        #     pobj = douinf[dinfitem.sim_obj]
        #     stime = douinf[dinfitem.sim_time]
        #     opttime = douinf[dinfitem.optimizer_time]
        #
        #     print("Iterations: %-3d" % itrn)
        #     print("  Elapsed time: %6.2f(%.2f)" % (opttime, stime))
        #     print("  Obj.: %-18.6e" % pobj)
        # elif caller == callbackcode.end_primal_simplex:
        #     print("Primal simplex optimizer finished.")
        # elif caller == callbackcode.begin_dual_simplex:
        #     print("Dual simplex optimizer started.")
        # elif caller == callbackcode.update_dual_simplex:
        #     itrn = intinf[iinfitem.sim_dual_iter]
        #     pobj = douinf[dinfitem.sim_obj]
        #     stime = douinf[dinfitem.sim_time]
        #     opttime = douinf[dinfitem.optimizer_time]
        #     print("Iterations: %-3d" % itrn)
        #     print("  Elapsed time: %6.2f(%.2f)" % (opttime, stime))
        #     print("  Obj.: %-18.6e" % pobj)
        # elif caller == callbackcode.end_dual_simplex:
        #     print("Dual simplex optimizer finished.")
        # elif caller == callbackcode.new_int_mio:
        #     print("New integer solution has been located.")
        #     task.getxx(soltype.itg, xx)
        #     print(xx)
        #     print("Obj.: %f" % douinf[dinfitem.mio_obj_int])
        # else:
        #     pass
        #
        # if opttime >= maxtime:
        #     # mosek is spending too much time. Terminate it.
        #     print("Terminating.")
        #     return 1

        return 0
    return userCallback

def LadderMOSEK_test(number_of_nodes=3, numberOfBins=20, numberOfUnaries=10, interval=(-5, 18)):


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



        # --------------------------------------- MOSEK ----------------------------

    # Make a MOSEK environment
    with mosek.Env() as env:
        # Attach a printer to the environment
        env.set_Stream(mosek.streamtype.log, streamprinter)

        # Create a task
        with env.Task(0, 0) as task:
            # Attach a printer to the task
            task.set_Stream(mosek.streamtype.log, streamprinter)

            numberVariablesRVs = (2 * number_of_nodes + 1) * numberOfBins * numberOfUnaries

            # The variables will initially be fixed at zero (x=0).
            task.appendvars(numberVariablesRVs)

            # set variables to be boolean
            rvIndices = np.array(range(0, numberVariablesRVs))
            task.putvartypelist(rvIndices,
                                [mosek.variabletype.type_int] * numberVariablesRVs)

            startingNodes = []
            # set objective function
            for gate in range(0, number_of_nodes + 1):
                g = histogramGenerator.get_gauss_bins_UNARY(input_means[gate], input_stds[gate], numberOfBins,
                                                            n_samples,
                                                            interval, numberOfUnaries)
                bins = np.zeros((numberOfBins, numberOfUnaries)).astype(int)

                for bin in range(0, numberOfBins):
                    for unary in range(0, numberOfUnaries):
                        variableIndex = gate * numberOfBins * numberOfUnaries + bin * numberOfUnaries + unary

                        # Set the bounds on variable
                        # 0 <= x_j <= 1

                        task.putvarbound(variableIndex, mosek.boundkey.ra, 0.0, g.bins[bin, unary])

                        # save index to the bins
                        bins[bin, unary] = variableIndex

                node = Node(RandomVariableMOSEK(bins, g.edges, task))
                startingNodes.append(node)

            offset = (number_of_nodes + 1) * numberOfBins * numberOfUnaries

            # generetate nodes
            generatedNodes = []
            for gate in range(0, number_of_nodes):
                g = histogramGenerator.get_gauss_bins_UNARY(gateParams[0], gateParams[1], numberOfBins, n_samples,
                                                            interval,
                                                            numberOfUnaries)
                bins = np.zeros((numberOfBins, numberOfUnaries)).astype(int)

                for bin in range(0, numberOfBins):
                    for unary in range(0, numberOfUnaries):
                        variableIndex = offset + gate * numberOfBins * numberOfUnaries + bin * numberOfUnaries + unary

                        # Set the bounds on variable
                        # 0 <= x_j <= 1

                        task.putvarbound(variableIndex, mosek.boundkey.ra, 0.0, g.bins[bin, unary])

                        # save index to the bins
                        bins[bin, unary] = variableIndex

                node = Node(RandomVariableMOSEK(bins, g.edges, task))
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

            delays, newNofVariables = SSTA.calculateCircuitDelay(startingNodes, cvxpy=False, unary=True,
                                                                 mosekStatus=(numberVariablesRVs, 0),
                                                                 withSymmetryConstr=True)
            delays = delays[number_of_nodes + 1:]

            # setting objective
            # startingIndex = numberOfGates + 2
            sum = np.array([]).astype(int)
            for gate in range(0, number_of_nodes):
                sum = np.append(sum, np.concatenate(delays[gate].bins[:, :]))

            task.putclist(sum, [1] * sum.shape[0])

            # Input the objective sense (minimize/maximize)
            task.putobjsense(mosek.objsense.maximize)
                # set mip gap to 1%
            task.putdouparam(dparam.mio_tol_rel_gap, 1.0e-2)

            # Solve the problem
            task.optimize()
            # Print a summary containing information
            # about the solution for debugging purposes
            task.solutionsummary(mosek.streamtype.msg)

            prosta = task.getprosta(mosek.soltype.itg)
            solsta = task.getsolsta(mosek.soltype.itg)

            # Output a solution
            xx = np.array([0.] * newNofVariables)
            task.getxx(mosek.soltype.itg, xx)

            # delays[-1].bins = xx[delays[-1].bins]

            # rv = RandomVariable(delays[-1].bins, edges=delays[-1].edges, unary=True)
            # actual = [rv.mean, rv.std]
            #
            rvs = []


            for gate in range(0, len(delays)):  # construct RVs

                finalBins = xx[delays[gate].bins[:, :]]

                rvs.append(RandomVariable(finalBins, generatedNodes[0].randVar.edges, unary=True))

            print("\n MOSEK UNARY VALUES: \n")
            for i in range(0, number_of_nodes):
                print(rvs[i].mean, rvs[i].std)


    # ---------------------------- numpy ----------------------------------


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

    rvs2 = []

    for gate in range(0, number_of_nodes):  # construct RVs

        finalBins = np.zeros((numberOfBins, numberOfUnaries))
        for bin in range(0, numberOfBins):
            for unary in range(0, numberOfUnaries):
                finalBins[bin, unary] = ((delays[gate].bins)[bin])[unary]

        rvs2.append(RandomVariable(finalBins, generatedNodes[0].randVar.edges, unary=True))


    print("\n NUMPY UNARY VALUES: \n")
    for i in range(0, number_of_nodes):
        print(rvs2[i].mean, rvs2[i].std)

        # print

    # plot
    rvMOSEK = histogramGenerator.get_Histogram_from_UNARY(rvs[0])
    rvNump = histogramGenerator.get_Histogram_from_UNARY(rvs2[0])

    print(rvMOSEK.bins.shape)
    print(rvMOSEK.edges.shape)

    plt.hist(rvMOSEK.edges[:-1], rvMOSEK.edges, weights=rvMOSEK.bins, density="PDF", color='blue')
    plt.hist(rvNump.edges[:-1], rvNump.edges, weights=rvNump.bins, density="PDF",alpha=0.2, color='orange')
    plt.show()

    # ------------------- monte carlo ----------------------------------

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

def LadderMOSEK_maxconv_test(number_of_nodes=1, numberOfBins=13, numberOfUnaries=14, interval=(-5, 18)):

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



        # --------------------------------------- MOSEK ----------------------------

    # Make a MOSEK environment
    with mosek.Env() as env:
        # Attach a printer to the environment
        env.set_Stream(mosek.streamtype.log, streamprinter)

        # Create a task
        with env.Task(0, 0) as task:
            # Attach a printer to the task
            task.set_Stream(mosek.streamtype.log, streamprinter)

            numberVariablesRVs = (2 * number_of_nodes + 1) * numberOfBins * numberOfUnaries

            # The variables will initially be fixed at zero (x=0).
            task.appendvars(numberVariablesRVs)

            # set variables to be boolean
            rvIndices = np.array(range(0, numberVariablesRVs))
            task.putvartypelist(rvIndices,
                                [mosek.variabletype.type_int] * numberVariablesRVs)

            startingNodes = []
            # set objective function
            for gate in range(0, number_of_nodes + 1):
                g = histogramGenerator.get_gauss_bins_UNARY(input_means[gate], input_stds[gate], numberOfBins,
                                                            n_samples,
                                                            interval, numberOfUnaries)
                bins = np.zeros((numberOfBins, numberOfUnaries)).astype(int)

                for bin in range(0, numberOfBins):
                    for unary in range(0, numberOfUnaries):
                        variableIndex = gate * numberOfBins * numberOfUnaries + bin * numberOfUnaries + unary

                        # Set the bounds on variable
                        # 0 <= x_j <= 1

                        task.putvarbound(variableIndex, mosek.boundkey.ra, 0.0, g.bins[bin, unary])

                        # save index to the bins
                        bins[bin, unary] = variableIndex

                node = Node(RandomVariableMOSEK(bins, g.edges, task))
                startingNodes.append(node)

            offset = (number_of_nodes + 1) * numberOfBins * numberOfUnaries

            # generetate nodes
            generatedNodes = []
            for gate in range(0, number_of_nodes):
                g = histogramGenerator.get_gauss_bins_UNARY(gateParams[0], gateParams[1], numberOfBins, n_samples,
                                                            interval,
                                                            numberOfUnaries)
                bins = np.zeros((numberOfBins, numberOfUnaries)).astype(int)

                for bin in range(0, numberOfBins):
                    for unary in range(0, numberOfUnaries):
                        variableIndex = offset + gate * numberOfBins * numberOfUnaries + bin * numberOfUnaries + unary

                        # Set the bounds on variable
                        # 0 <= x_j <= 1

                        task.putvarbound(variableIndex, mosek.boundkey.ra, 0.0, g.bins[bin, unary])

                        # save index to the bins
                        bins[bin, unary] = variableIndex

                node = Node(RandomVariableMOSEK(bins, g.edges, task))
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

            delays, newNofVariables = SSTA.calculateCircuitDelay(startingNodes, cvxpy=False, unary=True,
                                                                 mosekStatus=(numberVariablesRVs, 0), mosekTRI=True)
            delays = delays[number_of_nodes + 1:]

            # setting objective
            # startingIndex = numberOfGates + 2
            sum = np.array([]).astype(int)
            for gate in range(0, number_of_nodes):
                sum = np.append(sum, np.concatenate(delays[gate].bins[:, :]))

            task.putclist(sum, [1] * sum.shape[0])

            # Input the objective sense (minimize/maximize)
            task.putobjsense(mosek.objsense.maximize)
                # set mip gap to 1%
            task.putdouparam(dparam.mio_tol_rel_gap, 1.0e-2)

            # Solve the problem
            task.optimize()
            # Print a summary containing information
            # about the solution for debugging purposes
            task.solutionsummary(mosek.streamtype.msg)

            prosta = task.getprosta(mosek.soltype.itg)
            solsta = task.getsolsta(mosek.soltype.itg)

            # Output a solution
            xx = np.array([0.] * newNofVariables)
            task.getxx(mosek.soltype.itg, xx)

            # delays[-1].bins = xx[delays[-1].bins]

            # rv = RandomVariable(delays[-1].bins, edges=delays[-1].edges, unary=True)
            # actual = [rv.mean, rv.std]
            #
            rvs = []


            for gate in range(0, len(delays)):  # construct RVs

                finalBins = xx[delays[gate].bins[:, :]]

                rvs.append(RandomVariable(finalBins, generatedNodes[0].randVar.edges, unary=True))

            print("\n MOSEK UNARY VALUES: \n")
            for i in range(0, number_of_nodes):
                print(rvs[i].mean, rvs[i].std)


    # ---------------------------- numpy ----------------------------------


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

    delays = SSTA.calculateCircuitDelay(startingNodes, cvxpy=False, unary=True, mosekTRI=True)

    delays = delays[number_of_nodes + 1:]

    rvs2 = []

    for gate in range(0, number_of_nodes):  # construct RVs

        finalBins = np.zeros((numberOfBins, numberOfUnaries))
        for bin in range(0, numberOfBins):
            for unary in range(0, numberOfUnaries):
                finalBins[bin, unary] = ((delays[gate].bins)[bin])[unary]

        rvs2.append(RandomVariable(finalBins, generatedNodes[0].randVar.edges, unary=True))


    print("\n NUMPY UNARY VALUES: \n")
    for i in range(0, number_of_nodes):
        print(rvs2[i].mean, rvs2[i].std)

        # print

    # plot
    rvMOSEK = histogramGenerator.get_Histogram_from_UNARY(rvs[-1])
    rvNump = histogramGenerator.get_Histogram_from_UNARY(rvs2[-1])

    print(rvMOSEK.bins.shape)
    print(rvMOSEK.edges.shape)

    plt.hist(rvMOSEK.edges[:-1], rvMOSEK.edges, weights=rvMOSEK.bins, density="PDF", color='blue')
    plt.hist(rvNump.edges[:-1], rvNump.edges, weights=rvNump.bins, density="PDF",alpha=0.7, color='orange')
    plt.show()

    # ------------------- monte carlo ----------------------------------

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

    # mainCVXPY()
    mainCVXPY_GP()
    # mainCVXPYMcCormick()
    # LadderNumpy()
    # LadderMOSEK_test()
    # LadderMOSEK_maxconv_test(number_of_nodes=2, numberOfBins=10, numberOfUnaries=12, interval=(-5, 18))

    # print("All tests passed!")
