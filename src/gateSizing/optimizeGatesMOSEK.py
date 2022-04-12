import numpy as np
import cvxpy as cp

from node import Node

from cvxpyVariable import RandomVariableCVXPY

import histogramGenerator
import SSTA
import networkx as nx
import mosek
from test_SSTA_Numpy import putTuplesIntoArray
from randomVariableHist_Numpy import RandomVariable
from mosek import *
import sys
from mosekVariable import RandomVariableMOSEK
import matplotlib.pyplot as plt

from examples_monteCarlo.montecarlo import get_inputs, get_unknown_nodes, simulation, preprocess



"""
This module includes gate sizing optimization of the c17 circuit in a function 'optimizeGates'.
Some test for MOSEK ssta can be found aswell. 
Implemented in MOSEK

"""

# Define a stream printer to grab output from MOSEK
def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()



def test_SSTA_MAX():

    with mosek.Env() as env:
        # Attach a printer to the environment
        env.set_Stream(mosek.streamtype.log, streamprinter)

        # Create a task
        with env.Task(0, 0) as task:

            numberOfGates = 6

            numberOfBins = 8
            numberOfUnaries = 8
            binsInterval = (0, 10)
            numberOfSamples = 20000000

            # Attach a printer to the task
            task.set_Stream(mosek.streamtype.log, streamprinter)

            numberVariablesRVs = numberOfGates * numberOfBins * numberOfUnaries

            # The variables will initially be fixed at zero (x=0).
            task.appendvars(numberVariablesRVs)

            # set variables to be boolean
            rvIndices = np.array(range(0, numberVariablesRVs))
            task.putvartypelist(rvIndices,
                                [mosek.variabletype.type_int] * numberVariablesRVs)

            gateNodes = []
            # set objective function
            for gate in range(0, numberOfGates):
                g = histogramGenerator.get_gauss_bins_UNARY(3, 1, numberOfBins,
                                                            numberOfSamples,
                                                            binsInterval, numberOfUnaries)
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
                gateNodes.append(node)



            # set circuit design
            # gateNodes[0].setNextNodes([gateNodes[2], gateNodes[3]])
            # gateNodes[1].setNextNodes([gateNodes[2], gateNodes[4]])
            # gateNodes[2].setNextNodes([gateNodes[3], gateNodes[4]])

            gateNodes[0].setNextNodes([gateNodes[4]])
            gateNodes[1].setNextNodes([gateNodes[2], gateNodes[2], gateNodes[3], gateNodes[3]])
            gateNodes[2].setNextNodes([gateNodes[4], gateNodes[5]])
            gateNodes[3].setNextNodes([gateNodes[5]])

            # print(gateNodes[1].nextNodes)

            # calculate delay with ssta
            startingNodes = [gateNodes[1], gateNodes[0]]

            delays, newNofVariables = SSTA.calculateCircuitDelay(startingNodes, cvxpy=False, unary=True,
                                                                 mosekStatus=(numberVariablesRVs, 0), mosekTRI=True)

            # delays=delays[-1]

            # setting objective
            # startingIndex = numberOfGates + 2
            sum = np.array([]).astype(int)
            # for gate in range(0, numberOfGates + 1):
            #     sum = np.append(sum, np.concatenate(delays[gate].bins[:, :]))
            # sum = np.append(sum, np.concatenate(delays[-3].bins[:, :]))
            # sum = np.append(sum, np.concatenate(delays[-2].bins[:, :]))
            sum = np.append(sum, np.concatenate(delays[-1].bins[:, :]))

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

            rvs = []

            for gate in range(0, len(delays)):  # construct RVs

                finalBins = xx[delays[gate].bins[:, :]]

                rvs.append(RandomVariable(finalBins, gateNodes[0].randVar.edges, unary=True))

            print("\n MOSEK UNARY VALUES: \n")
            for i in range(0, numberOfGates + 1):
                print(rvs[i].mean, rvs[i].std)


    # numpy

    g1 = histogramGenerator.get_gauss_bins_UNARY(3, 1, numberOfBins, numberOfSamples,
                                           binsInterval, numberOfUnaries)  # g1, g2 INPUT gates, g3 middle
    g2 = histogramGenerator.get_gauss_bins_UNARY(3, 1, numberOfBins, numberOfSamples,
                                           binsInterval, numberOfUnaries)  # g4 output - inputs: g3 g1
    g3 = histogramGenerator.get_gauss_bins_UNARY(3, 1, numberOfBins, numberOfSamples,
                                           binsInterval, numberOfUnaries)  # g5 output - inputs: g3, g2
    g4 = histogramGenerator.get_gauss_bins_UNARY(3, 1, numberOfBins, numberOfSamples, binsInterval, numberOfUnaries)
    g5 = histogramGenerator.get_gauss_bins_UNARY(3, 1, numberOfBins, numberOfSamples, binsInterval, numberOfUnaries)
    g6 = histogramGenerator.get_gauss_bins_UNARY(3, 1, numberOfBins, numberOfSamples, binsInterval, numberOfUnaries)

    n1 = Node(g1)
    n2 = Node(g2)
    n3 = Node(g3)
    n4 = Node(g4)
    n5 = Node(g5)
    n6 = Node(g6)

    # set circuit design
    n1.setNextNodes([n5])
    n2.setNextNodes([n3, n3, n4, n4])
    n3.setNextNodes([n5, n6])
    n4.setNextNodes([n6])


    delays = SSTA.calculateCircuitDelay([n2, n1], cvxpy=False, unary=True, mosekTRI=True)

    # actual = putTuplesIntoArray(rvs=delays)

    rvs2 = []

    for gate in range(0, numberOfGates + 1):  # construct RVs

        finalBins = np.zeros((numberOfBins, numberOfUnaries))
        for bin in range(0, numberOfBins):
            for unary in range(0, numberOfUnaries):
                finalBins[bin, unary] = ((delays[gate].bins)[bin])[unary]

        rvs2.append(RandomVariable(finalBins, gateNodes[0].randVar.edges, unary=True))

    print("\n NUMPY UNARY VALUES: \n")
    for i in range(0, numberOfGates + 1):
        print(rvs2[i].mean, rvs2[i].std)


def test_SSTA_MIN():

    with mosek.Env() as env:
        # Attach a printer to the environment
        env.set_Stream(mosek.streamtype.log, streamprinter)

        # Create a task
        with env.Task(0, 0) as task:

            numberOfGates = 6

            numberOfBins = 8
            numberOfUnaries = 8
            binsInterval = (0, 10)
            numberOfSamples = 20000000

            # Attach a printer to the task
            task.set_Stream(mosek.streamtype.log, streamprinter)

            numberVariablesRVs = numberOfGates * numberOfBins * numberOfUnaries

            # The variables will initially be fixed at zero (x=0).
            task.appendvars(numberVariablesRVs)

            # set variables to be boolean
            rvIndices = np.array(range(0, numberVariablesRVs))
            task.putvartypelist(rvIndices,
                                [mosek.variabletype.type_int] * numberVariablesRVs)

            gateNodes = []
            # set objective function
            for gate in range(0, numberOfGates):
                g = histogramGenerator.get_gauss_bins_UNARY(3, 1, numberOfBins,
                                                            numberOfSamples,
                                                            binsInterval, numberOfUnaries)
                bins = np.zeros((numberOfBins, numberOfUnaries)).astype(int)

                for bin in range(0, numberOfBins):
                    for unary in range(0, numberOfUnaries):
                        variableIndex = gate * numberOfBins * numberOfUnaries + bin * numberOfUnaries + unary

                        # Set the bounds on variable
                        # 0 <= x_j <= 1

                        task.putvarbound(variableIndex, mosek.boundkey.ra, g.bins[bin, unary], 1)

                        # save index to the bins
                        bins[bin, unary] = variableIndex

                node = Node(RandomVariableMOSEK(bins, g.edges, task))
                gateNodes.append(node)



            # set circuit design
            # gateNodes[0].setNextNodes([gateNodes[2], gateNodes[3]])
            # gateNodes[1].setNextNodes([gateNodes[2], gateNodes[4]])
            # gateNodes[2].setNextNodes([gateNodes[3], gateNodes[4]])

            gateNodes[0].setNextNodes([gateNodes[4]])
            gateNodes[1].setNextNodes([gateNodes[2], gateNodes[2], gateNodes[3], gateNodes[3]])
            gateNodes[2].setNextNodes([gateNodes[4], gateNodes[5]])
            gateNodes[3].setNextNodes([gateNodes[5]])

            # print(gateNodes[1].nextNodes)

            # calculate delay with ssta
            startingNodes = [gateNodes[1], gateNodes[0]]

            delays, newNofVariables = SSTA.calculateCircuitDelay(startingNodes, cvxpy=False, unary=True,
                                                                 mosekStatus=(numberVariablesRVs, 0), mosekTRI=True)

            # delays=delays[-1]

            # setting objective
            # startingIndex = numberOfGates + 2
            sum = np.array([]).astype(int)
            # for gate in range(0, numberOfGates + 1):
            #     sum = np.append(sum, np.concatenate(delays[gate].bins[:, :]))
            # sum = np.append(sum, np.concatenate(delays[-3].bins[:, :]))
            # sum = np.append(sum, np.concatenate(delays[-2].bins[:, :]))
            sum = np.append(sum, np.concatenate(delays[-1].bins[:, :]))

            task.putclist(sum, [1] * sum.shape[0])

            # Input the objective sense (minimize/maximize)
            task.putobjsense(mosek.objsense.minimize)
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

            rvs = []

            for gate in range(0, len(delays)):  # construct RVs

                finalBins = xx[delays[gate].bins[:, :]]

                rvs.append(RandomVariable(finalBins, gateNodes[0].randVar.edges, unary=True))

            print("\n MOSEK UNARY VALUES: \n")
            for i in range(0, numberOfGates + 1):
                print(rvs[i].mean, rvs[i].std)


    # numpy

    # g1 = histogramGenerator.get_gauss_bins_UNARY(3, 1, numberOfBins, numberOfSamples,
    #                                        binsInterval, numberOfUnaries)  # g1, g2 INPUT gates, g3 middle
    # g2 = histogramGenerator.get_gauss_bins_UNARY(3, 1, numberOfBins, numberOfSamples,
    #                                        binsInterval, numberOfUnaries)  # g4 output - inputs: g3 g1
    # g3 = histogramGenerator.get_gauss_bins_UNARY(3, 1, numberOfBins, numberOfSamples,
    #                                        binsInterval, numberOfUnaries)  # g5 output - inputs: g3, g2
    # g4 = histogramGenerator.get_gauss_bins_UNARY(3, 1, numberOfBins, numberOfSamples, binsInterval, numberOfUnaries)
    # g5 = histogramGenerator.get_gauss_bins_UNARY(3, 1, numberOfBins, numberOfSamples, binsInterval, numberOfUnaries)
    # g6 = histogramGenerator.get_gauss_bins_UNARY(3, 1, numberOfBins, numberOfSamples, binsInterval, numberOfUnaries)


    g1 = histogramGenerator.get_gauss_bins(3, 1, numberOfBins, numberOfSamples,
                                           binsInterval)  # g1, g2 INPUT gates, g3 middle

    n1 = Node(g1)
    n2 = Node(g1)
    n3 = Node(g1)
    n4 = Node(g1)
    n5 = Node(g1)
    n6 = Node(g1)

    # set circuit design
    n1.setNextNodes([n5])
    n2.setNextNodes([n3, n3, n4, n4])
    n3.setNextNodes([n5, n6])
    n4.setNextNodes([n6])


    delays = SSTA.calculateCircuitDelay([n2, n1], cvxpy=False, unary=False, mosekTRI=False)

    actual = putTuplesIntoArray(rvs=delays)
    print(actual)

    edges = rvs[-1].edges
    mosekBins = histogramGenerator.get_Histogram_from_UNARY(rvs[-1]).bins
    numpyBins = delays[-1].bins


    plt.hist(delays[-1].edges[:-1], delays[-1].edges[:-1], weights=delays[-1].bins, alpha=0.4, density="PDF")
    plt.hist(delays[-1].edges[:-1], delays[-1].edges[:-1], weights=mosekBins, density="PDF")
    plt.legend(['Numpy', 'Histogram approximation'])
    plt.show()

    # rvs2 = []
    #
    # for gate in range(0, numberOfGates + 1):  # construct RVs
    #
    #     finalBins = np.zeros((numberOfBins, numberOfUnaries))
    #     for bin in range(0, numberOfBins):
    #         for unary in range(0, numberOfUnaries):
    #             finalBins[bin, unary] = ((delays[gate].bins)[bin])[unary]
    #
    #     rvs2.append(RandomVariable(finalBins, gateNodes[0].randVar.edges, unary=True))
    #
    # print("\n NUMPY UNARY VALUES: \n")
    # for i in range(0, numberOfGates + 1):
    #     print(rvs2[i].mean, rvs2[i].std)


def optimizeGates():

    with mosek.Env() as env:
        # Attach a printer to the environment
        env.set_Stream(mosek.streamtype.log, streamprinter)

        # Create a task
        with env.Task(0, 0) as task:

            numberOfGates = 6

            numberOfBins = 7
            numberOfUnaries = 8
            binsInterval = (0, 10)
            numberOfSamples = 20000000

            # Attach a printer to the task
            task.set_Stream(mosek.streamtype.log, streamprinter)

            numberVariablesRVs = numberOfGates * numberOfBins * numberOfUnaries

            # The variables will initially be fixed at zero (x=0).
            task.appendvars(numberVariablesRVs)

            # set variables to be boolean
            rvIndices = np.array(range(0, numberVariablesRVs))
            task.putvartypelist(rvIndices,
                                [mosek.variabletype.type_int] * numberVariablesRVs)

            gateNodes = []

            g = histogramGenerator.get_gauss_bins_UNARY(3, 1, numberOfBins,
                                                        numberOfSamples,
                                                        binsInterval, numberOfUnaries)

            for gate in range(0, numberOfGates):
                bins = np.zeros((numberOfBins, numberOfUnaries)).astype(int)

                for bin in range(0, numberOfBins):
                    for unary in range(0, numberOfUnaries):
                        variableIndex = gate * numberOfBins * numberOfUnaries + bin * numberOfUnaries + unary

                        # Set the bounds on variable
                        # 0 <= x_j <= 1

                        task.putvarbound(variableIndex, mosek.boundkey.ra, 0, 1)
                        # task.putvarbound(variableIndex, mosek.boundkey.ra, g.bins[bin][unary], 1)

                        # save index to the bins
                        bins[bin, unary] = variableIndex

                node = Node(RandomVariableMOSEK(bins, g.edges, task))
                gateNodes.append(node)



            # set circuit design

            gateNodes[0].setNextNodes([gateNodes[4]])
            gateNodes[1].setNextNodes([gateNodes[2], gateNodes[2], gateNodes[3], gateNodes[3]])
            gateNodes[2].setNextNodes([gateNodes[4], gateNodes[5]])
            gateNodes[3].setNextNodes([gateNodes[5]])

            # print(gateNodes[1].nextNodes)

            # calculate delay with ssta
            startingNodes = [gateNodes[1], gateNodes[0]]

            delays, newNofVariables, newNoConstr = SSTA.calculateCircuitDelay(startingNodes, cvxpy=False, unary=True,
                                                                 mosekStatus=(numberVariablesRVs, 0), mosekTRI=True)

            # delays=delays[-1]

            # setting objective
            # startingIndex = numberOfGates + 2
            sum = np.array([]).astype(int)
            # for gate in range(0, numberOfGates + 1):
            #     sum = np.append(sum, np.concatenate(delays[gate].bins[:, :]))

                # last two
            sum = np.append(sum, np.concatenate(delays[-3].bins[:, :]))
            sum = np.append(sum, np.concatenate(delays[-2].bins[:, :]))

                # sink gate
            # sum = np.append(sum, np.concatenate(delays[-1].bins[:, :]))

            task.putclist(sum, [1] * sum.shape[0])


            # set the area and power constraints

            f = np.array([4, 0.8, 1, 0.8, 1.7, 0.5])
            e = np.array([1, 2, 1, 1.5, 1.5, 1])
            a = np.ones(numberOfGates)
            Amax = 30
            Pmax = 55

            # create sizing variables
            inf = 0.0
            task.appendvars(numberOfGates)
            sizingVariables = np.array(range(newNofVariables, newNofVariables + numberOfGates)).astype(int)

            task.putvarboundlist(sizingVariables, [mosek.boundkey.lo] * numberOfGates,
                                 [1] * numberOfGates, [+inf]*numberOfGates)  # binary
            newNofVariables += numberOfGates

            # create constraints

            task.appendcons(2)

                # constraint for area
            task.putarow(newNoConstr, sizingVariables,  a)
            task.putconbound(newNoConstr, mosek.boundkey.up, 0.0, Amax)

                # constraint for power
            task.putarow(newNoConstr + 1, sizingVariables, np.multiply(f, e))
            task.putconbound(newNoConstr + 1, mosek.boundkey.up, 0.0, Pmax)

            newNoConstr += 2

            # connect sizing constraints to histograms

            task.appendcons(numberOfGates*numberOfBins)

            print(g.bins)


            for gate in range(0, numberOfGates):

                curNode = gateNodes[gate]

                for bin in range(0, numberOfBins):

                    generatedValue = np.sum(g.bins[bin, :])

                    # print(generatedValues)

                    row = curNode.randVar.bins[bin, :]

                    task.putarow(newNoConstr + gate*numberOfBins + bin, row, [1]* row.size)
                    task.putconbound(newNoConstr + gate*numberOfBins + bin, mosek.boundkey.fx, generatedValue, generatedValue)



                # node = Node(RandomVariableMOSEK(bins, g.edges, task))
                # gateNodes.append(node)

            # exit(-1)

            #######################
            ######OPTIMIZE#########
            #######################


            # Input the objective sense (minimize/maximize)
            task.putobjsense(mosek.objsense.minimize)
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

            print('Sizing parameters:')
            print(xx[sizingVariables])

            rvs = []

            for gate in range(0, len(delays)):  # construct RVs

                finalBins = xx[delays[gate].bins[:, :]]

                rvs.append(RandomVariable(finalBins, gateNodes[0].randVar.edges, unary=True))

            print("\n MOSEK UNARY VALUES: \n")
            for i in range(0, numberOfGates + 1):
                print(rvs[i].mean, rvs[i].std)


    # numpy

    # g1 = histogramGenerator.get_gauss_bins_UNARY(3, 1, numberOfBins, numberOfSamples,
    #                                        binsInterval, numberOfUnaries)  # g1, g2 INPUT gates, g3 middle
    # g2 = histogramGenerator.get_gauss_bins_UNARY(3, 1, numberOfBins, numberOfSamples,
    #                                        binsInterval, numberOfUnaries)  # g4 output - inputs: g3 g1
    # g3 = histogramGenerator.get_gauss_bins_UNARY(3, 1, numberOfBins, numberOfSamples,
    #                                        binsInterval, numberOfUnaries)  # g5 output - inputs: g3, g2
    # g4 = histogramGenerator.get_gauss_bins_UNARY(3, 1, numberOfBins, numberOfSamples, binsInterval, numberOfUnaries)
    # g5 = histogramGenerator.get_gauss_bins_UNARY(3, 1, numberOfBins, numberOfSamples, binsInterval, numberOfUnaries)
    # g6 = histogramGenerator.get_gauss_bins_UNARY(3, 1, numberOfBins, numberOfSamples, binsInterval, numberOfUnaries)


    g1 = histogramGenerator.get_gauss_bins(3, 1, numberOfBins, numberOfSamples,
                                           binsInterval)  # g1, g2 INPUT gates, g3 middle

    n1 = Node(g1)
    n2 = Node(g1)
    n3 = Node(g1)
    n4 = Node(g1)
    n5 = Node(g1)
    n6 = Node(g1)

    # set circuit design
    n1.setNextNodes([n5])
    n2.setNextNodes([n3, n3, n4, n4])
    n3.setNextNodes([n5, n6])
    n4.setNextNodes([n6])


    delays = SSTA.calculateCircuitDelay([n2, n1], cvxpy=False, unary=False, mosekTRI=False)

    actual = putTuplesIntoArray(rvs=delays)
    print(actual)


if __name__ == "__main__":

    # test_SSTA_MAX()
    # test_SSTA_MIN()
    optimizeGates()
