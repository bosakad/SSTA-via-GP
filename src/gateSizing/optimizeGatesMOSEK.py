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
    """
    This function optimizes the c17 circuit.
    """

    numberOfGates = 8

    numberOfBins = 10
    numberOfUnaries = 10
    binsInterval = (0, 20)
    numberOfSamples = 20000000

    coef = np.load('Inputs.outputs/model.npz')
    model = coef['coef']

    f = np.array([4, 0.8, 1, 0.8, 1.7, 0.5])
    e = np.array([1, 2, 1, 1.5, 1.5, 1])
    a = np.ones(numberOfGates - 2)

    # generate gates
    rv1 = histogramGenerator.generateAccordingToModel(model, 1, f[0]*e[0], x_i=10, int=binsInterval, nUnaries=numberOfUnaries)
    rv2 = histogramGenerator.generateAccordingToModel(model, 1, f[1]*e[1], x_i=2.5, int=binsInterval, nUnaries=numberOfUnaries)
    rv3 = histogramGenerator.generateAccordingToModel(model, 1, f[2]*e[2], x_i=4, int=binsInterval, nUnaries=numberOfUnaries)
    rv4 = histogramGenerator.generateAccordingToModel(model, 1, f[3]*e[3], x_i=2, int=binsInterval, nUnaries=numberOfUnaries)
    rv5 = histogramGenerator.generateAccordingToModel(model, 1, f[4]*e[4], x_i=2, int=binsInterval, nUnaries=numberOfUnaries)
    rv6 = histogramGenerator.generateAccordingToModel(model, 1, f[5]*e[5], x_i=2, int=binsInterval, nUnaries=numberOfUnaries)

    # generate inputs
    in1 = histogramGenerator.generateAccordingToModel(model, 1, 4, x_i=2.5, int=binsInterval, nUnaries=numberOfUnaries)
    in2 = histogramGenerator.generateAccordingToModel(model, 1, 4, x_i=2.5, int=binsInterval, nUnaries=numberOfUnaries)


    #######################
    ######MOSEK############
    #######################

    with mosek.Env() as env:
        # Attach a printer to the environment
        env.set_Stream(mosek.streamtype.log, streamprinter)

        # Create a task
        with env.Task(0, 0) as task:

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

            RVs = [rv1, rv2, rv3, rv4, rv5, rv6, in1, in2]
            for gate in range(0, numberOfGates):
                bins = np.zeros((numberOfBins, numberOfUnaries)).astype(int)

                g = RVs[gate]

                for bin in range(0, numberOfBins):
                    for unary in range(0, numberOfUnaries):
                        variableIndex = gate * numberOfBins * numberOfUnaries + bin * numberOfUnaries + unary

                        # Set the bounds on variable
                        # 0 <= x_j <= 1

                        if gate == 6 or gate == 7 or gate == 1 or gate == 2 or gate == 3 or gate == 4 or gate == 5:
                            task.putvarbound(variableIndex, mosek.boundkey.ra, g.bins[bin][unary], 1)
                        else:
                            task.putvarbound(variableIndex, mosek.boundkey.ra, 0, 1)

                        # save index to the bins
                        bins[bin, unary] = variableIndex

                node = Node(RandomVariableMOSEK(bins, rv1.edges, task))
                gateNodes.append(node)


                # set circuit design
            gateNodes[0].setNextNodes([gateNodes[4]])
            gateNodes[1].setNextNodes([gateNodes[2], gateNodes[3]])
            gateNodes[2].setNextNodes([gateNodes[4], gateNodes[5]])
            gateNodes[3].setNextNodes([gateNodes[5]])

            IN1 = gateNodes[-2]
            IN2 = gateNodes[-1]

            IN1.setNextNodes([gateNodes[2]])
            IN2.setNextNodes([gateNodes[3]])

            startingNodes = [gateNodes[1], IN1, IN2, gateNodes[0]]

                # calculate delay
            delays, newNofVariables, newNoConstr = SSTA.calculateCircuitDelay(startingNodes, cvxpy=False, unary=True,
                                                                 mosekStatus=(numberVariablesRVs, 0), mosekTRI=True)

            numberOfGates = 6

            # delays=delays[-1]

            # setting objective
            # startingIndex = numberOfGates + 2
            sum = np.array([]).astype(int)
            for gate in range(0, numberOfGates + 3):
                sum = np.append(sum, np.concatenate(delays[gate].bins[:, :]))

                # last two
            # sum = np.append(sum, np.concatenate(delays[-3].bins[:, :]))
            # sum = np.append(sum, np.concatenate(delays[0].bins[:, :]))

                # sink gate
            # sum = np.append(sum, np.concatenate(delays[-2].bins[:, :]))
            # sum = np.append(sum, np.concatenate(delays[-3].bins[:, :]))
            # sum = np.append(sum, np.concatenate(delays[-1].bins[:, :]))

            midPoints = np.array(0.5 * (g.edges[1:] + g.edges[:-1]))
            midPoints = np.tile(midPoints, (numberOfUnaries, 1)).T
            midPoints = np.concatenate(midPoints)

            # task.putclist(sum, np.square(midPoints))
                # todo: zkus maly vahy na zacatku, velky na konci
            task.putclist(sum, [1] * sum.shape[0])


            # set the area and power constraints
            # Amax = 1200
            # Pmax = 10000
            #
            #
            # # create sizing variables
            # inf = 0.0
            # task.appendvars(numberOfGates)
            # sizingVariables = np.array(range(newNofVariables, newNofVariables + numberOfGates)).astype(int)
            #
            # # task.putvarboundlist(sizingVariables, [mosek.boundkey.fx] * numberOfGates,
            # #                      [3] * numberOfGates, [3]*numberOfGates)
            # # task.putvarboundlist(sizingVariables, [mosek.boundkey.lo] * numberOfGates,
            #                      # [9, 1.5, 2.5, 2.5, 1.5, 1.5], [10]*numberOfGates)
            #                      # [9.5, 9, 9, 9, 9, 9], [10]*numberOfGates)
            #                      # [9.5], [10])
            #
            # newNofVariables += numberOfGates
            #
            # # create constraints
            #
            # task.appendcons(2)
            #
            #     # constraint for area
            # task.putarow(newNoConstr, sizingVariables,  a[:1])
            # task.putconbound(newNoConstr, mosek.boundkey.up, 0.0, Amax)
            #
            #     # constraint for power
            # task.putarow(newNoConstr + 1, sizingVariables, np.multiply(f, e)[:1])
            # task.putconbound(newNoConstr + 1, mosek.boundkey.up, 0.0, Pmax)
            #
            # newNoConstr += 2




            # connect sizing constraints to histograms
            task.appendcons(2*numberOfGates*numberOfBins)



                # introduce fit constraints
            for gate in range(0, 1):

                curNode = gateNodes[gate]
                # x_i = sizingVariables[gate]
                a_i = a[gate]
                f_i = f[gate]
                e_i = e[gate]

                for bin in range(0, numberOfBins):

                    generatedValues = np.sum(rv1.bins[bin, :])
                    model = loadModel('Inputs.outputs/model.npz')
                    # model = model

                    round = 0.5
                    shift = model[bin, 0]
                    areaCoef = model[bin, 1]
                    powerCoef = model[bin, 2]


                    row = curNode.randVar.bins[bin, :]

                    offset1 = newNoConstr + gate * numberOfBins + bin
                    offset2 = newNoConstr + numberOfGates * numberOfBins + gate * numberOfBins + bin

                    task.putarow(offset1, row, [1] * row.size)
                    task.putarow(offset2, row, [1] * row.size)

                    task.putconbound(offset1, mosek.boundkey.up, generatedValues + 0.1, generatedValues+0.1)
                    task.putconbound(offset2, mosek.boundkey.lo, generatedValues - 0.1, generatedValues - 0.1)

                    # sizingValue = -numberOfUnaries * (areaCoef * a_i + powerCoef * f_i * e_i)
                    # task.putaij(offset1, x_i, sizingValue)
                    # task.putaij(offset2, x_i, sizingValue)
                    #
                    # task.putconbound(offset1, mosek.boundkey.up, 0.0, shift * numberOfUnaries + round)
                    # task.putconbound(offset2, mosek.boundkey.lo, shift * numberOfUnaries - round, 0.0)



            newNofConstr = newNoConstr + numberOfGates * numberOfBins * 2
            task.appendcons(numberOfGates * numberOfBins * (numberOfUnaries - 1))


            # introduce separation constraints
            for gate in range(0, numberOfGates):
                curNode = gateNodes[gate]

                # symmetry constraints
                for bin in range(0, numberOfBins):
                    for unary in range(0, numberOfUnaries - 1):
                        offset = bin * (numberOfUnaries - 1) + unary

                        # (maximum[bin])[unary] >= (maximum[bin])[unary + 1])
                        task.putaij(newNofConstr + offset, curNode.randVar.bins[bin, unary], 1)
                        task.putaij(newNofConstr + offset, curNode.randVar.bins[bin, unary + 1], -1)

                        task.putconbound(newNofConstr + offset, mosek.boundkey.lo, 0, 0.0)

            newNofConstr += (numberOfUnaries - 1) * numberOfBins



            #######################
            ######OPTIMIZE#########
            #######################



            task.putobjsense(mosek.objsense.minimize)
            # set mip gap to 1%
            task.putdouparam(dparam.mio_tol_rel_gap, 1.0e-2)

            # Solve the problem
            task.optimize()


            task.solutionsummary(mosek.streamtype.msg)

            prosta = task.getprosta(mosek.soltype.itg)
            solsta = task.getsolsta(mosek.soltype.itg)

            # Output a solution
            xx = np.array([0.] * newNofVariables)
            task.getxx(mosek.soltype.itg, xx)

            # print('Sizing parameters:')
            # print(xx[sizingVariables])

            rvs = []

            for gate in range(0, len(delays)):  # construct RVs

                finalBins = xx[delays[gate].bins[:, :]]
                rvs.append(RandomVariable(finalBins, gateNodes[0].randVar.edges, unary=True))

            print("\n MOSEK UNARY VALUES: \n")
            for i in range(0, numberOfGates + 3):
                print(rvs[i].mean, rvs[i].std)





    #######################
    ######NUMPY############
    #######################

    n1 = Node(rv1)
    n2 = Node(rv2)
    n3 = Node(rv3)
    n4 = Node(rv4)
    n5 = Node(rv5)
    n6 = Node(rv6)

    IN1 = Node(in1)
    IN2 = Node(in2)

    # set circuit design
    n1.setNextNodes([n5])
    # n2.setNextNodes([n3, n3, n4, n4])
    IN1.setNextNodes([n3])
    IN2.setNextNodes([n4])
    n2.setNextNodes([n3, n4])
    n3.setNextNodes([n5, n6])
    n4.setNextNodes([n6])


    delays = SSTA.calculateCircuitDelay([n2,IN1, IN2, n1], cvxpy=False, unary=True, mosekTRI=True)

    actual = putTuplesIntoArray(rvs=delays)
    print(actual)

def optimizeCVXPY_GP():

    numberOfBins = 6
    binsInterval = (0, 20)
    numberOfSamples = 20000000
    numberOfGates = 8

    coef = np.load('Inputs.outputs/model.npz')
    model = coef['coef']

    f = np.array([4, 0.8, 1, 0.8, 1.7, 0.5])
    e = np.array([1, 2, 1, 1.5, 1.5, 1])
    a = np.ones(numberOfGates - 2)

    # generate gates
    rv1 = histogramGenerator.generateAccordingToModel(model, 1, f[0] * e[0], x_i=10, int=binsInterval)

    rv2 = histogramGenerator.generateAccordingToModel(model, 1, f[1] * e[1], x_i=2.5, int=binsInterval)
    rv3 = histogramGenerator.generateAccordingToModel(model, 1, f[2] * e[2], x_i=4, int=binsInterval)
    rv4 = histogramGenerator.generateAccordingToModel(model, 1, f[3] * e[3], x_i=2, int=binsInterval)
    rv5 = histogramGenerator.generateAccordingToModel(model, 1, f[4] * e[4], x_i=2, int=binsInterval)
    rv6 = histogramGenerator.generateAccordingToModel(model, 1, f[5] * e[5], x_i=2, int=binsInterval)

    # generate inputs
    in1 = histogramGenerator.generateAccordingToModel(model, 1, 4, x_i=2.5, int=binsInterval)
    in2 = histogramGenerator.generateAccordingToModel(model, 1, 4, x_i=2.5, int=binsInterval)


    gateNodes = []
    constr = []

    RVs = [rv1, rv2, rv3, rv4, rv5, rv6, in1, in2]

    for gate in range(0, numberOfGates):
        bins = {}

        g = RVs[gate]

        for bin in range(0, numberOfBins):

            var = cp.Variable(pos=True)

            if g.bins[bin] == 0: g.bins[bin] += 0.0000000000000000000000000000001

            # if gate >= 6:   # for input gates
            constr.append(var >= g.bins[bin])

            bins[bin] = var

        node = Node(RandomVariableCVXPY(bins, rv1.edges))
        gateNodes.append(node)

        # set circuit design
    gateNodes[0].setNextNodes([gateNodes[4]])
    gateNodes[1].setNextNodes([gateNodes[2], gateNodes[3]])
    gateNodes[2].setNextNodes([gateNodes[4], gateNodes[5]])
    gateNodes[3].setNextNodes([gateNodes[5]])

    IN1 = gateNodes[-2]
    IN2 = gateNodes[-1]

    IN1.setNextNodes([gateNodes[2]])
    IN2.setNextNodes([gateNodes[3]])

    startingNodes = [gateNodes[1], IN1, IN2, gateNodes[0]]

    # calculate delay
    delays, newConstr = SSTA.calculateCircuitDelay(startingNodes, cvxpy=True, GP=True)
    constr.extend(newConstr)

    sum = 0
    norm = 0
    midPoints = 0.5 * (delays[-1].edges[1:] + delays[-1].edges[:-1])  # midpoints of the edges of hist.
    # for gate in range(0, len(delays)):

    for bin in range(0, numberOfBins):
        sum += delays[-1].bins[bin]                # minimize the mean value
        norm += delays[-1].bins[bin]

    # sum = sum / (delays[-1].bins[0] + delays[-1].bins[1])




    # create sizing parameters

    Amax = 100
    Pmax = 100

    sizingVariables = cp.Variable((6,), pos=True)
    constr.append(sizingVariables >= 1)


    power = cp.sum(cp.multiply((cp.multiply(f, sizingVariables)), e))
    area = cp.sum(cp.multiply(a, sizingVariables))

    constr.append(power <= Pmax)
    constr.append(area <= Amax)

    model = loadModel('Inputs.outputs/model.npz')

    # for gate in range(0, 6):
    for gate in range(0, 6):
        curGate = gateNodes[gate]
        bins = curGate.randVar.bins

        x_i = sizingVariables[gate]
        a_i = a[gate]
        f_i = f[gate]
        e_i = e[gate]

        for bin in range(0, numberOfBins):

            # constr.append(bins[bin] >= rv1.bins[bin])


            shift = model[bin, 0]
            areaCoef = model[bin, 1]
            powerCoef = model[bin, 2]

            prob = shift + areaCoef * x_i * a_i + powerCoef * x_i * e_i * f_i

            # constr.append(bins[bin] >= prob)

    # solve

    objective = cp.Minimize(sum)
    prob = cp.Problem(objective, constr)

    prob.solve(verbose=True, solver=cp.MOSEK, gp=True,
               mosek_params={  'MSK_DPAR_INTPNT_CO_TOL_MU_RED': 0.1,
               'MSK_DPAR_OPTIMIZER_MAX_TIME': 1200}  # max time
               )


    # print('Mean:')
    # print(prob.value)
    #
    # print(midPoints)
    sum = 0
    for bin in range(0, numberOfBins):
        print(delays[-1].bins[bin].value)
        sum += delays[-1].bins[bin].value  # minimize the mean value
    print(sum)

    rvs = []
    for gate in range(0, len(delays)):  # construct RVs

        finalBins = np.zeros(numberOfBins)
        for bin in range(0, numberOfBins):
            # if ((delays[gate].bins)[bin]) != 0:
            finalBins[bin] = ((delays[gate].bins)[bin]).value
            # else:
            #     finalBins[bin] = 0
            # print(finalBins)
            # print(rv1.bins)
            print(finalBins >= rv1.bins)
            # exit(-1)
        last = finalBins
        rvs.append(RandomVariable(finalBins, delays[0].edges))

    print('Sizing parameters')
    print(sizingVariables.value)

    print("\n APRX. VALUES: \n")
    for i in range(0, len(delays)):
        print(rvs[i].mean, rvs[i].std)


    #######################
    ######NUMPY############
    #######################

    n1 = Node(rv1)
    n2 = Node(rv2)
    n3 = Node(rv3)
    n4 = Node(rv4)
    n5 = Node(rv5)
    n6 = Node(rv6)

    IN1 = Node(in1)
    IN2 = Node(in2)

    # set circuit design
    n1.setNextNodes([n5])
    # n2.setNextNodes([n3, n3, n4, n4])
    IN1.setNextNodes([n3])
    IN2.setNextNodes([n4])
    n2.setNextNodes([n3, n4])
    n3.setNextNodes([n5, n6])
    n4.setNextNodes([n6])

    delays = SSTA.calculateCircuitDelay([n2, IN1, IN2, n1])

    # print(np.sum(delays[-1].bins))

    plt.hist(delays[-1].edges[:-1], delays[-1].edges, weights=delays[-1].bins, density="PDF", alpha=0.2, color='orange')
    plt.hist(rvs[-1].edges[:-1], rvs[-1].edges, weights=last, density="PDF", color='blue')

    print(np.sum(last)*(1))
    print(np.sum(delays[-1].bins)*(rvs[-1].edges[1] - rvs[-1].edges[0]))
    plt.show()

    actual = putTuplesIntoArray(rvs=delays)
    print(actual)


def loadModel(path):
    """
    Loads linear regression model.
    :param path: relative path to the model
    :return model: matrix [numberOfBins, 3] with model parameters
    """

    coef = np.load(path)
    model = coef['coef']

    return model


if __name__ == "__main__":

    # test_SSTA_MAX()
    # test_SSTA_MIN()
    # optimizeGates()
    optimizeCVXPY_GP()
    # loadModel("Inputs.outputs/model.npz")
