import numpy as np
import cvxpy as cp

from node import Node

from cvxpyVariable import RandomVariableCVXPY

import histogramGenerator
import SSTA
import networkx as nx
import mosek
from test_SSTA_Numpy import putTuplesIntoArray
from examples_monteCarlo.infinite_ladder_montecarlo import MonteCarlo_inputs, MonteCarlo_nodes, get_moments_from_simulations

from randomVariableHist_Numpy import RandomVariable
from mosek import *
import sys
from mosekVariable import RandomVariableMOSEK
import matplotlib.pyplot as plt




"""
This module includes gate sizing optimization of the c17 circuit in functions 'optimizeCVXPY_GP'
and optimization of a toy circuit in 'optimizeGates_MIXED_INT'.

"""

# Define a stream printer to grab output from MOSEK
def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()



def optimizeGates_MIXED_INT(dec=3):
    """
    Optimizes a toy circuit using Mixed-integer programming
    """

    numberOfGates = 3
    interval = (0, 16)
    numberOfBins = 10
    numberOfUnaries = 10

    coef = np.load('Inputs.outputs/model_MIXED_INT.npz')
    model = coef['coef']

        # generate inputs and 2 for control
    rv1 = histogramGenerator.generateAccordingToModel(model, 1, 4, x_i=1.5, int=interval, nUnaries=numberOfUnaries)
    rv2 = histogramGenerator.generateAccordingToModel(model, 1, 4, x_i=1, int=interval, nUnaries=numberOfUnaries)
    rv3 = histogramGenerator.generateAccordingToModel(model, 1, 4, x_i=1, int=interval, nUnaries=numberOfUnaries)


    # Make a MOSEK environment
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

            bins1 = np.zeros((numberOfBins, numberOfUnaries)).astype(int)
            bins2 = np.zeros((numberOfBins, numberOfUnaries)).astype(int)
            bins3 = np.zeros((numberOfBins, numberOfUnaries)).astype(int)

            gates = [rv1, rv2, rv3]
            bins = [bins1, bins2, bins3]

            # set fitting constraints
            for gate in range(0, 3):
                currentBins = bins[gate]
                generatedRV = gates[gate]
                for bin in range(0, numberOfBins):
                    for unary in range(0, numberOfUnaries):

                        variableIndex = gate * numberOfBins * numberOfUnaries + bin * numberOfUnaries + unary

                        if gate == 2 or gate == 3:
                            task.putvarbound(variableIndex, mosek.boundkey.ra, 0, 1)
                        else:
                            task.putvarbound(variableIndex, mosek.boundkey.ra, generatedRV.bins[bin, unary], 1)

                            # save index to the bins
                        currentBins[bin, unary] = variableIndex

            RV1 = RandomVariableMOSEK(bins1, rv1.edges, task)
            RV2 = RandomVariableMOSEK(bins2, rv1.edges, task)
            RV3 = RandomVariableMOSEK(bins3, rv1.edges, task)

            # set circuit design
            n1 = Node(RV1)
            n2 = Node(RV2)
            n3 = Node(RV3)

            n1.setNextNodes([n3])
            n2.setNextNodes([n3])

            start = [n1, n2]

            delays, newNofVariables, newNofConstr = SSTA.calculateCircuitDelay(start, cvxpy=False, unary=True,
                                                                 mosekStatus=(numberVariablesRVs, 0), mosekTRI=True)
            maximum = delays[-1]

            # create the objective function
            sum = np.array([]).astype(int)
            for gate in range(0, 3):
                sum = np.append(sum, np.concatenate(delays[gate].bins[:, :]))

            firstWeights = np.ones(2 * numberOfUnaries * numberOfBins + numberOfUnaries * 6) * 1e-21
            midPoints = np.ones(numberOfUnaries * 4) * 3

            # rough approximation of CVaR min
            weights = np.append(firstWeights, midPoints)
            task.putclist(sum, weights)



            # create upper bound constraints

            f = np.array([4, 0.8])
            e = np.array([1, 2])
            a = np.ones(numberOfGates)
            Amax = 1.5
            Pmax = 50

            # create sizing variables
            inf = 0.0
            task.appendvars(1)
            sizingVariables = np.array(range(newNofVariables, newNofVariables + 1)).astype(int)

            task.putvarboundlist(sizingVariables, [mosek.boundkey.lo],
                                 [1] * 1, [inf])
            newNofVariables += 1


            task.appendcons(2)

            # constraint for area
            task.putarow(newNofConstr, sizingVariables, a[:1])
            task.putconbound(newNofConstr, mosek.boundkey.up, 0.0, Amax)

            # constraint for power
            task.putarow(newNofConstr + 1, sizingVariables, np.multiply(f, e)[:1])
            task.putconbound(newNofConstr + 1, mosek.boundkey.up, 0.0, Pmax)

            newNofConstr += 2

            task.appendcons(numberOfGates * numberOfBins * 2)

            gateNodes = [bins1, bins2, bins3]

                # set sizing constraints
            numberOfGates = 1
            for gate in range(0, 1):

                curNode = gateNodes[2 + gate]
                x_i = sizingVariables[gate]
                a_i = a[gate]
                f_i = f[gate]
                e_i = e[gate]

                for bin in range(0, numberOfBins):
                    round = 0.5
                    shift = model[bin, 0]
                    areaCoef = model[bin, 1]
                    powerCoef = model[bin, 2]

                    row = curNode[bin, :]

                    offset1 = newNofConstr + gate * numberOfBins + bin
                    offset2 = newNofConstr + numberOfGates * numberOfBins + gate * numberOfBins + bin

                    task.putarow(offset1, row, [1] * row.size)
                    task.putarow(offset2, row, [1] * row.size)

                    sizingValue = -numberOfUnaries * (areaCoef * a_i + powerCoef * f_i * e_i)
                    task.putaij(offset1, x_i, sizingValue)
                    task.putaij(offset2, x_i, sizingValue)

                    task.putconbound(offset1, mosek.boundkey.up, 0.0, shift * numberOfUnaries + round)
                    task.putconbound(offset2, mosek.boundkey.lo, shift * numberOfUnaries - round, 0.0)

            newNofConstr = newNofConstr + numberOfGates * numberOfBins * 2

            numberOfGates = 3
            task.appendcons(numberOfGates * numberOfBins * (numberOfUnaries - 1))

            for gate in range(0, numberOfGates):
                curNode = gateNodes[gate]

                # separation constraints
                for bin in range(0, numberOfBins):
                    for unary in range(0, numberOfUnaries - 1):
                        offset = bin * (numberOfUnaries - 1) + unary

                        # (maximum[bin])[unary] >= (maximum[bin])[unary + 1])
                        task.putaij(newNofConstr + offset, curNode[bin, unary], 1)
                        task.putaij(newNofConstr + offset, curNode[bin, unary + 1], -1)

                        task.putconbound(newNofConstr + offset, mosek.boundkey.lo, 0, 0.0)

            newNofConstr += (numberOfUnaries - 1) * numberOfBins

            # solve problem

            # Input the objective sense (minimize/maximize)
            task.putobjsense(mosek.objsense.minimize)

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

            if solsta in [mosek.solsta.integer_optimal]:
                pass
                # print("Optimal solution: %s" % xx)
            elif solsta == mosek.solsta.prim_feas:
                pass
                # print("Feasible solution: %s" % xx)
            elif mosek.solsta.unknown:
                if prosta == mosek.prosta.prim_infeas_or_unbounded:
                    print("Problem status Infeasible or unbounded.\n")
                elif prosta == mosek.prosta.prim_infeas:
                    print("Problem status Infeasible.\n")
                elif prosta == mosek.prosta.unkown:
                    print("Problem status unkown.\n")
                else:
                    print("Other problem status.\n")
            else:
                print("Other solution status")

            print('Sizing parameters:')
            print(xx[sizingVariables])




def optimizeCVXPY_GP(modelType='Gauss', precomputedSizing=[2.11, 5.1, 5.0, 5.39, 3.55, 8.78]):

    """
    This function optimizes the c17 circuit using the GP model

    :param modelType: 'Gauss' or 'LogNormal' - type of least squares model
    :param precomputedSizing: precomputed sizing factors for monte carlo
    :return delay: delay distribution of the last gate
    :return values: monte carlo values
    """

    numberOfBins = 35
    binsInterval = (0, 35)
    numberOfGates = 11

        # lognormals model
    if modelType == "LogNormal":
        coef = np.load('Inputs.outputs/model.npz')
    elif modelType == "Gauss":
            # normal model
        coef = np.load('Inputs.outputs/model_Normal.npz')

    model = coef['coef']

    f = np.array([4, 0.8, 1, 0.8, 1.7, 0.5])
    e = np.array([1, 2, 1, 1.5, 1.5, 1])
    a = np.ones(6)

    # generate gates
    rv1 = histogramGenerator.generateAccordingToModel(model, 1, f[0] * e[0], x_i=precomputedSizing[0], int=binsInterval)
    rv2 = histogramGenerator.generateAccordingToModel(model, 1, f[1] * e[1], x_i=precomputedSizing[1], int=binsInterval)
    rv3 = histogramGenerator.generateAccordingToModel(model, 1, f[2] * e[2], x_i=precomputedSizing[2], int=binsInterval)
    rv4 = histogramGenerator.generateAccordingToModel(model, 1, f[3] * e[3], x_i=precomputedSizing[3], int=binsInterval)
    rv5 = histogramGenerator.generateAccordingToModel(model, 1, f[4] * e[4], x_i=precomputedSizing[4], int=binsInterval)
    rv6 = histogramGenerator.generateAccordingToModel(model, 1, f[5] * e[5], x_i=precomputedSizing[5], int=binsInterval)


    # generate random inputs
    in1 = histogramGenerator.generateAccordingToModel(model, 1, 0.2, x_i=40, int=binsInterval)
    in2 = histogramGenerator.generateAccordingToModel(model, 1, 0.2, x_i=35, int=binsInterval)
    in3 = histogramGenerator.generateAccordingToModel(model, 1, 0.2, x_i=35, int=binsInterval)
    in4 = histogramGenerator.generateAccordingToModel(model, 1, 0.2, x_i=35, int=binsInterval)
    in5 = histogramGenerator.generateAccordingToModel(model, 1, 0.2, x_i=35, int=binsInterval)

    gateNodes = []
    constr = []

    RVs = [rv1, rv2, rv3, rv4, rv5, rv6, in1, in2, in3, in4, in5]

        # set the 'fitting' constraints for the input gates
    for gate in range(0, numberOfGates):
        bins = {}

        g = RVs[gate]

        for bin in range(0, numberOfBins):

            var = cp.Variable(pos=True)

            if gate >= 6:   # for input gates
                constr.append(var >= g.bins[bin])


            bins[bin] = var

        node = Node(RandomVariableCVXPY(bins, rv1.edges))
        gateNodes.append(node)

        # set circuit design
    gateNodes[0].setNextNodes([gateNodes[4]])
    gateNodes[1].setNextNodes([gateNodes[2], gateNodes[3]])

    gateNodes[2].setNextNodes([gateNodes[5]])
    gateNodes[3].setNextNodes([gateNodes[4], gateNodes[5]])

    IN1 = gateNodes[-5]
    IN2 = gateNodes[-4]
    IN3 = gateNodes[-3]

    IN1.setNextNodes([gateNodes[0]])
    IN2.setNextNodes([gateNodes[0], gateNodes[1]])
    IN3.setNextNodes([gateNodes[1]])

    IN4 = gateNodes[-2]
    IN5 = gateNodes[-1]

    IN4.setNextNodes([gateNodes[3]])
    IN5.setNextNodes([gateNodes[2]])

    startingNodes = [IN1, IN2, IN3, IN4, IN5]

    # calculate delay using SSTA
    delays, newConstr = SSTA.calculateCircuitDelay(startingNodes, cvxpy=True, GP=True)
    constr.extend(newConstr)

        # formulate the objective function
    midPoints = 0.5 * (delays[-1].edges[1:] + delays[-1].edges[:-1])  # midpoints of the edges of hist.
    nLast = 9
    finalMidPoints = np.append(np.ones((numberOfBins - nLast,))*1.0e-2, np.power(midPoints[-nLast:], 2))

    sum = 0
    for bin in range(0, numberOfBins):
        sum += delays[-1].bins[bin] * finalMidPoints[bin]                # minimize the mean value


    # create sizing parameters
    Amax = 35
    Pmax = 55

    sizingVariables = cp.Variable((6,), pos=True)
    constr.append(sizingVariables >= 1)

    power = cp.sum(cp.multiply((cp.multiply(f, sizingVariables)), e))
    area = cp.sum(cp.multiply(a, sizingVariables))

    constr.append(power <= Pmax)
    constr.append(area <= Amax)

        # create the 'fitting' constraints for the sizable gates
    for gate in range(0, 6):
        curGate = gateNodes[gate]
        bins = curGate.randVar.bins

        x_i = sizingVariables[gate]
        a_i = a[gate]
        f_i = f[gate]
        e_i = e[gate]

        for bin in range(0, numberOfBins):

            a1 = model[bin, 0]
            p1 = model[bin, 1]
            a2 = model[bin, 2]
            p2 = model[bin, 3]

                # zero prob. of a bin - no sizing should be included
            if (model[bin, :] == 1.00000000e-27).all():
                prob = 1.00000000e-27
            else:
                prob = a1 * x_i*a_i + p1 * x_i*e_i*f_i + a2 * (1/(x_i*a_i)) + p2 * (1/(x_i*e_i*f_i))

            constr.append(bins[bin] >= prob)

    # solve

    objective = cp.Minimize(sum)
    prob = cp.Problem(objective, constr)

    prob.solve(verbose=True, solver=cp.MOSEK, gp=True,
               mosek_params={  'MSK_DPAR_INTPNT_CO_TOL_MU_RED': 0.1,
               'MSK_DPAR_OPTIMIZER_MAX_TIME': 1200}  # max time
               )



        # print out the values
    rvs = []
    for gate in range(0, len(delays)):  # construct RVs

        finalBins = np.zeros(numberOfBins)
        for bin in range(0, numberOfBins):

            finalBins[bin] = ((delays[gate].bins)[bin]).value

        last = finalBins
        rvs.append(RandomVariable(finalBins, delays[0].edges))


    print('Sizing parameters')
    print(sizingVariables.value)
    print('total area:')
    print(np.sum(np.multiply(sizingVariables.value, a)))
    print('total power:')
    print(np.sum(np.multiply(sizingVariables.value, np.multiply(f, e))))

    print("\n APRX. VALUES: \n")
    for i in range(0, len(delays)):
        print(rvs[i].mean, rvs[i].std)


    #######################
    ######NUMPY############
    #######################
    # uncomment following to test with numpy exact version

    # n1 = Node(rv1)
    # n2 = Node(rv2)
    # n3 = Node(rv3)
    # n4 = Node(rv4)
    # n5 = Node(rv5)
    # n6 = Node(rv6)
    #
    # IN1 = Node(in1)
    # IN2 = Node(in2)
    #
    # # set circuit design
    # n1.setNextNodes([n5])
    # # n2.setNextNodes([n3, n3, n4, n4])
    # IN1.setNextNodes([n4])
    # IN2.setNextNodes([n3])
    # n2.setNextNodes([n3, n4])
    # n3.setNextNodes([n6])
    # n4.setNextNodes([n5, n6])
    #
    # delays = SSTA.calculateCircuitDelay([n2, IN1, IN2, n1])

    #######################
    ######Monte Carlo############
    #######################

    n_samples = 1000000

    n1 = Node(histogramGenerator.getValuesForMonteCarlo(rv1, n_samples))
    n2 = Node(histogramGenerator.getValuesForMonteCarlo(rv2, n_samples))
    n3 = Node(histogramGenerator.getValuesForMonteCarlo(rv3, n_samples))
    n4 = Node(histogramGenerator.getValuesForMonteCarlo(rv4, n_samples))
    n5 = Node(histogramGenerator.getValuesForMonteCarlo(rv5, n_samples))
    n6 = Node(histogramGenerator.getValuesForMonteCarlo(rv6, n_samples))

    IN4 = Node(histogramGenerator.getValuesForMonteCarlo(in1, n_samples))
    IN5 = Node(histogramGenerator.getValuesForMonteCarlo(in2, n_samples))
    IN1 = Node(histogramGenerator.getValuesForMonteCarlo(in1, n_samples))
    IN2 = Node(histogramGenerator.getValuesForMonteCarlo(in2, n_samples))
    IN3 = Node(histogramGenerator.getValuesForMonteCarlo(in1, n_samples))

    # set circuit design
    n1.setNextNodes([n5])
    IN4.setNextNodes([n4])
    IN4.setNextNodes([n3])
    IN1.setNextNodes([n1])
    IN2.setNextNodes([n1, n2])
    IN3.setNextNodes([n2])
    n2.setNextNodes([n3, n4])
    n3.setNextNodes([n6])
    n4.setNextNodes([n5, n6])

    # mc = SSTA.calculateCircuitDelay([n2, IN1, IN2, n1], monteCarlo=True)
    mc = SSTA.calculateCircuitDelay([IN1, IN2, IN3, IN4, IN5], monteCarlo=True)
    values = mc[-1]


    result = []
    for data in mc:
        result.append([np.mean(data), np.std(data)])

    print("Monte carlo")
    print(result)

        # set it real edges
    rvs[-1].edges = delays[-1].edges / 1e11

    return rvs[-1], values



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

    # optimizeGates_MIXED_INT()
    optimizeCVXPY_GP(modelType="Gauss")
