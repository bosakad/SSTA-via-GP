from queue import Queue

from randomVariableHist_Numpy import RandomVariable
from node import Node
import cvxpy as cp
from mosekVariable import RandomVariableMOSEK
import numpy as np

import cvxpyVariable


def calculateCircuitDelay(rootNodes: [Node], cvxpy=False, unary=False, mosekStatus=(-1, -1),
                                        withSymmetryConstr=False, mosekTRI=False, GP=False, monteCarlo=False) -> [Node]:
    """
    Compute circuit delay using SSTA algorithm
    Function executes the algorithm for finding out the PDF of a circuit delay.

    :param rootNodes: array of root nodes, [Node], Node includes next nodes and previous nodes
    :param mosekStatus: current number of MOSEK variables and constraints (tuple)
    :param cvxpy: Bool, true if cvxpy objects are used, false if just RandomVariables class is used
    :param unary: Bool, true if unary (M 0/1-bin repr.) is used, false otherwise
    :param mosekTRI: Bool, true if trilinear constraints are used, false otherwise
    :param GP: Bool, true if using the GP version of the SSTA, false otherwise
    :param monteCarlo: Bool, true if using the monte carlo, false otherwise

    :return newConstr: new constraints for CVXPY(optional)
    :return newNofVariables: new number of MOSEK variables(optional)
    :return newDelays: array of new RVs
    """

        # pointer to a max and convolution functions - depend on current tool and algorithm
    if mosekStatus[0] >= 0:  MaximumF = maxOfDistributionsMOSEK_UNARY(withSymmetryConstr)
    elif cvxpy and unary:     MaximumF = maxOfDistributionsCVXPY_UNARY(withSymmetryConstr)
    elif cvxpy and not unary and not GP: MaximumF = maxOfDistributionsCVXPY_McCormick
    elif cvxpy and GP:        MaximumF = maxOfDistributionsCVXPY_GP
    elif not cvxpy and unary and not mosekTRI: MaximumF = maxOfDistributionsUNARY
    elif monteCarlo:          MaximumF = maxMonteCarlo
    else:                     MaximumF = maxOfDistributionsFORM
    if mosekStatus[0] >= 0:  ConvolutionF = Convolution_UNARY_MOSEK(withSymmetryConstr)
    elif cvxpy and unary:     ConvolutionF = Convolution_UNARY(withSymmetryConstr)
    elif cvxpy and not unary and not GP: ConvolutionF = Convolution_McCormick
    elif cvxpy and GP:        ConvolutionF = Convolution_GP
    elif not cvxpy and unary: ConvolutionF = RandomVariable.convolutionOfTwoVarsNaiveSAME_UNARY
    elif monteCarlo:          ConvolutionF = np.add
    else:                     ConvolutionF = RandomVariable.convolutionOfTwoVarsShift

    if mosekStatus[0] >= 0:
        usingMosek = True
        curNofVariables = mosekStatus[0]
        curNofConstr = mosekStatus[1]
    else:
        usingMosek = False
        curNofConstr = 0

        # init. data structures
    queue = Queue()
    sink = []
    newDelays = []
    closedList = set()
    AllConstr = []

        # put root nodes into Queue
    putIntoQueue(queue, rootNodes)

        # BFS
    while not queue.empty():

        tmpNode = queue.get()                                       # get data
        currentRandVar = tmpNode.randVar

        if tmpNode in closedList:
            continue

        if tmpNode.prevDelays:                                      # calculate maximum + convolution - depends on the tool

                # two-term multiplication alg. - unary encoding
            if usingMosek and not mosekTRI:

                maxDelay, curNofVariables, curNofConstr = MaximumF(tmpNode.prevDelays, curNofVariables, curNofConstr)
                currentRandVar, curNofVariables, curNofConstr = ConvolutionF(currentRandVar, maxDelay, curNofVariables, curNofConstr)

                #  - unary encoding
            elif usingMosek and mosekTRI:
                        # works just for 2 inputs for maximum
                RV1 = tmpNode.prevDelays[0]
                RV2 = tmpNode.prevDelays[1]
                RV3 = currentRandVar
                currentRandVar, curNofVariables, curNofConstr = RV1.maximum_AND_Convolution_VECTORIZED_MIN(RV2, RV3, curNofVariables, curNofConstr)

                # numpy version of the three-term multiplication alg.- unary encoding
            elif not usingMosek and mosekTRI:

                RV1 = tmpNode.prevDelays[0]
                RV2 = tmpNode.prevDelays[1]
                RV3 = currentRandVar
                currentRandVar = RV1.maximum_AND_Convolution_UNARY(RV2, RV3)

                # two-term multiplication alg. - unary encoding in CVXPY
            elif cvxpy and not GP:
                maxDelay, newConstraints = MaximumF(tmpNode.prevDelays)
                AllConstr.extend(newConstraints)
                currentRandVar, newConstraints = ConvolutionF(currentRandVar, maxDelay)
                AllConstr.extend(newConstraints)

                # SSTA using GP in CVXPY tool
            elif cvxpy and GP:
                maxDelay, AllConstr = MaximumF(tmpNode.prevDelays, AllConstr)
                currentRandVar, AllConstr = ConvolutionF(currentRandVar, maxDelay, AllConstr)

                # SSTA using exact computation
            else:
                maxDelay = MaximumF(tmpNode.prevDelays)
                currentRandVar = ConvolutionF(currentRandVar, maxDelay)



        for nextNode in tmpNode.nextNodes:                          # append this node as a previous
            nextNode.appendPrevDelays(currentRandVar)

        if not tmpNode.nextNodes:                                   # save for later ouput delays
            sink.append(currentRandVar)

            # continue BFS
        closedList.add(tmpNode)
        putIntoQueue(queue, tmpNode.nextNodes)
        newDelays.append(currentRandVar)


        # if more output gates - calculate maximum
    if (len(sink) != 1):

        if usingMosek:
            sinkDelay, curNofVariables, curNofConstr = MaximumF(sink, curNofVariables, curNofConstr)

        elif not usingMosek and mosekTRI:
            sinkDelay = maxOfDistributionsUNARY(sink)

        elif cvxpy and not GP:
            sinkDelay, newConstraints = MaximumF(sink)
            AllConstr.extend(newConstraints)

        elif cvxpy and GP:
            sinkDelay, AllConstr = MaximumF(sink, AllConstr)

        else:
            sinkDelay = MaximumF(sink)

        newDelays.append(sinkDelay)


        # return depends on the current tool
    if usingMosek:
        return newDelays, curNofVariables, curNofConstr
    elif cvxpy:
        return newDelays, AllConstr
    else:
        return newDelays



def Convolution_UNARY_MOSEK(withSymmetryConstr=False):
    """
    Calculates convolution of an array of PDFs of cvxpy variable, is for clean code.

    :param x1: RandomVariableMOSEK class
    :param x2: RandomVariableMOSEK class
    :return curNofVariables: current number of variables
    :return curNofVariables: current number of constraints
    :param withSymmetryConstr: boolean, whether symmetry constraints should be used

    :return convolution: RandomVariableMOSEK class
    :return curNofVariables: new number of variables
    :return curNofVariables: new number of constraints
    """

    def convolutionF(x1: RandomVariableMOSEK, x2: RandomVariableMOSEK, curNofVariables, curNofConstr):
        return x1.convolution_UNARY_MAX_DIVIDE_VECTORIZED(x2, curNofVariables, curNofConstr, withSymmetryConstr=withSymmetryConstr)

    return convolutionF

def Convolution_UNARY(withSymmetryConstr=False):
    """
    Calculates convolution of an array of PDFs of cvxpy variable, is for clean code sake.

    :param x1: RandomVariableCVXPY class
    :param x2: RandomVariableCVXPY class
    :param withSymmetryConstr: boolean, whether symmetry constraints should be used
    :return convolution: RandomVariableCVXPY class
    """

    def convolutionF(x1: cvxpyVariable.RandomVariableCVXPY, x2: cvxpyVariable.RandomVariableCVXPY):
        return x1.convolution_UNARY_DIVIDE(x2, withSepConstr=withSymmetryConstr, asMin=False)

    return convolutionF

def Convolution_McCormick(x1: cvxpyVariable.RandomVariableCVXPY,
                      x2: cvxpyVariable.RandomVariableCVXPY) -> cvxpyVariable.RandomVariableCVXPY:
    """
    Calculates convolution of an array of PDFs of cvxpy variable, is for clean code sake.

    :param x1: RandomVariableCVXPY class
    :param x2: RandomVariableCVXPY class
    :return convolution: RandomVariableCVXPY class
    """

    return x1.convolution_McCormick(x2)

def Convolution_GP(x1: cvxpyVariable.RandomVariableCVXPY,
                      x2: cvxpyVariable.RandomVariableCVXPY, constr) -> (cvxpyVariable.RandomVariableCVXPY, []):
    """
    Calculates convolution of an array of PDFs of cvxpy variable, is for clean code sake.

    :param x1: RandomVariableCVXPY class
    :param x2: RandomVariableCVXPY class
    :param constr: old constraints
    :return convolution: RandomVariableCVXPY class
    :return constr: new constraints
    """

    return x1.convolution_GP_OPT(x2, constr)

def maxOfDistributionsMOSEK_UNARY(withSymmetryConstr=False) -> cp.Variable:
    """
    Calculates maximum of an array of PDFs of cvxpy variable

    :param delays: array of cvxpy variables (n, m), n gates, m bins
    :return curNofVariables: current number of variables
    :return curNofVariables: current number of constraints

    :return maximum:  mosek variable (1, m)
    :return curNofVariables: new number of variables
    :return curNofVariables: new number of constraints
    """


    def maximumF(delays: [RandomVariableMOSEK], curNofVariables, curNofConstr):
        size = len(delays)
        for i in range(0, size - 1):
            newRV, newNofVariables, newNofConstr = delays[i].maximum_UNARY_MAX_DIVIDE_VECTORIZED(delays[i + 1],curNofVariables, curNofConstr,
                                                                withSymmetryConstr=withSymmetryConstr, asMin=True)
            delays[i + 1] = newRV

        maximum = delays[-1]

        return maximum, newNofVariables, newNofConstr

    return maximumF

def maxOfDistributionsCVXPY_UNARY(withSymmetryConstr=False) -> cp.Variable:
    """
    Calculates maximum of an array of PDFs of cvxpy variable

    :param delays: array of cvxpy variables (n, m), n gates, m bins
    :return maximum:  cvxpy variable (1, m)
    """


    def maximumF(delays: [cvxpyVariable.RandomVariableCVXPY]):
        size = len(delays)
        for i in range(0, size - 1):
            newRV = delays[i].maximum_QUAD_UNARY_DIVIDE(delays[i + 1], withSymmetryConstr=withSymmetryConstr, asMin=False)
            delays[i + 1] = newRV

        maximum = delays[-1]

        return maximum

    return maximumF

def maxOfDistributionsCVXPY_McCormick(delays: [cvxpyVariable.RandomVariableCVXPY]) -> cp.Variable:
    """
    Calculates maximum of an array of PDFs of cvxpy variable

    :param delays: array of cvxpy variables (n, m), n gates, m bins
    :return maximum:  cvxpy variable (1, m)
    """

    size = len(delays)
    for i in range(0, size - 1):
        newRV = delays[i].maximum_McCormick(delays[i + 1])
        delays[i + 1] = newRV

    maximum = delays[-1]

    return maximum

def maxOfDistributionsCVXPY_GP(delays: [cvxpyVariable.RandomVariableCVXPY], constr) -> (cp.Variable, []):
    """
    Calculates maximum of an array of PDFs of cvxpy variable

    :param delays: array of cvxpy variables (n, m), n gates, m bins
    :param constr: old constraints
    :return maximum:  cvxpy variable (1, m)
    :return constr: new constraints
    """

    size = len(delays)
    for i in range(0, size - 1):
        newRV, constr = delays[i].maximum_GP_OPT(delays[i + 1], constr)
        delays[i + 1] = newRV

    maximum = delays[-1]

    return maximum, constr


def maxOfDistributionsELEMENTWISE(delays: [RandomVariable]) -> RandomVariable:
    """
    Calculates maximum of an array of PDFs of cvxpy variable
    Using elementwise maximum - np.maximum

    :param delays: array of RandomVariables
    :return maximum: RandomVariable - maximum delay
    """

    size = len(delays)
    for i in range(0, size - 1):
        newRV = delays[i].maxOfDistributionsELEMENTWISE(delays[i + 1])
        delays[i + 1] = newRV

    maximum = delays[-1]

    return maximum

def maxOfDistributionsUNARY(delays: [RandomVariable]) -> RandomVariable:
    """
    Calculates maximum of an array of PDFs of cvxpy variable
    Using formula - look up function maxOfDistributionsFORM

    :param delays: array of RandomVariables
    :return maximum: RandomVariable - maximum delay
    """


    size = len(delays)

    for i in range(0, size - 1):
        newRV = delays[i].maxOfDistributionsQUAD_FORMULA_UNARY(delays[i + 1])
        delays[i + 1] = newRV

    maximum = delays[-1]

    return maximum


def maxMonteCarlo(delays):
    """
    Calculates maximum for monte carlo

    :param delays: 2d array of numbers
    :return maximum:  maximum delay
    """


    size = len(delays)

    for i in range(0, size - 1):
        newRV = np.maximum(delays[i], delays[i + 1])
        delays[i + 1] = newRV

    maximum = delays[-1]

    return maximum

def maxOfDistributionsFORM(delays: [RandomVariable]) -> RandomVariable:
    """
    Calculates maximum of an array of PDFs of cvxpy variable
    Using formula - look up function maxOfDistributionsFORM

    :param delays: array of RandomVariables
    :return maximum: RandomVariable - maximum delay
    """


    size = len(delays)

    for i in range(0, size - 1):
        newRV = delays[i].maxOfDistributionsFORM(delays[i + 1])
        delays[i + 1] = newRV

    maximum = delays[-1]

    return maximum


def maxOfDistributionsQUAD(delays: [RandomVariable]) -> RandomVariable:
    """
    Calculates maximum of an array of PDFs of cvxpy variable
    Using maxOfDistributionsQUAD, quadratic algorithm

    :param delays: array of RandomVariables
    :return maximum: RandomVariable - maximum delay
    """

    size = len(delays)

    for i in range(0, size - 1):

        newRV = delays[i].maxOfDistributionsQUAD(delays[i + 1])
        delays[i + 1] = newRV

    max = delays[-1]

    return max


def putIntoQueue(queue: Queue, list: [Node]) -> None:
    """
    Function puts list into queue.

    :param queue: Queue
    :return list: array of Node class
    """

    for item in list:
        queue.put(item)




