from queue import Queue

from randomVariableHist_Numpy import RandomVariable
from node import Node
import cvxpy as cp
from mosekVariable import RandomVariableMOSEK

import cvxpyVariable


def calculateCircuitDelay(rootNodes: [Node], cvxpy=False, unary=False, curNofVariables=-1, withSymmetryConstr=False) -> [Node]:
    """
    Compute circuit delay using PDFs algorithm
    Function executes the algorithm for finding out the PDF of a circuit delay.

    :param rootNodes: array of root nodes, [Node], Node includes next nodes and previous nodes
    :param curNofVariables: current number of MOSEK variables
    :param cvxpy: Bool, true if cvxpy objects are used, false if just RandomVariables class is used
    :param unary: Bool, true if unary (M 0/1-bin repr.) is used, false otherwise

    :return newConstr: new constraints for CVXPY(optional)
    :return newNofVariables: new number of MOSEK variables(optional)
    :return newDelays: array of new RVs
    """

    # pointer to a max and convolution functions
    if curNofVariables >= 0:  MaximumF = maxOfDistributionsMOSEK_UNARY(withSymmetryConstr)
    elif cvxpy and unary:     MaximumF = maxOfDistributionsCVXPY_UNARY(withSymmetryConstr)
    elif cvxpy and not unary: MaximumF = maxOfDistributionsCVXPY_McCormick
    elif not cvxpy and unary: MaximumF = maxOfDistributionsUNARY
    else:                     MaximumF = maxOfDistributionsFORM
    if curNofVariables >= 0:  ConvolutionF = Convolution_UNARY_MOSEK(withSymmetryConstr)
    elif cvxpy and unary:     ConvolutionF = Convolution_UNARY(withSymmetryConstr)
    elif cvxpy and not unary: ConvolutionF = Convolution_McCormick
    elif not cvxpy and unary: ConvolutionF = RandomVariable.convolutionOfTwoVarsNaiveSAME_UNARY
    else:                     ConvolutionF = RandomVariable.convolutionOfTwoVarsShift

    if curNofVariables >= 0:
        usingMosek = True
        curNofConstr = 0
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

    while not queue.empty():

        tmpNode = queue.get()                                       # get data
        currentRandVar = tmpNode.randVar

        if tmpNode in closedList:
            continue

        if tmpNode.prevDelays:                                      # get maximum + convolution

            if usingMosek:

                maxDelay, curNofVariables, curNofConstr = MaximumF(tmpNode.prevDelays, curNofVariables, curNofConstr)

                currentRandVar, curNofVariables, curNofConstr = ConvolutionF(currentRandVar, maxDelay, curNofVariables, curNofConstr)

            elif cvxpy:
                maxDelay, newConstraints = MaximumF(tmpNode.prevDelays)
                AllConstr.extend(newConstraints)
                currentRandVar, newConstraints = ConvolutionF(currentRandVar, maxDelay)
                AllConstr.extend(newConstraints)

            else:
                maxDelay = MaximumF(tmpNode.prevDelays)
                currentRandVar = ConvolutionF(currentRandVar, maxDelay)


        for nextNode in tmpNode.nextNodes:                          # append this node as a previous
            nextNode.appendPrevDelays(currentRandVar)

        if not tmpNode.nextNodes:                                   # save for later ouput delays
            sink.append(currentRandVar)

        closedList.add(tmpNode)
        putIntoQueue(queue, tmpNode.nextNodes)
        newDelays.append(currentRandVar)


        # if more output gate
    if (len(sink) != 1):

        # make max into sink
        if usingMosek:
            maxDelay, curNofVariables, curNofConstr = MaximumF(tmpNode.prevDelays, curNofVariables, curNofConstr)

        elif cvxpy:
            sinkDelay, newConstraints = MaximumF(sink)
            AllConstr.extend(newConstraints)
        else:
            sinkDelay = MaximumF(sink)

        newDelays.append(sinkDelay)
    else:
        pass
        # newDelays.append(sink[0])

    if usingMosek:
        return newDelays, curNofVariables
    elif cvxpy:
        return newDelays, AllConstr
    else:
        return newDelays



def Convolution_UNARY_MOSEK(withSymmetryConstr=False):
    """
    Calculates convolution of an array of PDFs of cvxpy variable, is for clean code sake.

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
        return x1.convolution_UNARY_DIVIDE(x2, withSymmetryConstr=withSymmetryConstr, asMin=False)

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
            newRV, newNofVariables, newNofConstr = delays[i].maximum_UNARY_MAX_DIVIDE(delays[i + 1],curNofVariables, curNofConstr,
                                                                withSymmetryConstr=withSymmetryConstr)
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




