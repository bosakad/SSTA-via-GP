from queue import Queue

from randomVariableHist import RandomVariable
from node import Node
import cvxpy as cp

import sys
# setting path
sys.path.append('../Cvxpy')

import cvxpyVariable



def calculateCircuitDelay(rootNodes: [Node], cvxpy=False, unified=False) -> [Node]:
    """
    Compute circuit delay using PDFs algorithm
    Function executes the algorithm for finding out the PDF of a circuit delay.

    :param rootNodes: array of root nodes, [Node], Node includes next nodes and previous nodes
    :param cvxpy: Bool, true if cvxpy objects are used, false if just RandomVariables class is used
    :param unified: Bool, true if unified (M 0/1-bin repr.) is used, false otherwise
    :return newDelays: array of new RVs
    """

    # pointer to a max and convolution functions
    MaximumF = maxOfDistributionsCVXPY_UNIFIED if (cvxpy and unified) else maxOfDistributionsFORM
    ConvolutionF = Convolution_UNIFIED if (cvxpy and unified) else RandomVariable.convolutionOfTwoVarsShift

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

            if cvxpy:
                maxDelay, newConstraints = MaximumF(tmpNode.prevDelays)   # old code with constr.
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

        # make max into sink
    sinkDelay = MaximumF(sink)
    newDelays.append(sinkDelay)


        # return dependent on cvxpy / numpy     # old code with constr, might of use in the future
    if cvxpy:
        return newDelays, AllConstr
    else:
        return newDelays



def Convolution_UNIFIED(x1: cvxpyVariable.RandomVariableCVXPY, x2: cvxpyVariable.RandomVariableCVXPY) -> cvxpyVariable.RandomVariableCVXPY:
    """
    Calculates convolution of an array of PDFs of cvxpy variable, is for clean code sake.

    :param x1: RandomVariableCVXPY class
    :param x2: RandomVariableCVXPY class
    :return convolution: RandomVariableCVXPY class
    """

    return x1.convolution_UNIFIED_NEW_MAX(x2)


def maxOfDistributionsCVXPY_UNIFIED(delays: [cvxpyVariable.RandomVariableCVXPY]) -> cp.Variable:
    """
    Calculates maximum of an array of PDFs of cvxpy variable

    :param delays: array of cvxpy variables (n, m), n gates, m bins
    :return maximum:  cvxpy variable (1, m)
    """

    size = len(delays)
    for i in range(0, size - 1):
        newRV = delays[i].maximum_QUAD_UNIFIED_NEW_MAX(delays[i + 1])
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




