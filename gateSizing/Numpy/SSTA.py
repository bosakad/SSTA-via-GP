from queue import Queue

from randomVariableHist import RandomVariable
from node import Node
import cvxpy as cp

import sys
# setting path
sys.path.append('../Cvxpy')

import cvxpyVariable



def calculateCircuitDelay(rootNodes: [Node], cvxpy=False) -> [Node]:
    """
    Compute circuit delay using PDFs algorithm
    Function executes the algorithm for finding out the PDF of a circuit delay.

    :param rootNodes: array of root nodes, [Node], Node includes next nodes and previous nodes
    :param cvxpy: Bool, true if cvxpy objects are used, false if just RandomVariables class is used
    :return newDelays: array of new RVs
    """

    # pointer to a max and convolution functions
    MaximumF = maxOfDistributionsCVXPY if cvxpy else maxOfDistributionsFORM
    ConvolutionF = cvxpyVariable.convolutionCVXPY if cvxpy else RandomVariable.convolutionOfTwoVarsShift

    # init. data structures
    queue = Queue()
    sink = []
    newDelays = []
    closedList = set()

    # put root nodes into Queue
    putIntoQueue(queue, rootNodes)

    while not queue.empty():

        tmpNode = queue.get()                                       # get data
        currentRandVar = tmpNode.randVar

        if tmpNode in closedList:
            continue

        if tmpNode.prevDelays:                                      # get maximum + convolution
            maxDelay = MaximumF(tmpNode.prevDelays)
            currentRandVar = ConvolutionF(currentRandVar, maxDelay)


        for nextNode in tmpNode.nextNodes:                          # append this node as a previous
            nextNode.appendPrevDelays(currentRandVar)

        if not tmpNode.nextNodes:                                   # save for later ouput delays
            sink.append(currentRandVar)

        closedList.add(tmpNode)
        putIntoQueue(queue, tmpNode.nextNodes)
        newDelays.append(currentRandVar)


    sinkDelay = MaximumF(sink)
    newDelays.append(sinkDelay)

    return newDelays






def maxOfDistributionsCVXPY(delays: [{cp.Expression}]) -> cp.Variable:
    """
    Calculates maximum of an array of PDFs of cvxpy variable

    :param delays: array of cvxpy variables (n, m), n gates, m bins
    :return maximum:  cvxpy variable (1, m)
    """

    size = len(delays)
    for i in range(0, size - 1):
        newRV = cvxpyVariable.maximumCVXPY(delays[i], delays[i + 1])
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




