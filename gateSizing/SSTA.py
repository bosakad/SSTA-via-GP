from queue import Queue
from randomVariableHist import RandomVariable
from node import Node
import matplotlib.pyplot as plt
import histogramGenerator

""" Compute circuit delay using PDFs algorithm

Function executes the algorithm for finding out the PDF of a circuit delay.

    Params:
        rootNodes: graph of the circuit - array of root nodes
                      nodes include:
                        class 'RandomVariable',
                        array of next nodes,
                        array of previous nodes

    Return:
        # mean: mean value of the circuits PDF
        newRandomVariables: array of new RVs, with computed mean and variance

"""


def calculateCircuitDelay(rootNodes: [Node]) -> [Node]:
    queue = Queue()

    sink = []
    newDelays = []
    closedList = []

    putIntoQueue(queue, rootNodes)

    while not queue.empty():

        tmpNode = queue.get()                                       # get data
        currentRandVar = tmpNode.randVar

        if tmpNode in closedList:
            continue

        if tmpNode.prevDelays:                                      # get maximum + convolution
            # print(tmpNode.prevDelays[0].mean, tmpNode.prevDelays[0].std)
            # print(tmpNode.prevDelays[1].mean, tmpNode.prevDelays[0].std)
            # print(len(tmpNode.prevDelays))

            maxDelay = maxOfDistributionsFORM(tmpNode.prevDelays)
            print("mean, std of max: " + str(maxDelay.mean) + ", " + str(maxDelay.std))
            currentRandVar = currentRandVar.convolutionOfTwoVarsShift(maxDelay)
            # currentRandVar = currentRandVar.convolutionOfTwoVarsUnion(maxDelay)

            print("mean, std of conmaxvolution: " + str(currentRandVar.mean) + ", " + str(currentRandVar.std))

        for nextNode in tmpNode.nextNodes:                          # append this node as a previous
            nextNode.appendPrevDelays(currentRandVar)

        if not tmpNode.nextNodes:                                   # save for later ouput delays
            sink.append(currentRandVar)

        closedList.append(tmpNode)
        putIntoQueue(queue, tmpNode.nextNodes)
        newDelays.append(currentRandVar)

    # plt.hist(sink[0].edges[:-1], sink[0].edges, weights=sink[0].bins, density="PDF")
    # plt.hist(sink[1].edges[:-1], sink[1].edges, weights=sink[1].bins, density="PDF")
    #
    # plt.show()

    sinkDelay = maxOfDistributionsFORM(sink)
    # g5 = histogramGenerator.get_gauss_bins(5, 0.5, 3000, 1000000, (-20, 120))
    # sinkDelay = sinkDelay.convolutionOfTwoVarsShift(g5)
    newDelays.append(sinkDelay)

    return newDelays






""" Calculates maximum of an array of PDFs

    Using elementwise maximum - np.maximum

    Params: 
        delays: array of RandomVariables

    Return:
        max: RandomVariable - maximum delay

"""


def maxOfDistributionsELEMENTWISE(delays: [RandomVariable]) -> RandomVariable:

    size = len(delays)
    for i in range(0, size - 1):
        newRV = delays[i].maxOfDistributionsELEMENTWISE(delays[i + 1])
        delays[i + 1] = newRV

    max = delays[-1]

    return max


"""
    Using formula - look up function maxOfDistributionsFORM
"""
def maxOfDistributionsFORM(delays: [RandomVariable]) -> RandomVariable:

    size = len(delays)

    for i in range(0, size - 1):
        newRV = delays[i].maxOfDistributionsFORM(delays[i + 1])
        delays[i + 1] = newRV

    max = delays[-1]

    return max

"""
    Using maxOfDistributionsQUAD, quadratic algorithm
"""
def maxOfDistributionsQUAD(delays: [RandomVariable]) -> RandomVariable:

    size = len(delays)

    for i in range(0, size - 1):

        newRV = delays[i].maxOfDistributionsQUAD(delays[i + 1])
        delays[i + 1] = newRV

    max = delays[-1]

    return max


""" Put into queue
        
    Function puts list into queue. 
        
"""
def putIntoQueue(queue: Queue, list: [Node]) -> None:

    for item in list:
        queue.put(item)




