from queue import Queue
from randomVariableHist import RandomVariable
from node import Node

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
            maxDelay = maxOfDistributions3(tmpNode.prevDelays)
            # print("mean, std of max: " + str(maxDelay.mean) + ", " + str(maxDelay.std))
            currentRandVar = currentRandVar.convolutionOfTwoVars2(maxDelay)

            # print("mean, std of convolution: " + str(currentRandVar.mean) + ", " + str(currentRandVar.std))

        for nextNode in tmpNode.nextNodes:                          # append this node as a previous
            nextNode.appendPrevDelays(currentRandVar)

        if not tmpNode.nextNodes:                                   # save for later ouput delays
            sink.append(currentRandVar)

        closedList.append(tmpNode)
        putIntoQueue(queue, tmpNode.nextNodes)
        newDelays.append(currentRandVar)


    sinkDelay = maxOfDistributions4(sink)
    newDelays.append(sinkDelay)

    return newDelays


""" Calculates maximum of an array of PDFs

    Params: 
        delays: array of RandomVariables
    
    Return:
        max: RandomVariable - maximum delay

"""

def maxOfDistributions(delays: [RandomVariable]) -> RandomVariable:

    size = len(delays)
    for i in range(0, size - 1):
        val, newRV = delays[i].getMaximum(delays[i + 1])
        delays[i+1] = newRV

    max = delays[-1]

    return max


""" Calculates maximum of an array of PDFs

    Params: 
        delays: array of RandomVariables

    Return:
        max: RandomVariable - maximum delay

"""


def maxOfDistributions2(delays: [RandomVariable]) -> RandomVariable:

    size = len(delays)
    for i in range(0, size - 1):
        newRV = delays[i].getMaximum2(delays[i + 1])
        delays[i + 1] = newRV

    max = delays[-1]

    return max

def maxOfDistributions3(delays: [RandomVariable]) -> RandomVariable:

    size = len(delays)

    for i in range(0, size - 1):
        newRV = delays[i].getMaximum4(delays[i + 1])
        delays[i + 1] = newRV

    max = delays[-1]

    return max

def maxOfDistributions4(delays: [RandomVariable]) -> RandomVariable:

    size = len(delays)

    for i in range(0, size - 1):
        newRV = delays[i].getMaximum5(delays[i + 1])
        delays[i + 1] = newRV

    max = delays[-1]

    return max


""" Put into queue
        
    Function puts list into queue. 
        
"""
def putIntoQueue(queue: Queue, list: [Node]) -> None:

    for item in list:
        queue.put(item)




