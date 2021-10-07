from queue import Queue
from randomVariableHist import RandomVariable

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


def calculateCircuitDelay(rootNodes):
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


        # print(queue, tmpNode)

        if tmpNode.prevDelays:                                      # get maximum + convolution
            print(tmpNode.prevDelays[0].mean)
            print(tmpNode.prevDelays[1].mean)
            maxDelay = maxOfDistributions3(tmpNode.prevDelays)
            print(maxDelay.mean)
            print(currentRandVar.mean)
            currentRandVar = currentRandVar.convolutionOfTwoVars(maxDelay)
            print(currentRandVar.mean)
            print()

        for nextNode in tmpNode.nextNodes:                          # append this node as a previous
            nextNode.appendPrevDelays(currentRandVar)

        if not tmpNode.nextNodes:                                   # save for later ouput delays
            sink.append(currentRandVar)

        closedList.append(tmpNode)
        putIntoQueue(queue, tmpNode.nextNodes)
        newDelays.append(currentRandVar)

    print(len(sink))
    sinkDelay = maxOfDistributions3(sink)

    return [newDelays, sinkDelay]


""" Calculates maximum of an array of PDFs

    Params: 
        delays: array of RandomVariables
    
    Return:
        max: RandomVariable - maximum delay

"""

def maxOfDistributions(delays):

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


def maxOfDistributions2(delays):

    size = len(delays)
    for i in range(0, size - 1):
        newRV = delays[i].getMaximum2(delays[i + 1])
        delays[i + 1] = newRV

    max = delays[-1]

    return max

def maxOfDistributions3(delays):

    size = len(delays)
    for i in range(0, size - 1):
        newRV = delays[i].getMaximum3(delays[i + 1])
        delays[i + 1] = newRV

    max = delays[-1]

    return max


""" Put into queue
        
    Function puts list into queue. 
        
"""
def putIntoQueue(queue, list):

    for item in list:
        queue.put(item)




