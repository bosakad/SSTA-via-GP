import cvxpy as cp
import sys
import numpy as np

from Main.node import Node
from queue import Queue

def FindMaxDelayGates():
    """
    Computes maximum possible path. CVXPY variables as gates
    :return:
    """

    values = np.array([0, 3, 2, 4, 4, 10, 0])

    # define cvxpy variable for each gate
    path = cp.Variable((7,), boolean=True)

    source = Node([0, path[0]])
    n1 = Node([3, path[1]])
    n2 = Node([2, path[2]])
    n3 = Node([4, path[3]])
    n4 = Node([4, path[4]])
    n5 = Node([10, path[5]])
    sink = Node([0, path[6]])

    # set circuit design
    n1.setNextNodes([n3, n4])
    n2.setNextNodes([n3, n5])
    n3.setNextNodes([n4, n5])

    # set sink and source
    n4.setNextNodes([sink])
    n5.setNextNodes([sink])
    source.setNextNodes([n1, n2])

    # set constraints for fan-out

    queue = Queue()
    queue.put(source)
    closedList = set()

    constr = []

    while not queue.empty():

        currentGate = queue.get()
        nextNodes = currentGate.nextNodes
        prevNodes = currentGate.prevDelays

        if len(nextNodes) == 0:          # reached sink
            sum = 0
            for prevGate in prevNodes:
                sum += prevGate.randVar[1]
            constr.append( sum <= 1 )       # only 1 edge into the sink
            continue


        # compute fanout
        fanout = 0
        for nextGate in nextNodes:
            fanout += nextGate.randVar[1]

            # add prev gates
            nextGate.appendPrevDelays(currentGate)

            # update queue
            if nextGate not in closedList:
                queue.put(nextGate)
                closedList.add(nextGate)

        # only 1 path can be chosen
        constr.append( fanout <= 1 )

        if len(prevNodes) == 0: continue     # reached source


        gateCost = currentGate.randVar[1]

        # compute fanin
        fanin = 0
        for prevGate in prevNodes:
            fanin += prevGate.randVar[1]


        # set driving constraint
        constr.append( fanin - gateCost >= 0 )


    # formulate LP

    obj = cp.Maximize( cp.sum( cp.multiply(path, values) ) )


    # solve
    problem = cp.Problem(obj, constr)
    problem.solve(solver=cp.MOSEK)


    # print out values
    print("prob value: ", problem.value)
    print("path: ", path.value)


def FindMaxDelayEdges(source):
    """
    Computes maximum possible path. CVXPY variables as edges
    :param: Source of the graph - Node class
    :return: integer, maximum delay
    """


    # set constraints for fan-out

    queue = Queue()
    queue.put(source)
    closedList = set()

    circuitDelay = 0
    constr = []
    variables = []

    while not queue.empty():

        currentGate = queue.get()
        nextNodes = currentGate.nextNodes
        prevEdges = currentGate.prevDelays

        numOfNextNodes = len(nextNodes)

        if numOfNextNodes == 0:          # reached sink
            sum = 0
            for prevEdge in prevEdges:
                sum += prevEdge
            constr.append( sum <= 1 )       # only 1 edge into the sink
            continue


        # compute fanout
        fanout = 0
        for nextGate in nextNodes:

            edge = cp.Variable((1, ), nonneg=True)
            variables.append(edge)

            fanout += edge

            delay = nextGate.randVar

            # add to objective
            circuitDelay += edge * delay

            # add prev gates
            nextGate.appendPrevDelays(edge)


            # update queue
            if nextGate not in closedList:
                queue.put(nextGate)
                closedList.add(nextGate)

        # only 1 path can be chosen
        constr.append( fanout <= 1 )

        if len(prevEdges) == 0: continue     # reached source

        # compute fanin
        fanin = 0
        for prevEdge in prevEdges:
            fanin += prevEdge


        # set driving constraint
        constr.append( fanin - fanout >= 0 )


    # formulate LP

    obj = cp.Maximize( circuitDelay )

    # solve
    problem = cp.Problem(obj, constr)
    problem.solve(solver=cp.MOSEK)


    # print out values
    print("prob value: ", problem.value)
    print("path: \n")
    for variable in variables:
        print(variable.value)


    return problem.value


def putIntoQueue(queue: Queue, list: [Node]) -> None:
    """
    Function puts list into queue.

    :param queue: Queue
    :return list: array of Node class
    """

    for item in list:
        queue.put(item)




# call function

if __name__ == "__main__":

    source = Node(0)
    n1 = Node(3)
    n2 = Node(2)
    n3 = Node(1)
    n4 = Node(4)
    n5 = Node(10)
    sink = Node(0)

    # set circuit design
    n1.setNextNodes([n3, n4])
    n2.setNextNodes([n3, n5])
    n3.setNextNodes([n4, n5])

    # set sink and source
    n4.setNextNodes([sink])
    n5.setNextNodes([sink])
    source.setNextNodes([n1, n2])

    # call the function
    FindMaxDelayEdges(source)
