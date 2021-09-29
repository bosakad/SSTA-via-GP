import randomVariableHist

""" Node class

    Class representing a node.
    Class includes: 
        class 'RandomVariable',
        array of next gates,

"""


class Node:

    def __init__(self, randomVar):
        self.randVar = randomVar
        self.nextNodes = []
        self.prevDelays = []

    def setNextNodes(self, nextNodes):
        self.nextNodes = nextNodes


    def setPrevDelays(self, prevDelays):
        self.prevDelays = prevDelays

    def appendPrevDelays(self, prevDelays):
        self.prevDelays.append(prevDelays)