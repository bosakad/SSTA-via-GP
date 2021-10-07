from randomVariableHist import RandomVariable
import parser
import sys
import numpy as np
import histogramGenerator
from node import Node
import SSTA
import matplotlib.pyplot as plt


"""
Script main.py is used as a main method for circuit optimization.
Script calls all fundamental methods as well as parsers.
"""


def main(argv):

    # circuitMatrix = parser.getIncidenceMatrixFromNetlist(argv)

    # h1 = RandomVariable([2, 3, 4], [0, 1, 3, 7])
    # h2 = RandomVariable([3, 8, 6], [0, 1, 3, 7])

    numberOfBins = 100
    numberOfSamples = int(100000)

    g1 = histogramGenerator.get_gauss_bins(1, 0.45, numberOfBins, numberOfSamples)  # g1, g2 INPUT gates, g3 middle
    g2 = histogramGenerator.get_gauss_bins(0.5, 0.3, numberOfBins, numberOfSamples)  # g4 output - inputs: g3 g1
    g3 = histogramGenerator.get_gauss_bins(0.5, 0.5, numberOfBins, numberOfSamples)  # g5 output - inputs: g3, g2
    g4 = histogramGenerator.get_gauss_bins(0.5, 0.5, numberOfBins, numberOfSamples)
    g5 = histogramGenerator.get_gauss_bins(0.5, 0.5, numberOfBins, numberOfSamples)


    n1 = Node(g1)
    n2 = Node(g2)
    n3 = Node(g3)
    n4 = Node(g4)
    n5 = Node(g5)


        # set circuit design
    n1.setNextNodes( [n3, n4] )
    n2.setNextNodes( [n3, n5] )
    n3.setNextNodes( [n4, n5] )

    delays = SSTA.calculateCircuitDelay([n1, n2])

    sinkDelay = delays[-1]

    print(f'The mean delay is {sinkDelay.mean}')
    print(f'The std of a delay is {sinkDelay.std}')

    plt.hist(sinkDelay.edges[:-1], sinkDelay.edges, weights=sinkDelay.bins)
    plt.ylabel('PDF of delay', size=14)
    plt.xlabel('time', size=14)
    plt.title('Histogram of the MAX delay', size=16)
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])
