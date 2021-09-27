from parser import Parser
from randomVariableHist import RandomVariable
import sys
import numpy as np

"""
Script main.py is used as a main method for circuit optimization.
Script calls all fundamental methods as well as parsers.
"""


def main(argv):
    parser = Parser()
    netListString = parser.parseFileName(argv)
    # matrix = parser.parseNetListIntoMatrix(netListString)

    # h1 = RandomVariable([2, 3, 4], [0, 1, 3, 7])
    # h2 = RandomVariable([3, 8, 6], [0, 1, 3, 7])





if __name__ == "__main__":
    main(sys.argv[1:])
