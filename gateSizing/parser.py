import io
import sys
import re
import numpy

""" Get incidence matrix from netlist.

    Function calls parser functions, return a cell-edge incidence matrix of a circuit.

    Params:
        argv: arguments given to main function
    Return:
        matrix: incidence matrix 

"""

def getIncidenceMatrixFromNetlist(argv):
    netListString = parseFileName(argv)

    matrix = parseNetListIntoMatrix(netListString)
    return matrix



""" Parses file name. Does edge casing. Returns content of the file if
    no errors appeared.

      Params:
        argv: list of arguments
      Return: name of the file  """

def parseFileName(argv):

    # edge casing

    if len(argv) == 0:
        print('Error: Name of the .bench file needed!')
        sys.exit(-1)

    elif len(argv) > 1:
        print('Error: Too many arguments!')
        sys.exit(-2)

    elif len(argv) > 1:
        print('Error: Too many arguments!')
        sys.exit(-2)

    elif not argv[0].endswith('.bench'):
        print('Error: Wrong file! Script accepts only .bench files.')
        sys.exit(-3)

    try:
        # loading file
        fileName = argv[0]
        file = open(fileName, "r")
        ret = file.read()
        file.close()

    except IOError:
        print("Error: File does not appear to exist.")
        sys.exit(-4)

    return ret

""" TODO: FINISH

Parses string and puts the information into cell-edge incidence matrix.

      Params:
        netList: string with circuit information
      Return: cell-edge incidence matrix  """

def parseNetListIntoMatrix(netList):
    global gateIndex, edgeIndex, mappingList, matrixDict
    buf = io.StringIO(netList)
    readLine = ""

    inputGates = []

    # skip empty lines or comments
    while not readLine.startswith("INPUT"):
        readLine = buf.readline()


        # read inputs
    while readLine.startswith("INPUT"):

        gateNum = int(re.search(r'\d+', readLine).group())
        inputGates.append(gateNum)   # input gates are -1

        readLine = buf.readline()


        # skip empty lines or comments
    while not readLine.startswith("OUTPUT"):
        readLine = buf.readline()

        # read outputs
    while readLine.startswith("OUTPUT"):
        # todo: save output gates
        readLine = buf.readline()

        # skip empty lines or comments
    while readLine.startswith("\n"):
        readLine = buf.readline()



        matrixDict = {} # matrix in dict
        mappingList = {} # list to map a gate to a matrix index

        edgeIndex = 0
        gateIndex = 0

        # parse circuit - for circuit information
    while readLine != "":
        gates = list(map(int, re.findall(r'\d+', readLine)))    # get line gates

        newGate = gates[0]                  # get new gate and create a mapping
        mappingList[newGate] = gateIndex

        if gates[1] not in inputGates:
            matrixDict[(gateIndex, edgeIndex)] = -1
            matrixDict[(mappingList[gates[1]], edgeIndex)] = 1
            edgeIndex += 1
        if gates[2] not in inputGates:
            matrixDict[(gateIndex, edgeIndex)] = -1
            matrixDict[(mappingList[gates[2]], edgeIndex)] = 1
            edgeIndex += 1

        gateIndex += 1

        readLine = buf.readline()


    # put dict. into matrix

    matrix = numpy.array([[0]* edgeIndex] * gateIndex)

    for (key, value) in matrixDict.items():
        c = key[0]
        e = key[1]
        matrix[c, e] = value


    return matrix


""" Parse gate properties from .txt

    Function should parse all gates properties for VLSI optimization such as
    -- alphas, betas, gammas constants
    -- energyLoss, frequencies

"""

def parseGatesPropertiesFromTXT(fileName, gates):
    #TODO
    pass



