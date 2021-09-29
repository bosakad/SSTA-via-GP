import io
import sys


""" Get incidence matrix from netlist.

    Function calls parser functions, return a cell-edge incidence matrix of a circuit.

    Params:
        argv: arguments given to main function
    Return:
        matrix: incidence matrix 

"""

def getIncidenceMatrixFromNetlist(argv):
    netListString = parseFileName(argv)

    # matrix = parseNetListIntoMatrix(netListString)
    # return matrix

    return None



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
    buf = io.StringIO(netList)
    readLine = ""

    # skip empty lines or comments
    while not readLine.startswith("INPUT"):
        readLine = buf.readline()

        # read inputs
    while readLine.startswith("INPUT"):
        print(readLine)
        # todo: save input gates
        readLine = buf.readline()

        # skip empty lines or comments
    while not readLine.startswith("OUTPUT"):
        readLine = buf.readline()

        # read outputs
    while readLine.startswith("OUTPUT"):
        print(readLine)
        # todo: save output gates
        readLine = buf.readline()

        # skip empty lines or comments
    while readLine.startswith("\n"):
        readLine = buf.readline()

        # parse circuit
    while readLine != "":
        print(readLine)
        # todo: save circuit into matrix
        readLine = buf.readline()


""" Parse gate properties from .txt

    Function should parse all gates properties for VLSI optimization such as
    -- alphas, betas, gammas constants
    -- energyLoss, frequencies

"""

def parseGatesPropertiesFromTXT(fileName, gates):
    #TODO
    pass