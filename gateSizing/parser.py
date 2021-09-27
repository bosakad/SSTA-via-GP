import io
import sys


"""
    Class representing a parser - a clean way to store all the parser methods in python.
"""

class Parser:
    def __init__(self):
        pass

    """ Parses file name. Does edge casing. Returns content of the file if
        no errors appeared.

          Params:
            argv: list of arguments
          Return: name of the file  """

    def parseFileName(self, argv):

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

    def parseNetListIntoMatrix(self, netList):
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
