import test_infiniteLadder
import numpy as np

"""
    This Module includes functions that test infiniteLadder on on 2 algorithms and dependant on number of gates.
    Indicator is number of nonzero constraints dependant on number of gates
"""

def testAlgorithms():

        # number of testing
    numberOfIterations = 4
    step = 2

    numberOfGatesStart = 10
    numberOfBins = 10
    numberOfUnaries = 10

    interval = (-8, 35)

        # for saving values of the last gates and calculating error
    rvs_nonPrecise = np.zeros((numberOfIterations, 2))
    rvs_Precise = np.zeros((numberOfIterations, 2))
    rvs_MonteCarlo = np.zeros((numberOfIterations, 2))

    results = {}



    print("NON-PRECISE\n\n")
        # test non-precise
    for iter in range(0, numberOfIterations):
        print(str(iter) + ". iteration: \n\n")

            # calculating
        numGates = numberOfGatesStart + iter * step
        numNonZeros, ObjVal, lastGate = test_infiniteLadder.main(numGates, numberOfUnaries, numberOfBins, interval, precise=False)

            # saving values
        rvs_nonPrecise[iter, 0] = lastGate[0]
        rvs_nonPrecise[iter, 1] = lastGate[1]
        results[(numGates, False)] = (numNonZeros, ObjVal)

        # print results
    print("\n\n" + str(results))

    print("PRECISE\n\n")
        # test precise
    for iter in range(0, numberOfIterations):
        print("\n\n" + str(iter) + ". iteration: \n\n")

        numGates = numberOfGatesStart + iter*step
        numNonZeros, ObjVal, lastGate = test_infiniteLadder.main(numGates, numberOfUnaries, numberOfBins, interval, precise=True)

        # saving values
        rvs_Precise[iter, 0] = lastGate[0]
        rvs_Precise[iter, 1] = lastGate[1]
        results[(numGates, True)] = (numNonZeros, ObjVal)


    # MonteCarlo
    for iter in range(0, numberOfIterations):
        print("\n\n" + str(iter) + ". iteration: \n\n")

        numGates = numberOfGatesStart + iter * step
        lastGate = test_infiniteLadder.MonteCarlo(numGates)

        # saving values
        rvs_MonteCarlo[iter, 0] = lastGate[0]
        rvs_MonteCarlo[iter, 1] = lastGate[1]


    print("\nNON-PRECISE:\n")
    print(rvs_nonPrecise)

    print("\nPRECISE:\n")
    print(rvs_Precise)

    print("\nGROUND-TRUTH:\n")
    print(rvs_MonteCarlo)

    print("\nNON-PRECISE DIFF:\n")
    print(np.abs(rvs_nonPrecise - rvs_MonteCarlo))

    print("\nPRECISE DIFF:\n")
    print(np.abs(rvs_Precise - rvs_MonteCarlo) )


        # print results
    print("\n\n" + str(results))



if __name__ == "__main__":

    testAlgorithms()