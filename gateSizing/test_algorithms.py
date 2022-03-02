import test_infiniteLadder
import numpy as np

"""
    This Module includes functions that test infiniteLadder on on 2 algorithms and dependant on number of gates.
    Indicator is number of nonzero constraints dependant on number of gates
"""

def testAlgorithms():

        # number of testing
    numberOfIterations = 3
    step = 2

    numberOfGatesStart = 10
    numberOfBins = 5
    numberOfUnaries = 2

    interval = (-8, 35)

        # for saving values of the last gates and calculating error
    rvs_nonPrecise = np.zeros((numberOfIterations, 2))
    rvs_Precise = np.zeros((numberOfIterations, 2))
    rvs_MonteCarlo = np.zeros((numberOfIterations, 2))

    results = {}

    # MonteCarlo
    for iter in range(0, numberOfIterations):
        print("\n\n" + str(iter) + ". iteration: \n\n")

        numGates = numberOfGatesStart + iter * step
        lastGate = test_infiniteLadder.MonteCarlo(numGates)

        # saving values
        rvs_MonteCarlo[iter, 0] = lastGate[0]
        rvs_MonteCarlo[iter, 1] = lastGate[1]


    print("NO SYMMETRY\n\n")
        # test non-precise
    for iter in range(0, numberOfIterations):
        print(str(iter) + ". iteration: \n\n")

            # calculating
        numGates = numberOfGatesStart + iter * step
        numNonZeros, ObjVal, lastGate = test_infiniteLadder.main(numGates, numberOfUnaries, numberOfBins, interval,
                                                                 precise=False, withSymmetryConstr=False)

            # saving values
        rvs_nonPrecise[iter, 0] = lastGate[0]
        rvs_nonPrecise[iter, 1] = lastGate[1]

        error = np.sum(np.abs( rvs_nonPrecise[iter, :] - rvs_MonteCarlo[iter, :] ))
        results[(numGates, False)] = (numNonZeros, ObjVal, error)

        # print results
    print("\n\n" + str(results))

    print("SYMMETRY\n\n")
        # test precise
    for iter in range(0, numberOfIterations):
        print("\n\n" + str(iter) + ". iteration: \n\n")

        numGates = numberOfGatesStart + iter*step
        numNonZeros, ObjVal, lastGate = test_infiniteLadder.main(numGates, numberOfUnaries, numberOfBins, interval,
                                                                 precise=False, withSymmetryConstr=True)

        # saving values
        rvs_Precise[iter, 0] = lastGate[0]
        rvs_Precise[iter, 1] = lastGate[1]

        error = np.sum(np.abs( rvs_Precise[iter, :] - rvs_MonteCarlo[iter, :] ))
        results[(numGates, True)] = (numNonZeros, ObjVal, error)





    print("\nNON-symmetry:\n")
    print(rvs_nonPrecise)

    print("\nsymmetry:\n")
    print(rvs_Precise)

    print("\nGROUND-TRUTH:\n")
    print(rvs_MonteCarlo)

    print("\n no symmetry DIFF:\n")
    print(np.abs(rvs_nonPrecise - rvs_MonteCarlo))

    print("\nsymmetry DIFF:\n")
    print(np.abs(rvs_Precise - rvs_MonteCarlo) )


        # print results
    print("\n\n" + str(results))



if __name__ == "__main__":

    testAlgorithms()