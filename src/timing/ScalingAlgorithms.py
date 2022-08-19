import src.timing.infiniteLadder as infiniteLadder
import numpy as np


"""
    This Module includes functions that test scalability infiniteLadder on on 2 algorithms and dependant on number of gates.
    Indicator is number of nonzero constraints dependant on number of gates. Also tests optimization of the circuit. 
"""


def AlgorithmsScaling():

    # number of testing
    numberOfIterations = 1
    step = 1

    numberOfGatesStart = 1
    numberOfBins = 10
    numberOfUnaries = 10

    interval = (0, 35)

    # for saving values of the last gates and calculating error
    rvs_nonPrecise = np.zeros((numberOfIterations, 2))
    rvs_Precise = np.zeros((numberOfIterations, 2))
    rvs_MonteCarlo = np.zeros((numberOfIterations, 2))

    results = {}

    # MonteCarlo
    for iter in range(0, numberOfIterations):

        numGates = numberOfGatesStart + iter * step
        lastGate = infiniteLadder.MonteCarlo(numGates)

        # saving values
        rvs_MonteCarlo[iter, 0] = lastGate[0]
        rvs_MonteCarlo[iter, 1] = lastGate[1]

    print("SYMMETRY\n\n")
    # test precise
    for iter in range(0, numberOfIterations):
        print("\n\n" + str(iter) + ". iteration: \n\n")

        numGates = numberOfGatesStart + iter * step
        numNonZeros, ObjVal, lastGate, time = infiniteLadder.main(
            numGates, numberOfUnaries, numberOfBins, interval, withSymmetryConstr=True
        )

        # saving values
        rvs_Precise[iter, 0] = lastGate[0]
        rvs_Precise[iter, 1] = lastGate[1]

        MAPE = 100 * np.abs(
            np.divide(
                rvs_Precise[iter, :] - rvs_MonteCarlo[iter, :], rvs_MonteCarlo[iter, :]
            )
        )

        if iter != 0:
            prevError = np.zeros(2)
            prevError[0] = results[(numGates - step, True)][2]
            prevError[1] = results[(numGates - step, True)][3]

            MAPE = ((MAPE + prevError) * iter) / (iter + 1)

        results[(numGates, True)] = (numNonZeros, ObjVal, time, MAPE[0], MAPE[1])

    print("NO SYMMETRY\n\n")
    # test non-precise
    for iter in range(0, numberOfIterations):
        print(str(iter) + ". iteration: \n\n")

        # calculating
        numGates = numberOfGatesStart + iter * step
        numNonZeros, ObjVal, lastGate, time = infiniteLadder.main(
            numGates, numberOfUnaries, numberOfBins, interval, withSymmetryConstr=False
        )
        print(time)
        # saving values
        rvs_nonPrecise[iter, 0] = lastGate[0]
        rvs_nonPrecise[iter, 1] = lastGate[1]

        MAPE = 100 * np.abs(
            np.divide(
                rvs_nonPrecise[iter, :] - rvs_MonteCarlo[iter, :],
                rvs_MonteCarlo[iter, :],
            )
        )

        if iter != 0:
            prevError = np.zeros(2)
            prevError[0] = results[(numGates - step, False)][2]
            prevError[1] = results[(numGates - step, False)][3]

            MAPE = ((MAPE + prevError) * iter) / (iter + 1)

        results[(numGates, False)] = (numNonZeros, ObjVal, time, MAPE[0], MAPE[1])

        # print results
    print("\n\n" + str(results))

    print("\nNON-symmetry:\n")
    print(rvs_nonPrecise)

    print("\nsymmetry:\n")
    print(rvs_Precise)

    print("\nGROUND-TRUTH:\n")
    print(rvs_MonteCarlo)

    print("\n no symmetry MAPE:\n")
    print(100 * np.abs(np.divide(rvs_nonPrecise - rvs_MonteCarlo, rvs_MonteCarlo)))

    print("\nsymmetry MAPE:\n")
    print(100 * np.abs(np.divide(rvs_Precise - rvs_MonteCarlo, rvs_MonteCarlo)))

    # print results
    print("\n\n" + str(results))


def AlgorithmsScaling_MOSEK():

    # number of testing
    numberOfIterations = 1
    step = 1
    # prevMape = np.array([3.0627844482985895, 31.477761094020565])

    numberOfGatesStart = 3
    numberOfBins = 10
    numberOfUnaries = 10

    interval = (-4, 19)

    # for saving values of the last gates and calculating error
    rvs_nonPrecise = np.zeros((numberOfIterations, 2))
    rvs_Precise = np.zeros((numberOfIterations, 2))
    rvs_MonteCarlo = np.zeros((numberOfIterations, 2))

    results = {}

    # MonteCarlo
    for iter in range(0, numberOfIterations):
        numGates = numberOfGatesStart + iter * step
        lastGate = infiniteLadder.MonteCarlo(numGates)

        # saving values
        rvs_MonteCarlo[iter, 0] = lastGate[0]
        rvs_MonteCarlo[iter, 1] = lastGate[1]

    print("SYMMETRY\n\n")
    # test precise
    for iter in range(0, numberOfIterations):
        print("\n\n" + str(iter) + ". iteration: \n\n")

        numGates = numberOfGatesStart + iter * step
        (
            numNonZeros,
            ObjVal,
            lastGate,
            time,
            numVars,
            numConstr,
            mipGapRoot,
            nVarsPresolve,
            nConstrPresolve,
        ) = infiniteLadder.mainMOSEK(
            numGates,
            numberOfUnaries,
            numberOfBins,
            interval,
            TRI=False,
            withSymmetryConstr=True,
        )

        # saving values
        rvs_Precise[iter, 0] = lastGate[0]
        rvs_Precise[iter, 1] = lastGate[1]

        print(
            np.abs(rvs_Precise[iter, :] - rvs_MonteCarlo[iter, :])
            / rvs_MonteCarlo[iter, :]
        )

        MAPE = 100 * np.abs(
            np.divide(
                rvs_Precise[iter, :] - rvs_MonteCarlo[iter, :], rvs_MonteCarlo[iter, :]
            )
        )

        print(MAPE)

        if iter != 0:
            prevError = np.zeros(2)
            prevError[0] = results[(numGates - step, True)][3]
            prevError[1] = results[(numGates - step, True)][4]

            MAPE = (MAPE + (prevError) * (iter)) / (iter + 1)
        elif iter == 0:

            MAPE = (MAPE) / (iter + 1)

        results[(numGates, True)] = (
            numNonZeros,
            ObjVal,
            time,
            MAPE[0],
            MAPE[1],
            numVars,
            numConstr,
            mipGapRoot,
            nVarsPresolve,
            nConstrPresolve,
        )

        # print results
        print("\n\n" + str(results))

    # print("NO SYMMETRY\n\n")
    # # test non-precise
    # for iter in range(0, numberOfIterations):
    #     print(str(iter) + ". iteration: \n\n")
    #
    #     # calculating
    #     numGates = numberOfGatesStart + iter * step
    #     numNonZeros, ObjVal, lastGate, time, numVars, numConstr, mipGapRoot,nVarsPresolve, nConstrPresolve   = \
    #                                             test_infiniteLadder.mainMOSEK(numGates, numberOfUnaries, numberOfBins,
    #                                                                    interval,
    #                                                                    withSymmetryConstr=False)
    #     # saving values
    #     rvs_nonPrecise[iter, 0] = lastGate[0]
    #     rvs_nonPrecise[iter, 1] = lastGate[1]
    #
    #     MAPE = 100 * np.abs(np.divide(rvs_nonPrecise[iter, :] - rvs_MonteCarlo[iter, :], rvs_MonteCarlo[iter, :]))
    #
    #     if iter != 0:
    #         prevError = np.zeros(2)
    #         prevError[0] = results[(numGates - step, False)][2]
    #         prevError[1] = results[(numGates - step, False)][3]
    #
    #         MAPE = ((MAPE + prevError) * iter) / (iter + 1)
    #
    #     results[(numGates, False)] = (numNonZeros, ObjVal, time, MAPE[0], MAPE[1], numVars, numConstr, mipGapRoot,
    #                                                                                     nVarsPresolve, nConstrPresolve)
    #
    #     # print results
    #     print("\n\n" + str(results))

    # print("\nNON-symmetry:\n")
    # print(rvs_nonPrecise)
    #
    # print("\nsymmetry:\n")
    # print(rvs_Precise)
    #
    # print("\nGROUND-TRUTH:\n")
    # print(rvs_MonteCarlo)
    #
    # print("\n no symmetry MAPE:\n")
    # print(100 * np.abs(np.divide(rvs_nonPrecise - rvs_MonteCarlo, rvs_MonteCarlo)))
    #
    # print("\nsymmetry MAPE:\n")
    # print(100 * np.abs(np.divide(rvs_Precise - rvs_MonteCarlo, rvs_MonteCarlo)))

    # print results
    # print("\n\n" + str(results))


def testAlgorithms_CVXPY_GP():

    # number of testing
    numberOfIterations = 15
    step = 1

    numberOfGatesStart = 10
    numberOfBins = 15

    interval = (-4, 25)

    # for saving values of the last gates and calculating error
    rvs_nonPrecise = np.zeros((numberOfIterations, 2))
    rvs_Precise = np.zeros((numberOfIterations, 2))
    rvs_MonteCarlo = np.zeros((numberOfIterations, 2))

    results = {}

    # MonteCarlo
    for iter in range(0, numberOfIterations):
        numGates = numberOfGatesStart + iter * step
        lastGate = infiniteLadder.MonteCarlo(numGates)

        # saving values
        rvs_MonteCarlo[iter, 0] = lastGate[0]
        rvs_MonteCarlo[iter, 1] = lastGate[1]

    # test precise
    for iter in range(0, numberOfIterations):
        print("\n\n" + str(iter) + ". iteration: \n\n")

        numGates = numberOfGatesStart + iter * step
        lastGate, time = infiniteLadder.mainCVXPY_GP(numGates, numberOfBins, interval)

        # saving values
        rvs_Precise[iter, 0] = lastGate[0]
        rvs_Precise[iter, 1] = lastGate[1]

        # print(np.abs(rvs_Precise[iter, :] - rvs_MonteCarlo[iter, :]) / rvs_MonteCarlo[iter, :])

        MAPE = 100 * np.abs(
            np.divide(
                rvs_Precise[iter, :] - rvs_MonteCarlo[iter, :], rvs_MonteCarlo[iter, :]
            )
        )

        print(MAPE)

        if iter != 0:
            prevError = np.zeros(2)
            prevError[0] = results[(numGates - step, True)][3]
            prevError[1] = results[(numGates - step, True)][4]

            MAPE = (MAPE + (prevError) * (iter)) / (iter + 1)

        results[(numGates, True)] = (-1, -1, time, MAPE[0], MAPE[1], -1, -1, -1, -1, -1)

        # print results
        print("\n\n" + str(results))


def scalingBins_CVXPY_GP():

    # number of testing
    numberOfIterations = 5
    step = 4

    numberOfGates = 2
    numBinsStart = 10

    interval = (-4, 25)

    # for saving values of the last gates and calculating error
    rvs_nonPrecise = np.zeros((numberOfIterations, 2))
    rvs_Precise = np.zeros((numberOfIterations, 2))
    rvs_MonteCarlo = np.zeros((1, 2))

    results = {}

    # MonteCarlo
    for iter in range(0, 1):
        numGates = numberOfGates
        lastGate = infiniteLadder.MonteCarlo(numGates)

        # saving values
        rvs_MonteCarlo[iter, 0] = lastGate[0]
        rvs_MonteCarlo[iter, 1] = lastGate[1]

    print("SYMMETRY\n\n")

    BINS = [5, 8, 10]
    numberOfIterations = len(BINS)
    # test precise
    for iter in range(0, numberOfIterations):
        print("\n\n" + str(iter) + ". iteration: \n\n")

        # numBins = numBinsStart + iter * step
        numBins = BINS[iter]
        print(numBins)
        lastGate, time = infiniteLadder.mainCVXPY_GP(numberOfGates, numBins, interval)

        # saving values
        rvs_Precise[iter, 0] = lastGate[0]
        rvs_Precise[iter, 1] = lastGate[1]

        # print(np.abs(rvs_Precise[iter, :] - rvs_MonteCarlo[iter, :]) / rvs_MonteCarlo[iter, :])

        MAPE = 100 * np.abs(
            np.divide(rvs_Precise[iter, :] - rvs_MonteCarlo[0, :], rvs_MonteCarlo[0, :])
        )

        print(MAPE)

        # if iter != 0:
        #     prevError = np.zeros(2)
        #     lastNumBins = BINS[iter-1]
        #
        #     prevError[0] = results[(lastNumBins, True)][3]
        #     prevError[1] = results[(lastNumBins, True)][4]
        #     prevError[0] = results[(numBins - step, True)][3]
        #     prevError[1] = results[(numBins - step, True)][4]
        #
        # MAPE = (MAPE + (prevError) *(iter)) / (iter + 1)

        results[(numBins, True)] = (-1, -1, time, MAPE[0], MAPE[1], -1, -1, -1, -1, -1)

        # print results
        print("\n\n" + str(results))


def scalingOptimization_CVXPY_GP():

    # number of testing
    numberOfBins = 20

    interval = (0, 28)

    results = {}

    GATES = [1, 4, 7, 10, 12]
    numberOfIterations = len(GATES)
    # test precise
    for iter in range(0, numberOfIterations):
        print("\n\n" + str(iter) + ". iteration: \n\n")

        # numBins = numBinsStart + iter * step
        numGates = GATES[iter]
        print(numGates)
        # lastGate, time = test_infiniteLadder.mainCVXPY_GP_Sizing(numGates, numberOfBins, interval)
        lastGate, time = infiniteLadder.mainCVXPY_GP(numGates, numberOfBins, interval)

        results[(numGates, False)] = (-1, -1, time, 0, 0, -1, -1, -1, -1, -1)

        # print results
        print("\n\n" + str(results))


def testAlgorithms_PRESOLVE():

    # number of testing
    numberOfIterations = 1
    step = 1
    numberOfGatesStart = 1

    numIterPass = 1
    numberOfPassesStart = 100000
    passesStep = 100

    numberOfBins = 15
    numberOfUnaries = 15

    interval = (-5, 18)

    results = {}

    for passes in range(0, numIterPass):
        curPasses = numberOfPassesStart * (passesStep**passes)

        print("\n\n" + str(passes) + " pass: \n\n")
        # test precise
        for iter in range(0, numberOfIterations):

            print("\n\n" + str(iter) + ". iteration: \n\n")

            numGates = numberOfGatesStart + iter * step
            (
                numNonZeros,
                ObjVal,
                lastGate,
                time,
                numVars,
                numConstr,
            ) = infiniteLadder.mainMOSEK(
                numGates,
                numberOfUnaries,
                numberOfBins,
                interval,
                withSymmetryConstr=True,
                presolvePasses=curPasses,
            )

            results[(numGates, curPasses)] = (numNonZeros, numVars, numConstr)

        # print results
    print("\n\n" + str(results))


#     np.testing.assert_almost_equal(desired, actual, decimal=5)


def computeMAPE(n_bins, n_unaries, start, function):

    MAPE_mean = np.zeros((n_bins, n_unaries))
    MAPE_std = np.zeros((n_bins, n_unaries))

    jump = 2

    for bins in range(0, n_bins):
        for unaries in range(0, n_unaries):
            actual, desired = function(start + jump * bins, start + jump * unaries)

            print("bins:" + str(start + jump * bins))
            print("unaries: " + str(start + jump * unaries))
            print(actual)
            print(desired)

            curMAPE = 100 * np.abs((actual - desired) / desired)

            print(curMAPE)

            MAPE_mean[bins, unaries] = curMAPE[0]
            MAPE_std[bins, unaries] = curMAPE[1]

    return MAPE_mean, MAPE_std


if __name__ == "__main__":

    scalingBins_CVXPY_GP()
    # scalingOptimization_CVXPY_GP()

    # testAlgorithms_PRESOLVE()
