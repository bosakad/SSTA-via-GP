
import numpy as np
import sys

import cvxpyVariable
import cvxpy as cp

sys.path.append('../Numpy')

from randomVariableHist import RandomVariable
import histogramGenerator


def test_CVXPY_MAX(dec: int):

    mu1 = 12.98553396
    sigma1 = 4.76804456

    mu2 = 13.98483475
    sigma2 = 4.802585

    interval = (-10, 50)

    numberOfSamples = 2000000
    numberOfBins = 20
    numberOfUnions = 50


    # DESIRED

    rv1 = histogramGenerator.get_gauss_bins_UNIFIED(mu1, sigma1, numberOfBins, numberOfSamples, interval, numberOfUnions)
    rv2 = histogramGenerator.get_gauss_bins_UNIFIED(mu2, sigma2, numberOfBins, numberOfSamples, interval, numberOfUnions)

    test1 = histogramGenerator.get_gauss_bins(mu1, sigma1, numberOfBins, numberOfSamples, interval)
    test2 = histogramGenerator.get_gauss_bins(mu2, sigma2, numberOfBins, numberOfSamples, interval)

    # max1 = rv1.maxOfDistributionsQUAD_FORMULA_UNIFIED(rv2)
    max1 = test1.maxOfDistributionsFORM(test2)
    desired = [max1.mean, max1.std]


    # ACTUAL

        # init

    x1 = {}

    for bin in range(0, numberOfBins):
        x1[bin] = {}
        for union in range(0, numberOfUnions):
            (x1[bin])[union] = cp.Variable(boolean=True)

    x2 = {}

    for bin in range(0, numberOfBins):
        x2[bin] = {}
        for union in range(0, numberOfUnions):
            (x2[bin])[union] = cp.Variable(boolean=True)


        # GET obj. function and constr

    maximum, constr = cvxpyVariable.maximumCVXPY_QUAD_UNIFIED(x1, x2)

        # FORMULATE

    # objective function
    sum = 0
    for bin in range(0, numberOfBins):
        for union in range(0, numberOfUnions):
            sum += (maximum[bin])[union]

    # other constraints

    for bin in range(0, numberOfBins):
        for union in range(0, numberOfUnions):
            constr.append( (x1[bin])[union] >= rv1.bins[bin, union] )    # set lower constr.

    for bin in range(0, numberOfBins):
        for union in range(0, numberOfUnions):
            constr.append( (x2[bin])[union] >= rv2.bins[bin, union] )    # set lower constr.



        # solve
    objective = cp.Minimize( sum )
    prob = cp.Problem(objective, constr)
    prob.solve(verbose=True, solver=cp.MOSEK)

        # print out the values
    print("Problem value: " + str(prob.value))

    # x1_cp = np.zeros((numberOfBins, numberOfUnions))
    # for bin in range(0, numberOfBins):
    #     for union in range(0, numberOfUnions):
    #         x1_cp[bin, union] = (x1[bin])[union].value
    # print()
    # x2_cp = np.zeros((numberOfBins, numberOfUnions))
    # for bin in range(0, numberOfBins):
    #     for union in range(0, numberOfUnions):
    #         x2_cp[bin, union] = (x1[bin])[union].value
    # print(x1_cp)
    # print(x2_cp)

    maxBins = np.zeros((numberOfBins, numberOfUnions))
    for bin in range(0, numberOfBins):
        for union in range(0, numberOfUnions):
            maxBins[bin, union] = (maximum[bin])[union].value

    edges = np.linspace(interval[0], interval[1], numberOfBins + 1)
    maxRV = RandomVariable(maxBins, edges, unified=True)

    # print(maxRV.bins)

    actual = [maxRV.mean, maxRV.std]

    # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

    return None



if __name__ == "__main__":
        # dec param is Desired precision

    test_CVXPY_MAX(dec=5)


    print("All tests passed!")