
import numpy as np
import sys

from cvxpyVariable import RandomVariableCVXPY
import cvxpy as cp

sys.path.append('../Numpy')

from randomVariableHist import RandomVariable
import histogramGenerator


def test_CVXPY_MAX_UNIFIED_OLD(dec: int):

    mu1 = 12.98553396
    sigma1 = 4.76804456

    mu2 = 13.98483475
    sigma2 = 4.802585

    interval = (-10, 50)

    numberOfSamples = 2000000
    numberOfBins = 10
    numberOfUnions = 10


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

    RV1 = RandomVariableCVXPY(x1, rv1.edges)
    RV2 = RandomVariableCVXPY(x2, rv1.edges)
    maximum, constr = RV1.maximum_QUAD_UNIFIED_OLD(RV2)
    maximum = maximum.bins

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

        # PRINT OUT THE VALUES

    print("Problem value: " + str(prob.value))

    maxBins = np.zeros((numberOfBins, numberOfUnions))
    for bin in range(0, numberOfBins):
        for union in range(0, numberOfUnions):
            maxBins[bin, union] = (maximum[bin])[union].value

    edges = np.linspace(interval[0], interval[1], numberOfBins + 1)
    maxRV = RandomVariable(maxBins, edges, unified=True)

    print(maxRV.bins)
    actual = [maxRV.mean, maxRV.std]

    # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

    return None


def test_CVXPY_MAX_UNIFIED_NEW_AS_MAX(dec: int):

    mu1 = 12.98553396
    sigma1 = 4.76804456

    mu2 = 13.98483475
    sigma2 = 4.802585

    interval = (-10, 50)

    numberOfSamples = 2000000
    numberOfBins = 10
    numberOfUnions = 5


    # DESIRED

    rv1 = histogramGenerator.get_gauss_bins_UNIFIED(mu1, sigma1, numberOfBins, numberOfSamples, interval, numberOfUnions)
    rv2 = histogramGenerator.get_gauss_bins_UNIFIED(mu2, sigma2, numberOfBins, numberOfSamples, interval, numberOfUnions)

    test1 = histogramGenerator.get_gauss_bins(mu1, sigma1, numberOfBins, numberOfSamples, interval)
    test2 = histogramGenerator.get_gauss_bins(mu2, sigma2, numberOfBins, numberOfSamples, interval)

    max1 = rv1.maxOfDistributionsQUAD_FORMULA_UNIFIED(rv2)
    # max1 = test1.maxOfDistributionsFORM(test2)
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

    RV1 = RandomVariableCVXPY(x1, rv1.edges)
    RV2 = RandomVariableCVXPY(x2, rv1.edges)
    maximum, constr = RV1.maximum_QUAD_UNIFIED_NEW_MAX(RV2)
    maximum = maximum.bins

        # FORMULATE

    # objective function
    sum = 0
    for bin in range(0, numberOfBins):
        for union in range(0, numberOfUnions):
            sum += (maximum[bin])[union]

    # other constraints

    for bin in range(0, numberOfBins):
        for union in range(0, numberOfUnions):
            constr.append( (x1[bin])[union] <= rv1.bins[bin, union] )    # set lower constr.

    for bin in range(0, numberOfBins):
        for union in range(0, numberOfUnions):
            constr.append( (x2[bin])[union] <= rv2.bins[bin, union] )    # set lower constr.



        # solve
    objective = cp.Maximize( sum )
    prob = cp.Problem(objective, constr)
    prob.solve(verbose=False, solver=cp.MOSEK)

        # PRINT OUT THE VALUES

    print("Problem value: " + str(prob.value))


    # x2_S = np.zeros((numberOfBins, numberOfUnions))
    # for bin in range(0, numberOfBins):
    #     for union in range(0, numberOfUnions):
    #         x2_S[bin, union] = (x2[bin])[union].value
    #
    #
    # x1_S = np.zeros((numberOfBins, numberOfUnions))
    # for bin in range(0, numberOfBins):
    #     for union in range(0, numberOfUnions):
    #         x1_S[bin, union] = (x1[bin])[union].value

    maxBins = np.zeros((numberOfBins, numberOfUnions))
    for bin in range(0, numberOfBins):
        for union in range(0, numberOfUnions):
            maxBins[bin, union] = (maximum[bin])[union].value

    print(maxBins)

    edges = np.linspace(interval[0], interval[1], numberOfBins + 1)
    maxRV = RandomVariable(maxBins, edges, unified=True)



    actual = [maxRV.mean, maxRV.std]

    # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

    return None

def test_CVXPY_MAX_UNIFIED_NEW_AS_MIN(dec: int):

    mu1 = 12.98553396
    sigma1 = 4.76804456

    mu2 = 13.98483475
    sigma2 = 4.802585

    interval = (-10, 50)

    numberOfSamples = 2000000
    numberOfBins = 14
    numberOfUnions = 25


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

    RV1 = RandomVariableCVXPY(x1, rv1.edges)
    RV2 = RandomVariableCVXPY(x2, rv1.edges)
    maximum, constr = RV1.maximum_QUAD_UNIFIED_NEW_MIN(RV2)
    maximum = maximum.bins

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
    prob.solve(verbose=False, solver=cp.MOSEK)

        # PRINT OUT THE VALUES

    print("Problem value: " + str(prob.value))


    # x2_S = np.zeros((numberOfBins, numberOfUnions))
    # for bin in range(0, numberOfBins):
    #     for union in range(0, numberOfUnions):
    #         x2_S[bin, union] = (x2[bin])[union].value
    #
    #
    # x1_S = np.zeros((numberOfBins, numberOfUnions))
    # for bin in range(0, numberOfBins):
    #     for union in range(0, numberOfUnions):
    #         x1_S[bin, union] = (x1[bin])[union].value

    maxBins = np.zeros((numberOfBins, numberOfUnions))
    for bin in range(0, numberOfBins):
        for union in range(0, numberOfUnions):
            maxBins[bin, union] = (maximum[bin])[union].value

    print(maxBins)

    edges = np.linspace(interval[0], interval[1], numberOfBins + 1)
    maxRV = RandomVariable(maxBins, edges, unified=True)



    actual = [maxRV.mean, maxRV.std]

    # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

    return None


def test_CVXPY_CONVOLUTION_UNIFIED_MAX(dec: int):
    """
    Problem is formulated as maximization
    """

    mu1 = 22.98553396
    sigma1 = 2.76804456

    mu2 = 18.98483475
    sigma2 = 5.802585

    interval = (-5, 40)

    numberOfSamples = 2000000
    numberOfBins = 15
    numberOfUnions = 15


    # DESIRED

    rv1 = histogramGenerator.get_gauss_bins_UNIFIED(mu1, sigma1, numberOfBins, numberOfSamples, interval, numberOfUnions)
    rv2 = histogramGenerator.get_gauss_bins_UNIFIED(mu2, sigma2, numberOfBins, numberOfSamples, interval, numberOfUnions)

    test1 = histogramGenerator.get_gauss_bins(mu1, sigma1, numberOfBins, numberOfSamples, interval)
    test2 = histogramGenerator.get_gauss_bins(mu2, sigma2, numberOfBins, numberOfSamples, interval)

    # max1 = rv1.convolutionOfTwoVarsNaiveSAME_UNIFIED(rv2)
    max1 = test1.convolutionOfTwoVarsShift(test2)
    desired = [max1.mean, max1.std]

    print(max1.bins)

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

    RV1 = RandomVariableCVXPY(x1, rv1.edges)
    RV2 = RandomVariableCVXPY(x2, rv1.edges)

        # GET obj. function and constr

    convolution, constr = RV1.convolution_UNIFIED_NEW_MAX(RV2)
    convolution = convolution.bins

        # FORMULATE

    # objective function
    sum = 0
    for bin in range(0, numberOfBins):
        for union in range(0, numberOfUnions):
            sum += (convolution[bin])[union]

    # other constraints

    for bin in range(0, numberOfBins):
        for union in range(0, numberOfUnions):
            constr.append( (x1[bin])[union] <= rv1.bins[bin, union] )    # set lower constr.

    for bin in range(0, numberOfBins):
        for union in range(0, numberOfUnions):
            constr.append( (x2[bin])[union] <= rv2.bins[bin, union] )    # set lower constr.



        # solve
    objective = cp.Maximize( sum )
    prob = cp.Problem(objective, constr)
    prob.solve(verbose=False, solver=cp.MOSEK)

        # PRINT OUT THE VALUES
    print("Problem value: " + str(prob.value))

    convBins = np.zeros((numberOfBins, numberOfUnions))
    for bin in range(0, numberOfBins):
        for union in range(0, numberOfUnions):
            convBins[bin, union] = (convolution[bin])[union].value

    edges = np.linspace(interval[0], interval[1], numberOfBins + 1)
    convRV = RandomVariable(convBins, edges, unified=True)

    print(convRV.bins)

    actual = [convRV.mean, convRV.std]

    # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

    return None


def test_CVXPY_CONVOLUTION_UNIFIED_MIN(dec: int):
    """
    Problem is formulated as maximization
    """

    mu1 = 22.98553396
    sigma1 = 2.76804456

    mu2 = 18.98483475
    sigma2 = 2.802585

    interval = (-5, 40)

    numberOfSamples = 2000000
    numberOfBins = 15
    numberOfUnions = 5


    # DESIRED

    rv1 = histogramGenerator.get_gauss_bins_UNIFIED(mu1, sigma1, numberOfBins, numberOfSamples, interval, numberOfUnions)
    rv2 = histogramGenerator.get_gauss_bins_UNIFIED(mu2, sigma2, numberOfBins, numberOfSamples, interval, numberOfUnions)

    test1 = histogramGenerator.get_gauss_bins(mu1, sigma1, numberOfBins, numberOfSamples, interval)
    test2 = histogramGenerator.get_gauss_bins(mu2, sigma2, numberOfBins, numberOfSamples, interval)

    # max1 = rv1.convolutionOfTwoVarsNaiveSAME_UNIFIED(rv2)
    max1 = test1.convolutionOfTwoVarsShift(test2)
    desired = [max1.mean, max1.std]

    print(max1.bins)

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

    RV1 = RandomVariableCVXPY(x1, rv1.edges)
    RV2 = RandomVariableCVXPY(x2, rv1.edges)

        # GET obj. function and constr

    convolution, constr = RV1.convolution_UNIFIED_NEW_MIN(RV2)
    convolution = convolution.bins

        # FORMULATE

    # objective function
    sum = 0
    for bin in range(0, numberOfBins):
        for union in range(0, numberOfUnions):
            sum += (convolution[bin])[union]

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
    prob.solve(verbose=False, solver=cp.MOSEK)

        # PRINT OUT THE VALUES
    print("Problem value: " + str(prob.value))

    convBins = np.zeros((numberOfBins, numberOfUnions))
    for bin in range(0, numberOfBins):
        for union in range(0, numberOfUnions):
            convBins[bin, union] = (convolution[bin])[union].value

    edges = np.linspace(interval[0], interval[1], numberOfBins + 1)
    convRV = RandomVariable(convBins, edges, unified=True)

    print(convRV.bins)

    actual = [convRV.mean, convRV.std]

    # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

    return None



if __name__ == "__main__":
        # dec param is Desired precision

    # test_CVXPY_MAX_UNIFIED_OLD(dec=5)
    # test_CVXPY_MAX_UNIFIED_NEW_AS_MAX(dec=5)

    test_CVXPY_CONVOLUTION_UNIFIED_MAX(dec=5)

    print("All tests passed!")