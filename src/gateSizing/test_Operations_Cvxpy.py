import numpy as np

from cvxpyVariable import RandomVariableCVXPY
import cvxpy as cp

from randomVariableHist_Numpy import RandomVariable
import histogramGenerator


def test_CVXPY_MAX_UNARY_OLD(dec: int):

    mu1 = 12.98553396
    sigma1 = 4.76804456

    mu2 = 13.98483475
    sigma2 = 4.802585

    interval = (-10, 50)

    numberOfSamples = 2000000
    numberOfBins = 10
    numberOfUnaries = 10


    # DESIRED

    rv1 = histogramGenerator.get_gauss_bins_UNARY(mu1, sigma1, numberOfBins, numberOfSamples, interval, numberOfUnaries)
    rv2 = histogramGenerator.get_gauss_bins_UNARY(mu2, sigma2, numberOfBins, numberOfSamples, interval, numberOfUnaries)

    test1 = histogramGenerator.get_gauss_bins(mu1, sigma1, numberOfBins, numberOfSamples, interval)
    test2 = histogramGenerator.get_gauss_bins(mu2, sigma2, numberOfBins, numberOfSamples, interval)

    # max1 = rv1.maxOfDistributionsQUAD_FORMULA_UNARY(rv2)
    max1 = test1.maxOfDistributionsFORM(test2)
    desired = [max1.mean, max1.std]


    # ACTUAL

        # init

    x1 = {}

    for bin in range(0, numberOfBins):
        x1[bin] = {}
        for unary in range(0, numberOfUnaries):
            (x1[bin])[unary] = cp.Variable(boolean=True)

    x2 = {}

    for bin in range(0, numberOfBins):
        x2[bin] = {}
        for unary in range(0, numberOfUnaries):
            (x2[bin])[unary] = cp.Variable(boolean=True)


        # GET obj. function and constr

    RV1 = RandomVariableCVXPY(x1, rv1.edges)
    RV2 = RandomVariableCVXPY(x2, rv1.edges)
    maximum, constr = RV1.maximum_QUAD_UNARY_OLD(RV2)
    maximum = maximum.bins

        # FORMULATE

    # objective function
    sum = 0
    for bin in range(0, numberOfBins):
        for unary in range(0, numberOfUnaries):
            sum += (maximum[bin])[unary]

    # other constraints

    for bin in range(0, numberOfBins):
        for unary in range(0, numberOfUnaries):
            constr.append( (x1[bin])[unary] >= rv1.bins[bin, unary] )    # set lower constr.

    for bin in range(0, numberOfBins):
        for unary in range(0, numberOfUnaries):
            constr.append( (x2[bin])[unary] >= rv2.bins[bin, unary] )    # set lower constr.



        # solve
    objective = cp.Minimize( sum )
    prob = cp.Problem(objective, constr)
    prob.solve(verbose=True, solver=cp.MOSEK)

        # PRINT OUT THE VALUES

    print("Problem value: " + str(prob.value))

    maxBins = np.zeros((numberOfBins, numberOfUnaries))
    for bin in range(0, numberOfBins):
        for unary in range(0, numberOfUnaries):
            maxBins[bin, unary] = (maximum[bin])[unary].value

    edges = np.linspace(interval[0], interval[1], numberOfBins + 1)
    maxRV = RandomVariable(maxBins, edges, unary=True)

    print(maxRV.bins)
    actual = [maxRV.mean, maxRV.std]

    # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

    return None

def test_CVXPY_MAX_UNARY_NEW_AS_MAX_FORM(dec: int):

    mu1 = 12.98553396
    sigma1 = 4.76804456

    mu2 = 13.98483475
    sigma2 = 4.802585

    interval = (-10, 50)

    numberOfSamples = 2000000
    numberOfBins = 5
    numberOfUnaries = 5


    # DESIRED

    rv1 = histogramGenerator.get_gauss_bins_UNARY(mu1, sigma1, numberOfBins, numberOfSamples, interval, numberOfUnaries)
    rv2 = histogramGenerator.get_gauss_bins_UNARY(mu2, sigma2, numberOfBins, numberOfSamples, interval, numberOfUnaries)

    test1 = histogramGenerator.get_gauss_bins(mu1, sigma1, numberOfBins, numberOfSamples, interval)
    test2 = histogramGenerator.get_gauss_bins(mu2, sigma2, numberOfBins, numberOfSamples, interval)

    # max1 = rv1.maxOfDistributionsFORM_UNARY(rv2)
    max1 = test1.maxOfDistributionsFORM(test2)
    desired = [max1.mean, max1.std]

    print(desired)

    # ACTUAL

        # init
    x1 = {}

    for bin in range(0, numberOfBins):
        x1[bin] = {}
        for unary in range(0, numberOfUnaries):
            (x1[bin])[unary] = cp.Variable(boolean=True)

    x2 = {}

    for bin in range(0, numberOfBins):
        x2[bin] = {}
        for unary in range(0, numberOfUnaries):
            (x2[bin])[unary] = cp.Variable(boolean=True)


        # GET obj. function and constr

    RV1 = RandomVariableCVXPY(x1, rv1.edges)
    RV2 = RandomVariableCVXPY(x2, rv1.edges)

    maximum, constr = RV1.maximum_FORM_UNARY_NEW_MAX(RV2, precise=False)
    # maximum, constr = RV1.maximum_QUAD_UNARY_NEW_MAX(RV2, precise=False)
    maximum = maximum.bins

        # FORMULATE

    # objective function
    sum = 0
    for bin in range(0, numberOfBins):
        for unary in range(0, numberOfUnaries):
            sum += (maximum[bin])[unary]

    # other constraints

    for bin in range(0, numberOfBins):
        for unary in range(0, numberOfUnaries):
            constr.append( (x1[bin])[unary] <= rv1.bins[bin, unary] )    # set lower constr.

    for bin in range(0, numberOfBins):
        for unary in range(0, numberOfUnaries):
            constr.append( (x2[bin])[unary] <= rv2.bins[bin, unary] )    # set lower constr.

        # solve
    objective = cp.Maximize( sum )
    prob = cp.Problem(objective, constr)
    prob.solve(verbose=True, solver=cp.GUROBI)

        # PRINT OUT THE VALUES

    print("Problem value: " + str(prob.value))


    # x2_S = np.zeros((numberOfBins, numberOfUnaries))
    # for bin in range(0, numberOfBins):
    #     for unary in range(0, numberOfUnaries):
    #         x2_S[bin, unary] = (x2[bin])[unary].value
    #
    #
    # x1_S = np.zeros((numberOfBins, numberOfUnaries))
    # for bin in range(0, numberOfBins):
    #     for unary in range(0, numberOfUnaries):
    #         x1_S[bin, unary] = (x1[bin])[unary].value

    maxBins = np.zeros((numberOfBins, numberOfUnaries))
    for bin in range(0, numberOfBins):
        for unary in range(0, numberOfUnaries):
            maxBins[bin, unary] = (maximum[bin])[unary].value


    edges = np.linspace(interval[0], interval[1], numberOfBins + 1)
    maxRV = RandomVariable(maxBins, edges, unary=True)



    actual = [maxRV.mean, maxRV.std]
    print(actual)

    # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

    return None

def test_CVXPY_MAX_UNARY_NEW_AS_MAX(dec: int):

    mu1 = 12.98553396
    sigma1 = 4.76804456

    mu2 = 25.98483475
    sigma2 = 3.802585

    interval = (-10, 40)

    numberOfSamples = 2000000
    numberOfBins = 10
    numberOfUnaries = 10


    # DESIRED

    rv1 = histogramGenerator.get_gauss_bins_UNARY(mu1, sigma1, numberOfBins, numberOfSamples, interval, numberOfUnaries)
    rv2 = histogramGenerator.get_gauss_bins_UNARY(mu2, sigma2, numberOfBins, numberOfSamples, interval, numberOfUnaries)

    test1 = histogramGenerator.get_gauss_bins(mu1, sigma1, numberOfBins, numberOfSamples, interval)
    test2 = histogramGenerator.get_gauss_bins(mu2, sigma2, numberOfBins, numberOfSamples, interval)

    max1 = rv1.maxOfDistributionsQUAD_FORMULA_UNARY(rv2)
    # max1 = test1.maxOfDistributionsFORM(test2)
    desired = [max1.mean, max1.std]

    print(desired)

    # ACTUAL

        # init
    x1 = {}

    for bin in range(0, numberOfBins):
        x1[bin] = {}
        for unary in range(0, numberOfUnaries):
            (x1[bin])[unary] = cp.Variable(boolean=True)

    x2 = {}

    for bin in range(0, numberOfBins):
        x2[bin] = {}
        for unary in range(0, numberOfUnaries):
            (x2[bin])[unary] = cp.Variable(boolean=True)


        # GET obj. function and constr

    RV1 = RandomVariableCVXPY(x1, rv1.edges)
    RV2 = RandomVariableCVXPY(x2, rv1.edges)
    maximum, constr = RV1.maximum_QUAD_UNARY_NEW_MAX(RV2, precise=False)
    maximum = maximum.bins

        # FORMULATE

    # objective function
    sum = 0
    for bin in range(0, numberOfBins):
        for unary in range(0, numberOfUnaries):
            sum += (maximum[bin])[unary]

    # other constraints

    for bin in range(0, numberOfBins):
        for unary in range(0, numberOfUnaries):
            constr.append( (x1[bin])[unary] <= rv1.bins[bin, unary] )    # set lower constr.

    for bin in range(0, numberOfBins):
        for unary in range(0, numberOfUnaries):
            constr.append( (x2[bin])[unary] <= rv2.bins[bin, unary] )    # set lower constr.



        # solve
    objective = cp.Maximize( sum )
    prob = cp.Problem(objective, constr)
    prob.solve(verbose=False, solver=cp.MOSEK)

        # PRINT OUT THE VALUES

    print("Problem value: " + str(prob.value))


    # x2_S = np.zeros((numberOfBins, numberOfUnaries))
    # for bin in range(0, numberOfBins):
    #     for unary in range(0, numberOfUnaries):
    #         x2_S[bin, unary] = (x2[bin])[unary].value
    #
    #
    # x1_S = np.zeros((numberOfBins, numberOfUnaries))
    # for bin in range(0, numberOfBins):
    #     for unary in range(0, numberOfUnaries):
    #         x1_S[bin, unary] = (x1[bin])[unary].value

    maxBins = np.zeros((numberOfBins, numberOfUnaries))
    for bin in range(0, numberOfBins):
        for unary in range(0, numberOfUnaries):
            maxBins[bin, unary] = (maximum[bin])[unary].value


    edges = np.linspace(interval[0], interval[1], numberOfBins + 1)
    maxRV = RandomVariable(maxBins, edges, unary=True)



    actual = [maxRV.mean, maxRV.std]
    print(actual)

    # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

    return None

def test_CVXPY_MAX_UNARY_NEW_AS_MIN(dec: int):

    mu1 = 12.98553396
    sigma1 = 4.76804456

    mu2 = 13.98483475
    sigma2 = 4.802585

    interval = (-10, 50)

    numberOfSamples = 2000000
    numberOfBins = 10
    numberOfUnaries = 10


    # DESIRED

    rv1 = histogramGenerator.get_gauss_bins_UNARY(mu1, sigma1, numberOfBins, numberOfSamples, interval, numberOfUnaries)
    rv2 = histogramGenerator.get_gauss_bins_UNARY(mu2, sigma2, numberOfBins, numberOfSamples, interval, numberOfUnaries)

    test1 = histogramGenerator.get_gauss_bins(mu1, sigma1, numberOfBins, numberOfSamples, interval)
    test2 = histogramGenerator.get_gauss_bins(mu2, sigma2, numberOfBins, numberOfSamples, interval)

    max1 = rv1.maxOfDistributionsQUAD_FORMULA_UNARY(rv2)
    # max1 = test1.maxOfDistributionsFORM(test2)
    desired = [max1.mean, max1.std]

    print(max1.bins)
    print(desired)

    # ACTUAL

        # init
    x1 = {}

    for bin in range(0, numberOfBins):
        x1[bin] = {}
        for unary in range(0, numberOfUnaries):
            (x1[bin])[unary] = cp.Variable(boolean=True)

    x2 = {}

    for bin in range(0, numberOfBins):
        x2[bin] = {}
        for unary in range(0, numberOfUnaries):
            (x2[bin])[unary] = cp.Variable(boolean=True)


        # GET obj. function and constr

    RV1 = RandomVariableCVXPY(x1, rv1.edges)
    RV2 = RandomVariableCVXPY(x2, rv1.edges)
    maximum, constr = RV1.maximum_QUAD_UNARY_CUT(RV2, asMin=False)
    maximum = maximum.bins

        # FORMULATE

    # objective function
    sum = 0
    for bin in range(0, numberOfBins):
        for unary in range(0, numberOfUnaries):
            sum += (maximum[bin])[unary]

    # other constraints

    for bin in range(0, numberOfBins):
        for unary in range(0, numberOfUnaries):
            constr.append( (x1[bin])[unary] <= rv1.bins[bin, unary] )    # set lower constr.

    for bin in range(0, numberOfBins):
        for unary in range(0, numberOfUnaries):
            constr.append( (x2[bin])[unary] <= rv2.bins[bin, unary] )    # set lower constr.



        # solve
    objective = cp.Maximize( sum )
    prob = cp.Problem(objective, constr)
    prob.solve(verbose=False, solver=cp.MOSEK)

        # PRINT OUT THE VALUES

    print("Problem value: " + str(prob.value))


    # x2_S = np.zeros((numberOfBins, numberOfUnaries))
    # for bin in range(0, numberOfBins):
    #     for unary in range(0, numberOfUnaries):
    #         x2_S[bin, unary] = (x2[bin])[unary].value
    #
    #
    # x1_S = np.zeros((numberOfBins, numberOfUnaries))
    # for bin in range(0, numberOfBins):
    #     for unary in range(0, numberOfUnaries):
    #         x1_S[bin, unary] = (x1[bin])[unary].value

    maxBins = np.zeros((numberOfBins, numberOfUnaries))
    for bin in range(0, numberOfBins):
        for unary in range(0, numberOfUnaries):
            maxBins[bin, unary] = (maximum[bin])[unary].value

    # print(maxBins)

    edges = np.linspace(interval[0], interval[1], numberOfBins + 1)
    maxRV = RandomVariable(maxBins, edges, unary=True)



    actual = [maxRV.mean, maxRV.std]
    print(actual)
    print(maxRV.bins)
    # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

    return None


def test_CVXPY_CONVOLUTION_UNARY_MAX(dec: int):
    """
    Problem is formulated as maximization
    """

    mu1 = 2.98553396
    sigma1 = 2.76804456

    mu2 = 3.98483475
    sigma2 = 1.802585

    interval = (-5, 10)

    numberOfSamples = 2000000
    numberOfBins = 10
    numberOfUnaries = 17


    # DESIRED

    rv1 = histogramGenerator.get_gauss_bins_UNARY(mu1, sigma1, numberOfBins, numberOfSamples, interval, numberOfUnaries)
    rv2 = histogramGenerator.get_gauss_bins_UNARY(mu2, sigma2, numberOfBins, numberOfSamples, interval, numberOfUnaries)

    test1 = histogramGenerator.get_gauss_bins(mu1, sigma1, numberOfBins, numberOfSamples, interval)
    test2 = histogramGenerator.get_gauss_bins(mu2, sigma2, numberOfBins, numberOfSamples, interval)

    max1 = rv1.convolutionOfTwoVarsNaiveSAME_UNARY(rv2)
    # max1 = test1.convolutionOfTwoVarsShift(test2)
    desired = [max1.mean, max1.std]
    #
    print(desired)

    # ACTUAL
        # init
    x1 = {}

    for bin in range(0, numberOfBins):
        x1[bin] = {}
        for unary in range(0, numberOfUnaries):
            (x1[bin])[unary] = cp.Variable(boolean=True)

    x2 = {}

    for bin in range(0, numberOfBins):
        x2[bin] = {}
        for unary in range(0, numberOfUnaries):
            (x2[bin])[unary] = cp.Variable(boolean=True)

    RV1 = RandomVariableCVXPY(x1, rv1.edges)
    RV2 = RandomVariableCVXPY(x2, rv1.edges)

        # GET obj. function and constr

    convolution, constr = RV1.convolution_UNARY_NEW_MAX(RV2)
    convolution = convolution.bins

        # FORMULATE

    # objective function
    sum = 0
    for bin in range(0, numberOfBins):
        for unary in range(0, numberOfUnaries):
            sum += (convolution[bin])[unary]

    # other constraints

    for bin in range(0, numberOfBins):
        for unary in range(0, numberOfUnaries):
            constr.append( (x1[bin])[unary] <= rv1.bins[bin, unary] )    # set lower constr.

    for bin in range(0, numberOfBins):
        for unary in range(0, numberOfUnaries):
            constr.append( (x2[bin])[unary] <= rv2.bins[bin, unary] )    # set lower constr.

    # symmetry constraints
    # for bin in range(0, numberOfBins):
    #     for unary in range(0, numberOfUnaries - 1):
    #         constr.append((x2[bin])[unary] >= (x2[bin])[unary + 1])  # set lower constr.
    #
    # for bin in range(0, numberOfBins):
    #     for unary in range(0, numberOfUnaries - 1):
    #         constr.append((x1[bin])[unary] >= (x1[bin])[unary + 1])  # set lower constr.

    # for bin in range(0, numberOfBins):
    #     for unary in range(0, numberOfUnaries - 1):
    #         constr.append((convolution[bin])[unary] >= (convolution[bin])[unary + 1])  # set lower constr.

        # solve
    objective = cp.Maximize( sum )
    prob = cp.Problem(objective, constr)
    prob.solve(verbose=False, solver=cp.MOSEK)

        # PRINT OUT THE VALUES
    print("Problem value: " + str(prob.value))

    convBins = np.zeros((numberOfBins, numberOfUnaries))
    for bin in range(0, numberOfBins):
        for unary in range(0, numberOfUnaries):
            convBins[bin, unary] = (convolution[bin])[unary].value

    edges = np.linspace(interval[0], interval[1], numberOfBins + 1)
    convRV = RandomVariable(convBins, edges, unary=True)

    # print(convRV.bins)

    actual = [convRV.mean, convRV.std]

    print(actual)

    # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

    return None


def test_CVXPY_CONVOLUTION_UNARY_MIN(dec: int):
    """
    Problem is formulated as maximization
    """

    mu1 = 2.98553396
    sigma1 = 2.76804456

    mu2 = 3.98483475
    sigma2 = 1.802585

    interval = (-5, 10)

    numberOfSamples = 2000000
    numberOfBins = 8
    numberOfUnaries = 8


    # DESIRED

    rv1 = histogramGenerator.get_gauss_bins_UNARY(mu1, sigma1, numberOfBins, numberOfSamples, interval, numberOfUnaries)
    rv2 = histogramGenerator.get_gauss_bins_UNARY(mu2, sigma2, numberOfBins, numberOfSamples, interval, numberOfUnaries)

    test1 = histogramGenerator.get_gauss_bins(mu1, sigma1, numberOfBins, numberOfSamples, interval)
    test2 = histogramGenerator.get_gauss_bins(mu2, sigma2, numberOfBins, numberOfSamples, interval)

    max1 = rv1.maxOfDistributionsQUAD_FORMULA_UNARY(rv2)
    # max1 = test1.convolutionOfTwoVarsShift(test2)
    desired = [max1.mean, max1.std]

    print(desired)
    print(max1.bins)

    # ACTUAL
        # init
    x1 = {}

    for bin in range(0, numberOfBins):
        x1[bin] = {}
        for unary in range(0, numberOfUnaries):
            (x1[bin])[unary] = cp.Variable(boolean=True)

    x2 = {}

    for bin in range(0, numberOfBins):
        x2[bin] = {}
        for unary in range(0, numberOfUnaries):
            (x2[bin])[unary] = cp.Variable(boolean=True)

    RV1 = RandomVariableCVXPY(x1, rv1.edges)
    RV2 = RandomVariableCVXPY(x2, rv1.edges)

        # GET obj. function and constr

    convolution, constr = RV1.maximum_QUAD_UNARY_DIVIDE(RV2, asMin=False)
    convolution = convolution.bins

        # FORMULATE

    # objective function
    sum = 0
    for bin in range(0, numberOfBins):
        for unary in range(0, numberOfUnaries):
            sum += (convolution[bin])[unary]

    # other constraints

    for bin in range(0, numberOfBins):
        for unary in range(0, numberOfUnaries):
            constr.append( (x1[bin])[unary] <= rv1.bins[bin, unary] )    # set lower constr.

    for bin in range(0, numberOfBins):
        for unary in range(0, numberOfUnaries):
            constr.append( (x2[bin])[unary] <= rv2.bins[bin, unary] )    # set lower constr.

        # solve
    objective = cp.Maximize( sum )
    prob = cp.Problem(objective, constr)
    prob.solve(verbose=True, solver=cp.MOSEK)

        # PRINT OUT THE VALUES
    print("Problem value: " + str(prob.value))

    convBins = np.zeros((numberOfBins, numberOfUnaries))
    for bin in range(0, numberOfBins):
        for unary in range(0, numberOfUnaries):
            convBins[bin, unary] = (convolution[bin])[unary].value

    edges = np.linspace(interval[0], interval[1], numberOfBins + 1)
    convRV = RandomVariable(convBins, edges, unary=True)

    print(convRV.bins)

    actual = [convRV.mean, convRV.std]
    print(actual)

    # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

    return None

def test_CVXPY_MAXIMUM_McCormick(dec: int):
    """
    Problem is formulated as minimization
    """

    mu1 = 22.98553396
    sigma1 = 2.76804456

    mu2 = 18.98483475
    sigma2 = 2.802585

    interval = (-5, 40)

    numberOfSamples = 2000000
    numberOfBins = 10


    # DESIRED

    test1 = histogramGenerator.get_gauss_bins(mu1, sigma1, numberOfBins, numberOfSamples, interval)
    test2 = histogramGenerator.get_gauss_bins(mu2, sigma2, numberOfBins, numberOfSamples, interval)

    print(test1.bins)
    print(test2.bins)

    # max1 = rv1.convolutionOfTwoVarsNaiveSAME_UNARY(rv2)
    max1 = test1.maxOfDistributionsFORM(test2)
    desired = [max1.mean, max1.std]

    print(max1.bins)

    # ACTUAL
        # init
    x1 = {}

    for bin in range(0, numberOfBins):
        x1[bin] = cp.Variable(nonneg=True)

    x2 = {}

    for bin in range(0, numberOfBins):
        x2[bin] = cp.Variable(nonneg=True)

    RV1 = RandomVariableCVXPY(x1, test1.edges, test1.bins)
    RV2 = RandomVariableCVXPY(x2, test1.edges, test2.bins)

        # GET obj. function and constr

    maximum, constr = RV1.maximum_McCormick(RV2)
    maximum = maximum.bins

        # FORMULATE

    # objective function
    sum = 0
    for bin in range(0, numberOfBins):
        sum += maximum[bin]

    # other constraints


        # solve
    objective = cp.Minimize( sum )
    prob = cp.Problem(objective, constr)
    prob.solve(verbose=False, solver=cp.MOSEK)

        # PRINT OUT THE VALUES
    print("Problem value: " + str(prob.value))

    convBins = np.zeros(numberOfBins)
    for bin in range(0, numberOfBins):
        convBins[bin] = maximum[bin].value

    edges = np.linspace(interval[0], interval[1], numberOfBins + 1)
    maxRV = RandomVariable(convBins, edges)

    print(maxRV.bins)

    actual = [maxRV.mean, maxRV.std]

    # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

    return None

def test_CVXPY_CONVOLUTION_McCormick(dec: int):
    """
    Problem is formulated as minimization
    """

    mu1 = 22.98553396
    sigma1 = 2.76804456

    mu2 = 18.98483475
    sigma2 = 2.802585

    interval = (-5, 40)

    numberOfSamples = 2000000
    numberOfBins = 10


    # DESIRED

    test1 = histogramGenerator.get_gauss_bins(mu1, sigma1, numberOfBins, numberOfSamples, interval)
    test2 = histogramGenerator.get_gauss_bins(mu2, sigma2, numberOfBins, numberOfSamples, interval)

    max1 = test1.convolutionOfTwoVarsShift(test2)
    desired = [max1.mean, max1.std]

    print(max1.bins)

    # ACTUAL
        # init
    x1 = {}

    for bin in range(0, numberOfBins):
        x1[bin] = cp.Variable(nonneg=True)

    x2 = {}

    for bin in range(0, numberOfBins):
        x2[bin] = cp.Variable(nonneg=True)

    RV1 = RandomVariableCVXPY(x1, test1.edges, test1.bins)
    RV2 = RandomVariableCVXPY(x2, test1.edges, test2.bins)

        # GET obj. function and constr

    maximum, constr = RV1.convolution_McCormick(RV2)
    maximum = maximum.bins

        # FORMULATE

    # objective function
    sum = 0
    for bin in range(0, numberOfBins):
        sum += maximum[bin]

    # other constraints


        # solve
    objective = cp.Minimize( sum )
    prob = cp.Problem(objective, constr)
    prob.solve(verbose=False, solver=cp.MOSEK)

        # PRINT OUT THE VALUES
    print("Problem value: " + str(prob.value))

    convBins = np.zeros(numberOfBins)
    for bin in range(0, numberOfBins):
        convBins[bin] = maximum[bin].value

    edges = np.linspace(interval[0], interval[1], numberOfBins + 1)
    maxRV = RandomVariable(convBins, edges)

    print(maxRV.bins)

    actual = [maxRV.mean, maxRV.std]

    # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

    return None

def test_CVXPY_MULTIPLICATION_McCormick(dec: int):
    """
    Problem is formulated as minimization
    """

    x1 = 0.5
    x2 = 0.5

    desired = x1*x2

    # actual

    constr = []

    x = cp.Variable(nonneg=True)
    y = cp.Variable(nonneg=True)

    slackMult = cp.Variable(nonneg=True)

    # mccormick envelope

    x_L = x1
    x_U = 1
    y_L = x2
    y_U = 1


    # McCormick constraints
    constr.append(slackMult >= x_L * y + x * y_L - x_L * y_L)
    constr.append(slackMult >= x_U * y + x * y_U - x_U * y_U)
    constr.append(slackMult <= x_U * y + x * y_L - x_U * y_L)
    constr.append(slackMult <= x * y_U + x_L * y - x_L * y_U)




    # constr.append(slackMult >= x1 * y + x * x2 - x1 * x2)
    # constr.append(slackMult >= x_U * y + x * y_U - x_U * y_U)
    # constr.append(slackMult <= x_U * y + x * x2 - x_U * x2)
    # constr.append(slackMult <= x * y_U + x1 * y - x1 * y_U)


    constr.append(slackMult <= 1)
    constr.append( x >= 0 )
    constr.append( y >= 0 )
    constr.append( x <= 1 )
    constr.append( y <= 1 )

    # FORMULATE

    # other constraints

    constr.append( x >= x1 )
    constr.append( y >= x2 )

        # solve
    objective = cp.Minimize( slackMult )
    prob = cp.Problem(objective, constr)
    prob.solve(verbose=True, solver=cp.MOSEK)

        # PRINT OUT THE VALUES
    print(x.value)
    print(y.value)


    actual = prob.value

    # TESTING

    np.testing.assert_almost_equal(actual, desired, decimal=dec)

    return None

if __name__ == "__main__":
        # dec param is Desired precision

    # test_CVXPY_MAX_UNARY_OLD(dec=5)
    # test_CVXPY_MAX_UNARY_NEW_AS_MAX(dec=
    # test_CVXPY_MAX_UNARY_NEW_AS_MIN(dec=5)
    test_CVXPY_CONVOLUTION_UNARY_MIN(dec=5)
    # test_CVXPY_CONVOLUTION_UNARY_MAX(dec=5)
    # test_CVXPY_MAXIMUM_McCormick(dec=5)
    # test_CVXPY_MULTIPLICATION_McCormick(5)
    # test_CVXPY_CONVOLUTION_McCormick(5)
    # test_CVXPY_MAX_UNARY_NEW_AS_MAX_FORM(3)

    print("All tests passed!")