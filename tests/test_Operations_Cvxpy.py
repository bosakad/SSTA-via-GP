import numpy as np

from src.timing.cvxpyVariable import RandomVariableCVXPY
import cvxpy as cp

from src.timing.randomVariableHist_Numpy import RandomVariable
import src.utility_tools.histogramGenerator as histogramGenerator


def test_CVXPY_MAXIMUM_McCormick(dec=2):
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

    test1 = histogramGenerator.get_gauss_bins(
        mu1, sigma1, numberOfBins, numberOfSamples, interval
    )
    test2 = histogramGenerator.get_gauss_bins(
        mu2, sigma2, numberOfBins, numberOfSamples, interval
    )

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
    objective = cp.Minimize(sum)
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


def test_CVXPY_CONVOLUTION_McCormick(dec=2):
    """
    Problem is formulated as minimization
    """

    mu1 = 22.98553396
    sigma1 = 2.76804456

    mu2 = 18.98483475
    sigma2 = 2.802585

    mu3 = 10.98553396
    sigma3 = 3.76804456

    mu4 = 18.98483475
    sigma4 = 1.802585

    interval = (0, 20)

    numberOfSamples = 2000000
    numberOfBins = 15

    # DESIRED

    test1 = histogramGenerator.get_gauss_bins(
        mu1, sigma1, numberOfBins, numberOfSamples, interval
    )
    test2 = histogramGenerator.get_gauss_bins(
        mu2, sigma2, numberOfBins, numberOfSamples, interval
    )

    test3 = histogramGenerator.get_gauss_bins(
        mu3, sigma3, numberOfBins, numberOfSamples, interval
    )
    test4 = histogramGenerator.get_gauss_bins(
        mu4, sigma4, numberOfBins, numberOfSamples, interval
    )

    max1 = test1.maxOfDistributionsFORM(test2)

    max2 = test3.maxOfDistributionsFORM(test4)

    max1 = max1.maxOfDistributionsFORM(max2)

    desired = [max1.mean, max1.std]

    # print(max1.bins)

    # ACTUAL
    # init
    x1 = {}

    for bin in range(0, numberOfBins):
        x1[bin] = cp.Variable(nonneg=True)

    x2 = {}

    for bin in range(0, numberOfBins):
        x2[bin] = cp.Variable(nonneg=True)

    x3 = {}

    for bin in range(0, numberOfBins):
        x3[bin] = cp.Variable(nonneg=True)

    x4 = {}

    for bin in range(0, numberOfBins):
        x4[bin] = cp.Variable(nonneg=True)

    RV1 = RandomVariableCVXPY(x1, test1.edges, test1.bins)
    RV2 = RandomVariableCVXPY(x2, test1.edges, test2.bins)

    RV3 = RandomVariableCVXPY(x3, test1.edges, test3.bins)
    RV4 = RandomVariableCVXPY(x4, test1.edges, test4.bins)

    # GET obj. function and constr

    maximum, constr1 = RV1.maximum_McCormick(RV2)
    maximum2, constr2 = RV3.maximum_McCormick(RV4)
    # maximum, constr3 = maximum.convolution_McCormick(RV3)

    maximum, constr3 = maximum.maximum_McCormick(maximum2)
    maximum = maximum.bins

    # print(len(constr1))
    # print(len(constr2))
    # constr = constr1 + constr3
    constr = constr1 + constr2 + constr3
    # print(len(constr))

    # FORMULATE

    # objective function
    sum = 0
    for bin in range(0, numberOfBins):
        sum += maximum[bin]

    # other constraints

    # solve
    objective = cp.Minimize(sum)
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


def test_CVXPY_CONVOLUTION_GP(dec=2):
    """
    Problem is formulated as minimization
    """

    mu1 = 22.98553396
    sigma1 = 2.76804456

    mu2 = 10.98483475
    sigma2 = 2.802585

    mu3 = 10.98553396
    sigma3 = 3.76804456

    mu4 = 10.98483475
    sigma4 = 1.802585

    interval = (0, 60)

    numberOfSamples = 2000000
    numberOfBins = 41

    # DESIRED

    test1 = histogramGenerator.get_gauss_bins(
        mu1, sigma1, numberOfBins, numberOfSamples, interval, forGP=True
    )
    test2 = histogramGenerator.get_gauss_bins(
        mu2, sigma2, numberOfBins, numberOfSamples, interval, forGP=True
    )

    test3 = histogramGenerator.get_gauss_bins(
        mu3, sigma3, numberOfBins, numberOfSamples, interval, forGP=True
    )
    test4 = histogramGenerator.get_gauss_bins(
        mu4, sigma4, numberOfBins, numberOfSamples, interval, forGP=True
    )

    max1 = test1.maxOfDistributionsFORM(test2)

    # max1 = test1.maxOfDistributionsFORM(test2)
    max2 = test3.maxOfDistributionsFORM(test4)

    max1 = max1.convolutionOfTwoVarsNaiveSAME(max2)

    desired = [max1.mean, max1.std]

    # print(max1.bins)

    # ACTUAL
    # init
    x1 = {}
    inv1 = {}
    constr = []

    for bin in range(0, numberOfBins):

        x1[bin] = cp.Variable(pos=True)
        inv1[bin] = 1 / x1[bin]
        # if test1.bins[bin] == 0:
        #     test1.bins[bin] += 0.00000000000000001
        constr.append(x1[bin] >= test1.bins[bin])

    x2 = {}
    inv2 = {}

    for bin in range(0, numberOfBins):
        x2[bin] = cp.Variable(pos=True)
        inv2[bin] = 1 / x2[bin]
        # if test2.bins[bin] == 0:
        #     test2.bins[bin] += 0.000000000000000001
        constr.append(x2[bin] >= test2.bins[bin])

    x3 = {}

    for bin in range(0, numberOfBins):

        x3[bin] = cp.Variable(pos=True)

        # if test3.bins[bin] == 0:
        #     test3.bins[bin] += 0.000000000000000001
        constr.append(x3[bin] >= test3.bins[bin])

    x4 = {}

    for bin in range(0, numberOfBins):
        x4[bin] = cp.Variable(pos=True)

        # if test4.bins[bin] == 0:
        #     test4.bins[bin] += 0.000000000000000001
        constr.append(x4[bin] >= test4.bins[bin])

    RV1 = RandomVariableCVXPY(x1, test1.edges)
    RV2 = RandomVariableCVXPY(x2, test1.edges)

    RV3 = RandomVariableCVXPY(x3, test1.edges)
    RV4 = RandomVariableCVXPY(x4, test1.edges)

    # GET obj. function and constr

    # maximum, constr1 = RV1.maximum_GP(RV2)

    print(len(constr))
    maximum, constr = RV1.maximum_GP_OPT(RV2, constr)
    # maximum = RV1.maximum_GP(RV2)
    print(len(constr))

    maximum2, constr = RV3.maximum_GP_OPT(RV4, constr)
    # maximum2 = RV3.maximum_GP(RV4)
    maximum, constr = maximum.convolution_GP_OPT(maximum2, constr)
    # maximum, constr = maximum.convolution_GP(maximum2, constr)
    # maximum2, constr2 = RV3.maximum_McCormick(RV4)
    # maximum, constr3 = maximum.convolution_McCormick(RV3)

    # maximum, constr3 = maximum.maximum_McCormick(maximum2)
    maximum = maximum.bins

    # print(len(constr1))
    # print(len(constr2))
    # constr = constr1 + constr3
    # constr = constr1 + constr2 + constr3
    # print(len(constr))

    # FORMULATE

    # objective function
    sum = 0
    for bin in range(0, numberOfBins):
        sum += maximum[bin]

    # other constraints

    # solve
    objective = cp.Minimize(sum)
    prob = cp.Problem(objective, constr)
    prob.solve(
        gp=True,
        verbose=True,
        mosek_params={
            "MSK_DPAR_INTPNT_CO_TOL_MU_RED": 0.1,
            "MSK_DPAR_OPTIMIZER_MAX_TIME": 1200,
        },  # max time
    )

    # PRINT OUT THE VALUES
    print("Problem value: " + str(prob.value))

    print(np.sum(max1.bins))

    convBins = np.zeros(numberOfBins)
    for bin in range(0, numberOfBins):
        convBins[bin] = maximum[bin].value

    edges = np.linspace(interval[0], interval[1], numberOfBins + 1)
    maxRV = RandomVariable(convBins, edges)

    actual = [maxRV.mean, maxRV.std]

    # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

    return None


def test_CVXPY_MULTIPLICATION_McCormick(dec=2):
    """
    Problem is formulated as minimization
    """

    x1 = 0.5
    x2 = 0.5

    desired = x1 * x2

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
    constr.append(x >= 0)
    constr.append(y >= 0)
    constr.append(x <= 1)
    constr.append(y <= 1)

    # FORMULATE

    # other constraints

    constr.append(x >= x1)
    constr.append(y >= x2)

    # solve
    objective = cp.Minimize(slackMult)
    prob = cp.Problem(objective, constr)
    prob.solve(verbose=True, solver=cp.MOSEK)

    # PRINT OUT THE VALUES
    print(x.value)
    print(y.value)

    actual = prob.value

    # TESTING

    np.testing.assert_almost_equal(actual, desired, decimal=dec)

    return None


def test_CVXPY_MULTIPLICATION_GP(dec=2):
    """
    Problem is formulated as minimization
    """

    x1 = 0.00000001
    x2 = 0.00000001

    desired = x1 * x2

    # actual

    constr = []

    x = cp.Variable(pos=True)
    y = cp.Variable(pos=True)

    constr.append(x >= x1)
    constr.append(y >= x2)

    # solve
    objective = cp.Minimize(x * y)
    prob = cp.Problem(objective, constr)
    prob.solve(gp=True, verbose=True, solver=cp.GUROBI)

    # PRINT OUT THE VALUES
    print(x.value)
    print(y.value)

    actual = prob.value
    print(actual)

    # TESTING

    np.testing.assert_almost_equal(actual, desired, decimal=dec)

    return None
