import numpy
import numpy as np
import histogramGenerator
from randomVariableHist_Numpy import RandomVariable

# import matplotlib.pyplot as plt


def maxOfDistributionsELEMENTWISE(dec=1):

    mu = 10
    sigma = 0.5
    numberOfSamples = 1000000
    numberOfBins = 1000

    STATIC_BINS = np.linspace(-20, 40, numberOfBins)

    # DESIRED

    rv1 = np.random.normal(mu, sigma, numberOfSamples)
    rv2 = np.random.normal(mu, sigma, numberOfSamples)

    max = np.maximum(rv1, rv2)
    desired = [np.mean(max), np.std(max)]

    # ACTUAL

    data, edges = np.histogram(rv1, bins=STATIC_BINS)
    dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))
    h1 = RandomVariable(dataNorm, edges)

    data, edges = np.histogram(rv2, bins=STATIC_BINS)
    dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))
    h2 = RandomVariable(dataNorm, edges)

    max2 = h1.maxOfDistributionsELEMENTWISE(h2)

    actual = [max2.mean, max2.std]

    # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

    return None


def maxOfDistributionsFORM_MultiMax(dec=1):

    mu1 = 25
    sigma1 = 0.5

    mu2 = 29
    sigma2 = 0.5

    mu3 = 34
    sigma3 = 5

    mu4 = 54
    sigma4 = 20

    mu5 = 100
    sigma5 = 0.9

    numberOfSamples = 2000000
    numberOfBins = 2000

    STATIC_BINS = np.linspace(40, 250, numberOfBins)

    # DESIRED

    rv1 = np.random.normal(mu1, sigma1, numberOfSamples)
    rv2 = np.random.normal(mu2, sigma2, numberOfSamples)
    rv3 = np.random.normal(mu3, sigma3, numberOfSamples)
    rv4 = np.random.normal(mu4, sigma4, numberOfSamples)
    rv5 = np.random.normal(mu5, sigma5, numberOfSamples)

    desired = np.zeros((4, 2))

    max = rv1 + rv2
    max2 = rv3 + max
    max3 = rv4 + max2
    max4 = np.maximum(rv5, max3)

    # desired[0, :] = [np.mean(max), np.std(max)]
    # desired[1, :] = [np.mean(max2), np.std(max2)]
    # desired[2, :] = [np.mean(max3), np.std(max3)]
    desired[3, :] = [np.mean(max4), np.std(max4)]

    # ACTUAL
    #
    # data, edges = np.histogram(rv1, bins=STATIC_BINS)
    # dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))
    # h1 = RandomVariable(dataNorm, edges)

    # data, edges = np.histogram(rv2, bins=STATIC_BINS)
    # dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))
    # h2 = RandomVariable(dataNorm, edges)
    #
    # data, edges = np.histogram(rv3, bins=STATIC_BINS)
    # dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))
    # h3 = RandomVariable(dataNorm, edges)
    #
    # data, edges = np.histogram(rv4, bins=STATIC_BINS)
    # dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))
    # h4 = RandomVariable(dataNorm, edges)
    #
    # data, edges = np.histogram(rv5, bins=STATIC_BINS)
    # dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))
    # h5 = RandomVariable(dataNorm, edges)

    data, edges = np.histogram(max3, bins=STATIC_BINS)
    dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))
    h5 = RandomVariable(dataNorm, edges)

    data, edges = np.histogram(rv5, bins=STATIC_BINS)
    dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))
    max23 = RandomVariable(dataNorm, edges)

    # max21 = h1.convolutionOfTwoVarsUnion(h2)
    # max22 = max21.convolutionOfTwoVarsUnion(h3)
    # max23 = max22.convolutionOfTwoVarsUnion(h4)
    max24 = max23.maxOfDistributionsFORM(h5)

    # plt.hist(h1.edges[:-1], h1.edges, weights=h1.bins, density="PDF")
    # plt.hist(h2.edges[:-1], h2.edges, weights=h2.bins, density="PDF")
    # plt.hist(max21.edges[:-1], max21.edges, weights=max21.bins, density="PDF")
    # plt.hist(h3.edges[:-1], h3.edges, weights=h3.bins, density="PDF")
    # plt.hist(max22.edges[:-1], max22.edges, weights=max22.bins, density="PDF")

    # plt.show()

    actual = np.zeros((4, 2))
    # actual[0, :] = [max21.mean, max21.std]
    # actual[1, :] = [max22.mean, max22.std]
    # actual[2, :] = [max23.mean, max23.std]
    actual[3, :] = [max24.mean, max24.std]

    # TESTING

    np.testing.assert_almost_equal(
        desired,
        actual,
        decimal=dec,
        err_msg="Monte Carlo: \n" + str(desired) + "\n\n" + "SSTA: \n" + str(actual),
    )

    return None


def testConvolutionGaussShift(dec=1):

    mu1 = 12
    sigma1 = 2

    mu2 = 6
    sigma2 = 3

    numberOfSamples = 20000000
    numberOfBins = 10000

    # DESIRED

    rv1 = np.random.normal(mu1, sigma1, numberOfSamples)
    rv2 = np.random.normal(mu2, sigma2, numberOfSamples)

    convolution = rv1 + rv2

    desired = [np.mean(convolution), np.std(convolution)]

    # ACTUAL

    STATIC_BINS = np.linspace(-50, 90, numberOfBins)

    # histogram1
    data, edges = np.histogram(rv1, bins=STATIC_BINS)
    dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))
    h1 = RandomVariable(dataNorm, edges)

    # histogram2
    data, edges = np.histogram(rv2, bins=STATIC_BINS)
    dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))
    h2 = RandomVariable(dataNorm, edges)

    convolution = h1.convolutionOfTwoVarsShift(h2)

    actual = [convolution.mean, convolution.std]

    # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

    return None


def testMAX_UNARY(dec=1):
    mu1 = 12
    sigma1 = 2

    mu2 = 6
    sigma2 = 3

    numberOfSamples = 20000000
    numberOfBins = 100
    numberOfUnaries = 100

    interval = (-5, 20)

    # DESIRED

    rv1 = np.random.normal(mu1, sigma1, numberOfSamples)
    rv2 = np.random.normal(mu2, sigma2, numberOfSamples)

    max1 = np.maximum(rv1, rv2)
    desired = [np.mean(max1), np.std(max1)]
    print(desired)

    h3 = histogramGenerator.get_gauss_bins(
        mu1, sigma1, numberOfBins, numberOfSamples, interval
    )
    h4 = histogramGenerator.get_gauss_bins(
        mu2, sigma2, numberOfBins, numberOfSamples, interval
    )
    max1 = h3.maxOfDistributionsFORM(h4)

    desired = [max1.mean, max1.std]
    print(desired)

    # ACTUAL

    # histogram1
    h1 = histogramGenerator.get_gauss_bins_UNARY(
        mu1, sigma1, numberOfBins, numberOfSamples, interval, numberOfUnaries
    )
    h2 = histogramGenerator.get_gauss_bins_UNARY(
        mu2, sigma2, numberOfBins, numberOfSamples, interval, numberOfUnaries
    )

    # max2 = h3.maxOfDistributionsQUAD_FORMULA(h4)
    # max2 = h3.maxOfDistributionsQUAD(h4)
    max2 = h1.maxOfDistributionsQUAD_FORMULA_UNARY(h2)

    actual = [max2.mean, max2.std]
    print(actual)

    # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

    return None


def testUniteEdgesNormal(dec=1):

    mu1 = 12
    sigma1 = 2

    numberOfSamples = 10000000
    numberOfBins = 1000

    rv1 = np.random.normal(mu1, sigma1, numberOfSamples)

    STATIC_BINS = np.linspace(-40, 90, numberOfBins)

    data, edges = np.histogram(rv1, bins=STATIC_BINS)
    dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))
    h1 = RandomVariable(dataNorm, edges)

    h2 = RandomVariable(dataNorm, edges - 20)

    desired = numpy.array([[h1.mean, h1.std], [h2.mean, h2.std]])

    h1.uniteEdges(h2)

    actual = numpy.array([[h1.mean, h1.std], [h2.mean, h2.std]])

    # TEST

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

    """ Test with different edges """


def testUniteEdgesStudent(dec=1):

    chi = 6

    numberOfSamples = 1000000
    numberOfBins = 1000

    # DESIRED

    rv1 = np.random.standard_t(chi, numberOfSamples)

    STATIC_BINS = np.linspace(-40, 90, numberOfBins)

    data, edges = np.histogram(rv1, bins=STATIC_BINS)
    dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))
    h1 = RandomVariable(dataNorm, edges)

    h2 = RandomVariable(dataNorm, edges - 20)

    desired = numpy.array([[h1.mean, h1.std], [h2.mean, h2.std]])

    # plt.hist(h1.edges[:-1], h1.edges, weights=h1.bins, density="PDF")
    # plt.hist(h2.edges[:-1], h2.edges, weights=h2.bins, density="PDF")
    # plt.show()

    # ACTUAL

    h1.uniteEdges(h2)

    actual = numpy.array([[h1.mean, h1.std], [h2.mean, h2.std]])

    # TEST

    # plt.hist(h1.edges[:-1], h1.edges, weights=h1.bins, density="PDF")
    # plt.hist(h2.edges[:-1], h2.edges, weights=h2.bins, density="PDF")
    # plt.show()

    np.testing.assert_almost_equal(desired, actual, decimal=dec)


def testMaxAndConvolution(dec=1):
    mu1 = 40
    sigma1 = 0.5

    mu2 = 22
    sigma2 = 3

    mu3 = 14
    sigma3 = 0.5

    numberOfSamples = 2000000
    numberOfBins = 2000

    # DESIRED

    rv1 = np.random.normal(mu1, sigma1, numberOfSamples)
    rv2 = np.random.normal(mu2, sigma2, numberOfSamples)
    rv3 = np.random.normal(mu3, sigma3, numberOfSamples)

    maximum = np.maximum(rv1, rv2)
    convolution = maximum + rv3

    desired = np.zeros((2, 2))
    desired[0, :] = [np.mean(maximum), np.std(maximum)]
    desired[1, :] = [np.mean(convolution), np.std(convolution)]

    # ACTUAL

    STATIC_BINS = np.linspace(-10, 80, numberOfBins)

    # histogram1
    data, edges = np.histogram(rv1, bins=STATIC_BINS)
    dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))
    h1 = RandomVariable(dataNorm, edges)

    # histogram2
    data, edges = np.histogram(rv2, bins=STATIC_BINS)
    dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))
    h2 = RandomVariable(dataNorm, edges)

    data, edges = np.histogram(rv3, bins=STATIC_BINS)
    dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))
    h3 = RandomVariable(dataNorm, edges)

    maximum2 = h1.maxOfDistributionsFORM(h2)
    convolution2 = maximum2.convolutionOfTwoVarsShift(h3)

    actual = np.zeros((2, 2))
    actual[0, :] = [maximum2.mean, maximum2.std]
    actual[1, :] = [convolution2.mean, convolution2.std]

    # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

    return None


def testMeanGauss(dec=1):

    mu = 1
    sigma = 0.5
    numberOfSamples = 2000000
    numberOfBins = 50
    numberOfUnaries = numberOfBins**2

    interval = (-10, 10)

    # DESIRED

    rv1 = np.random.normal(mu, sigma, numberOfSamples)
    desired = np.std(rv1)
    # desired = np.mean(rv1)

    # ACTUAL

    h1 = histogramGenerator.get_gauss_bins_UNARY(
        mu, sigma, numberOfBins, numberOfSamples, interval, numberOfUnaries
    )

    actual = h1.std
    # actual = h1.mean

    print(desired)
    print(actual)
    # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

    return None


def testMeanUniform(dec=1):

    low = 2
    high = 10
    numberOfSamples = 1000
    numberOfBins = 100

    STATIC_BINS = np.linspace(low, high, numberOfBins)

    # DESIRED

    rv1 = np.random.uniform(low, high, numberOfSamples)
    desired = np.mean(rv1)

    # ACTUAL

    data, edges = np.histogram(rv1, bins=STATIC_BINS)
    dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))
    h1 = RandomVariable(dataNorm, edges)

    actual = h1.mean

    # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

    return None


def testStdGauss(dec=1):

    mu = 1
    sigma = 0.5
    numberOfSamples = 10000
    numberOfBins = 150

    STATIC_BINS = np.linspace(-4, 6, numberOfBins)

    # DESIRED

    rv1 = np.random.normal(mu, sigma, numberOfSamples)
    desired = np.std(rv1)

    # ACTUAL

    data, edges = np.histogram(rv1, bins=STATIC_BINS)
    dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))
    h1 = RandomVariable(dataNorm, edges)

    actual = h1.std

    # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

    return None


def testStdUniform(dec=1):

    low = -2
    high = 8
    numberOfSamples = 10000
    numberOfBins = 100

    STATIC_BINS = np.linspace(low, high, numberOfBins)

    # DESIRED

    rv1 = np.random.uniform(low, high, numberOfSamples)
    desired = np.std(rv1)

    # ACTUAL

    data, edges = np.histogram(rv1, bins=STATIC_BINS)
    dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))
    h1 = RandomVariable(dataNorm, edges)

    actual = h1.std

    # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

    return None


def testmaxOfDistributionsQUAD(dec=1):
    mu1 = 21.98553396
    sigma1 = 0.76804456

    mu2 = 21.98483475
    sigma2 = 0.76802585

    numberOfSamples = 1000000
    numberOfBins = 2000

    STATIC_BINS1 = np.linspace(-20, 100, numberOfBins)
    STATIC_BINS2 = np.linspace(-20, 100, numberOfBins)

    # DESIRED

    rv1 = np.random.normal(mu1, sigma1, numberOfSamples)
    rv2 = np.random.normal(mu2, sigma2, numberOfSamples)

    max = np.maximum(rv1, rv2)
    desired = [np.mean(max), np.std(max)]

    # ACTUAL

    data, edges = np.histogram(rv1, bins=STATIC_BINS1)
    dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))
    h1 = RandomVariable(dataNorm, edges)

    data, edges = np.histogram(rv2, bins=STATIC_BINS2)
    dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))
    h2 = RandomVariable(dataNorm, edges)

    max2 = h1.maxOfDistributionsQUAD(h2)

    actual = [max2.mean, max2.std]

    # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

    return None


def testMAX_QUAD_FORMULA(dec=1):
    mu1 = 12.98553396
    sigma1 = 0.76804456

    mu2 = 26.98483475
    sigma2 = 4.802585

    numberOfSamples = 1000000
    numberOfBins = 2000

    STATIC_BINS1 = np.linspace(-20, 100, numberOfBins)
    STATIC_BINS2 = np.linspace(-20, 100, numberOfBins)

    # DESIRED

    rv1 = np.random.normal(mu1, sigma1, numberOfSamples)
    rv2 = np.random.normal(mu2, sigma2, numberOfSamples)

    max = np.maximum(rv1, rv2)
    desired = [np.mean(max), np.std(max)]

    # ACTUAL

    data, edges = np.histogram(rv1, bins=STATIC_BINS1)
    dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))
    h1 = RandomVariable(dataNorm, edges)

    data, edges = np.histogram(rv2, bins=STATIC_BINS2)
    dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))
    h2 = RandomVariable(dataNorm, edges)

    max2 = h1.maxOfDistributionsQUAD_FORMULA(h2)

    actual = [max2.mean, max2.std]

    # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

    return None


def testMAX_QUAD_FORMULA_UNARY(dec=0):

    mu1 = 12.98553396
    sigma1 = 4.76804456

    mu2 = 13.984834752020
    sigma2 = 4.802585

    interval = (-10, 50)

    numberOfSamples = 2000000
    numberOfBins = 1000
    numberOfUnaries = 1000

    # DESIRED

    rv1 = histogramGenerator.get_gauss_bins(
        mu1, sigma1, numberOfBins, numberOfSamples, interval
    )
    rv2 = histogramGenerator.get_gauss_bins(
        mu2, sigma2, numberOfBins, numberOfSamples, interval
    )

    max1 = rv1.maxOfDistributionsFORM(rv2)
    desired = [max1.mean, max1.std]

    # ACTUAL

    rv3 = histogramGenerator.get_gauss_bins_UNARY(
        mu1, sigma1, numberOfBins, numberOfSamples, interval, numberOfUnaries
    )
    rv4 = histogramGenerator.get_gauss_bins_UNARY(
        mu2, sigma2, numberOfBins, numberOfSamples, interval, numberOfUnaries
    )

    max2 = rv3.maxOfDistributionsQUAD_FORMULA_UNARY(rv4)
    actual = [max2.mean, max2.std]

    # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

    return None


def testMAX_QUAD_FORMULA_UNARY2(dec=0):

    mu1 = 12.98553396
    sigma1 = 4.76804456

    mu2 = 13.98483475
    sigma2 = 4.802585

    mu3 = 20.98483475
    sigma3 = 0.802585

    interval = (-10, 50)

    numberOfSamples = 2000000
    numberOfBins = 200
    numberOfUnaries = 600

    # DESIRED

    rv1 = histogramGenerator.get_gauss_bins(
        mu1, sigma1, numberOfBins, numberOfSamples, interval
    )
    rv2 = histogramGenerator.get_gauss_bins(
        mu2, sigma2, numberOfBins, numberOfSamples, interval
    )
    rv3 = histogramGenerator.get_gauss_bins(
        mu3, sigma3, numberOfBins, numberOfSamples, interval
    )

    max1 = rv1.maxOfDistributionsFORM(rv2)
    max1 = max1.maxOfDistributionsFORM(rv3)
    desired = [max1.mean, max1.std]

    # ACTUAL

    rv4 = histogramGenerator.get_gauss_bins_UNARY(
        mu1, sigma1, numberOfBins, numberOfSamples, interval, numberOfUnaries
    )
    rv5 = histogramGenerator.get_gauss_bins_UNARY(
        mu2, sigma2, numberOfBins, numberOfSamples, interval, numberOfUnaries
    )
    rv6 = histogramGenerator.get_gauss_bins_UNARY(
        mu3, sigma3, numberOfBins, numberOfSamples, interval, numberOfUnaries
    )

    max2 = rv4.maxOfDistributionsQUAD_FORMULA_UNARY(rv5)
    max2 = rv5.maxOfDistributionsQUAD_FORMULA_UNARY(rv6)
    actual = [max2.mean, max2.std]

    # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

    return None


def test_Convolution_UNARY(dec=0):

    mu1 = 22.98553396
    sigma1 = 2.76804456

    mu2 = 22.98483475
    sigma2 = 2.802585

    interval = (-2, 50)

    numberOfSamples = 2000000
    numberOfBins = 100
    numberOfUnaries = 100

    # DESIRED

    rv1 = histogramGenerator.get_gauss_bins(
        mu1, sigma1, numberOfBins, numberOfSamples, interval
    )
    rv2 = histogramGenerator.get_gauss_bins(
        mu2, sigma2, numberOfBins, numberOfSamples, interval
    )

    con1 = rv1.convolutionOfTwoVarsShift(rv2)
    desired = [con1.mean, con1.std]

    # ACTUAL

    rv3 = histogramGenerator.get_gauss_bins_UNARY(
        mu1, sigma1, numberOfBins, numberOfSamples, interval, numberOfUnaries
    )
    rv4 = histogramGenerator.get_gauss_bins_UNARY(
        mu2, sigma2, numberOfBins, numberOfSamples, interval, numberOfUnaries
    )

    con2 = rv3.convolutionOfTwoVarsNaiveSAME_UNARY(rv4)
    actual = [con2.mean, con2.std]

    # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

    return None
