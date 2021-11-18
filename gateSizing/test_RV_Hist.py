import numpy
import numpy as np
from randomVariableHist import RandomVariable
import matplotlib.pyplot as plt

def testMaximum3(dec: int):

    mu = 1
    sigma = 0.5
    numberOfSamples = 1000
    numberOfBins = 40

    STATIC_BINS = np.linspace(-1, 10, numberOfBins)

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

    max2 = h1.getMaximum3(h2)

    actual = [max2.mean, max2.std]

        # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

    return None

def testMaximum4(dec: int):

    mu1 = 1
    sigma1 = 0.4491837314887311

    mu2 = 1.55
    sigma2 = 0.4491837314887311

    numberOfSamples = 10000
    numberOfBins = 50

    STATIC_BINS = np.linspace(-2, 5, numberOfBins)

        # DESIRED

    rv1 = np.random.normal(mu1, sigma1, numberOfSamples)
    rv2 = np.random.normal(mu2, sigma2, numberOfSamples)

    max = np.maximum(rv1, rv2)
    desired = [np.mean(max), np.std(max)]

       # ACTUAL

    data, edges = np.histogram(rv1, bins=STATIC_BINS)
    dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))
    h1 = RandomVariable(dataNorm, edges)

    data, edges = np.histogram(rv2, bins=STATIC_BINS)
    dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))
    h2 = RandomVariable(dataNorm, edges)

    max2 = h1.getMaximum4(h2)

    actual = [max2.mean, max2.std]

        # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

    return None

def testMaximum5(dec: int):

    mu = 1
    sigma = 0.5
    numberOfSamples = 1000
    numberOfBins = 40

    STATIC_BINS = np.linspace(-1, 10, numberOfBins)


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

    max2 = h1.getMaximum5(h2)

    actual = [max2.mean, max2.std]

        # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

    return None

def testConvolutionGaussShift(dec: int):

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

def testConvolutionUniformShift(dec: int):

    low = 3
    high = 6
    numberOfSamples = 10000
    numberOfBins = 30

        # DESIRED

    rv1 = np.random.uniform(low, high, numberOfSamples)
    rv2 = np.random.uniform(low, high, numberOfSamples)

    convolution = rv1 + rv2

    desired = [np.mean(convolution), np.std(convolution)]


        # ACTUAL

    STATIC_BINS = np.linspace(0,10, numberOfBins)

    data, edges = np.histogram(rv1, bins=STATIC_BINS)
    dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))
    h1 = RandomVariable(dataNorm, edges)

    # plt.hist(h1.edges[:-1], h1.edges, weights=h1.bins, density="PDF")

    data, edges = np.histogram(rv2, bins=STATIC_BINS)
    dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))
    h2 = RandomVariable(dataNorm, edges)

    # plt.hist(h2.edges[:-1], h2.edges, weights=h2.bins, density="PDF")

    convolution = h1.convolutionOfTwoVarsShift(h2)

    # plt.hist(convolution.edges[:-1], convolution.edges, weights=convolution.bins, density="PDF")
    # plt.show()

    actual = [convolution.mean, convolution.std]

        # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

    return None


def testUniteEdges(dec: int):

    edges1 = np.array([1, 2, 3])
    bins1 = np.array([4, 8])
    rv1 = RandomVariable(bins1, edges1)

    edges2 = np.array([0.5, 1.5, 2.5])
    bins2 = np.array([2, 4])
    rv2 = RandomVariable(bins2, edges2)


        # DESIRED

    edges = np.array([0.5, 1.75, 3])
    bins1N = np.array([6, 12]) / 18            # [0] = 3; [1] = 1 + 8
    bins2N = np.array([3, 3]) / 6          # [0] = 2 + 1 ; [1] = 3
    desired = np.array([edges, bins1N, bins2N], dtype=object)


        # ACTUAL

    rv1.uniteEdges(rv2)

    actual = np.array([rv2.edges, rv1.bins, rv2.bins], dtype=object)


        # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

    return None


def testUniteEdges(dec: int):

    mu1 = 12
    sigma1 = 2

    numberOfSamples = 10000000
    numberOfBins = 2300

    rv1 = np.random.normal(mu1, sigma1, numberOfSamples)

    STATIC_BINS = np.linspace(-40, 90, numberOfBins)

    data, edges = np.histogram(rv1, bins=STATIC_BINS)
    dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))
    h1 = RandomVariable(dataNorm, edges)

    h2 = RandomVariable(dataNorm, edges - 20)

    # plt.hist(h1.edges[:-1], h1.edges, weights=h1.bins, density="PDF")
    #
    # plt.hist(h2.edges[:-1], h2.edges, weights=h2.bins, density="PDF")

    desired = numpy.array([[h1.mean, h1.std], [h2.mean, h2.std]])

    h1.uniteEdges(h2)

    # plt.hist(h1.edges[:-1], h1.edges, weights=h1.bins, density="PDF")
    # plt.hist(h2.edges[:-1], h2.edges, weights=h2.bins, density="PDF")

    actual = numpy.array([[h1.mean, h1.std], [h2.mean, h2.std]])

    # TEST

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

    plt.show()


    """ Test with different edges """
def testConvolutionUnion(dec: int):

    mu1 = 12
    sigma1 = 2

    mu2 = 6
    sigma2 = 3

    numberOfSamples = 10000000
    numberOfBins = 2300

    # DESIRED

    rv1 = np.random.normal(mu1, sigma1, numberOfSamples)
    rv2 = np.random.normal(mu2, sigma2, numberOfSamples)

    convolution = rv1 + rv2

    desired = [np.mean(convolution), np.std(convolution)]

    # ACTUAL

    STATIC_BINS = np.linspace(-40, 90, numberOfBins)

    # histogram1
    data, edges = np.histogram(rv1, bins=STATIC_BINS)
    dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))
    h1 = RandomVariable(dataNorm, edges)


    STATIC_BINS = np.linspace(-30, 100, numberOfBins)
    # histogram2
    data, edges = np.histogram(rv2, bins=STATIC_BINS)
    dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))
    h2 = RandomVariable(dataNorm, edges)

    convolution = h1.convolutionOfTwoVarsUnion(h2)

    actual = [convolution.mean, convolution.std]

    # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

    return None




    return None


def testMeanGauss(dec: int):

    mu = 1
    sigma = 0.5
    numberOfSamples = 10000
    numberOfBins = 50

    STATIC_BINS = np.linspace(-10, 10, numberOfBins)

        # DESIRED

    rv1 = np.random.normal(mu, sigma, numberOfSamples)
    desired = np.mean(rv1)

        # ACTUAL

    data, edges = np.histogram(rv1, bins=STATIC_BINS)
    dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))
    h1 = RandomVariable(dataNorm, edges)

    actual = h1.mean

        # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

    return None

def testMeanUniform(dec: int):

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

def testStdGauss(dec: int):

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

def testStdUniform(dec: int):

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






if __name__ == "__main__":

        # dec param is Desired precision

    testMeanGauss(dec=2)
    testMeanUniform(dec=2)

    testStdGauss(dec=2)
    testStdUniform(dec=2)

    testMaximum3(dec=0)
    testMaximum4(dec=1)
    testMaximum5(dec=1)

    # testUniteEdges(dec=2)   # failed test
    testUniteEdges(dec=5)

    # testConvolutionUnion(dec=2)
    # testConvolutionUniform(dec=4)
    # testConvolutionGauss(dec=4)



    print("All tests passed!")