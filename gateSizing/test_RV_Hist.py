import numpy as np
from randomVariableHist import RandomVariable

def testMaximum3(dec: int):

    mu = 1
    sigma = 0.5
    numberOfSamples = 1000
    numberOfEdges = 40

    STATIC_BINS = np.arange(-1, 10, 12 / numberOfEdges)

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

    mu = 1
    sigma = 0.5
    numberOfSamples = 1000
    numberOfEdges = 40

    STATIC_BINS = np.arange(-1, 10, 12 / numberOfEdges)

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

    max2 = h1.getMaximum4(h2)

    actual = [max2.mean, max2.std]

        # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

    return None

def testMaximum5(dec: int):

    mu = 1
    sigma = 0.5
    numberOfSamples = 1000
    numberOfEdges = 40

    STATIC_BINS = np.arange(-1, 10, 12 / numberOfEdges)

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

def testConvolution(dec: int):

    mu = 0.5
    sigma = 0.3
    numberOfSamples = 1000
    numberOfBins = 40

        # DESIRED

    rv1 = np.random.normal(mu, sigma, numberOfSamples)
    rv2 = np.random.normal(mu, sigma, numberOfSamples)

    convolution = rv1 + rv2

    desired = [np.mean(convolution), np.std(convolution)]


        # ACTUAL

    minVal = min(min(rv1), min(rv2))
    maxVal = max(max(rv1), max(rv2))

    STATIC_BINS = np.linspace(minVal, maxVal, numberOfBins)

    STATIC_BINS = np.linspace(0, 5, 100)

    data, edges = np.histogram(rv1, bins=STATIC_BINS)
    dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))
    h1 = RandomVariable(dataNorm, edges)


    data, edges = np.histogram(rv2, bins=STATIC_BINS)
    dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))
    h2 = RandomVariable(dataNorm, edges)

    convolution2 = h1.convolutionOfTwoVars2(h2)

    actual = [convolution2.mean, convolution2.std]

        # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

    return None



def testMean(dec: int):

    mu = 1
    sigma = 0.5
    numberOfSamples = 1000
    numberOfEdges = 40

    STATIC_BINS = np.arange(-1, 10, 12 / numberOfEdges)

        # DESIRED

    rv1 = np.random.normal(mu, sigma, numberOfSamples)
    desired = np.mean(rv1)

        # ACTUAL

    data, edges = np.histogram(rv1, bins=STATIC_BINS)
    dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))
    h1= RandomVariable(dataNorm, edges)

    actual = h1.mean

        # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

    return None

def testStd(dec: int):

    mu = 1
    sigma = 0.5
    numberOfSamples = 1000
    numberOfEdges = 40

    STATIC_BINS = np.arange(-1, 10, 12 / numberOfEdges)

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



if __name__ == "__main__":

        # dec param is Desired precision

    testMean(dec=2)
    testStd(dec=2)

    testMaximum3(dec=0)
    testMaximum4(dec=1)
    testMaximum5(dec=2)

    testConvolution(dec=3)

    print("All tests passed!")