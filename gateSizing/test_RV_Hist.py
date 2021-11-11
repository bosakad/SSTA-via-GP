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

def testConvolutionGauss(dec: int):

    mu1 = 0.5
    sigma1 = 0.5

    mu2 = 1.0527713798460994
    sigma2 = 0.3920577124415191

    numberOfSamples = 1000000
    numberOfBins = 1000

        # DESIRED

    rv1 = np.random.normal(mu1, sigma1, numberOfSamples)
    rv2 = np.random.normal(mu2, sigma2, numberOfSamples)

    convolution = rv1 + rv2
    
    desired = [np.mean(convolution), np.std(convolution)]



        # ACTUAL

    STATIC_BINS = np.linspace(-10, 10, numberOfBins)

    # histogram1
    data, edges = np.histogram(rv1, bins=STATIC_BINS)
    dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))
    h1 = RandomVariable(dataNorm, edges)

    # histogram2
    data, edges = np.histogram(rv2, bins=STATIC_BINS)
    dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))
    h2 = RandomVariable(dataNorm, edges)

    convolution = h1.convolutionOfTwoVars2(h2)

    actual = [convolution.mean, convolution.std]

        # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

    return None

def testConvolutionUniform(dec: int):

    low = 5
    high = 8
    numberOfSamples = 100000
    numberOfBins = 500

        # DESIRED

    rv1 = np.random.uniform(low, high, numberOfSamples)
    rv2 = np.random.uniform(low, high, numberOfSamples)

    convolution = rv1 + rv2

    desired = [np.mean(convolution), np.std(convolution)]


        # ACTUAL

    STATIC_BINS = np.linspace(5, 12, numberOfBins)

    data, edges = np.histogram(rv1, bins=STATIC_BINS)
    dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))
    h1 = RandomVariable(dataNorm, edges)

    data, edges = np.histogram(rv2, bins=STATIC_BINS)
    dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))
    h2 = RandomVariable(dataNorm, edges)

    convolution = h1.convolutionOfTwoVars2(h2)

    actual = [convolution.mean, convolution.std]

        # TESTING

    np.testing.assert_almost_equal(desired, actual, decimal=dec)

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

    testConvolutionUniform(dec=2)
    testConvolutionGauss(dec=2)



    print("All tests passed!")