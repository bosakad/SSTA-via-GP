import math

import numpy as np
from randomVariableHist_Numpy import RandomVariable


def get_gauss_bins(mu: float, sigma: float, numberOfBins: int, numberOfSamples: int, binsInterval: tuple
                                                                            , distr="Gauss") -> RandomVariable:
    """
    Generates a randomly generated gaussian histogram with given mean and standard deviation.

    :param mu: mean
    :param sigma: std
    :param numberOfBins: -
    :param numberOfSamples: number of samples used for generating
    :param binsInterval: static bins interval - should be large enough
    :param distr: string - "Gauss" / "LogNormal"
    :return randomVar: new RV
    """

    if distr == "Gauss":
        s = np.random.normal(mu, sigma, numberOfSamples)
    elif distr == "LogNormal":
        s = np.random.lognormal(mu, sigma, numberOfSamples)


    STATIC_BINS = np.linspace(binsInterval[0], binsInterval[1], numberOfBins+1)

    data, edges = np.histogram(s, bins=STATIC_BINS)
    dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))

    randomVar = RandomVariable(dataNorm, edges)

    return randomVar

def get_gauss_bins_UNARY(mu: float, sigma: float, numberOfBins: int, numberOfSamples: int, binsInterval: tuple,
                         numberOfUnaries: int, distr="Gauss") -> RandomVariable:
    """
    Generates a randomly generated gaussian histogram with given mean and standard deviation.
    Each bin is represented by M 0/1-bins.

    :param mu: mean
    :param sigma: std
    :param numberOfBins: -
    :param numberOfSamples: number of samples used for generating
    :param binsInterval: static bins interval - should be large enough
    :param numberOfUnaries: number of representative bins for each bin
    :param distr: string - "Gauss" / "LogNormal"
    :return randomVar: new RV
    """

    if distr == "Gauss":
        s = np.random.normal(mu, sigma, numberOfSamples)
    elif distr == "LogNormal":
        s = np.random.lognormal(mu, sigma, numberOfSamples)

    STATIC_BINS = np.linspace(binsInterval[0], binsInterval[1], numberOfBins+1)

    data, edges = np.histogram(s, bins=STATIC_BINS)
    dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]) )


        #testing
    # import matplotlib.pyplot as plt
    # plt.hist(edges[:-1], edges, weights=dataNorm)

    finalBins = np.zeros((numberOfBins, numberOfUnaries))


    for bin in range(0, numberOfBins):

        numberOfOnes = int(round(dataNorm[bin] * numberOfUnaries))
        # numberOfOnes = math.floor(round(dataNorm[bin] * numberOfUnaries))

        finalBins[bin, :numberOfOnes] = 1

        # for unary in range(0, numberOfOnes):  # non-vectorized version
        #     finalBins[bin, unary] = 1

    randomVar = RandomVariable(finalBins, edges, unary=True)

        # testing
    # rv = get_Histogram_from_UNARY(randomVar)
    # plt.hist(rv.edges[:-1], rv.edges, weights=rv.bins)

    return randomVar


def get_Histogram_from_UNARY(unaryHist):
    """
    :param unaryHist: unary encoded histogram, dtype = RV
    :return result: normal bin approximation derived from unary encoded histogram
    """


    numberOfBins, numberOfUnaries = unaryHist.bins.shape

    norm = np.sum(unaryHist.bins) * (unaryHist.edges[1] - unaryHist.edges[0])

    resultBins = np.zeros(numberOfBins)

    for bin in range(0, numberOfBins):
        resultBins[bin] = np.sum(unaryHist.bins[bin, :]) / norm

    result = RandomVariable(resultBins, unaryHist.edges)

    return result


