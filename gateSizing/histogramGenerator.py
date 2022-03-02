import math

import numpy as np
from randomVariableHist_Numpy import RandomVariable


def get_gauss_bins(mu: float, sigma: float, numberOfBins: int, numberOfSamples: int, binsInterval: tuple) -> RandomVariable:
    """
    Generates a randomly generated gaussian histogram with given mean and standard deviation.

    :param mu: mean
    :param sigma: std
    :param numberOfBins: -
    :param numberOfSamples: number of samples used for generating
    :param binsInterval: static bins interval - should be large enough
    :return randomVar: new RV
    """

    s = np.random.normal(mu, sigma, numberOfSamples)

    STATIC_BINS = np.linspace(binsInterval[0], binsInterval[1], numberOfBins+1)

    data, edges = np.histogram(s, bins=STATIC_BINS)
    dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))

    randomVar = RandomVariable(dataNorm, edges)

    return randomVar

def get_gauss_bins_UNARY(mu: float, sigma: float, numberOfBins: int, numberOfSamples: int, binsInterval: tuple,
                         numberOfUnaries: int) -> RandomVariable:
    """
    Generates a randomly generated gaussian histogram with given mean and standard deviation.
    Each bin is represented by M 0/1-bins.

    :param mu: mean
    :param sigma: std
    :param numberOfBins: -
    :param numberOfSamples: number of samples used for generating
    :param binsInterval: static bins interval - should be large enough
    :param numberOfUnaries: number of representative bins for each bin
    :return randomVar: new RV
    """

    s = np.random.normal(mu, sigma, numberOfSamples)

    STATIC_BINS = np.linspace(binsInterval[0], binsInterval[1], numberOfBins+1)

    data, edges = np.histogram(s, bins=STATIC_BINS)
    dataNorm = np.array(data) / (np.sum(data))

    # print(dataNorm)

    finalBins = np.zeros((numberOfBins, numberOfUnaries))


    for bin in range(0, numberOfBins):

        numberOfOnes = math.floor(dataNorm[bin] * numberOfUnaries)

        finalBins[bin, :numberOfOnes] = 1

        # for unary in range(0, numberOfOnes):  # non-vectorized version
        #     finalBins[bin, unary] = 1


    randomVar = RandomVariable(finalBins, edges, unary=True)

    return randomVar




