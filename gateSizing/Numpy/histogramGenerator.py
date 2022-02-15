import math

import numpy as np
from randomVariableHist import RandomVariable


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

def get_gauss_bins_UNIFIED(mu: float, sigma: float, numberOfBins: int, numberOfSamples: int, binsInterval: tuple,
                                                                    numberOfUnions: int) -> RandomVariable:
    """
    Generates a randomly generated gaussian histogram with given mean and standard deviation.
    Each bin is represented by M 0/1-bins.

    :param mu: mean
    :param sigma: std
    :param numberOfBins: -
    :param numberOfSamples: number of samples used for generating
    :param binsInterval: static bins interval - should be large enough
    :param numberOfUnions: number of representative bins for each bin
    :return randomVar: new RV
    """

    s = np.random.normal(mu, sigma, numberOfSamples)

    STATIC_BINS = np.linspace(binsInterval[0], binsInterval[1], numberOfBins+1)

    data, edges = np.histogram(s, bins=STATIC_BINS)
    dataNorm = np.array(data) / (np.sum(data))

    finalBins = np.zeros((numberOfBins, numberOfUnions))


    for bin in range(0, numberOfBins):

        numberOfOnes = int(math.floor(dataNorm[bin] * numberOfUnions))

        finalBins[bin, :numberOfOnes] = 1

        # for union in range(0, numberOfOnes):  # non-vectorized version
        #     finalBins[bin, union] = 1


    randomVar = RandomVariable(finalBins, edges, unified=True)

    return randomVar




