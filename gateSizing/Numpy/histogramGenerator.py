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



