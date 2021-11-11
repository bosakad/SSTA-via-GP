import numpy as np
from randomVariableHist import RandomVariable


""" Get gaussian bins.
Generates a randomly generated gaussian histogram with given mean and standard deviation.

    Args:
        mu: mean
        sigma: standard deviation
        numberOfEdges: number of bins in histogram
        numberOfSamples: number of samples in histogram
    
    Return:
        histogram: returns histogram 

"""


def get_gauss_bins(mu: float, sigma: float, numberOfBins: int, numberOfSamples: int, binsInterval: tuple) -> RandomVariable:
    s = np.random.normal(mu, sigma, numberOfSamples)

    STATIC_BINS = np.linspace(binsInterval[0], binsInterval[1], numberOfBins)

    data, edges = np.histogram(s, bins=STATIC_BINS)
    dataNorm = np.array(data) / (np.sum(data) * (edges[1] - edges[0]))

    randomVar = RandomVariable(dataNorm, edges)

    return randomVar


