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


def get_gauss_bins(mu: float, sigma: float, numberOfEdges: int, numberOfSamples: int) -> RandomVariable:
    s = np.random.normal(mu, sigma, numberOfSamples)

    STATIC_BINS = np.arange(-1, 10, 12 / numberOfEdges)

    data, edges = np.histogram(s, bins=STATIC_BINS)
    dataNorm = np.array(data) / np.sum(data)

    randomVar = RandomVariable(dataNorm, edges)

    return randomVar



