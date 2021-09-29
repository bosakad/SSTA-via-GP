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


def get_gauss_bins(mu, sigma, numberOfEdges, numberOfSamples):
    s = np.random.normal(mu, sigma, numberOfSamples)
    data, edges = np.histogram(s, numberOfEdges)

    randomVar = RandomVariable(data, edges)

    return randomVar



