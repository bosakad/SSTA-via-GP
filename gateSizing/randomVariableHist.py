import cvxpy as cp
import numpy as np


""" Random variable 

    Class representing a random variable given by histogram.
    Class includes: 
        bins: len n of frequencies
        edges: len n+1 of histogram edges
        mean: computed sample mean
        variance: computed sample variance

"""

class RandomVariable:

    def __init__(self, bins, edges):
        self.bins = np.array(bins)
        self.edges = np.array(edges)
        self.mean = self.calculateMean()
        self.variance = self.calculateVariance()


    """ Maximum of 2 distribution functions
    
    Input:
        secondHistogram: class Histogram with bins and edges data
    
    Output: 
        max_x { intervalEnds[x] | x in [0, N-1]; max_i { frequencies[i][x]} > 0 }
    
    """

    def getMaximum(self, secondVariable):
        frequencies = [self.bins, secondVariable.bins]
        intervalEnds = self.edges[1:]

        F = np.matrix(frequencies)
        E = np.matrix(intervalEnds)

        L = F > 0  # logical matrix

        B = len(frequencies[0])
        frequency = cp.Variable((1, B))

        objective = cp.Minimize(cp.max(cp.multiply(frequency, E)))
        constraints = [frequency >= L[0], frequency >= L[1]]

        prob = cp.Problem(objective, constraints)
        prob.solve()

        # sum columns of 2 given histograms
        newHistogram = np.sum(frequencies, axis=0)
        norm = np.linalg.norm(newHistogram)
        newHistogram = newHistogram / norm

        # norm check
        if (np.sum(newHistogram) <= 1):
            print("Wrong data!")
            return -1

        maxDelay = RandomVariable(newHistogram, self.edges  )
        return prob.value, maxDelay


    """ Convolution of two independent random variables
    
    Input:
        frequencies: 2xB array, where B is number of bins of given a histogram
    
    Output:
    
    (f*g)(z) = sum{k=-inf, inf} ( f(k)g(z-k)  )
    
    """

    def convolutionOfTwoVars(self, secondVariable):
        f = self.bins
        g = secondVariable.bins

        size = len(f)
        newHistogram = []

        for z in range(0, size):
            newHistogram.append(0)
            for k in range(0, z + 1):
                newHistogram[z] += f[k] * g[z - k]

        return RandomVariable(newHistogram, self.edges)


    """ Calculate mean
    
    Function calculates sample mean of the random variable.
    Calculation: weighted average of the frequencies, edges being the weights    
        
    """
    def calculateMean(self):
        midPoints = 0.5 * (self.edges[1:] + self.edges[:-1])    # midpoints of the edges of hist.
        mean = np.average(midPoints, weights=self.bins)
        return mean


    """ Calculate variance

        Function calculates sample variance of the random variable.
        Calculation: weighted average of the frequencies, edges being the weights    

        """

    def calculateVariance(self):
        midPoints = 0.5 * (self.edges[1:] + self.edges[:-1])    # midpoints of the edges of hist.
        variance = np.average((midPoints - self.mean) ** 2, weights=self.bins)
        return variance









