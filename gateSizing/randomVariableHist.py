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
        self.std = self.calculateSTD()

    # def __init__(self, bins, edges, numbers=None):
    #     self.bins = np.array(bins)
    #     self.edges = np.array(edges)
    #     self.numbers = numbers
    #     self.mean = np.mean(numbers)
    #     self.std = np.std(numbers)


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

    def getMaximum2(self, secondVariable):


        max = np.maximum(self.numbers, secondVariable.numbers)

        data, edges = np.histogram(max, self.edges)

        maxDelay = RandomVariable(data, self.edges, max)
        return maxDelay


    def getMaximum3(self, secondVariable):

        maxBins = np.maximum(self.bins, secondVariable.bins)

        maxDelay = RandomVariable(maxBins, self.edges)
        return maxDelay


    def getMaximum4(self, secondVariable):

        n = len(self.bins)

        f1 = self.bins
        f2 = secondVariable.bins

        max = np.array([0.]* n)

        for i in range(0, n):
            F2 = np.sum(f2[:i+1])
            F1 = np.sum(f1[:i])     # only for discrete - not to count with 1 number twice

            max[i] = f1[i] * F2 + f2[i] * F1

        maxDelay = RandomVariable(max, self.edges)
        return maxDelay




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
        newHistogram = np.array([0.] * size)

        for z in range(0, size):
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

    def calculateSTD(self):
        midPoints = 0.5 * (self.edges[1:] + self.edges[:-1])    # midpoints of the edges of hist.
        variance = np.average((midPoints - self.mean) ** 2, weights=self.bins)
        return np.sqrt(variance)









