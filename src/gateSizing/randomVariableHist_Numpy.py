import math

import numpy as np
import scipy.stats


class RandomVariable:
    """
    Class representing a random variable given by histogram.
    Class includes:

    Class includes:
        bins: len n of frequencies
        edges: len n+1 of histogram edges
        mean: computed sample mean
        variance: computed sample variance
    """


    def __init__(self, bins, edges, unary=False):


        if (unary == True):
            self.bins = np.array(bins)
            self.edges = np.array(edges)
            self.mean = self.calculateMean_UNARY()
            self.std = self.calculateSTD_UNARY()

        else:   # normal histogram
            self.bins = np.array(bins)
            self.edges = np.array(edges)
            self.mean = self.calculateMean()
            self.std = self.calculateSTD()



    def recalculateParams(self):
        """
        Recalculates parameters mean and std after their change
        :return:
        """
        self.mean = self.calculateMean()
        self.std = self.calculateSTD()



    def maxOfDistributionsELEMENTWISE(self, secondVariable):
        """
        Maximum of 2 distribution functions using elementwise max

        :param self: random variable class
        :param secondVariable: random variable class
        :return maxDelay: random variable class, elementwise maximum of 2 histograms
        """

        self.uniteEdges(secondVariable)

        # normalize
        self.bins = self.bins / (np.sum(self.bins) * (self.edges[1] - self.edges[0]))

        # normalize
        secondVariable.bins = secondVariable.bins / (np.sum(secondVariable.bins) * (self.edges[1] - self.edges[0]))

        maxBins = np.maximum(self.bins, secondVariable.bins)


        # normalize
        maxBins = maxBins / (np.sum(maxBins) * (self.edges[1] - self.edges[0]))

        maxDelay = RandomVariable(maxBins, self.edges)
        return maxDelay


    def maxOfDistributionsFORM(self, secondVariable):
        """
        Maximum of 2 distribution functions using formula

        :param self: random variable class
        :param secondVariable: random variable class
        :return maxDelay: random variable class, maximum of 2 histograms
        """


        # unite
        self.uniteEdges(secondVariable)
        f1 = self.bins
        f2 = secondVariable.bins


        n = self.bins.shape[0]

        f1 = f1 / (np.sum(f1) * (self.edges[1] - self.edges[0]))
        f2 = f2 / (np.sum(f2) * (self.edges[1] - self.edges[0]))
        maximum = np.zeros(n)
        for i in range(0, n):
            F2 = np.sum(f2[:i+1])
            F1 = np.sum(f1[:i])                 # only for discrete - not to count with 1 number twice
            maximum[i] = f1[i] * F2 + f2[i] * F1

        # normalize
        maximum = maximum / (np.sum(maximum) * (self.edges[1] - self.edges[0]))

        # unite
        # self.uniteEdges(secondVariable)
        # f1 = self.bins
        # f2 = secondVariable.bins
        #
        # # vectorized code from above
        # F2 = np.cumsum(f2)
        # F1 = np.cumsum(f1)
        #
        # F1[1:] = F1[:-1]
        # F1[0] = 0
        # maximum = np.multiply(f1, F2) + np.multiply(f2, F1)

        maxDelay = RandomVariable(maximum, self.edges)
        return maxDelay

    def maxOfDistributionsQUAD(self, secondVariable):
        """
        Maximum of 2 distribution functions using quadratic algorithm

        :param self: random variable class
        :param secondVariable: random variable class
        :return maxDelay: random variable class, maximum of 2 histograms
        """

        # unite
        self.uniteEdges(secondVariable)

        # get data
        n = len(self.bins)


        bins1 = self.bins
        edges1 = self.edges
        midPoints1 = 0.5 * (edges1[1:] + edges1[:-1])    # midpoints of the edges of hist.

        bins2 = secondVariable.bins
        edges2 = secondVariable.edges
        midPoints2 = 0.5 * (edges2[1:] + edges2[:-1])  # midpoints of the edges of hist.


        # prealloc
        maximum = np.zeros(n, dtype=np.double)

        # calc. maximum
        for i in range(0, n):
            for j in range(0, n):
                e1 = midPoints1[i]
                e2 = midPoints2[j]

                if e1 >= e2:
                    maximum[i] += bins1[i] * bins2[j]
                elif e1 < e2:
                    maximum[j] += bins1[i] * bins2[j]

        # normalize
        maximum = maximum / (np.sum(maximum) * (self.edges[1] - self.edges[0]))

        return RandomVariable(maximum, self.edges)


    def maxOfDistributionsQUAD_FORMULA(self, secondVariable):
        """
        Maximum of 2 distribution functions using quadratic algorithm and simplified version - in formula

        :param self: random variable class
        :param secondVariable: random variable class
        :return maxDelay: random variable class, maximum of 2 histograms
        """

        # unite
        self.uniteEdges(secondVariable)

        # get data
        n = len(self.bins)

        bins1 = self.bins
        bins2 = secondVariable.bins


        # prealloc
        maximum = np.zeros(n, dtype=np.double)

        for i in range(0, n):
            for j in range(0, i + 1):

                maximum[i] += bins1[i] * bins2[j]

                if i != j:
                    maximum[i] += bins1[j] * bins2[i]



        # for i in range(0, n):
        #     for j in range(0, i + 1):
        #
        #         maximum[i] += bins1[j] * bins2[i]

        # calc. maximum
        # for i in range(0, n):
        #     for j in range(0, i + 1):
        #
        #         maximum[i] += bins1[i] * bins2[j]
        #
        # for i in range(0, n):
        #     for j in range(i + 1, n):
        #
        #         maximum[j] += bins1[i] * bins2[j]


        # normalize
        maximum = maximum / (np.sum(maximum) * (self.edges[1] - self.edges[0]))

        return RandomVariable(maximum, self.edges)

    def maxOfDistributionsQUAD_FORMULA_UNARY(self, secondVariable):
        """
        Maximum of 2 distribution functions using quadratic algorithm and simplified version - in formula
        with unary bins - meaning each bin is represented by M 0/1-bins. M and number of bins is same for all variables
        Uses division to get into unary form .

        :param self: random variable class
        :param secondVariable: random variable class
        :return maxDelay: random variable class, maximum of 2 histograms
        """

        binMatrix1 = self.bins
        binMatrix2 = secondVariable.bins

        numberOfBins = binMatrix1.shape[0]
        numberOfUnaries = binMatrix1.shape[1]

        # prealloc
        maximum = np.zeros((numberOfBins, numberOfUnaries))

        # calc. maximum

        # simple, non-vectorized
        # for i in range(0, numberOfBins):
        #     for j in range(0, i + 1):
        #
        #         for unary in range(0, numberOfUnaries):
        #             for unary2 in range(0, numberOfUnaries):
        #                 maximum[i, unary] += binMatrix1[i, unary] * binMatrix2[j, unary2]
        #
        #                 if i != j:
        #                     maximum[i, unary] += binMatrix1[j, unary] * binMatrix2[i, unary2]


            # vectorized
        for i in range(0, numberOfBins):
            for j in range(0, i + 1):

                maximum[i, :] += binMatrix1[i, :] * np.sum(binMatrix2[j, :])

                if i != j:
                    maximum[i, :] += binMatrix1[j, :] * np.sum(binMatrix2[i, :])


        maximum = self.unarizeDivide(maximum, convolution=False)

        return RandomVariable(maximum, self.edges, unary=True)

    def maximum_AND_Convolution_UNARY(self, secondVariable, thirdVariable):
        """
        Maximum AND Convolution of two independent random variables naively - using 2 for loops:
            (f*g)(z) = sum{k=-inf, inf} ( f(k)g(z-k)  )
        with unary bins - meaning each bin is represented by M 0/1-bins. M and number of bins is same for all variables
        Uses division to get into unary form.

        :param self: random variable class
        :param secondVariable: random variable class
        :param thirdVariable: random variable class
        :return convolution: random variable class of convolution
        """

        binMatrix1 = self.bins
        binMatrix2 = secondVariable.bins

        numberOfBins = binMatrix1.shape[0]
        numberOfUnaries = binMatrix1.shape[1]

        # prealloc
        maximum = np.zeros((numberOfBins, numberOfUnaries))

        # vectorized
        for i in range(0, numberOfBins):
            for j in range(0, i + 1):

                maximum[i, :] += binMatrix1[i, :] * np.sum(binMatrix2[j, :])

                if i != j:
                    maximum[i, :] += binMatrix1[j, :] * np.sum(binMatrix2[i, :])



        # create unary matrices

        maximum = np.sum(maximum, axis=1).astype(int)
        maxNumber = int(np.max(maximum))

        newMaximum = np.zeros((numberOfBins, maxNumber)).astype(int)
        for i in range(0, numberOfBins):
            newMaximum[i, :maximum[i]] = 1

        # perform convolution
        f = thirdVariable.bins

        convolution = np.zeros((numberOfBins, numberOfUnaries))
        for z in range(0, numberOfBins):
            for k in range(0, z + 1):

                convolution[z, :] += f[k, :] * np.sum(newMaximum [z - k, :])

        # print('maxconv')
        # Deal With Edges
        self.cutBins(self.edges, convolution)

        # unarize
        convolution = self.unarizeDivide(convolution, convolution=True, TRI=True)

        return RandomVariable(convolution, self.edges, unary=True)

    def convolutionOfTwoVarsNaiveSAME_UNARY(self, secondVariable):
        """
        'SAME' Convolution of two independent random variables naively - using 2 for loops:
            (f*g)(z) = sum{k=-inf, inf} ( f(k)g(z-k)  )
        with unary bins - meaning each bin is represented by M 0/1-bins. M and number of bins is same for all variables
        Uses division to get into unary form.

        :param self: random variable class
        :param secondVariable: random variable class
        :return convolution: random variable class of convolution
        """

        f = self.bins
        g = secondVariable.bins

        numberOfBins = f.shape[0]
        numberOfUnaries = f.shape[1]

        convolution = np.zeros((numberOfBins, numberOfUnaries))

            # unvectorized
        # for z in range(0, numberOfBins):
        #     for k in range(0, z + 1):
        #
        #         for unary in range(0, numberOfUnaries):
        #             for unary2 in range(0, numberOfUnaries):
        #                 convolution[z, unary] += f[k, unary] * g [z - k, unary2]


            # vectorized
        for z in range(0, numberOfBins):
            for k in range(0, z + 1):

                convolution[z, :] += f[k, :] * np.sum(g [z - k, :])


        # Deal With Edges
        self.cutBins(self.edges, convolution)

            # unarize
        convolution = self.unarizeDivide(convolution, convolution=True)


        return RandomVariable(convolution, self.edges, unary=True)

    def convolutionOfTwoVarsNaiveFULL(self, secondVariable):
        """
        'Full' Convolution of two independent random variables naively - using 2 for loops:
            (f*g)(z) = sum{k=-inf, inf} ( f(k)g(z-k)  )

        :param self: random variable class
        :param secondVariable: random variable class
        :return convolution: random variable class of convolution
        """

        f = self.bins
        g = secondVariable.bins

        diff = self.edges[1] - self.edges[0]

        N = len(f)
        M = len(g)

        finalSize = N + M - 1
        convolution = np.array([0.] * finalSize)

        for z in range(0, finalSize):
            for k in range(0, z + 1):
                if k >= N:
                    convolution[z] += 0
                elif z - k >= M:
                    convolution[z] += 0
                else:
                    convolution[z] += f[k] * g[z - k]

        convolution = convolution[:f.size]    # get the wanted range


        # Deal With Edges
        self.cutBins(self.edges, convolution)

        # normalize
        convolution = convolution / (np.sum(convolution) * diff)

        return RandomVariable(convolution, self.edges)

    def convolutionOfTwoVarsNaiveSAME(self, secondVariable):
        """
        'SAME' Convolution of two independent random variables naively - using 2 for loops:
            (f*g)(z) = sum{k=-inf, inf} ( f(k)g(z-k)  )

        :param self: random variable class
        :param secondVariable: random variable class
        :return convolution: random variable class of convolution
        """

        f = self.bins
        g = secondVariable.bins

        finalSize = len(f)
        convolution = np.array([0.] * finalSize)

        for z in range(0, finalSize):
            for k in range(0, z + 1):
                convolution[z] += f[k] * g[z - k]

        # Deal With Edges
        self.cutBins(self.edges, convolution)

        return RandomVariable(convolution, self.edges)


    def convolutionOfTwoVarsShift(self, secondVariable):
        """
        Convolution of two independent random variables numpy and shift afterwards.
            (f*g)(z) = sum{k=-inf, inf} ( f(k)g(z-k)  )

        :param self: random variable class
        :param secondVariable: random variable class
        :return convolution: random variable class of convolution
        """

        f = self.bins
        g = secondVariable.bins

        diff = self.edges[1] - self.edges[0]

        # Convolve
        convolution = np.convolve(f, g, mode="full")[:f.size]   # get the same range

        # Deal With Edges
        self.cutBins(self.edges, convolution)

        # normalize
        convolution = convolution / (np.sum(convolution) * diff)

        return RandomVariable(convolution, self.edges)


    def convolutionOfTwoVarsUnion(self, secondVariable):
        """
        Convolution of two independent random variables numpy and union of edges afterwards:
            (f*g)(z) = sum{k=-inf, inf} ( f(k)g(z-k)  )

        :param self: random variable class
        :param secondVariable: random variable class
        :return convolution: random variable class of convolution
        """


        # Unite Edges
        self.uniteEdges(secondVariable)

        f = self.bins
        g = secondVariable.bins
        diff = self.edges[1] - self.edges[0]

        # Convolve
        convolution = np.convolve(f, g, mode="full")[:f.size]  # get the same range

        # Deal With Edges
        edges = self.edges + self.edges[0]    # shift made by indexing from 0

        convolution = convolution / (np.sum(convolution) * diff)

        return RandomVariable(convolution, edges)


    def uniteEdges(self, secondVariable):
        """
        Makes a union of two histograms.
        Edges are considered to have the same difference and same length.

        :param self: random variable class
        :param secondVariable: random variable class
        :returns None
        """


        edges1 = self.edges
        bins1 = self.bins
        edges2 = secondVariable.edges
        bins2 = secondVariable.bins

        numberOfBins = bins1.size

        if edges1[0] == edges2[0]:    # edges are same, no union needed
            return None


            # get lower and upper bounds
        minE = min(edges1[0], edges2[0])
        maxE = max(edges1[-1], edges2[-1])

            # create new edges
        edgesN = np.linspace(minE, maxE, numberOfBins + 1)
        self.edges = edgesN
        secondVariable.edges = edgesN

            # create new values
        cdf1 = scipy.stats.rv_histogram((bins1, edges1)).cdf
        cdf2 = scipy.stats.rv_histogram((bins2, edges2)).cdf
        pdf1 = scipy.stats.rv_histogram((bins1, edges1)).pdf
        pdf2 = scipy.stats.rv_histogram((bins2, edges2)).pdf

        # for i in range(0, numberOfBins):
        #     value1 = (cdf1(edgesN[i+1]) - cdf1(edgesN[i]))
        #     value2 = (cdf2(edgesN[i+1]) - cdf2(edgesN[i]))
        #     # value1 = pdf1((edgesN[i+1] + edgesN[i]) / 2)
        #     # value2 = pdf2((edgesN[i+1] + edgesN[i]) / 2)
        #
        #     bins1[i] = value1
        #     bins2[i] = value2for i in range(0, numberOfBins):


        eIndex = 1
        e1 = 1
        diff1 = edges1[1] - edges1[0]
        e2 = 1
        diff2 = edges2[1] - edges2[0]

        bins1New = np.zeros(numberOfBins)
        bins2New = np.zeros(numberOfBins)
        i = 0
        # exact computation
        while i != numberOfBins:

            if edgesN[eIndex + i] > edges1[e1]:
                bins1New[i] += bins1[e1-1]*(edges1[e1] - edges1[e1 - 1])
                e1 += 1

            elif edgesN[eIndex + i] <= edges1[e1]:
                # bins1New[i] += bins1[e1-1]* (edgesN[eIndex + i] - edges1[e1 - 1])
                # e1 += 1
                i += 1


        i = 0
        while i != numberOfBins:
            if edgesN[eIndex + i] > edges2[e2]:
                bins2New[i] += bins2[e2-1]*(edges2[e2] - edges2[e2 - 1])
                e2 += 1

            elif edgesN[eIndex + i] <= edges2[e2]:
                # bins2New[i] += bins2[e2-1]* (edgesN[eIndex + i] - edges2[e2 - 1])
                # e2 += 1
                i += 1

                # end
            if e2 >= numberOfBins + 1:
                bins2New[i] += bins2[e2 - 2] * (edgesN[eIndex + i] - edges2[e2 - 1])
                break

        bins1 = bins1New
        bins2 = bins2New



        # set new bins
        self.bins = bins1
        secondVariable.bins = bins2

        self.recalculateParams()
        secondVariable.recalculateParams()

        return None


    @staticmethod
    def unarizeDivide(bins, convolution, TRI=False):
        """
            Make a non unarized histogram in the unary form. This is done using the divison of the numbers.
        """

        numberOfBins, numberOfUnaries = bins.shape

        # find divider
        # old version - does not work with cvxpy
        sum = np.sum(bins, axis=1)
        maximum = np.max(sum)
        norm =  maximum / numberOfUnaries
        # print(norm)
        # bins = bins / norm    # non-unarized version
        # print(convolution)
        # print(norm)


            # technique with maximum possible
        if not TRI and convolution:
            divisor = numberOfUnaries*numberOfBins / 30

        elif TRI and convolution:
            # divisor = (numberOfUnaries*numberOfBins) / 2.4
            # divisor = norm
            # divisor = 61
            divisor = 50
            # print('norm')
            # print(norm)
            # divisor = 239

        else:   # maximum
            divisor = numberOfBins*numberOfUnaries / 22
            # divisor = 1
        bins = bins / divisor



        # doableSum = np.ceil(np.sum(bins, axis=1)).astype(int)
        doableSum = np.sum(bins, axis=1)

        newBins = np.zeros((numberOfBins, numberOfUnaries))

        for bin in range(0, numberOfBins):
            # newBins[bin, :int(round(doableSum[bin]))] = 1
            newBins[bin, :round(doableSum[bin])] = 1

        return newBins


    @staticmethod
    def unarizeCut(bins):
        """
            Make a non unarized histogram in the unary form. This is done using cutting the additional information.
            Please see 'unarizeDivide' for a better alternative.
        """


        numberOfBins, numberOfUnaries = bins.shape
        newBins = np.zeros((numberOfBins, numberOfUnaries))

        for bin in range(0, numberOfBins):

            sum = int(np.sum(bins[bin, :]))

            if sum >= numberOfUnaries:
                newBins[bin, :] = 1
            else:
                newBins[bin, :sum] = 1


        return newBins

    @staticmethod
    def uniteEdgesNaive(edges1, edges2, convolution, g):
        """
        Makes a union of two histograms, Naive implementation
        Edges are considered to have the same difference and same length.

        :param self: random variable class
        :param secondVariable: random variable class
        :returns None
        """


        # pick the largest edges
        if edges2.size > edges1.size:
            edges1 = edges2

            # append new edges
        diff = edges1[1] - edges1[0]
        appended = []

        if len(edges1) != len(convolution) + 1:
            end = edges1[-1]
            appended = np.linspace(end + diff, end + diff * (len(g) - 1), len(g) - 1)

        edges = np.append(edges1, appended)

        return edges

    @staticmethod
    def cutBins_UNARY_Vectorized(edges, bins):
        """
        Cuts bins depending on edge[0]
        if edge[0] < 0: cuts left bins and adds zeros to the end
        if edge[0] > 0: cuts right bins and adds zeros to the beginning
        Works for only unary random variable

        :param edges: (1, n+1) numpy array of edges
        :param bins: (n, m) numpy array of bins
        :returns None
        """

        diff = edges[1] - edges[0]

        if edges[0] > 0:  # cut bins

            numberOfBinsNeeded = math.floor(abs(edges[0]) / diff)
            bins[numberOfBinsNeeded:, :] = bins[:-numberOfBinsNeeded, :]
            bins[:numberOfBinsNeeded, :] = 0

        elif edges[0] < 0:  # cut bins

            numberOfBinsNeeded = math.floor(abs(edges[0]) / diff)
            bins[:-numberOfBinsNeeded, :] = bins[numberOfBinsNeeded:, :]
            bins[-numberOfBinsNeeded:, :] = 0

    @staticmethod
    def cutBins_UNARY(edges, bins):
        """
        Cuts bins depending on edge[0]
        if edge[0] < 0: cuts left bins and adds zeros to the end
        if edge[0] > 0: cuts right bins and adds zeros to the beginning
        Works for only unary random variable

        :param edges: (1, n+1) numpy array of edges
        :param bins: (n, m) numpy array of bins
        :returns None
        """

        diff = edges[1] - edges[0]
        numberOfUnaries = bins.shape[1]
        numberOfBins = bins.shape[0]

        numberOfBinsNeeded = math.floor(abs(edges[0]) / diff)

        if edges[0] > 0:  # cut bins

            for i in range(numberOfBinsNeeded, numberOfBins):
                for unary in range(0, numberOfUnaries):
                    bins[i, unary] = bins[i - numberOfBinsNeeded, unary]

            for i in range(0, numberOfBinsNeeded):
                for unary in range(0, numberOfUnaries):
                    bins[i, unary] = 0


        elif edges[0] < 0:  # cut bins

            for i in range(numberOfBinsNeeded, numberOfBins):
                for unary in range(0, numberOfUnaries):
                    bins[i - numberOfBinsNeeded, unary] = bins[i, unary]

            for i in range(numberOfBins - numberOfBinsNeeded, numberOfBins):
                for unary in range(0, numberOfUnaries):
                    bins[i, unary] = 0

    @staticmethod
    def cutBins(edges, bins):
        """
        Cuts bins depending on edge[0]
        if edge[0] < 0: cuts left bins and adds zeros to the end
        if edge[0] > 0: cuts right bins and adds zeros to the beginning

        :param edges: (1, n+1) numpy array of edges
        :param bins: (1, n) numpy array of bins
        :returns None
        """

        diff = edges[1] - edges[0]

        if edges[0] > 0:  # cut bins

            numberOfBinsNeeded = math.floor(abs(edges[0]) / diff)
            bins[numberOfBinsNeeded:] = bins[:-numberOfBinsNeeded]
            bins[:numberOfBinsNeeded] = 0

        elif edges[0] < 0:  # cut bins

            numberOfBinsNeeded = math.floor(abs(edges[0]) / diff)
            bins[:-numberOfBinsNeeded] = bins[numberOfBinsNeeded:]
            bins[-numberOfBinsNeeded:] = 0


    def calculateMean(self):
        """
        Function calculates sample mean of the random variable.
        Calculation: weighted average of the frequencies, edges being the weights

        :param self: random variable class
        :returns mean: double value
        """

        midPoints = 0.5 * (self.edges[1:] + self.edges[:-1])    # midpoints of the edges of hist.
        mean = np.average(midPoints, weights=self.bins)
        return mean


    def calculateSTD(self):
        """
        Function calculates sample std of the random variable.
        Calculation: weighted average of the frequencies, edges being the weights

        :param self: random variable class
        :returns std: double value
        """

        midPoints = 0.5 * (self.edges[1:] + self.edges[:-1])    # midpoints of the edges of hist.
        variance = np.average(np.square(midPoints - self.mean), weights=self.bins)
        return np.sqrt(variance)

    def calculateMean_UNARY(self):
        """
        Function calculates sample mean of the random variable - represented as unary 0/1 bins
        Calculation: weighted average of the frequencies, edges being the weights

        :param self: random variable class
        :returns mean: double value
        """

        binMatrix = self.bins
        numberOfBins = binMatrix.shape[0]

        estimatedBins = np.zeros(numberOfBins)
        norm = np.sum(binMatrix) * (self.edges[1] - self.edges[0])

            # calculate prob. of each bin
        for bin in range(0, numberOfBins):
            numberOfOnes = np.sum( binMatrix[bin, :] )
            estimatedBins[bin] = numberOfOnes / norm

        midPoints = 0.5 * (self.edges[1:] + self.edges[:-1])  # midpoints of the edges of hist.
        mean = np.average(midPoints, weights=estimatedBins)

        return mean


    def calculateSTD_UNARY(self):
        """
        Function calculates sample std of the random variable - represented as unary 0/1 bins
        Calculation: weighted average of the frequencies, edges being the weights

        :param self: random variable class
        :returns std: double value
        """

        binMatrix = self.bins
        numberOfBins = binMatrix.shape[0]

        estimatedBins = np.zeros(numberOfBins)

        norm = np.sum(binMatrix) * (self.edges[1] - self.edges[0])

        # calculate prob. of each bin
        for bin in range(0, numberOfBins):
            numberOfOnes = np.sum(binMatrix[bin, :])
            estimatedBins[bin] = numberOfOnes / norm

        midPoints = 0.5 * (self.edges[1:] + self.edges[:-1])    # midpoints of the edges of hist.
        variance = np.average(np.square(midPoints - self.mean), weights=estimatedBins)
        return np.sqrt(variance)









