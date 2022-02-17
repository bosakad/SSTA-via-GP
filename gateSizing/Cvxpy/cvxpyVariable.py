import cvxpy as cp
import math
import numpy as np
import sys

sys.path.append('../Numpy')
import histogramGenerator


class RandomVariableCVXPY:
    """
    Class representing a random variable given by histogram represented as CVXPY dictionary.
    Class includes:

    Class includes:
        bins: len n of frequencies, dictionary of dictionary with cvxpy variables (1, 1)
        edges: len n+1 of histogram edges, dtype: 1-D np.array
    """


    def __init__(self, bins, edges):

        self.bins = bins
        self.edges = np.array(edges)



    def convolution_WITH_CONSTANT(self):
        """
        Calculates convolution of 2 PDFs of cvxpy variable

        :param self: class RandomVariableCVXPY
        :return convolution:  class RandomVariableCVXPY
        """

        x1 = self.bins
        size = len(x1.values())

        const = histogramGenerator.get_gauss_bins(8, 0.45, 5, 1000000, (0, 20)).bins

        convolution = {}
        for z in range(0, size):
            convolution[z] = 0

        for z in range(0, size):
            for k in range(0, z + 1):
                convolution[z] += x1[k] * const[z - k]

        return RandomVariableCVXPY(convolution, self.edges)


    def convolution_UNIFIED_NEW_MIN(self, secondVariable):
        """
        Calculates convolution of 2 PDFs of cvxpy variable. Works only for 2 identical edges. Is computed
        using the unified representation of bins - M 0/1-bins for each bin.
        IMPORTANT:
                WORKS ONLY FOR MINIMIZATION PROBLEM


        :param self: class RandomVariableCVXPY
        :param secondVariable: class RandomVariableCVXPY
        :return convolutionClass:  class RandomVariableCVXPY with cvxpy slack variables
        :return ConvConstraints: python array with inequalities - for computing the convolution
        """

        x1 = self.bins
        x2 = secondVariable.bins

        numberOfBins = len(x1.values())
        numberOfUnions = len(x1[0].values())

        ConvConstraints = []

        sumOfMultiplications = {}
        # allocation of convolution dictionary
        for i in range(0, numberOfBins):
            sumOfMultiplications[i] = 0


        # convolution

        for z in range(0, numberOfBins):
            for k in range(0, z + 1):

                for union in range(0, numberOfUnions):

                    # new variable - multiplication of x*y
                    slackMult = cp.Variable(boolean=True)
                    sumOfMultiplications[z] += slackMult

                    # help constraints
                    x = (x1[k])[union]
                    y = (x2[z-k])[union]

                    ConvConstraints.append( slackMult <= x          )
                    ConvConstraints.append( slackMult <= y          )
                    ConvConstraints.append( slackMult >= x + y - 1  )


        # cut edges
        self.cutBins(self.edges, sumOfMultiplications)

        convolution = {}
        # introducing constraint for convolution
        for i in range(0, numberOfBins):
            convolution[i] = {}
            sumOfNewVariables = 0
            for union in range(0, numberOfUnions):
                (convolution[i])[union] = cp.Variable(boolean=True)
                sumOfNewVariables += (convolution[i])[union]

            ConvConstraints.append(sumOfNewVariables >= sumOfMultiplications[i])  # as a min. problem - not working everytime

        convolutionClass = RandomVariableCVXPY(convolution, self.edges)

        return convolutionClass, ConvConstraints

    def convolution_UNIFIED_NEW_MAX(self, secondVariable):
        """
        Calculates convolution of 2 PDFs of cvxpy variable. Works only for 2 identical edges. Is computed
        using the unified representation of bins - M 0/1-bins for each bin.
        IMPORTANT:
                    WORKS ONLY FOR MAXIMIZATION PROBLEM

        :param self: class RandomVariableCVXPY
        :param secondVariable: class RandomVariableCVXPY
        :return convolutionClass:  class RandomVariableCVXPY with cvxpy slack variables
        :return ConvConstraints: python array with inequalities - for computing the convolution
        """

        x1 = self.bins
        x2 = secondVariable.bins

        numberOfBins = len(x1.values())
        numberOfUnions = len(x1[0].values())

        ConvConstraints = []

        sumOfMultiplications = {}
        # allocation of convolution dictionary
        for i in range(0, numberOfBins):
            sumOfMultiplications[i] = 0


        # convolution

        for z in range(0, numberOfBins):
            for k in range(0, z + 1):

                for union in range(0, numberOfUnions):

                    # new variable - multiplication of x*y
                    slackMult = cp.Variable(boolean=True)
                    sumOfMultiplications[z] += slackMult

                    # help constraints
                    x = (x1[k])[union]
                    y = (x2[z-k])[union]

                    ConvConstraints.append( slackMult <= x          )
                    ConvConstraints.append( slackMult <= y          )
                    ConvConstraints.append( slackMult >= x + y - 1  )


        # cut edges
        self.cutBins(self.edges, sumOfMultiplications)

        convolution = {}
        # introducing constraint for convolution
        for i in range(0, numberOfBins):
            convolution[i] = {}
            sumOfNewVariables = 0
            for union in range(0, numberOfUnions):
                (convolution[i])[union] = cp.Variable(boolean=True)
                sumOfNewVariables += (convolution[i])[union]

            ConvConstraints.append(sumOfNewVariables <= sumOfMultiplications[i])
            ConvConstraints.append(sumOfNewVariables <= numberOfUnions)

        convolutionClass = RandomVariableCVXPY(convolution, self.edges)

        return convolutionClass, ConvConstraints

    def convolution_UNIFIED_OLD(self, secondVariable):
        """
        Calculates convolution of 2 PDFs of cvxpy variable. Works only for 2 identical edges. Is computed
        using the unified representation of bins - M 0/1-bins for each bin.

        :param self: class RandomVariableCVXPY
        :param secondVariable: class RandomVariableCVXPY
        :return convolutionClass:  class RandomVariableCVXPY with cvxpy slack variables
        :return ConvConstraints: python array with inequalities - for computing the convolution
        """

        x1 = self.bins
        x2 = secondVariable.bins

        numberOfBins = len(x1.values())
        numberOfUnions = len(x1[0].values())


        convolution = {}
        ConvConstraints = []
        # allocation of convolution dictionary
        for i in range(0, numberOfBins):
            convolution[i] = {}
            for union in range(0, numberOfUnions):
                (convolution[i])[union] = 0


        # convolution

        for z in range(0, numberOfBins):
            for k in range(0, z + 1):

                for union in range(0, numberOfUnions):

                    # new variable - multiplication of x*y
                    slackMult = cp.Variable(boolean=True)
                    (convolution[z])[union] += slackMult

                    # help constraints
                    x = (x1[k])[union]
                    y = (x2[z-k])[union]

                    ConvConstraints.append( slackMult <= x          )
                    ConvConstraints.append( slackMult <= y          )
                    ConvConstraints.append( slackMult >= x + y - 1  )

        # cut edges
        self.cutBins_UNIFIED(self.edges, convolution)

        convolutionClass = RandomVariableCVXPY(convolution, self.edges)

        return convolutionClass, ConvConstraints

    def maximum_QUAD_UNIFIED_NEW_MAX(self, secondVariable):
        """
        Calculates maximum of 2 PDFs of cvxpy variable. Works only for 2 identical edges. Is computed
        using the 'quadratic' algorithm and unified representation of bins - M 0/1-bins for each bin.
        IMPORTANT:
                WORKS ONLY FOR MAXIMIZATION PROBLEM

        :param self: class RandomVariableCVXPY
        :param secondVariable: class RandomVariableCVXPY
        :return maximumClass: class RandomVariableCVXPY with cvxpy slack variables (1, 1)
        :return MaxConstraints: python array with inequalities - for computing the maximum
        """

        x1 = self.bins
        x2 = secondVariable.bins

        numberOfBins = len(x1.values())
        numberOfUnions = len(x1[0].values())

        MaxConstraints = []

        # allocation of help sum
        sumOfMultiplications = {}
        for i in range(0, numberOfBins):
            sumOfMultiplications[i] = 0

        # i >= j
        for i in range(0, numberOfBins):
            for j in range(0, i+1):

                for union in range(0, numberOfUnions):

                    # new variable - multiplication of x*y
                    slackMult = cp.Variable(boolean=True)
                    sumOfMultiplications[i] += slackMult

                    # help constraints
                    x = (x1[i])[union]
                    y = (x2[j])[union]

                    MaxConstraints.append(  slackMult <= x          )
                    MaxConstraints.append(  slackMult <= y          )
                    MaxConstraints.append(  slackMult >= x + y - 1  )      # driving constr.


        # i < j
        for i in range(0, numberOfBins):
            for j in range(i+1, numberOfBins):

                for union in range(0, numberOfUnions):

                    # new variable - multiplication of x*y
                    slackMult = cp.Variable(boolean=True)
                    sumOfMultiplications[j] += slackMult

                    # help constraints
                    x = (x1[i])[union]
                    y = (x2[j])[union]

                    MaxConstraints.append(  slackMult <= x          )
                    MaxConstraints.append(  slackMult <= y          )
                    MaxConstraints.append(  slackMult >= x + y - 1  )  # driving constr.


        maximum = {}
        # introducing constraint for maximum
        for i in range(0, numberOfBins):
            maximum[i] = {}
            sumOfNewVariables = 0
            for union in range(0, numberOfUnions):
                (maximum[i])[union] = cp.Variable(boolean=True)
                sumOfNewVariables += (maximum[i])[union]

            MaxConstraints.append(  sumOfNewVariables <= sumOfMultiplications[i]    )     # as a max. problem
            MaxConstraints.append(  sumOfNewVariables <= numberOfUnions             )


        maximumClass = RandomVariableCVXPY(maximum, self.edges)

        return maximumClass, MaxConstraints

    def maximum_QUAD_UNIFIED_NEW_MIN(self, secondVariable):
        """
        Calculates maximum of 2 PDFs of cvxpy variable. Works only for 2 identical edges. Is computed
        using the 'quadratic' algorithm and unified representation of bins - M 0/1-bins for each bin.
        IMPORTANT:
                    WORKS ONLY FOR MINIMIZATION PROBLEM
        :param self: class RandomVariableCVXPY
        :param secondVariable: class RandomVariableCVXPY
        :return maximumClass: class RandomVariableCVXPY with cvxpy slack variables (1, 1)
        :return MaxConstraints: python array with inequalities - for computing the maximum
        """

        x1 = self.bins
        x2 = secondVariable.bins

        numberOfBins = len(x1.values())
        numberOfUnions = len(x1[0].values())

        maximum = {}
        MaxConstraints = []


        # allocation of help sum
        sumOfMultiplications = {}
        for i in range(0, numberOfBins):
            sumOfMultiplications[i] = 0

        # i >= j
        for i in range(0, numberOfBins):
            for j in range(0, i+1):

                for union in range(0, numberOfUnions):

                    # new variable - multiplication of x*y
                    slackMult = cp.Variable(boolean=True)
                    sumOfMultiplications[i] += slackMult

                    # help constraints
                    x = (x1[i])[union]
                    y = (x2[j])[union]

                    MaxConstraints.append(  slackMult <= x          )
                    MaxConstraints.append(  slackMult <= y          )
                    MaxConstraints.append(  slackMult >= x + y - 1  )      # driving constr.


        # i < j
        for i in range(0, numberOfBins):
            for j in range(i+1, numberOfBins):

                for union in range(0, numberOfUnions):

                    # new variable - multiplication of x*y
                    slackMult = cp.Variable(boolean=True)
                    sumOfMultiplications[j] += slackMult

                    # help constraints
                    x = (x1[i])[union]
                    y = (x2[j])[union]

                    MaxConstraints.append(  slackMult <= x          )
                    MaxConstraints.append(  slackMult <= y          )
                    MaxConstraints.append(  slackMult >= x + y - 1  )  # driving constr.


        # introducing constraint for maximum
        for i in range(0, numberOfBins):
            maximum[i] = {}
            sumOfNewVariables = 0
            for union in range(0, numberOfUnions):
                (maximum[i])[union] = cp.Variable(boolean=True)
                sumOfNewVariables += (maximum[i])[union]

            MaxConstraints.append( sumOfNewVariables >= sumOfMultiplications[i])      # as a min. problem - not working everytime

        maximumClass = RandomVariableCVXPY(maximum, self.edges)

        return maximumClass, MaxConstraints

    def maximum_QUAD_UNIFIED_OLD(self, secondVariable):
        """
        Calculates maximum of 2 PDFs of cvxpy variable. Works only for 2 identical edges. Is computed
        using the 'quadratic' algorithm and unified representation of bins - M 0/1-bins for each bin.

        WARNING: this is an old version which does not work after more steps

        :param self: class RandomVariableCVXPY
        :param secondVariable: class RandomVariableCVXPY
        :return maximumClass: class RandomVariableCVXPY with cvxpy slack variables (1, 1)
        :return MaxConstraints: python array with inequalities - for computing the maximum
        """

        x1 = self.bins
        x2 = secondVariable.bins

        numberOfBins = len(x1.values())
        numberOfUnions = len(x1[0].values())

        maximum = {}
        MaxConstraints = []

            # allocation of maximum
        for i in range(0, numberOfBins):
            maximum[i] = {}
            for union in range(0, numberOfUnions):
                (maximum[i])[union] = 0


        # i >= j
        for i in range(0, numberOfBins):
            for j in range(0, i+1):

                for union in range(0, numberOfUnions):

                    # new variable - multiplication of x*y
                    slackMult = cp.Variable(boolean=True)
                    (maximum[i])[union] += slackMult

                    # help constraints
                    x = (x1[i])[union]
                    y = (x2[j])[union]

                    MaxConstraints.append(  slackMult <= x          )
                    MaxConstraints.append(  slackMult <= y          )
                    MaxConstraints.append(  slackMult >= x + y - 1  )      # driving constr.


        # i < j
        for i in range(0, numberOfBins):
            for j in range(i+1, numberOfBins):

                for union in range(0, numberOfUnions):

                    # new variable - multiplication of x*y
                    slackMult = cp.Variable(boolean=True)
                    (maximum[j])[union] += slackMult

                    # help constraints
                    x = (x1[i])[union]
                    y = (x2[j])[union]

                    MaxConstraints.append(  slackMult <= x          )
                    MaxConstraints.append(  slackMult <= y          )
                    MaxConstraints.append(  slackMult >= x + y - 1  )  # driving constr.

        maximumClass = RandomVariableCVXPY(maximum, self.edges)

        return maximumClass, MaxConstraints



    def maximum_ELEMENTWISE(self, secondVariable):
        """
        Calculates maximum of 2 PDFs of cvxpy variable. Works only for 2 identical edges.

        :param self: class RandomVariableCVXPY
        :param secondVariable: class RandomVariableCVXPY
        :return maximumClass:  class RandomVariableCVXPY
        :return MaxConstraints: python array with inequalities - for computing the maximum
        """

        return self, []

        x1 = self.bins
        x2 = secondVariable.bins

        size = len(x1.values())
        maximum = {}
        MaxConstraints = [0 <= 0] * 2 *  size   # allocation

        for i in range(0, size):
            # maximum[i] = cp.maximum(x1[i], x2[i])     # old version
            slackMax = cp.Variable(nonneg=True)
            maximum[i] = slackMax
            MaxConstraints[2*i] = x1[i] <= slackMax
            MaxConstraints[2*i + 1] = x2[i] <= slackMax

        maximumClass = RandomVariableCVXPY(maximum, self.edges)

        return maximumClass, MaxConstraints

    @staticmethod
    def cutBins(edges: np.array, bins: {cp.Expression}):
        """
        Cuts bins depending on edge[0]
        if edge[0] < 0: cuts left bins and adds zeros to the end
        if edge[0] > 0: cuts right bins and adds zeros to the beginning
        Works for only unified random variable

        :param edges: (1, n+1) numpy array of edges
        :param bins: dictionary of dictionray of cp.Variables (1,1)
        :returns None
        """

        diff = edges[1] - edges[0]
        numberOfBins = len(bins)

        numberOfBinsNeeded = math.floor(abs(edges[0]) / diff)

        if edges[0] > 0:  # cut bins

            for i in range(numberOfBinsNeeded, numberOfBins):
                bins[i] = bins[i - numberOfBinsNeeded]

            for i in range(0, numberOfBinsNeeded):
                    bins[i] = 0

        elif edges[0] < 0:  # cut bins

            for i in range(numberOfBinsNeeded, numberOfBins):
                bins[i - numberOfBinsNeeded] = bins[i]

            for i in range(numberOfBins - numberOfBinsNeeded, numberOfBins):
                    bins[i] = 0

    @staticmethod
    def cutBins_UNIFIED(edges: np.array, bins: {cp.Expression}):
        """
        Cuts bins depending on edge[0]
        if edge[0] < 0: cuts left bins and adds zeros to the end
        if edge[0] > 0: cuts right bins and adds zeros to the beginning
        Works for only unified random variable

        :param edges: (1, n+1) numpy array of edges
        :param bins: dictionary of dictionray of cp.Variables (1,1)
        :returns None
        """

        diff = edges[1] - edges[0]
        numberOfUnions = len(bins[0].values())
        numberOfBins = len(bins.values())

        numberOfBinsNeeded = math.floor(abs(edges[0]) / diff)

        if edges[0] > 0:  # cut bins

            for i in range(numberOfBinsNeeded, numberOfBins):
                for union in range(0, numberOfUnions):
                    (bins[i])[union] = (bins[i - numberOfBinsNeeded])[union]

            for i in range(0, numberOfBinsNeeded):
                for union in range(0, numberOfUnions):
                    # (bins[i])[union] = 0
                    (bins[i])[union] = cp.Variable(boolean=True)    # introducing new variable with no constraints == 0


        elif edges[0] < 0:  # cut bins

            for i in range(numberOfBinsNeeded, numberOfBins):
                for union in range(0, numberOfUnions):
                    (bins[i - numberOfBinsNeeded])[union] = (bins[i])[union]

            for i in range(numberOfBins - numberOfBinsNeeded, numberOfBins):
                for union in range(0, numberOfUnions):
                    # (bins[i])[union] = 0
                    (bins[i])[union] = cp.Variable(boolean=True)    # introducing new variable with no constraints == 0
