import cvxpy as cp
import math
import numpy as np

import histogramGenerator
from randomVariableHist_Numpy import RandomVariable as RandomVariableNumpy


class RandomVariableCVXPY:
    """
    Class representing a random variable given by histogram represented as CVXPY dictionary.
    Class also includes functions for convolution and maximum.

    Class includes:
        bins: len n of frequencies, dictionary of dictionary with cvxpy variables (1, 1)
        edges: len n+1 of histogram edges, dtype: 1-D np.array
        lowerBounds: numpy array of lowerbounds values, not a mandatory atribute
    """


    def __init__(self, bins: np.array, edges: np.array, lowerBounds: np.array = np.array([])):

        self.bins = bins
        self.edges = np.array(edges)
        self.lowerBounds = lowerBounds


    def convolution_WITH_CONSTANT(self):
        """
        Calculates convolution of 2 PDFs of cvxpy variable and a constant.

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

    def maximum_McCormick(self, secondVariable):
        """
        Calculates maximum of 2 PDFs of cvxpy variable. Works only for 2 identical edges. Is computed
        using the 'quadratic' algorithm and McCormick envelopes
        IMPORTANT:
                WORKS ONLY FOR MINIMIZATION PROBLEM

        :param self: class RandomVariableCVXPY
        :param secondVariable: class RandomVariableCVXPY
        :return maximumClass: class RandomVariableCVXPY with cvxpy slack variables (1, 1)
        :return MaxConstraints: python array with inequalities - for computing the maximum
        """

        x1 = self.bins
        lowerBounds1 = self.lowerBounds

        x2 = secondVariable.bins
        lowerBounds2 = secondVariable.lowerBounds

        numberOfBins = len(x1)

        MaxConstraints = []

        maximum = {}
        # allocation of convolution dictionary
        for i in range(0, numberOfBins):
            maximum[i] = 0

        # i >= j
        for i in range(0, numberOfBins):
            for j in range(0, i + 1):

                # new variable - multiplication of x*y
                slackMult = cp.Variable(nonneg=True)
                maximum[i] += slackMult

                # help constraints
                x = x1[i]
                y = x2[j]

                # formulating bound-values
                if lowerBounds1.size == 0:    x_L = 0
                else:                         x_L = lowerBounds1[i]
                if lowerBounds2.size == 0:    y_L = 0
                else:                         y_L = lowerBounds2[j]

                x_U = 1
                y_U = 1

                # McCormick constraints
                MaxConstraints.append( slackMult >= x_L*y + x*y_L - x_L*y_L )
                MaxConstraints.append( slackMult >= x_U*y + x*y_U - x_U*y_U )
                MaxConstraints.append( slackMult <= x_U*y + x*y_L - x_U*y_L )
                MaxConstraints.append( slackMult <= x*y_U + x_L*y - x_L*y_U )

                # creating bound-constraints
                MaxConstraints.append(slackMult <= 1)
                MaxConstraints.append(x <= x_U)
                MaxConstraints.append(x >= x_L)
                MaxConstraints.append(y <= y_U)
                MaxConstraints.append(y >= y_L)

                # i < j
        for i in range(0, numberOfBins):
            for j in range(i + 1, numberOfBins):

                # new variable - multiplication of x*y
                slackMult = cp.Variable(nonneg=True)
                maximum[j] += slackMult

                # help constraints
                debug = 0
                if lowerBounds1.size == 0:
                    x_L = 0
                    debug = debug + 1
                else:                         x_L = lowerBounds1[i]
                if lowerBounds2.size == 0:
                    y_L = 0
                    debug = debug + 1
                else:                         y_L = lowerBounds2[j]


                x = x1[i]
                y = x2[j]

                # formulating constraint-values
                # x_L = 0
                x_U = 1
                # y_L = 0
                y_U = 1

                # McCormick constraints
                MaxConstraints.append(slackMult >= x_L * y + x * y_L - x_L * y_L)
                MaxConstraints.append(slackMult >= x_U * y + x * y_U - x_U * y_U)
                MaxConstraints.append(slackMult <= x_U * y + x * y_L - x_U * y_L)
                MaxConstraints.append(slackMult <= x * y_U + x_L * y - x_L * y_U)

                # creating bound-constraints
                # MaxConstraints.append(slackMult <= 1)
                MaxConstraints.append(x <= x_U)
                MaxConstraints.append(x >= x_L)
                MaxConstraints.append(y <= y_U)
                MaxConstraints.append(y >= y_L)


        rv1 = RandomVariableNumpy(lowerBounds1, self.edges)
        rv2 = RandomVariableNumpy(lowerBounds2, self.edges)

        # maximumClass = RandomVariableCVXPY(maximum, self.edges)     # without exact comput.
        maximumClass = RandomVariableCVXPY(maximum, self.edges, rv1.maxOfDistributionsFORM(rv2).bins) # with exact comput.

        return maximumClass, MaxConstraints

    def convolution_GP(self, secondVariable, constr):
        """
        Calculates convolution of 2 PDFs of cvxpy variable. Works only for 2 identical edges. Is computed
        using the 'quadratic' algorithm and GP programming

        :param self: class RandomVariableCVXPY
        :param secondVariable: class RandomVariableCVXPY
        :return convClass: class RandomVariableCVXPY with cvxpy slack variables (1, 1)
        :return constr: python array with inequalities - for computing the convolution
        """

        x1 = self.bins
        x2 = secondVariable.bins

        numberOfBins = len(x1)

        convolution = {}
        # allocation of convolution dictionary
        for i in range(0, numberOfBins):
            convolution[i] = 0

        for z in range(0, numberOfBins):
            for k in range(0, z + 1):
                convolution[z] += x1[k] * x2[z - k]

            # shift
        constrNew = self.cutBins(self.edges, convolution, GP=True)
        constr.extend(constrNew)

        convClass = RandomVariableCVXPY(convolution, self.edges) # with exact comput.

        return convClass, constr

    def convolution_GP_OPT(self, secondVariable, constr):
        """
        Calculates convolution of 2 PDFs of cvxpy variable. Works only for 2 identical edges. Is computed
        using the 'quadratic' algorithm and GP programming - optimized version

        :param self: class RandomVariableCVXPY
        :param secondVariable: class RandomVariableCVXPY
        :return convClass: class RandomVariableCVXPY with cvxpy auxiliary variables (1, 1)
        :return constr: python array with inequalities - for computing the convolution
        """

        x1 = self.bins
        x2 = secondVariable.bins

        numberOfBins = len(x1)

        convolution = {}
        # allocation of convolution dictionary
        for i in range(0, numberOfBins):
            convolution[i] = 0


        for z in range(0, numberOfBins):
            for k in range(0, z + 1):
                convolution[z] += x1[k] * x2[z - k]

            # shift
        constrNew = self.cutBins(self.edges, convolution, GP=True)
        constr.extend(constrNew)

        newConv = {}
        for bin in range(0, numberOfBins):
            n_i = cp.Variable(pos=True)        # create auxiliary monomial
            constr.append(convolution[bin] <= n_i)
            newConv[bin] = n_i

        convClass = RandomVariableCVXPY(newConv, self.edges)

        return convClass, constr

    def maximum_GP(self, secondVariable):
        """
        Calculates maximum of 2 PDFs of cvxpy variable. Works only for 2 identical edges. Is computed
        using the 'quadratic' algorithm and GP programming

        :param self: class RandomVariableCVXPY
        :param secondVariable: class RandomVariableCVXPY
        :return maximumClass: class RandomVariableCVXPY with cvxpy slack variables (1, 1)
        """

        x1 = self.bins

        x2 = secondVariable.bins

        numberOfBins = len(x1)

        maximum = {}
        # allocation of maximum dictionary
        for i in range(0, numberOfBins):
            maximum[i] = 0

        for i in range(0, numberOfBins):
            for j in range(0, i + 1):

                maximum[i] += x1[i]*x2[j]

                if i != j:
                    maximum[i] += x1[j] * x2[i]


        maximumClass = RandomVariableCVXPY(maximum, self.edges) # with exact comput.

        return maximumClass

    def maximum_GP_OPT(self, secondVariable, constr):
        """
        Calculates maximum of 2 PDFs of cvxpy variable. Works only for 2 identical edges. Is computed
        using the 'quadratic' algorithm and GP programming - optimized version

        :param self: class RandomVariableCVXPY
        :param secondVariable: class RandomVariableCVXPY
        :return maximumClass: class RandomVariableCVXPY with cvxpy slack variables (1, 1)
        :return constr: python array with inequalities - for computing the maximum
        """

        x1 = self.bins
        x2 = secondVariable.bins

        numberOfBins = len(x1)

        maximum = {}
        # allocation of maximum dictionary
        for i in range(0, numberOfBins):
            maximum[i] = 0

        for i in range(0, numberOfBins):
            for j in range(0, i + 1):

                # new variable - multiplication of x*y
                maximum[i] += x1[i]*x2[j]

                if i != j:
                    maximum[i] += x1[j] * x2[i]

            # create upper bounds
        newMax = {}
        for bin in range(0, numberOfBins):
            n_i = cp.Variable(pos=True)  # create monomial
            constr.append(maximum[bin] <= n_i)
            newMax[bin] = n_i

        maximumClass = RandomVariableCVXPY(newMax, self.edges) # with exact comput.

        return maximumClass, constr

    def convolution_McCormick(self, secondVariable):
        """
        Calculates convolution of 2 PDFs of cvxpy variable. Works only for 2 identical edges. Is computed
        using the 'quadratic' algorithm and McCormick envelopes
        IMPORTANT:
                WORKS ONLY FOR MINIMIZATION PROBLEM

        :param self: class RandomVariableCVXPY
        :param secondVariable: class RandomVariableCVXPY
        :return maximumClass: class RandomVariableCVXPY with cvxpy slack variables (1, 1)
        :return MaxConstraints: python array with inequalities - for computing the maximum
        """

        x1 = self.bins
        lowerBounds1 = self.lowerBounds

        x2 = secondVariable.bins
        lowerBounds2 = secondVariable.lowerBounds

        numberOfBins = len(x1)

        ConvConstraints = []

        convolution = {}
        # allocation of convolution dictionary
        for i in range(0, numberOfBins):
            convolution[i] = 0

        for z in range(0, numberOfBins):
            for k in range(0, z + 1):

                slackMult = cp.Variable(nonneg=True)
                convolution[z] += slackMult

                x = x1[k]
                y = x2[z - k]

                # formulating bound-values
                if lowerBounds1.size == 0:    x_L = 0
                else:                         x_L = lowerBounds1[k]
                if lowerBounds2.size == 0:    y_L = 0
                else:                         y_L = lowerBounds2[z - k]

                # x_L = 0
                x_U = 1
                # y_L = 0
                y_U = 1
                # McCormick constraints
                ConvConstraints.append(slackMult >= x_L * y + x * y_L - x_L * y_L)
                ConvConstraints.append(slackMult >= x_U * y + x * y_U - x_U * y_U)
                ConvConstraints.append(slackMult <= x_U * y + x * y_L - x_U * y_L)
                ConvConstraints.append(slackMult <= x * y_U + x_L * y - x_L * y_U)

                # creating bound-constraints
                ConvConstraints.append(slackMult <= 1)
                ConvConstraints.append(x <= x_U)
                ConvConstraints.append(x >= x_L)
                ConvConstraints.append(y <= y_U)
                ConvConstraints.append(y >= y_L)


        # Deal With Edges
        self.cutBins(self.edges, convolution, mccormick=True)

        rv1 = RandomVariableNumpy(lowerBounds1, self.edges)
        rv2 = RandomVariableNumpy(lowerBounds2, self.edges)

        convClass = RandomVariableCVXPY(convolution, self.edges, rv1.convolutionOfTwoVarsShift(rv2).bins)


        return convClass, ConvConstraints



    def convolution_UNARY_DIVIDE(self, secondVariable, withSepConstr=False, asMin=False):
        """ Calculates convolution of 2 PDFs of cvxpy variable. Works only for 2 identical edges. Is computed
        using the unary representation of bins - M 0/1-bins for each bin. Unarization is kept using the divison.

        :param self: class RandomVariableCVXPY
        :param secondVariable: class RandomVariableCVXPY
        :param asMin: boolean, true for minimization problem, false for maximization problem
        :param withSepConstr: boolean whether a separation constraints should be included
        :return convolutionClass:  class RandomVariableCVXPY with cvxpy slack variables
        :return ConvConstraints: python array with inequalities - for computing the convolution
        """

        x1 = self.bins
        x2 = secondVariable.bins

        numberOfBins = len(x1.values())
        numberOfUnaries = len(x1[0].values())

        ConvConstraints = []

        sumOfMultiplications = {}
        # allocation of convolution dictionary
        for i in range(0, numberOfBins):
            sumOfMultiplications[i] = 0


        # convolution
        for z in range(0, numberOfBins):
            for k in range(0, z + 1):

                for unary in range(0, numberOfUnaries):
                    for unary2 in range(0, numberOfUnaries):

                        # new variable - multiplication of x*y
                        slackMult = cp.Variable(boolean=True)
                        sumOfMultiplications[z] += slackMult

                        # help constraints
                        x = (x1[k])[unary]
                        y = (x2[z - k])[unary2]

                        # two-term multiplication constraints
                        ConvConstraints.append(slackMult <= x)
                        ConvConstraints.append(slackMult <= y)
                        ConvConstraints.append(slackMult >= x + y - 1)



        # cut edges
        self.cutBins(self.edges, sumOfMultiplications)

        divisor = 2

        # introducing constraint for convolution
        convolution = {}
        for i in range(0, numberOfBins):
            convolution[i] = {}
            sumOfNewVariables = 0
            for unary in range(0, numberOfUnaries):
                (convolution[i])[unary] = cp.Variable(boolean=True)
                sumOfNewVariables += (convolution[i])[unary]

            if asMin:
                ConvConstraints.append(sumOfNewVariables >= sumOfMultiplications[i] / divisor)    # as a min possible
            elif not asMin:

                ceil = 0.999    # max possible for the solver not to round the number
                ConvConstraints.append(sumOfNewVariables <= sumOfMultiplications[i] / divisor + ceil)  # as a max possible
                ConvConstraints.append(sumOfNewVariables <= numberOfUnaries)

        # separation constraints
        if withSepConstr:
            for bin in range(0, numberOfBins):
                for unary in range(0, numberOfUnaries - 1):
                    ConvConstraints.append(
                        (convolution[bin])[unary] >= (convolution[bin])[unary + 1])  # set lower constr.


        convolutionClass = RandomVariableCVXPY(convolution, self.edges)

        return convolutionClass, ConvConstraints



    def maximum_QUAD_UNARY_DIVIDE(self, secondVariable, withSymmetryConstr=False, asMin=False):
        """
        Calculates maximum of 2 PDFs of cvxpy variable. Works only for 2 identical edges. Is computed
        using the 'quadratic' algorithm and unary representation of bins - M 0/1-bins for each bin.
        Unarization is kept using the divison.

        :param self: class RandomVariableCVXPY
        :param secondVariable: class RandomVariableCVXPY
        :param asMin: boolean, true for minimization problem, false for maximization problem
        :param withSymmetryConstr: boolean whether a symmetry constraints should be included
        :return maximumClass: class RandomVariableCVXPY with cvxpy slack variables (1, 1)
        :return MaxConstraints: python array with inequalities - for computing the maximum
        """

        x1 = self.bins
        x2 = secondVariable.bins

        numberOfBins = len(x1.values())
        numberOfUnaries = len(x1[0].values())

        MaxConstraints = []

        # allocation of help sum
        sumOfMultiplications = {}
        for i in range(0, numberOfBins):
            sumOfMultiplications[i] = 0

        # i >= j
        for i in range(0, numberOfBins):
            for j in range(0, i+1):

                for unary in range(0, numberOfUnaries):
                    for unary2 in range(0, numberOfUnaries):

                        # new variable - multiplication of x*y
                        slackMult = cp.Variable(boolean=True)
                        sumOfMultiplications[i] += slackMult

                        # help constraints
                        x = (x1[i])[unary]
                        y = (x2[j])[unary2]

                        MaxConstraints.append(slackMult <= x)
                        MaxConstraints.append(slackMult <= y)
                        MaxConstraints.append(slackMult >= x + y - 1)  # driving constr.

                        if i != j:

                            # new variable - multiplication of x*y
                            slackMult2 = cp.Variable(boolean=True)
                            sumOfMultiplications[i] += slackMult2

                            # help constraints
                            x = (x1[j])[unary]
                            y = (x2[i])[unary2]

                            MaxConstraints.append(slackMult2 <= x)
                            MaxConstraints.append(slackMult2 <= y)
                            MaxConstraints.append(slackMult2 >= x + y - 1)  # driving constr.


        # divisor = np.ceil((2*numberOfUnaries**2 - 2*numberOfUnaries + 1) / numberOfUnaries)
        divisor = 28

        # introducing constraint for maximum
        maximum = {}
        for i in range(0, numberOfBins):
            maximum[i] = {}
            sumOfNewVariables = 0
            for unary in range(0, numberOfUnaries):
                (maximum[i])[unary] = cp.Variable(boolean=True)
                sumOfNewVariables += (maximum[i])[unary]

            if asMin:
                MaxConstraints.append(  sumOfNewVariables >= sumOfMultiplications[i] / divisor   )     # as a max. problem
            elif not asMin:
                ceil = 0.999
                MaxConstraints.append(sumOfNewVariables <= sumOfMultiplications[i] / divisor + ceil)  # as a max. problem
                MaxConstraints.append(sumOfNewVariables <= numberOfUnaries)  # as a max. problem


        # symmetry constraints
        if withSymmetryConstr:
            for bin in range(0, numberOfBins):
                for unary in range(0, numberOfUnaries - 1):
                    MaxConstraints.append(
                        (maximum[bin])[unary] >= (maximum[bin])[unary + 1])  # set lower constr.


        maximumClass = RandomVariableCVXPY(maximum, self.edges)

        return maximumClass, MaxConstraints


    def convolution_UNARY_CUT(self, secondVariable, withSymmetryConstr=False, asMin=False):
        """
        Calculates convolution of 2 PDFs of cvxpy variable. Works only for 2 identical edges. Is computed
        using the unary representation of bins - M 0/1-bins for each bin. Unarization is kept using the information
        cutting. Maximization is always feasible, minimization is not.

        :param self: class RandomVariableCVXPY
        :param secondVariable: class RandomVariableCVXPY
        :param asMin: boolean, true for minimization problem, false for maximization problem
        :param withSymmetryConstr: boolean whether a symmetry constraints should be included
        :return convolutionClass:  class RandomVariableCVXPY with cvxpy slack variables
        :return ConvConstraints: python array with inequalities - for computing the convolution
        """

        x1 = self.bins
        x2 = secondVariable.bins

        numberOfBins = len(x1.values())
        numberOfUnaries = len(x1[0].values())

        ConvConstraints = []

        sumOfMultiplications = {}
        # allocation of convolution dictionary
        for i in range(0, numberOfBins):
            sumOfMultiplications[i] = 0


        # convolution

        for z in range(0, numberOfBins):
            for k in range(0, z + 1):

                for unary in range(0, numberOfUnaries):
                    for unary2 in range(0, numberOfUnaries):

                        # new variable - multiplication of x*y
                        slackMult = cp.Variable(boolean=True)
                        sumOfMultiplications[z] += slackMult

                        # help constraints
                        x = (x1[k])[unary]
                        y = (x2[z - k])[unary2]

                        ConvConstraints.append(slackMult <= x)
                        ConvConstraints.append(slackMult <= y)
                        ConvConstraints.append(slackMult >= x + y - 1)


        # cut edges
        self.cutBins(self.edges, sumOfMultiplications)

        convolution = {}
        # introducing constraint for convolution
        for i in range(0, numberOfBins):
            convolution[i] = {}
            sumOfNewVariables = 0
            for unary in range(0, numberOfUnaries):
                (convolution[i])[unary] = cp.Variable(boolean=True)
                sumOfNewVariables += (convolution[i])[unary]

            if asMin:
                ConvConstraints.append(sumOfNewVariables >= sumOfMultiplications[i])  # as a min. problem
            elif not asMin:
                ConvConstraints.append(sumOfNewVariables <= sumOfMultiplications[i])    # as max. problem
                ConvConstraints.append(sumOfNewVariables <= numberOfUnaries)


        # symmetry constraints
        if withSymmetryConstr:
            for bin in range(0, numberOfBins):
                for unary in range(0, numberOfUnaries - 1):
                    ConvConstraints.append(
                        (convolution[bin])[unary] >= (convolution[bin])[unary + 1])  # set lower constr.



        convolutionClass = RandomVariableCVXPY(convolution, self.edges)

        return convolutionClass, ConvConstraints

    def maximum_QUAD_UNARY_CUT(self, secondVariable, asMin=False, withSymmetryConstr=False):
        """
        Calculates maximum of 2 PDFs of cvxpy variable. Works only for 2 identical edges. Is computed
        using the 'quadratic' algorithm and unary representation of bins - M 0/1-bins for each bin.
        Unarization is kept using the information cutting. Maximization is always feasible, minimization is not.

        :param self: class RandomVariableCVXPY
        :param secondVariable: class RandomVariableCVXPY
            :param withSymmetryConstr: boolean whether a symmetry constraints should be included
        :param asMin: boolean, true for minimization problem, false for maximization problem
        :return maximumClass: class RandomVariableCVXPY with cvxpy slack variables (1, 1)
        :return MaxConstraints: python array with inequalities - for computing the maximum
        """

        x1 = self.bins
        x2 = secondVariable.bins

        numberOfBins = len(x1.values())
        numberOfUnaries = len(x1[0].values())

        maximum = {}
        MaxConstraints = []


        # allocation of help sum
        sumOfMultiplications = {}
        for i in range(0, numberOfBins):
            sumOfMultiplications[i] = 0


        for i in range(0, numberOfBins):    # represents pdf
            for j in range(0, i+1):         # represents cdf

                for unary in range(0, numberOfUnaries):
                    for unary2 in range(0, numberOfUnaries):

                        # new variable - multiplication of x*y
                        slackMult = cp.Variable(boolean=True)
                        sumOfMultiplications[i] += slackMult

                        # help constraints
                        x = (x1[i])[unary]
                        y = (x2[j])[unary2]

                        MaxConstraints.append(slackMult <= x)
                        MaxConstraints.append(slackMult <= y)
                        MaxConstraints.append(slackMult >= x + y - 1)  # driving constr.

                        if i != j:

                            # new variable - multiplication of x*y
                            slackMult2 = cp.Variable(boolean=True)
                            sumOfMultiplications[i] += slackMult2

                            # help constraints
                            x = (x1[j])[unary]
                            y = (x2[i])[unary2]

                            MaxConstraints.append(slackMult2 <= x)
                            MaxConstraints.append(slackMult2 <= y)
                            MaxConstraints.append(slackMult2 >= x + y - 1)


        # introducing constraint for maximum
        for i in range(0, numberOfBins):
            maximum[i] = {}
            sumOfNewVariables = 0
            for unary in range(0, numberOfUnaries):
                (maximum[i])[unary] = cp.Variable(boolean=True)
                sumOfNewVariables += (maximum[i])[unary]

            if asMin:
                MaxConstraints.append( sumOfNewVariables >= sumOfMultiplications[i])      # as a min. problem - not working everytime
            elif not asMin:
                MaxConstraints.append(  sumOfNewVariables <= sumOfMultiplications[i]    )     # as a max. problem
                MaxConstraints.append(  sumOfNewVariables <= numberOfUnaries            )


        # symmetry constraints
        if withSymmetryConstr:
            for bin in range(0, numberOfBins):
                for unary in range(0, numberOfUnaries - 1):
                    MaxConstraints.append(
                        (maximum[bin])[unary] >= (maximum[bin])[unary + 1])  # set lower constr.

        maximumClass = RandomVariableCVXPY(maximum, self.edges)

        return maximumClass, MaxConstraints


    # ----------------------------- below is just old functions and helper cut bins functions --------------


    def maximum_QUAD_UNARY_OLD(self, secondVariable):
        """
        Calculates maximum of 2 PDFs of cvxpy variable. Works only for 2 identical edges. Is computed
        using the 'quadratic' algorithm and unary representation of bins - M 0/1-bins for each bin.
        This is an old implementation.

        WARNING: this is an old version which does not work after more steps

        :param self: class RandomVariableCVXPY
        :param secondVariable: class RandomVariableCVXPY
        :return maximumClass: class RandomVariableCVXPY with cvxpy slack variables (1, 1)
        :return MaxConstraints: python array with inequalities - for computing the maximum
        """

        x1 = self.bins
        x2 = secondVariable.bins

        numberOfBins = len(x1.values())
        numberOfUnaries = len(x1[0].values())

        maximum = {}
        MaxConstraints = []

            # allocation of maximum
        for i in range(0, numberOfBins):
            maximum[i] = {}
            for unary in range(0, numberOfUnaries):
                (maximum[i])[unary] = 0


        # i >= j
        for i in range(0, numberOfBins):
            for j in range(0, i+1):

                for unary in range(0, numberOfUnaries):

                    # new variable - multiplication of x*y
                    slackMult = cp.Variable(boolean=True)
                    (maximum[i])[unary] += slackMult

                    # help constraints
                    x = (x1[i])[unary]
                    y = (x2[j])[unary]

                    MaxConstraints.append(  slackMult <= x          )
                    MaxConstraints.append(  slackMult <= y          )
                    MaxConstraints.append(  slackMult >= x + y - 1  )      # driving constr.


        # i < j
        for i in range(0, numberOfBins):
            for j in range(i+1, numberOfBins):

                for unary in range(0, numberOfUnaries):

                    # new variable - multiplication of x*y
                    slackMult = cp.Variable(boolean=True)
                    (maximum[j])[unary] += slackMult

                    # help constraints
                    x = (x1[i])[unary]
                    y = (x2[j])[unary]

                    MaxConstraints.append(  slackMult <= x          )
                    MaxConstraints.append(  slackMult <= y          )
                    MaxConstraints.append(  slackMult >= x + y - 1  )  # driving constr.

        maximumClass = RandomVariableCVXPY(maximum, self.edges)

        return maximumClass, MaxConstraints

    def convolution_UNARY_OLD(self, secondVariable):
        """
        Calculates convolution of 2 PDFs of cvxpy variable. Works only for 2 identical edges. Is computed
        using the unary representation of bins - M 0/1-bins for each bin. Old version.

        :param self: class RandomVariableCVXPY
        :param secondVariable: class RandomVariableCVXPY
        :return convolutionClass:  class RandomVariableCVXPY with cvxpy slack variables
        :return ConvConstraints: python array with inequalities - for computing the convolution
        """

        x1 = self.bins
        x2 = secondVariable.bins

        numberOfBins = len(x1.values())
        numberOfUnaries = len(x1[0].values())

        convolution = {}
        ConvConstraints = []
        # allocation of convolution dictionary
        for i in range(0, numberOfBins):
            convolution[i] = {}
            for unary in range(0, numberOfUnaries):
                (convolution[i])[unary] = 0

        # convolution

        for z in range(0, numberOfBins):
            for k in range(0, z + 1):

                for unary in range(0, numberOfUnaries):
                    # new variable - multiplication of x*y
                    slackMult = cp.Variable(boolean=True)
                    (convolution[z])[unary] += slackMult

                    # help constraints
                    x = (x1[k])[unary]
                    y = (x2[z - k])[unary]

                    ConvConstraints.append(slackMult <= x)
                    ConvConstraints.append(slackMult <= y)
                    ConvConstraints.append(slackMult >= x + y - 1)

        # cut edges
        self.cutBins_UNARY(self.edges, convolution)

        convolutionClass = RandomVariableCVXPY(convolution, self.edges)

        return convolutionClass, ConvConstraints

    def maximum_FORM_UNARY_OLD_MAX(self, secondVariable, withSymmetryConstr=False):
        """
        Calculates maximum of 2 PDFs of cvxpy variable. Works only for 2 identical edges. Is computed
        using the 'formula' algorithm and unary representation of bins - M 0/1-bins for each bin.
        This approach is significantly slower giving the same results.
        IMPORTANT:
                WORKS ONLY FOR MAXIMIZATION PROBLEM

        :param self: class RandomVariableCVXPY
        :param secondVariable: class RandomVariableCVXPY
        :param withSymmetryConstr: boolean whether a symmetry constraints should be included
        :return maximumClass: class RandomVariableCVXPY with cvxpy slack variables (1, 1)
        :return MaxConstraints: python array with inequalities - for computing the maximum
        """

        f1 = self.bins
        f2 = secondVariable.bins

        numberOfBins = len(f1.values())
        numberOfUnaries = len(f1[0].values())

        MaxConstraints = []

        # allocation of help cumsum and help maximum
        F1_HelpSum = {}
        F2_HelpSum = {}
        maximumHelpSum = {}
        for i in range(0, numberOfBins):
            F1_HelpSum[i] = 0
            F2_HelpSum[i] = 0
            maximumHelpSum[i] = 0

        # formulating the cumsum

        for unary in range(0, numberOfUnaries):  # set the first array
            F1_HelpSum[0] = f1[0][unary]
            F2_HelpSum[0] = f2[0][unary]

            # compute cumsum using dynamic programming
        for bin in range(1, numberOfBins):
            F1_HelpSum[bin] = F1_HelpSum[bin - 1]
            F2_HelpSum[bin] = F2_HelpSum[bin - 1]
            for unary in range(0, numberOfUnaries):
                F1_HelpSum[bin] += f1[bin][unary]
                F2_HelpSum[bin] += f2[bin][unary]

            # set cumsum constraints
        F1 = {}
        F2 = {}

        for i in range(0, numberOfBins):
            F1[i] = {}
            F2[i] = {}
            F1_newVariablesSum = 0
            F2_newVariablesSum = 0
            for unary in range(0, numberOfUnaries):
                (F1[i])[unary] = cp.Variable(boolean=True)
                (F2[i])[unary] = cp.Variable(boolean=True)
                F1_newVariablesSum += (F1[i])[unary]
                F2_newVariablesSum += (F2[i])[unary]

            MaxConstraints.append(F1_newVariablesSum <= F1_HelpSum[i])  # as a max. problem
            MaxConstraints.append(F1_newVariablesSum <= numberOfUnaries)

            MaxConstraints.append(F2_newVariablesSum <= F2_HelpSum[i])  # as a max. problem
            MaxConstraints.append(F2_newVariablesSum <= numberOfUnaries)

        # computing the maximum

        for bin in range(0, numberOfBins):
            for unary in range(0, numberOfUnaries):
                slackMult1 = cp.Variable(boolean=True)
                slackMult2 = cp.Variable(boolean=True)

                maximumHelpSum[bin] += slackMult1 + slackMult2

                x1 = f1[bin][unary]
                y1 = F2[bin][unary]
                x2 = f2[bin][unary]
                y2 = F1[bin][unary]

                # multiplication constraints
                MaxConstraints.append(slackMult1 <= x1)
                MaxConstraints.append(slackMult1 <= y1)
                MaxConstraints.append(slackMult1 >= x1 + y1 - 1)  # driving constr.

                MaxConstraints.append(slackMult2 <= x2)
                MaxConstraints.append(slackMult2 <= y2)
                MaxConstraints.append(slackMult2 >= x2 + y2 - 1)  # driving constr.

        # introducing constraint for maximum
        maximum = {}
        for i in range(0, numberOfBins):
            maximum[i] = {}
            max_newVariablesSum = 0
            for unary in range(0, numberOfUnaries):
                (maximum[i])[unary] = cp.Variable(boolean=True)
                max_newVariablesSum += (maximum[i])[unary]

            MaxConstraints.append(max_newVariablesSum <= maximumHelpSum[i])  # as a max. problem
            MaxConstraints.append(max_newVariablesSum <= numberOfUnaries)

        maximumClass = RandomVariableCVXPY(maximum, self.edges)

        return maximumClass, MaxConstraints


    def maximum_ELEMENTWISE(self, secondVariable):
        """
        Calculates elementwise maximum of 2 PDFs of cvxpy variable. Works only for 2 identical edges.

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
    def cutBins(edges: np.array, bins: {cp.Expression}, mccormick=False, GP=False):
        """
        Cuts bins depending on edge[0]
        if edge[0] < 0: cuts left bins and adds zeros to the end
        if edge[0] > 0: cuts right bins and adds zeros to the beginning
        Works for only unary random variable

        :param edges: (1, n+1) numpy array of edges
        :param bins: dictionary of dictionray of cp.Variables (1,1)
        :returns None
        """

        diff = edges[1] - edges[0]
        numberOfBins = len(bins)

        numberOfBinsNeeded = math.floor(abs(edges[0]) / diff)

        constr = []
        if edges[0] > 0:  # cut bins

            for i in range(numberOfBinsNeeded, numberOfBins):
                bins[i] = bins[i - numberOfBinsNeeded]

            for i in range(0, numberOfBinsNeeded):
                if mccormick:
                    bins[i] = cp.Variable(nonneg=True)
                elif GP:
                    bins[i] = cp.Variable(pos=True)
                    constr.append(bins[i] >= 0.0000000000000000001)
                else:
                    bins[i] = 0

        elif edges[0] < 0:  # cut bins

            for i in range(numberOfBinsNeeded, numberOfBins):
                bins[i - numberOfBinsNeeded] = bins[i]

            for i in range(numberOfBins - numberOfBinsNeeded, numberOfBins):
                if mccormick:
                    bins[i] = cp.Variable(nonneg=True)
                elif GP:
                    bins[i] = cp.Variable(pos=True)
                    constr.append(bins[i] >= 0.0000000000000000001)

                else:
                    bins[i] = 0

        if GP:
            return constr

    @staticmethod
    def cutBins_UNARY(edges: np.array, bins: {cp.Expression}):
        """
        Cuts bins depending on edge[0]
        if edge[0] < 0: cuts left bins and adds zeros to the end
        if edge[0] > 0: cuts right bins and adds zeros to the beginning
        Works for only unary random variable with unary notation

        :param edges: (1, n+1) numpy array of edges
        :param bins: dictionary of dictionary of cp.Variables (1,1)
        :returns None
        """

        diff = edges[1] - edges[0]
        numberOfUnaries = len(bins[0].values())
        numberOfBins = len(bins.values())

        numberOfBinsNeeded = math.floor(abs(edges[0]) / diff)

        if edges[0] > 0:  # cut bins

            for i in range(numberOfBinsNeeded, numberOfBins):
                for unary in range(0, numberOfUnaries):
                    (bins[i])[unary] = (bins[i - numberOfBinsNeeded])[unary]

            for i in range(0, numberOfBinsNeeded):
                for unary in range(0, numberOfUnaries):
                    # (bins[i])[unary] = 0
                    (bins[i])[unary] = cp.Variable(boolean=True)    # introducing new variable with no constraints == 0


        elif edges[0] < 0:  # cut bins

            for i in range(numberOfBinsNeeded, numberOfBins):
                for unary in range(0, numberOfUnaries):
                    (bins[i - numberOfBinsNeeded])[unary] = (bins[i])[unary]

            for i in range(numberOfBins - numberOfBinsNeeded, numberOfBins):
                for unary in range(0, numberOfUnaries):
                    # (bins[i])[unary] = 0
                    (bins[i])[unary] = cp.Variable(boolean=True)    # introducing new variable with no constraints == 0
