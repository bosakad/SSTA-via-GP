import math
import mosek
import sys
import numpy as np

"""
  This module includes same functions as module 'cvxpyVariable'. Is coded in MOSEK API for speed-up purposes.
  Unary variable is represented by numpy [n, m], where n is number of bins, m is number of unaries. For a MOSEK 
  variable bins is represented by [n, m] list of indices.
"""


class RandomVariableMOSEK:
    """
    Class representing a random variable given by histogram represented as MOSEK variable.
    Information about each variable is kept by their indices.

    Class includes:
      bins: numpy list of lists of integer indices to MOSEK variable list
      edges: len n+1 of histogram edges, dtype: 1-D np.array
      task: mosek.Task
    """


    def __init__(self, bins: np.array, edges: np.array, task: mosek.Task):
        self.bins = bins
        self.edges = edges
        self.task = task

        self.inf = 0.0   # symbolic purposes



    def convolution_UNARY_MAX_DIVIDE(self, secondVariable, curNofVariables, curNofConstr, withSymmetryConstr=False):
        """ Calculates convolution of 2 PDFs of random variable. Works only for 2 identical edges. Is computed
        using the unary representation of bins - M 0/1-bins for each bin. Unarization is kept using the divison.
        Is in MOSEK environment. Works only for maximization task.

        :param self: class RandomVariableMOSEK
        :param secondVariable: class RandomVariableMOSEK
        :param curNofVariables: current number of MOSEK variables - to know the indices
        :param curNofConstr: current number of MOSEK constraints - to know the indices
        :param withSymmetryConstr: boolean whether a symmetry constraints should be included
        :return convolutionClass:  class RandomVariableMOSEK with cvxpy slack variables
        :return newNofVariables: integer, new total number of MOSEK variables
        :return newNofConstr: integer, new total number of MOSEK constraints
        """

            # get data
        x1 = self.bins
        x2 = secondVariable.bins
        numberOfBins, numberOfUnaries = x1.shape
        task = self.task
        newNofVariables = curNofVariables


            # create slack multiplications
        nSlackMult = int((numberOfBins / 2)*(1+numberOfBins)*numberOfUnaries**2)
        slackMult = np.array(range(newNofVariables, newNofVariables + nSlackMult))
        newNofVariables += nSlackMult

            # create convolution variables
        nConvolutions = numberOfBins*numberOfUnaries
        convolution = np.array(range(newNofVariables, newNofVariables + nConvolutions))
        newNofVariables += nConvolutions
        newVariables = np.reshape(convolution, (numberOfBins, numberOfUnaries))  # reshape bins

            # append variables
        numberToCreate = nSlackMult + nConvolutions
        varsToCreateExtended = np.append(slackMult, convolution)
        task.appendvars(numberToCreate)
        task.putvartypelist(varsToCreateExtended, [mosek.variabletype.type_int]*numberToCreate)  # integer
        task.putvarboundlist(varsToCreateExtended, [mosek.boundkey.ra]* numberToCreate,
                                                        [0.0]*numberToCreate, [1.0]*numberToCreate)   # binary

            # convolution

        # allocate for storing the sum of multiplications
        sumOfMultiplications = {}

        # append sumOfMultiplications constraints
        if withSymmetryConstr:
            symN = (numberOfUnaries-1)*numberOfBins
        else:
            symN = 0

        task.appendcons(2*numberOfBins + 3*nSlackMult + symN)
        offset = 2*numberOfBins


        for z in range(0, numberOfBins):
            sumOfMultiplications[z] = np.array([]).astype(int)
            for k in range(0, z + 1):

                for unary in range(0, numberOfUnaries):
                    for unary2 in range(0, numberOfUnaries):

                            # constraints for sum of multiplications
                        indexToSlack = int((z / 2)*(1+z)*numberOfUnaries**2
                                       + k*numberOfUnaries**2 + unary*numberOfUnaries + unary2)
                        curSlackMult = slackMult[indexToSlack]

                        sumOfMultiplications[z] = np.append(sumOfMultiplications[z], np.array([curSlackMult]))

                                # help constraints
                        xIndex = x1[k, unary]
                        yIndex = x2[z - k, unary2]


                            # slackMult <= x
                        task.putaij(offset + 3*indexToSlack, xIndex, -1)
                        task.putaij(offset + 3*indexToSlack, curSlackMult, 1)

                        task.putconbound(offset + 3*indexToSlack, mosek.boundkey.up, -self.inf, 0)

                            # slackMult <= y
                        task.putaij(offset + 3*indexToSlack + 1, yIndex, -1)
                        task.putaij(offset + 3*indexToSlack + 1, curSlackMult, 1)

                        task.putconbound(offset + 3*indexToSlack + 1, mosek.boundkey.up, -self.inf, 0)

                            #slackMult >= x + y - 1
                        task.putaij(offset + 3*indexToSlack + 2, yIndex, -1)
                        task.putaij(offset + 3*indexToSlack + 2, xIndex, -1)
                        task.putaij(offset + 3*indexToSlack + 2, curSlackMult, 1)

                        task.putconbound(offset + 3*indexToSlack + 2, mosek.boundkey.lo, -1, self.inf)


        self.cutBins(self.edges, sumOfMultiplications)

        ceil = 0.999
        division = numberOfUnaries
        offset = curNofConstr

        for bin in range(0, numberOfBins):
            row = newVariables[bin, :]
            sumOfMultiplicationsRow = sumOfMultiplications[bin][:]

                # sumOfNewVariables <= sumOfMultiplications[i] / divisor + ceil
            task.putaijlist([bin + offset]*numberOfUnaries, row, [1]*numberOfUnaries)
            if sumOfMultiplicationsRow.size != 0:
                task.putaijlist([bin + offset]*sumOfMultiplicationsRow.size, sumOfMultiplicationsRow,
                                                                        [-1/division]*sumOfMultiplicationsRow.size)

            task.putconbound(bin + offset, mosek.boundkey.up, -self.inf, ceil)

                # sumOfNewVariables <= numberOfUnaries
            task.putaijlist([bin + offset + numberOfBins] * numberOfUnaries, row, [1] * numberOfUnaries)

            task.putconbound(bin + offset + numberOfBins, mosek.boundkey.up, -self.inf, numberOfUnaries)

        newNofConstr = curNofConstr + 2 * numberOfBins + 3 * nSlackMult


        if withSymmetryConstr:

            for bin in range(0, numberOfBins):
                for unary in range(0, numberOfUnaries - 1):
                    offset = bin * (numberOfUnaries - 1) + unary

                    # (maximum[bin])[unary] >= (maximum[bin])[unary + 1])
                    task.putaij(newNofConstr + offset, newVariables[bin, unary], 1)
                    task.putaij(newNofConstr + offset, newVariables[bin, unary + 1], -1)

                    task.putconbound(newNofConstr + offset, mosek.boundkey.lo, 0, self.inf)

            newNofConstr += (numberOfUnaries - 1) * numberOfBins


        convolutionClass = RandomVariableMOSEK(newVariables, self.edges, task)

        return convolutionClass, newNofVariables, newNofConstr

    def convolution_UNARY_MAX_DIVIDE_VECTORIZED(self, secondVariable, curNofVariables, curNofConstr, withSymmetryConstr=False):
        """ Calculates convolution of 2 PDFs of random variable. Works only for 2 identical edges. Is computed
        using the unary representation of bins - M 0/1-bins for each bin. Unarization is kept using the divison.
        Is in MOSEK environment. Does not work

        :param self: class RandomVariableMOSEK
        :param secondVariable: class RandomVariableMOSEK
        :param curNofVariables: current number of MOSEK variables - to know the indices
        :param curNofConstr: current number of MOSEK constraints - to know the indices
        :param withSymmetryConstr: boolean whether a symmetry constraints should be included
        :return convolutionClass:  class RandomVariableMOSEK with cvxpy slack variables
        :return newNofVariables: integer, new total number of MOSEK variables
        :return newNofConstr: integer, new total number of MOSEK constraints
        """

        # get data
        x1 = self.bins
        x2 = secondVariable.bins
        numberOfBins, numberOfUnaries = x1.shape
        task = self.task
        newNofVariables = curNofVariables

        # create slack multiplications
        nSlackMult = int((numberOfBins / 2) * (1 + numberOfBins) * numberOfUnaries ** 2)
        slackMult = np.array(range(newNofVariables, newNofVariables + nSlackMult))
        newNofVariables += nSlackMult

        # create convolution variables
        nConvolutions = numberOfBins * numberOfUnaries
        convolution = np.array(range(newNofVariables, newNofVariables + nConvolutions))
        newNofVariables += nConvolutions
        convolutionBins = np.reshape(convolution, (numberOfBins, numberOfUnaries))  # reshape bins

        # append variables
        numberToCreate = nSlackMult + nConvolutions
        varsToCreateExtended = np.append(slackMult, convolution)
        task.appendvars(numberToCreate)
        task.putvartypelist(varsToCreateExtended, [mosek.variabletype.type_int] * numberToCreate)  # integer
        task.putvarboundlist(varsToCreateExtended, [mosek.boundkey.ra] * numberToCreate,
                             [0.0] * numberToCreate, [1.0] * numberToCreate)  # binary

        # convolution

            # allocate dict for sum of multiplications
        sumOfMultiplications = {}

        # append constraints
        if withSymmetryConstr:
            symN = (numberOfUnaries-1)*numberOfBins
        else:
            symN = 0

        task.appendcons(2 * numberOfBins + 3 * nSlackMult + symN)
        offset = curNofConstr + 2 * numberOfBins


        for z in range(0, numberOfBins):
            sumOfMultiplications[z] = np.array([]).astype(int)
            for k in range(0, z + 1):

                unary = np.array(range(0, numberOfUnaries))
                unary = np.transpose(np.tile(unary, (numberOfUnaries, 1)))
                unary2 = np.array(range(0, numberOfUnaries))
                unary2 = np.tile(unary2, (numberOfUnaries, 1))

                # constraints for sum of multiplications
                indexToSlack = ((z / 2) * (1 + z) * numberOfUnaries ** 2
                                   + k * numberOfUnaries ** 2 + unary * numberOfUnaries + unary2).astype(int)
                curSlackMult = np.concatenate(slackMult[indexToSlack])

                # task.putaij(z, curSlackMult, - 1 / division)
                indexToSlack = np.concatenate(indexToSlack)
                sumOfMultiplications[z] = np.append(sumOfMultiplications[z], np.array([curSlackMult]))

                # help constraints
                xIndex = np.concatenate(x1[k, unary])
                yIndex = np.concatenate(x2[z - k, unary2])

                numberOfElements = indexToSlack.size

                # slackMult <= x
                task.putaijlist(offset + 3 * indexToSlack, xIndex, [-1]*numberOfElements)
                task.putaijlist(offset + 3 * indexToSlack, curSlackMult, [1]*numberOfElements)

                task.putconboundlist(offset + 3 * indexToSlack, [mosek.boundkey.up]*numberOfElements
                                                , [-self.inf]*numberOfElements, [0]*numberOfElements)

                # slackMult <= y
                task.putaijlist(offset + 3 * indexToSlack + 1, yIndex, [-1]*numberOfElements)
                task.putaijlist(offset + 3 * indexToSlack + 1, curSlackMult, [1]*numberOfElements)

                task.putconboundlist(offset + 3 * indexToSlack + 1, [mosek.boundkey.up]*numberOfElements,
                                                        [-self.inf]*numberOfElements, [0]*numberOfElements)

                # slackMult >= x + y - 1
                task.putaijlist(offset + 3 * indexToSlack + 2, yIndex, [-1]*numberOfElements)
                task.putaijlist(offset + 3 * indexToSlack + 2, xIndex, [-1]*numberOfElements)
                task.putaijlist(offset + 3 * indexToSlack + 2, curSlackMult, [1]*numberOfElements)

                task.putconboundlist(offset + 3 * indexToSlack + 2, [mosek.boundkey.lo]*numberOfElements,
                                                        [-1]*numberOfElements, [self.inf]*numberOfElements)



        self.cutBins(self.edges, sumOfMultiplications)

        ceil = 0.999
        division = numberOfUnaries

        offset = curNofConstr

        for bin in range(0, numberOfBins):
            row = convolutionBins[bin, :]
            sumOfMultiplicationsRow = sumOfMultiplications[bin][:]

            nSums = sumOfMultiplicationsRow.size

            # sumOfNewVariables <= sumOfMultiplications[i] / divisor + ceil
            task.putaijlist([bin + offset] * numberOfUnaries, row, [1] * numberOfUnaries)
            if sumOfMultiplicationsRow.size != 0:
                task.putaijlist([bin + offset] * nSums, sumOfMultiplicationsRow, [-1/division] * nSums)

            task.putconbound(bin + offset, mosek.boundkey.up, -self.inf, ceil)

            # sumOfNewVariables <= numberOfUnaries
            task.putaijlist([bin + offset + numberOfBins] * numberOfUnaries, row, [1] * numberOfUnaries)
            task.putconbound(bin + offset + numberOfBins, mosek.boundkey.up, -self.inf, numberOfUnaries)

        newNofConstr = curNofConstr + 2*numberOfBins + 3*nSlackMult

        if withSymmetryConstr:

            for bin in range(0, numberOfBins):
                for unary in range(0, numberOfUnaries - 1):
                    offset = bin * (numberOfUnaries - 1) + unary

                    # (maximum[bin])[unary] >= (maximum[bin])[unary + 1])
                    task.putaij(newNofConstr + offset, convolutionBins[bin, unary], 1)
                    task.putaij(newNofConstr + offset, convolutionBins[bin, unary + 1], -1)

                    task.putconbound(newNofConstr + offset, mosek.boundkey.lo, 0, self.inf)

            newNofConstr += (numberOfUnaries - 1) * numberOfBins


        convolutionClass = RandomVariableMOSEK(convolutionBins, self.edges, task)

        return convolutionClass, newNofVariables, newNofConstr

    def maximum_UNARY_MAX_DIVIDE(self, secondVariable, curNofVariables, curNofConstr, withSymmetryConstr=False):
        """
        Calculates maximum of 2 PDFs of cvxpy variable. Works only for 2 identical edges. Is computed
        using the 'quadratic' algorithm and unary representation of bins - M 0/1-bins for each bin.
        Unarization is kept using the divison.
        Is in MOSEK environment.

        :param self: class RandomVariableMOSEK
        :param secondVariable: class RandomVariableMOSEK
        :param curNofVariables: current number of MOSEK variables - to know the indices
        :param curNofConstr: current number of MOSEK constraints - to know the indices
        :param withSymmetryConstr: boolean whether a symmetry constraints should be included

        :return maximumClass: class RandomVariableMOSEK with cvxpy slack variables (1, 1)
        :return newNofVariables: integer, new total number of MOSEK variables
        :return newNofConstr: integer, new total number of MOSEK constraints
        """

        # get data
        x1 = self.bins
        x2 = secondVariable.bins
        numberOfBins, numberOfUnaries = x1.shape
        task = self.task
        newNofVariables = curNofVariables


        # create slack multiplications
        nSlackMult = int((numberOfBins / 2) * (1 + numberOfBins) * numberOfUnaries ** 2)
        slackMult = np.array(range(newNofVariables, newNofVariables + nSlackMult))
        newNofVariables += nSlackMult

        # create second slack multiplications
        nSlackMult2 = int(((numberOfBins-1) / 2) * (1 + (numberOfBins-1)) * (numberOfUnaries) ** 2 )
        slackMult2 = np.array(range(newNofVariables, newNofVariables + nSlackMult2))
        newNofVariables += nSlackMult2

        # create convolution variables
        nMaximums = numberOfBins * numberOfUnaries
        maximum = np.array(range(newNofVariables, newNofVariables + nMaximums))
        newNofVariables += nMaximums
        maximumBins = np.reshape(maximum, (numberOfBins, numberOfUnaries))  # reshape bins


        # append variables
        numberToCreate = nSlackMult + nSlackMult2 + nMaximums
        varsToCreateExtended = np.append(slackMult, maximum)
        varsToCreateExtended = np.append(varsToCreateExtended, slackMult2)
        task.appendvars(numberToCreate)
        task.putvartypelist(varsToCreateExtended, [mosek.variabletype.type_int] * numberToCreate)  # integer
        task.putvarboundlist(varsToCreateExtended, [mosek.boundkey.ra] * numberToCreate,
                             [0.0] * numberToCreate, [1.0] * numberToCreate)  # binary


        # allocate dict for sum of multiplications
        sumOfMultiplications = {}

        # append constraints
        if withSymmetryConstr:
            symN = (numberOfUnaries-1)*numberOfBins
        else:
            symN = 0
        task.appendcons(2*numberOfBins + 3*nSlackMult + 3*nSlackMult2 + symN)

            # set offsets
        constrOffset = curNofConstr + 2 * numberOfBins
        constrOffset2 = constrOffset + 3*nSlackMult


        for i in range(0, numberOfBins):
            sumOfMultiplications[i] = np.array([]).astype(int)
            for j in range(0, i + 1):

                for unary in range(0, numberOfUnaries):
                    for unary2 in range(0, numberOfUnaries):

                        # constraints for sum of multiplications
                        indexToSlack = int((i / 2) * (1 + i) * numberOfUnaries ** 2
                                           + j * numberOfUnaries ** 2 + unary * numberOfUnaries + unary2)
                        curSlackMult = slackMult[indexToSlack]

                        sumOfMultiplications[i] = np.append(sumOfMultiplications[i], np.array([curSlackMult]))

                        # help constraints
                        xIndex = x1[i, unary]
                        yIndex = x2[j, unary2]

                        # slackMult <= x
                        task.putaij(constrOffset + 3 * indexToSlack, xIndex, -1)
                        task.putaij(constrOffset + 3 * indexToSlack, curSlackMult, 1)

                        task.putconbound(constrOffset + 3 * indexToSlack, mosek.boundkey.up, -self.inf, 0)

                        # slackMult <= y
                        task.putaij(constrOffset + 3 * indexToSlack + 1, yIndex, -1)
                        task.putaij(constrOffset + 3 * indexToSlack + 1, curSlackMult, 1)

                        task.putconbound(constrOffset + 3 * indexToSlack + 1, mosek.boundkey.up, -self.inf, 0)

                        # slackMult >= x + y - 1
                        task.putaij(constrOffset + 3 * indexToSlack + 2, yIndex, -1)
                        task.putaij(constrOffset + 3 * indexToSlack + 2, xIndex, -1)
                        task.putaij(constrOffset + 3 * indexToSlack + 2, curSlackMult, 1)

                        task.putconbound(constrOffset + 3 * indexToSlack + 2, mosek.boundkey.lo, -1, self.inf)


                        if i != j:
                            # constraints for sum of multiplications
                            # print(slackMult2Offset)
                            indexToSlack = int(((i-1) / 2) * (1 + i-1) * numberOfUnaries ** 2
                                               + j * numberOfUnaries ** 2 + unary * numberOfUnaries + unary2)
                            curSlackMult = slackMult2[indexToSlack]


                            sumOfMultiplications[i] = np.append(sumOfMultiplications[i], np.array([curSlackMult]))

                            # help constraints
                            xIndex = x1[j, unary]
                            yIndex = x2[i, unary2]

                            # slackMult <= x
                            task.putaij(constrOffset2 + 3 * indexToSlack, xIndex, -1)
                            task.putaij(constrOffset2 + 3 * indexToSlack, curSlackMult, 1)

                            task.putconbound(constrOffset2 + 3 * indexToSlack, mosek.boundkey.up, -self.inf, 0)

                            # slackMult <= y
                            task.putaij(constrOffset2 + 3 * indexToSlack + 1, yIndex, -1)
                            task.putaij(constrOffset2 + 3 * indexToSlack + 1, curSlackMult, 1)

                            task.putconbound(constrOffset2 + 3 * indexToSlack + 1, mosek.boundkey.up, -self.inf, 0)

                            # slackMult >= x + y - 1
                            task.putaij(constrOffset2 + 3 * indexToSlack + 2, yIndex, -1)
                            task.putaij(constrOffset2 + 3 * indexToSlack + 2, xIndex, -1)
                            task.putaij(constrOffset2 + 3 * indexToSlack + 2, curSlackMult, 1)

                            task.putconbound(constrOffset2 + 3 * indexToSlack + 2, mosek.boundkey.lo, -1, self.inf)


        division = numberOfUnaries**2
        ceil = 0.999

        offset = curNofConstr

        for bin in range(0, numberOfBins):
            row = maximumBins[bin, :]
            sumOfMultiplicationsRow = sumOfMultiplications[bin][:]

            nSums = sumOfMultiplicationsRow.size

            # sumOfNewVariables <= sumOfMultiplications[i] / divisor + ceil
            task.putaijlist([bin + offset] * numberOfUnaries, row, [1] * numberOfUnaries)
            if nSums != 0:
                task.putaijlist([bin + offset] * nSums, sumOfMultiplicationsRow, [-1 / division] * nSums)

            task.putconbound(bin + offset, mosek.boundkey.up, -self.inf, ceil)

            # sumOfNewVariables <= numberOfUnaries
            task.putaijlist([bin + offset + numberOfBins] * numberOfUnaries, row, [1] * numberOfUnaries)
            task.putconbound(bin + offset + numberOfBins, mosek.boundkey.up, -self.inf, numberOfUnaries)



        newNofConstr = curNofConstr + 2 *numberOfBins + 3 *nSlackMult + 3*nSlackMult2

        if withSymmetryConstr:

            for bin in range(0, numberOfBins):
                for unary in range(0, numberOfUnaries-1):
                    offset = bin*(numberOfUnaries-1) + unary

                        #(maximum[bin])[unary] >= (maximum[bin])[unary + 1])
                    task.putaij(newNofConstr + offset, maximumBins[bin, unary], 1)
                    task.putaij(newNofConstr + offset, maximumBins[bin, unary+1], -1)

                    task.putconbound(newNofConstr + offset, mosek.boundkey.lo, 0, self.inf)


            newNofConstr += (numberOfUnaries-1)*numberOfBins


        maximumClass = RandomVariableMOSEK(maximumBins, self.edges, task)

        return maximumClass, newNofVariables, newNofConstr

    def maximum_UNARY_MAX_DIVIDE_VECTORIZED(self, secondVariable, curNofVariables, curNofConstr, withSymmetryConstr=False):
        """ Calculates maximum of 2 PDFs of random variable. Works only for 2 identical edges. Is computed
        using the unary representation of bins - M 0/1-bins for each bin. Unarization is kept using the divison.
        Is in MOSEK environment.

        :param self: class RandomVariableMOSEK
        :param secondVariable: class RandomVariableMOSEK
        :param curNofVariables: current number of MOSEK variables - to know the indices
        :param curNofConstr: current number of MOSEK constraints - to know the indices
        :param withSymmetryConstr: boolean whether a symmetry constraints should be included
        :return maximumClass:  class RandomVariableMOSEK with cvxpy slack variables
        :return newNofVariables: integer, new total number of MOSEK variables
        :return newNofConstr: integer, new total number of MOSEK constraints
        """

        # get data
        x1 = self.bins
        x2 = secondVariable.bins
        numberOfBins, numberOfUnaries = x1.shape
        task = self.task
        newNofVariables = curNofVariables

        # create slack multiplications
        nSlackMult = int((numberOfBins / 2) * (1 + numberOfBins) * numberOfUnaries ** 2)
        slackMult = np.array(range(newNofVariables, newNofVariables + nSlackMult))
        newNofVariables += nSlackMult

        # create second slack multiplications
        nSlackMult2 = int(((numberOfBins - 1) / 2) * (1 + (numberOfBins - 1)) * (numberOfUnaries) ** 2)
        slackMult2 = np.array(range(newNofVariables, newNofVariables + nSlackMult2))
        newNofVariables += nSlackMult2

        # create convolution variables
        nMaximums = numberOfBins * numberOfUnaries
        maximum = np.array(range(newNofVariables, newNofVariables + nMaximums))
        newNofVariables += nMaximums
        maximumBins = np.reshape(maximum, (numberOfBins, numberOfUnaries))  # reshape bins

        # append variables
        numberToCreate = nSlackMult + nSlackMult2 + nMaximums
        varsToCreateExtended = np.append(slackMult, maximum)
        varsToCreateExtended = np.append(varsToCreateExtended, slackMult2)
        task.appendvars(numberToCreate)
        task.putvartypelist(varsToCreateExtended, [mosek.variabletype.type_int] * numberToCreate)  # integer
        task.putvarboundlist(varsToCreateExtended, [mosek.boundkey.ra] * numberToCreate,
                             [0.0] * numberToCreate, [1.0] * numberToCreate)  # binary

        # allocate dict for sum of multiplications
        sumOfMultiplications = {}

        # append sumOfMultiplications constraints
        if withSymmetryConstr:
            symN = (numberOfUnaries-1)*numberOfBins
        else:
            symN = 0
        task.appendcons(2 * numberOfBins + 3 * nSlackMult + 3 * nSlackMult2 + symN)

        # set offsets
        constrOffset = curNofConstr + 2 * numberOfBins
        constrOffset2 = constrOffset + 3 * nSlackMult


        for i in range(0, numberOfBins):
            sumOfMultiplications[i] = np.array([]).astype(int)
            for j in range(0, i + 1):

                unary = np.array(range(0, numberOfUnaries))
                unary = np.transpose(np.tile(unary, (numberOfUnaries, 1)))
                unary2 = np.array(range(0, numberOfUnaries))
                unary2 = np.tile(unary2, (numberOfUnaries, 1))

                # constraints for sum of multiplications
                indexToSlack = ((i / 2) * (1 + i) * numberOfUnaries ** 2
                                   + i * numberOfUnaries ** 2 + unary * numberOfUnaries + unary2).astype(int)
                curSlackMult = np.concatenate(slackMult[indexToSlack])

                indexToSlack = np.concatenate(indexToSlack)
                sumOfMultiplications[i] = np.append(sumOfMultiplications[i], np.array([curSlackMult]))

                # help constraints
                xIndex = np.concatenate(x1[i, unary])
                yIndex = np.concatenate(x2[j, unary2])

                numberOfElements = indexToSlack.size

                # slackMult <= x
                task.putaijlist(constrOffset + 3 * indexToSlack, xIndex, [-1]*numberOfElements)
                task.putaijlist(constrOffset + 3 * indexToSlack, curSlackMult, [1]*numberOfElements)

                task.putconboundlist(constrOffset + 3 * indexToSlack, [mosek.boundkey.up]*numberOfElements
                                                , [-self.inf]*numberOfElements, [0]*numberOfElements)

                # slackMult <= y
                task.putaijlist(constrOffset + 3 * indexToSlack + 1, yIndex, [-1]*numberOfElements)
                task.putaijlist(constrOffset + 3 * indexToSlack + 1, curSlackMult, [1]*numberOfElements)

                task.putconboundlist(constrOffset + 3 * indexToSlack + 1, [mosek.boundkey.up]*numberOfElements,
                                                        [-self.inf]*numberOfElements, [0]*numberOfElements)

                # slackMult >= x + y - 1
                task.putaijlist(constrOffset + 3 * indexToSlack + 2, yIndex, [-1]*numberOfElements)
                task.putaijlist(constrOffset + 3 * indexToSlack + 2, xIndex, [-1]*numberOfElements)
                task.putaijlist(constrOffset + 3 * indexToSlack + 2, curSlackMult, [1]*numberOfElements)

                task.putconboundlist(constrOffset + 3 * indexToSlack + 2, [mosek.boundkey.lo]*numberOfElements,
                                                        [-1]*numberOfElements, [self.inf]*numberOfElements)

                if i != j:

                    # constraints for sum of multiplications
                    indexToSlack = (((i-1) / 2) * (1 + i-1) * numberOfUnaries ** 2
                                           + j * numberOfUnaries ** 2 + unary * numberOfUnaries + unary2).astype(int)
                    curSlackMult = slackMult2[indexToSlack]

                    sumOfMultiplications[i] = np.append(sumOfMultiplications[i], np.array([curSlackMult]))
                    curSlackMult = np.concatenate(slackMult2[indexToSlack])

                    indexToSlack = np.concatenate(indexToSlack)
                    sumOfMultiplications[i] = np.append(sumOfMultiplications[i], np.array([curSlackMult]))

                    # help constraints
                    xIndex = np.concatenate(x1[j, unary])
                    yIndex = np.concatenate(x2[i, unary2])

                    numberOfElements = indexToSlack.size

                    # slackMult <= x
                    task.putaijlist(constrOffset2 + 3 * indexToSlack, xIndex, [-1] * numberOfElements)
                    task.putaijlist(constrOffset2 + 3 * indexToSlack, curSlackMult, [1] * numberOfElements)

                    task.putconboundlist(constrOffset2 + 3 * indexToSlack, [mosek.boundkey.up] * numberOfElements
                                         , [-self.inf] * numberOfElements, [0] * numberOfElements)

                    # slackMult <= y
                    task.putaijlist(constrOffset2 + 3 * indexToSlack + 1, yIndex, [-1] * numberOfElements)
                    task.putaijlist(constrOffset2 + 3 * indexToSlack + 1, curSlackMult, [1] * numberOfElements)

                    task.putconboundlist(constrOffset2 + 3 * indexToSlack + 1, [mosek.boundkey.up] * numberOfElements,
                                         [-self.inf] * numberOfElements, [0] * numberOfElements)

                    # slackMult >= x + y - 1
                    task.putaijlist(constrOffset2 + 3 * indexToSlack + 2, yIndex, [-1] * numberOfElements)
                    task.putaijlist(constrOffset2 + 3 * indexToSlack + 2, xIndex, [-1] * numberOfElements)
                    task.putaijlist(constrOffset2 + 3 * indexToSlack + 2, curSlackMult, [1] * numberOfElements)

                    task.putconboundlist(constrOffset2 + 3 * indexToSlack + 2, [mosek.boundkey.lo] * numberOfElements,
                                         [-1] * numberOfElements, [self.inf] * numberOfElements)



        division = numberOfUnaries**2
        ceil = 0.999

        offset = curNofConstr

        for bin in range(0, numberOfBins):
            row = maximumBins[bin, :]
            sumOfMultiplicationsRow = sumOfMultiplications[bin][:]

            nSums = sumOfMultiplicationsRow.size

            # sumOfNewVariables <= sumOfMultiplications[i] / divisor + ceil
            task.putaijlist([bin + offset] * numberOfUnaries, row, [1] * numberOfUnaries)
            if nSums != 0:
                task.putaijlist([bin + offset] * nSums, sumOfMultiplicationsRow, [-1 / division] * nSums)

            task.putconbound(bin + offset, mosek.boundkey.up, -self.inf, ceil)

            # sumOfNewVariables <= numberOfUnaries
            task.putaijlist([bin + offset + numberOfBins] * numberOfUnaries, row, [1] * numberOfUnaries)
            task.putconbound(bin + offset + numberOfBins, mosek.boundkey.up, -self.inf, numberOfUnaries)



        newNofConstr = curNofConstr + 2 * numberOfBins + 3 * nSlackMult + 3 * nSlackMult2

        if withSymmetryConstr:

            for bin in range(0, numberOfBins):
                for unary in range(0, numberOfUnaries - 1):
                    offset = bin * (numberOfUnaries - 1) + unary

                    # (maximum[bin])[unary] >= (maximum[bin])[unary + 1])
                    task.putaij(newNofConstr + offset, maximumBins[bin, unary], 1)
                    task.putaij(newNofConstr + offset, maximumBins[bin, unary + 1], -1)

                    task.putconbound(newNofConstr + offset, mosek.boundkey.lo, 0, self.inf)

            newNofConstr += (numberOfUnaries - 1) * numberOfBins


        maximumClass = RandomVariableMOSEK(maximumBins, self.edges, task)
        return maximumClass, newNofVariables, newNofConstr


    @staticmethod
    def cutBins(edges: np.array, bins: {}):
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
        numberOfBins = len(edges) - 1

        numberOfBinsNeeded = math.floor(abs(edges[0]) / diff)

        if edges[0] > 0:  # cut bins

            for i in range(numberOfBinsNeeded, numberOfBins):
                bins[i] = bins[i - numberOfBinsNeeded]

            for i in range(0, numberOfBinsNeeded):
                bins[i] = np.array([]).astype(int)

        if edges[0] < 0:  # cut bins

            for i in range(numberOfBinsNeeded, numberOfBins):
                bins[i - numberOfBinsNeeded] = bins[i]

            for i in range(numberOfBins - numberOfBinsNeeded, numberOfBins):
                bins[i] = np.array([]).astype(int)








