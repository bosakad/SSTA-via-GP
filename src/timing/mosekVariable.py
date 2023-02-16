import math
import mosek
import numpy as np

"""
  This module includes same functions as module 'cvxpyVariable'. Is coded in MOSEK API for speed-up purposes.
  Unary variable is represented by numpy [n, m], where n is number of bins, m is number of unaries. For a MOSEK 
  variable bins is represented by [n, m] list of indices. Does not include the GP model - only mixed-integer model.
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

        self.inf = 0.0  # symbolic purposes

    def maximum_AND_Convolution(
        self, secondVariable, thirdVariable, curNofVariables, curNofConstr
    ):
        """Calculates maximum of 2 PDFs of random variable and a convolution with the third afterwards.
        Works only for 2 identical edges. Is computed
        using the unary representation of bins - M 0/1-bins for each bin. Unarization is kept using the divison.
        Is in MOSEK environment. Works only for maximization problem

        :param self: class RandomVariableMOSEK - one of maximums
        :param secondVariable: class RandomVariableMOSEK - one of maximums
        :param thirdVariable: class RandomVariableMOSEK - one to convolve
        :param curNofVariables: current number of MOSEK variables - to know the indices
        :param curNofConstr: current number of MOSEK constraints - to know the indices
        :return resultClass:  class RandomVariableMOSEK with mosek variables
        :return newNofVariables: integer, new total number of MOSEK variables
        :return newNofConstr: integer, new total number of MOSEK constraints
        """

        # get data
        x1 = self.bins
        x2 = secondVariable.bins
        numberOfBins, numberOfUnaries = x1.shape

        # allocate dict for sum of multiplications
        sumOfMaxs = {}

        for i in range(0, numberOfBins):
            sumOfMaxs[i] = []
            for j in range(0, i + 1):

                for unary in range(0, numberOfUnaries):
                    for unary2 in range(0, numberOfUnaries):

                        # help constraints
                        xIndex = x1[i, unary]
                        yIndex = x2[j, unary2]

                        # save the multiplication for later
                        sumOfMaxs[i].append([xIndex, yIndex])

                        if i != j:
                            # help constraints
                            xIndex = x1[j, unary]
                            yIndex = x2[i, unary2]

                            # save the multiplication for later
                            sumOfMaxs[i].append([xIndex, yIndex])

        ############# CONVOLUTION ##################

        # get data
        x3 = thirdVariable.bins
        task = self.task
        newNofVariables = curNofVariables

        sum = 0
        for z in range(0, numberOfBins):
            for k in range(0, z + 1):
                sum += numberOfUnaries * len(sumOfMaxs[z - k])

        # create auxiliary multiplications
        nAuxMult = int(sum)
        auxMult = np.array(range(newNofVariables, newNofVariables + nAuxMult))
        newNofVariables += nAuxMult

        # create result variables
        nResults = numberOfBins * numberOfUnaries
        convolution = np.array(range(newNofVariables, newNofVariables + nResults))
        newNofVariables += nResults
        convolutionBins = np.reshape(
            convolution, (numberOfBins, numberOfUnaries)
        )  # reshape bins

        # append variables
        numberToCreate = nAuxMult + nResults
        varsToCreateExtended = np.append(auxMult, convolution)
        task.appendvars(numberToCreate)
        task.putvartypelist(
            varsToCreateExtended, [mosek.variabletype.type_int] * numberToCreate
        )  # integer
        task.putvarboundlist(
            varsToCreateExtended,
            [mosek.boundkey.ra] * numberToCreate,
            [0.0] * numberToCreate,
            [1.0] * numberToCreate,
        )  # binary

        # convolution

        # allocate dict for sum of multiplications
        sumOfConvs = {}

        # append constraints
        symN = (numberOfUnaries - 1) * numberOfBins
        task.appendcons(numberOfBins + 4 * nAuxMult + symN)

        offset = curNofConstr + numberOfBins

        indexToAux = 0

        for z in range(0, numberOfBins):
            sumOfConvs[z] = np.array([]).astype(int)
            for k in range(0, z + 1):

                for unary in range(0, numberOfUnaries):
                    for unary2 in range(0, len(sumOfMaxs[z - k])):
                        curAuxMult = auxMult[indexToAux]
                        indexToAux += 1
                        sumOfConvs[z] = np.append(sumOfConvs[z], np.array([curAuxMult]))

                        # help constraints
                        xIndex = x3[k, unary]
                        yIndex = sumOfMaxs[z - k][unary2][0]
                        zIndex = sumOfMaxs[z - k][unary2][1]

                        # slackMult <= x
                        task.putaij(offset + 4 * indexToAux, xIndex, -1)
                        task.putaij(offset + 4 * indexToAux, curAuxMult, 1)

                        task.putconbound(
                            offset + 4 * indexToAux, mosek.boundkey.up, -self.inf, 0
                        )

                        # slackMult <= y
                        task.putaij(offset + 4 * indexToAux + 1, yIndex, -1)
                        task.putaij(offset + 4 * indexToAux + 1, curAuxMult, 1)

                        task.putconbound(
                            offset + 4 * indexToAux + 1, mosek.boundkey.up, -self.inf, 0
                        )

                        # slackMult <= z
                        task.putaij(offset + 4 * indexToAux + 2, zIndex, -1)
                        task.putaij(offset + 4 * indexToAux + 2, curAuxMult, 1)

                        task.putconbound(
                            offset + 4 * indexToAux + 2, mosek.boundkey.up, -self.inf, 0
                        )

                        # slackMult >= x + y + z - 2

                        task.putaij(offset + 4 * indexToAux + 3, yIndex, -1)
                        task.putaij(offset + 4 * indexToAux + 3, xIndex, -1)
                        task.putaij(offset + 4 * indexToAux + 3, zIndex, -1)
                        task.putaij(offset + 4 * indexToAux + 3, curAuxMult, 1)

                        task.putconbound(
                            offset + 4 * indexToAux + 3, mosek.boundkey.lo, -2, self.inf
                        )

        self.cutBins(self.edges, sumOfConvs)

        roundScalar = 0.5
        # division = numberOfUnaries * numberOfBins / 14
        division = 3

        offset = curNofConstr

        for bin in range(0, numberOfBins):
            row = convolutionBins[bin, :]
            sumOfMultiplicationsRow = sumOfConvs[bin][:]

            nSums = sumOfMultiplicationsRow.size

            # sumOfNewVariables <= sumOfMultiplications[i] / divisor + ceil
            task.putaijlist(
                [bin + offset] * numberOfUnaries, row, [1] * numberOfUnaries
            )
            if sumOfMultiplicationsRow.size != 0:
                task.putaijlist(
                    [bin + offset] * nSums,
                    sumOfMultiplicationsRow,
                    [-1 / division] * nSums,
                )

            task.putconbound(bin + offset, mosek.boundkey.up, -self.inf, roundScalar)

        newNofConstr = curNofConstr + numberOfBins + 4 * nAuxMult

        # separation constraints
        for bin in range(0, numberOfBins):
            for unary in range(0, numberOfUnaries - 1):
                offset = bin * (numberOfUnaries - 1) + unary

                # (maximum[bin])[unary] >= (maximum[bin])[unary + 1])
                task.putaij(newNofConstr + offset, convolutionBins[bin, unary], 1)
                task.putaij(newNofConstr + offset, convolutionBins[bin, unary + 1], -1)

                task.putconbound(newNofConstr + offset, mosek.boundkey.lo, 0, self.inf)

        newNofConstr += (numberOfUnaries - 1) * numberOfBins

        resultClass = RandomVariableMOSEK(convolutionBins, self.edges, task)

        return resultClass, newNofVariables, newNofConstr

    def maximum_AND_Convolution_VECTORIZED(
        self, secondVariable, thirdVariable, curNofVariables, curNofConstr
    ):
        """Calculates maximum of 2 PDFs of random variable and a convolution with the third afterwards.
        Works only for 2 identical edges. Is computed
        using the unary representation of bins - M 0/1-bins for each bin. Unarization is kept using the divison.
        Is in MOSEK environment. Works only for maximization problem

        :param self: class RandomVariableMOSEK - one of maximums
        :param secondVariable: class RandomVariableMOSEK - one of maximums
        :param thirdVariable: class RandomVariableMOSEK - one to convolve
        :param curNofVariables: current number of MOSEK variables - to know the indices
        :param curNofConstr: current number of MOSEK constraints - to know the indices
        :return resultClass:  class RandomVariableMOSEK with mosek variables
        :return newNofVariables: integer, new total number of MOSEK variables
        :return newNofConstr: integer, new total number of MOSEK constraints
        """

        # get data
        x1 = self.bins
        x2 = secondVariable.bins
        numberOfBins, numberOfUnaries = x1.shape

        # allocate dict for sum of multiplications
        sumOfMaxs = {}

        def myZip(x, y):
            return [x, y]

        for i in range(0, numberOfBins):
            sumOfMaxs[i] = []
            for j in range(0, i + 1):

                # help constraints

                unary = np.array(range(0, numberOfUnaries))
                unary = np.transpose(np.tile(unary, (numberOfUnaries, 1)))
                unary2 = np.array(range(0, numberOfUnaries))
                unary2 = np.tile(unary2, (numberOfUnaries, 1))

                # help constraints
                xIndex = np.concatenate(x1[i, unary])
                yIndex = np.concatenate(x2[j, unary2])

                xy = list(map(myZip, xIndex, yIndex))

                # save the multiplication for later
                sumOfMaxs[i].extend(xy)

                if i != j:
                    # help constraints
                    xIndex = np.concatenate(x1[j, unary])
                    yIndex = np.concatenate(x2[i, unary2])

                    xy = list(map(myZip, xIndex, yIndex))

                    # save the multiplication for later
                    sumOfMaxs[i].extend(xy)

        ############# CONVOLUTION ##################

        # get data
        x3 = thirdVariable.bins
        task = self.task
        newNofVariables = curNofVariables

        sum = 0
        for z in range(0, numberOfBins):
            for k in range(0, z + 1):
                sumOfMaxs[z - k] = np.array(sumOfMaxs[z - k])
                sum += numberOfUnaries * sumOfMaxs[z - k].shape[0]

        # create auxiliary multiplications
        nAuxMult = int(sum)
        auxMult = np.array(range(newNofVariables, newNofVariables + nAuxMult))
        newNofVariables += nAuxMult

        # create result variables
        nResults = numberOfBins * numberOfUnaries
        convolution = np.array(range(newNofVariables, newNofVariables + nResults))
        newNofVariables += nResults
        convolutionBins = np.reshape(
            convolution, (numberOfBins, numberOfUnaries)
        )  # reshape bins

        # append variables
        numberToCreate = nAuxMult + nResults
        varsToCreateExtended = np.append(auxMult, convolution)
        task.appendvars(numberToCreate)
        task.putvartypelist(
            varsToCreateExtended, [mosek.variabletype.type_int] * numberToCreate
        )  # integer
        task.putvarboundlist(
            varsToCreateExtended,
            [mosek.boundkey.ra] * numberToCreate,
            [0.0] * numberToCreate,
            [1.0] * numberToCreate,
        )  # binary

        # convolution

        # allocate dict for sum of multiplications
        sumOfConvs = {}

        # append constraints
        symN = (numberOfUnaries - 1) * numberOfBins
        task.appendcons(numberOfBins + 4 * nAuxMult + symN)

        offset = curNofConstr + numberOfBins

        blockIndex = 0

        for z in range(0, numberOfBins):
            sumOfConvs[z] = np.array([]).astype(int)
            for k in range(0, z + 1):
                size = sumOfMaxs[z - k].shape[0]

                unary = np.array(range(0, numberOfUnaries))
                unary = np.transpose(np.tile(unary, (size, 1)))
                unary2 = np.array(range(0, size))
                unary2 = np.tile(unary2, (numberOfUnaries, 1))

                index = blockIndex + unary * size + unary2

                curAuxMult = np.concatenate(auxMult[index])

                index = np.concatenate(index)
                sumOfConvs[z] = np.append(sumOfConvs[z], np.array([curAuxMult]))

                # help constraints
                xIndex = np.concatenate(x3[k, unary])
                yIndex = np.concatenate(sumOfMaxs[z - k][unary2][:, :, 0])
                zIndex = np.concatenate(sumOfMaxs[z - k][unary2][:, :, 1])

                numberOfElements = index.size

                # slackMult <= x
                # task.putaijlist([1, 0], [0, 0], [-1, 1])
                task.putaijlist(offset + 4 * index, xIndex, [-1] * numberOfElements)
                task.putaijlist(offset + 4 * index, curAuxMult, [1] * numberOfElements)

                task.putconboundlist(
                    offset + 4 * index,
                    [mosek.boundkey.up] * numberOfElements,
                    [-self.inf] * numberOfElements,
                    [0] * numberOfElements,
                )

                # slackMult <= y
                task.putaijlist(offset + 4 * index + 1, yIndex, [-1] * numberOfElements)
                task.putaijlist(
                    offset + 4 * index + 1, curAuxMult, [1] * numberOfElements
                )

                task.putconboundlist(
                    offset + 4 * index + 1,
                    [mosek.boundkey.up] * numberOfElements,
                    [-self.inf] * numberOfElements,
                    [0] * numberOfElements,
                )

                # slackMult <= z
                task.putaijlist(offset + 4 * index + 2, zIndex, [-1] * numberOfElements)
                task.putaijlist(
                    offset + 4 * index + 2, curAuxMult, [1] * numberOfElements
                )

                task.putconboundlist(
                    offset + 4 * index + 2,
                    [mosek.boundkey.up] * numberOfElements,
                    [-self.inf] * numberOfElements,
                    [0] * numberOfElements,
                )

                # slackMult >= x + y + z - 2

                task.putaijlist(offset + 4 * index + 3, yIndex, [-1] * numberOfElements)
                task.putaijlist(offset + 4 * index + 3, xIndex, [-1] * numberOfElements)
                task.putaijlist(offset + 4 * index + 3, zIndex, [-1] * numberOfElements)
                task.putaijlist(
                    offset + 4 * index + 3, curAuxMult, [1] * numberOfElements
                )

                task.putconboundlist(
                    offset + 4 * index + 3,
                    [mosek.boundkey.lo] * numberOfElements,
                    [-2] * numberOfElements,
                    [self.inf] * numberOfElements,
                )

                blockIndex += size * numberOfUnaries

        self.cutBins(self.edges, sumOfConvs)

        roundScalar = 0.5
        division = numberOfUnaries * numberOfBins / 14
        # division = 54
        # division = 50

        offset = curNofConstr

        for bin in range(0, numberOfBins):
            row = convolutionBins[bin, :]
            sumOfMultiplicationsRow = sumOfConvs[bin][:]

            nSums = sumOfMultiplicationsRow.size

            # sumOfNewVariables <= sumOfMultiplications[i] / divisor + ceil
            task.putaijlist(
                [bin + offset] * numberOfUnaries, row, [1] * numberOfUnaries
            )
            if sumOfMultiplicationsRow.size != 0:
                task.putaijlist(
                    [bin + offset] * nSums,
                    sumOfMultiplicationsRow,
                    [-1 / division] * nSums,
                )

            task.putconbound(bin + offset, mosek.boundkey.up, -self.inf, roundScalar)
            # task.putconbound(bin + offset, mosek.boundkey.fx, roundScalar, roundScalar)

        newNofConstr = curNofConstr + numberOfBins + 4 * nAuxMult

        # symmetry constraints
        for bin in range(0, numberOfBins):
            for unary in range(0, numberOfUnaries - 1):
                offset = bin * (numberOfUnaries - 1) + unary

                # (maximum[bin])[unary] >= (maximum[bin])[unary + 1])
                task.putaij(newNofConstr + offset, convolutionBins[bin, unary], 1)
                task.putaij(newNofConstr + offset, convolutionBins[bin, unary + 1], -1)

                task.putconbound(newNofConstr + offset, mosek.boundkey.lo, 0, self.inf)

        newNofConstr += (numberOfUnaries - 1) * numberOfBins

        resultClass = RandomVariableMOSEK(convolutionBins, self.edges, task)

        return resultClass, newNofVariables, newNofConstr

    def maximum_AND_Convolution_VECTORIZED_MIN(
        self, secondVariable, thirdVariable, curNofVariables, curNofConstr, VAR=False
    ):
        """Calculates maximum of 2 PDFs of random variable and a convolution with the third afterwards.
        Works only for 2 identical edges. Is computed
        using the unary representation of bins - M 0/1-bins for each bin. Unarization is kept using the divison.
        Is in MOSEK environment. This implementation works only for a minimization problem.

        :param self: class RandomVariableMOSEK - one of maximums
        :param secondVariable: class RandomVariableMOSEK - one of maximums
        :param thirdVariable: class RandomVariableMOSEK - one to convolve
        :param curNofVariables: current number of MOSEK variables - to know the indices
        :param curNofConstr: current number of MOSEK constraints - to know the indices
        :return resultClass:  class RandomVariableMOSEK with mosek variables
        :return newNofVariables: integer, new total number of MOSEK variables
        :return newNofConstr: integer, new total number of MOSEK constraints
        """

        # get data
        x1 = self.bins
        x2 = secondVariable.bins
        numberOfBins, numberOfUnaries = x1.shape

        # allocate dict for sum of multiplications
        sumOfMaxs = {}

        def myZip(x, y):
            return [x, y]

        for i in range(0, numberOfBins):
            sumOfMaxs[i] = []
            for j in range(0, i + 1):

                # help constraints

                unary = np.array(range(0, numberOfUnaries))
                unary = np.transpose(np.tile(unary, (numberOfUnaries, 1)))
                unary2 = np.array(range(0, numberOfUnaries))
                unary2 = np.tile(unary2, (numberOfUnaries, 1))

                # help constraints
                xIndex = np.concatenate(x1[i, unary])
                yIndex = np.concatenate(x2[j, unary2])

                xy = list(map(myZip, xIndex, yIndex))

                # save the multiplication for later
                sumOfMaxs[i].extend(xy)

                if i != j:
                    # help constraints
                    xIndex = np.concatenate(x1[j, unary])
                    yIndex = np.concatenate(x2[i, unary2])

                    xy = list(map(myZip, xIndex, yIndex))

                    # save the multiplication for later
                    sumOfMaxs[i].extend(xy)

        ############# CONVOLUTION ##################

        # get data
        x3 = thirdVariable.bins
        task = self.task
        newNofVariables = curNofVariables

        sum = 0
        for z in range(0, numberOfBins):
            for k in range(0, z + 1):
                sumOfMaxs[z - k] = np.array(sumOfMaxs[z - k])
                sum += numberOfUnaries * sumOfMaxs[z - k].shape[0]

        # create auxiliary multiplications
        nAuxMult = int(sum)
        auxMult = np.array(range(newNofVariables, newNofVariables + nAuxMult))
        newNofVariables += nAuxMult

        # create result variables
        nResults = numberOfBins * numberOfUnaries
        convolution = np.array(range(newNofVariables, newNofVariables + nResults))
        newNofVariables += nResults
        convolutionBins = np.reshape(
            convolution, (numberOfBins, numberOfUnaries)
        )  # reshape bins

        # append variables
        numberToCreate = nAuxMult + nResults
        varsToCreateExtended = np.append(auxMult, convolution)
        task.appendvars(numberToCreate)
        task.putvartypelist(
            varsToCreateExtended, [mosek.variabletype.type_int] * numberToCreate
        )  # integer
        task.putvarboundlist(
            varsToCreateExtended,
            [mosek.boundkey.ra] * numberToCreate,
            [0.0] * numberToCreate,
            [1.0] * numberToCreate,
        )  # binary

        # convolution

        # allocate dict for sum of multiplications
        sumOfConvs = {}

        # append constraints
        symN = (numberOfUnaries - 1) * numberOfBins
        if VAR:
            task.appendcons(2 * numberOfBins + 4 * nAuxMult + symN)
        else:
            task.appendcons(2 * numberOfBins + nAuxMult + symN)

        offset = curNofConstr + numberOfBins

        blockIndex = 0

        for z in range(0, numberOfBins):
            sumOfConvs[z] = np.array([]).astype(int)
            for k in range(0, z + 1):

                size = sumOfMaxs[z - k].shape[0]

                unary = np.array(range(0, numberOfUnaries))
                unary = np.transpose(np.tile(unary, (size, 1)))
                unary2 = np.array(range(0, size))
                unary2 = np.tile(unary2, (numberOfUnaries, 1))

                index = blockIndex + unary * size + unary2

                curAuxMult = np.concatenate(auxMult[index])

                index = np.concatenate(index)
                sumOfConvs[z] = np.append(sumOfConvs[z], np.array([curAuxMult]))

                # help constraints
                xIndex = np.concatenate(x3[k, unary])
                yIndex = np.concatenate(sumOfMaxs[z - k][unary2][:, :, 0])
                zIndex = np.concatenate(sumOfMaxs[z - k][unary2][:, :, 1])

                numberOfElements = index.size

                if VAR:
                    # slackMult <= x
                    task.putaijlist(offset + 4 * index, xIndex, [-1] * numberOfElements)
                    task.putaijlist(
                        offset + 4 * index, curAuxMult, [1] * numberOfElements
                    )

                    task.putconboundlist(
                        offset + 4 * index,
                        [mosek.boundkey.up] * numberOfElements,
                        [-self.inf] * numberOfElements,
                        [0] * numberOfElements,
                    )

                    # slackMult <= y
                    task.putaijlist(
                        offset + 4 * index + 1, yIndex, [-1] * numberOfElements
                    )
                    task.putaijlist(
                        offset + 4 * index + 1, curAuxMult, [1] * numberOfElements
                    )

                    task.putconboundlist(
                        offset + 4 * index + 1,
                        [mosek.boundkey.up] * numberOfElements,
                        [-self.inf] * numberOfElements,
                        [0] * numberOfElements,
                    )

                    # slackMult <= z
                    task.putaijlist(
                        offset + 4 * index + 2, zIndex, [-1] * numberOfElements
                    )
                    task.putaijlist(
                        offset + 4 * index + 2, curAuxMult, [1] * numberOfElements
                    )

                    task.putconboundlist(
                        offset + 4 * index + 2,
                        [mosek.boundkey.up] * numberOfElements,
                        [-self.inf] * numberOfElements,
                        [0] * numberOfElements,
                    )

                    # slackMult >= x + y + z - 2

                    task.putaijlist(
                        offset + 4 * index + 3, yIndex, [-1] * numberOfElements
                    )
                    task.putaijlist(
                        offset + 4 * index + 3, xIndex, [-1] * numberOfElements
                    )
                    task.putaijlist(
                        offset + 4 * index + 3, zIndex, [-1] * numberOfElements
                    )
                    task.putaijlist(
                        offset + 4 * index + 3, curAuxMult, [1] * numberOfElements
                    )

                    task.putconboundlist(
                        offset + 4 * index + 3,
                        [mosek.boundkey.lo] * numberOfElements,
                        [-2] * numberOfElements,
                        [self.inf] * numberOfElements,
                    )

                else:
                    task.putaijlist(offset + index, yIndex, [-1] * numberOfElements)
                    task.putaijlist(offset + index, xIndex, [-1] * numberOfElements)
                    task.putaijlist(offset + index, zIndex, [-1] * numberOfElements)
                    task.putaijlist(offset + index, curAuxMult, [1] * numberOfElements)

                    task.putconboundlist(
                        offset + index,
                        [mosek.boundkey.lo] * numberOfElements,
                        [-2] * numberOfElements,
                        [self.inf] * numberOfElements,
                    )

                blockIndex += size * numberOfUnaries

        self.cutBins(self.edges, sumOfConvs)

        roundScalar = 0.5
        division = numberOfUnaries * numberOfBins / 14
        # division = 10
        # division = 50

        offset = curNofConstr
        offset2 = curNofConstr + numberOfBins

        for bin in range(0, numberOfBins):
            row = convolutionBins[bin, :]
            sumOfMultiplicationsRow = sumOfConvs[bin][:]

            nSums = sumOfMultiplicationsRow.size

            # sumOfNewVariables <= sumOfMultiplications[i] / divisor + ceil
            task.putaijlist(
                [bin + offset] * numberOfUnaries, row, [1] * numberOfUnaries
            )
            # task.putaijlist([bin + offset2] * numberOfUnaries, row, [1] * numberOfUnaries)
            if sumOfMultiplicationsRow.size != 0:
                task.putaijlist(
                    [bin + offset] * nSums,
                    sumOfMultiplicationsRow,
                    [-1 / division] * nSums,
                )
                # task.putaijlist([bin + offset2] * nSums, sumOfMultiplicationsRow, [-1 / division] * nSums)

            # task.putconbound(bin + offset, mosek.boundkey.lo, -roundScalar, +self.inf)
                task.putconbound(newNofConstr + offset, mosek.boundkey.lo, 0, self.inf)
            # task.putconbound(bin + offset2, mosek.boundkey.up, -self.inf, +roundScalar)

        if VAR:
            newNofConstr = curNofConstr + 2 * numberOfBins + 4 * nAuxMult
        else:
            newNofConstr = curNofConstr + 2 * numberOfBins + nAuxMult

            # symmetry constraints
        for bin in range(0, numberOfBins):
            for unary in range(0, numberOfUnaries - 1):
                offset = bin * (numberOfUnaries - 1) + unary

                # (maximum[bin])[unary] >= (maximum[bin])[unary + 1])
                task.putaij(newNofConstr + offset, convolutionBins[bin, unary], 1)
                task.putaij(newNofConstr + offset, convolutionBins[bin, unary + 1], -1)

                task.putconbound(newNofConstr + offset, mosek.boundkey.lo, 0, self.inf)

        newNofConstr += (numberOfUnaries - 1) * numberOfBins

        resultClass = RandomVariableMOSEK(convolutionBins, self.edges, task)

        return resultClass, newNofVariables, newNofConstr

    def maximum_AND_Convolution_VECTORIZED_MEM_FREE(
        self, secondVariable, thirdVariable, curNofVariables, curNofConstr
    ):
        """Calculates maximum of 2 PDFs of random variable and a convolution with the third afterwards.
        Works only for 2 identical edges. Is computed
        using the unary representation of bins - M 0/1-bins for each bin. Unarization is kept using the divison.
        Is in MOSEK environment. This implementation works only for a maximization problem. Memory free version.

        :param self: class RandomVariableMOSEK - one of maximums
        :param secondVariable: class RandomVariableMOSEK - one of maximums
        :param thirdVariable: class RandomVariableMOSEK - one to convolve
        :param curNofVariables: current number of MOSEK variables - to know the indices
        :param curNofConstr: current number of MOSEK constraints - to know the indices
        :return resultClass:  class RandomVariableMOSEK with mosek variables
        :return newNofVariables: integer, new total number of MOSEK variables
        :return newNofConstr: integer, new total number of MOSEK constraints
        """

        # get data
        x1 = self.bins
        x2 = secondVariable.bins
        numberOfBins = len(x1)

        # allocate dict for sum of multiplications
        sumOfMaxs = {}

        for i in range(0, numberOfBins):
            sumOfMaxs[i] = []
            for j in range(0, i + 1):

                for unary in range(0, len(x1[i])):
                    for unary2 in range(0, len(x2[j])):
                        # help constraints
                        xIndex = x1[i][unary]
                        yIndex = x2[j][unary2]

                        # save the multiplication for later
                        sumOfMaxs[i].append([xIndex, yIndex])

        for i in range(0, numberOfBins):
            for j in range(0, i + 1):

                for unary in range(0, len(x1[j])):
                    for unary2 in range(0, len(x2[i])):

                        if i != j:
                            # help constraints
                            xIndex = x1[j, unary]
                            yIndex = x2[i, unary2]

                            # save the multiplication for later
                            sumOfMaxs[i].append([xIndex, yIndex])

        ############# CONVOLUTION ##################

        # get data
        x3 = thirdVariable.bins
        task = self.task
        newNofVariables = curNofVariables

        sum = 0
        for z in range(0, numberOfBins):
            for k in range(0, z + 1):
                sum += len(x3[k]) * len(sumOfMaxs[z - k])

        # create auxiliary multiplications
        nAuxMult = int(sum)
        auxMult = np.array(range(newNofVariables, newNofVariables + nAuxMult))
        newNofVariables += nAuxMult

        # append variables
        numberToCreate = nAuxMult
        task.appendvars(numberToCreate)
        task.putvartypelist(
            auxMult, [mosek.variabletype.type_int] * numberToCreate
        )  # integer
        task.putvarboundlist(
            auxMult,
            [mosek.boundkey.ra] * numberToCreate,
            [0.0] * numberToCreate,
            [1.0] * numberToCreate,
        )  # binary

        # convolution

        # allocate dict for sum of multiplications
        sumOfConvs = {}

        # append constraints
        task.appendcons(nAuxMult)

        offset = curNofConstr

        indexToAux = 0

        for z in range(0, numberOfBins):
            sumOfConvs[z] = np.array([]).astype(int)
            for k in range(0, z + 1):

                for unary in range(0, len(x3[k])):
                    for unary2 in range(0, len(sumOfMaxs[z - k])):
                        curAuxMult = auxMult[indexToAux]
                        sumOfConvs[z] = np.append(sumOfConvs[z], np.array([curAuxMult]))

                        # help constraints
                        xIndex = x3[k][unary]
                        yIndex = sumOfMaxs[z - k][unary2][0]
                        zIndex = sumOfMaxs[z - k][unary2][1]

                        task.putaij(offset + indexToAux, yIndex, -1)
                        task.putaij(offset + indexToAux, xIndex, -1)
                        task.putaij(offset + indexToAux, zIndex, -1)
                        task.putaij(offset + indexToAux, curAuxMult, 1)

                        task.putconbound(
                            offset + indexToAux, mosek.boundkey.lo, -2, self.inf
                        )

                        indexToAux += 1

        self.cutBins(self.edges, sumOfConvs)

        resultClass = RandomVariableMOSEK(sumOfConvs, self.edges, task)

        return resultClass, newNofVariables, curNofConstr

    def convolution_UNARY_MAX_DIVIDE(
        self, secondVariable, curNofVariables, curNofConstr, withSymmetryConstr=False
    ):
        """Calculates convolution of 2 PDFs of random variable. Works only for 2 identical edges. Is computed
        using the unary representation of bins - M 0/1-bins for each bin. Unarization is kept using the divison.
        Is in MOSEK environment. Works only for maximization task.

        :param self: class RandomVariableMOSEK
        :param secondVariable: class RandomVariableMOSEK
        :param curNofVariables: current number of MOSEK variables - to know the indices
        :param curNofConstr: current number of MOSEK constraints - to know the indices
        :param withSymmetryConstr: boolean whether a symmetry constraints should be included
        :return convolutionClass:  class RandomVariableMOSEK with variables
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
        nSlackMult = int((numberOfBins / 2) * (1 + numberOfBins) * numberOfUnaries**2)
        slackMult = np.array(range(newNofVariables, newNofVariables + nSlackMult))
        newNofVariables += nSlackMult

        # create convolution variables
        nConvolutions = numberOfBins * numberOfUnaries
        convolution = np.array(range(newNofVariables, newNofVariables + nConvolutions))
        newNofVariables += nConvolutions
        newVariables = np.reshape(
            convolution, (numberOfBins, numberOfUnaries)
        )  # reshape bins

        # append variables
        numberToCreate = nSlackMult + nConvolutions
        varsToCreateExtended = np.append(slackMult, convolution)
        task.appendvars(numberToCreate)
        task.putvartypelist(
            varsToCreateExtended, [mosek.variabletype.type_int] * numberToCreate
        )  # integer
        task.putvarboundlist(
            varsToCreateExtended,
            [mosek.boundkey.ra] * numberToCreate,
            [0.0] * numberToCreate,
            [1.0] * numberToCreate,
        )  # binary

        # convolution

        # allocate for storing the sum of multiplications
        sumOfMultiplications = {}

        # append sumOfMultiplications constraints
        if withSymmetryConstr:
            symN = (numberOfUnaries - 1) * numberOfBins
        else:
            symN = 0

        task.appendcons(2 * numberOfBins + 3 * nSlackMult + symN)
        offset = curNofConstr + 2 * numberOfBins

        for z in range(0, numberOfBins):
            sumOfMultiplications[z] = np.array([]).astype(int)
            for k in range(0, z + 1):

                for unary in range(0, numberOfUnaries):
                    for unary2 in range(0, numberOfUnaries):

                        # constraints for sum of multiplications
                        indexToSlack = int(
                            (z / 2) * (1 + z) * numberOfUnaries**2
                            + k * numberOfUnaries**2
                            + unary * numberOfUnaries
                            + unary2
                        )
                        curSlackMult = slackMult[indexToSlack]

                        sumOfMultiplications[z] = np.append(
                            sumOfMultiplications[z], np.array([curSlackMult])
                        )

                        # help constraints
                        xIndex = x1[k, unary]
                        yIndex = x2[z - k, unary2]

                        # slackMult <= x
                        task.putaij(offset + 3 * indexToSlack, xIndex, -1)
                        task.putaij(offset + 3 * indexToSlack, curSlackMult, 1)

                        task.putconbound(
                            offset + 3 * indexToSlack, mosek.boundkey.up, -self.inf, 0
                        )

                        # slackMult <= y
                        task.putaij(offset + 3 * indexToSlack + 1, yIndex, -1)
                        task.putaij(offset + 3 * indexToSlack + 1, curSlackMult, 1)

                        task.putconbound(
                            offset + 3 * indexToSlack + 1,
                            mosek.boundkey.up,
                            -self.inf,
                            0,
                        )

                        # slackMult >= x + y - 1
                        task.putaij(offset + 3 * indexToSlack + 2, yIndex, -1)
                        task.putaij(offset + 3 * indexToSlack + 2, xIndex, -1)
                        task.putaij(offset + 3 * indexToSlack + 2, curSlackMult, 1)

                        task.putconbound(
                            offset + 3 * indexToSlack + 2,
                            mosek.boundkey.lo,
                            -1,
                            self.inf,
                        )

        self.cutBins(self.edges, sumOfMultiplications)

        round = 0.5
        division = numberOfUnaries * numberOfBins / 30
        offset = curNofConstr

        for bin in range(0, numberOfBins):
            row = newVariables[bin, :]
            sumOfMultiplicationsRow = sumOfMultiplications[bin][:]

            # sumOfNewVariables <= sumOfMultiplications[i] / divisor + ceil
            task.putaijlist(
                [bin + offset] * numberOfUnaries, row, [1] * numberOfUnaries
            )
            if sumOfMultiplicationsRow.size != 0:
                task.putaijlist(
                    [bin + offset] * sumOfMultiplicationsRow.size,
                    sumOfMultiplicationsRow,
                    [-1 / division] * sumOfMultiplicationsRow.size,
                )

            task.putconbound(bin + offset, mosek.boundkey.up, -self.inf, round)

            # sumOfNewVariables <= numberOfUnaries
            task.putaijlist(
                [bin + offset + numberOfBins] * numberOfUnaries,
                row,
                [1] * numberOfUnaries,
            )

            task.putconbound(
                bin + offset + numberOfBins,
                mosek.boundkey.up,
                -self.inf,
                numberOfUnaries,
            )

        newNofConstr = curNofConstr + 2 * numberOfBins + 3 * nSlackMult

        if withSymmetryConstr:

            for bin in range(0, numberOfBins):
                for unary in range(0, numberOfUnaries - 1):
                    offset = bin * (numberOfUnaries - 1) + unary

                    # (maximum[bin])[unary] >= (maximum[bin])[unary + 1])
                    task.putaij(newNofConstr + offset, newVariables[bin, unary], 1)
                    task.putaij(newNofConstr + offset, newVariables[bin, unary + 1], -1)

                    task.putconbound(
                        newNofConstr + offset, mosek.boundkey.lo, 0, self.inf
                    )

            newNofConstr += (numberOfUnaries - 1) * numberOfBins

        convolutionClass = RandomVariableMOSEK(newVariables, self.edges, task)

        return convolutionClass, newNofVariables, newNofConstr

    def convolution_UNARY_MAX_DIVIDE_VECTORIZED(
        self,
        secondVariable,
        curNofVariables,
        curNofConstr,
        withSymmetryConstr=False,
        asMin=False,
    ):
        """Calculates convolution of 2 PDFs of random variable. Works only for 2 identical edges. Is computed
        using the unary representation of bins - M 0/1-bins for each bin. Unarization is kept using the divison.
        Is in MOSEK environment.

        :param self: class RandomVariableMOSEK
        :param secondVariable: class RandomVariableMOSEK
        :param curNofVariables: current number of MOSEK variables - to know the indices
        :param curNofConstr: current number of MOSEK constraints - to know the indices
        :param withSymmetryConstr: boolean whether a symmetry constraints should be included
        :return convolutionClass:  class RandomVariableMOSEK with variables
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
        nSlackMult = int((numberOfBins / 2) * (1 + numberOfBins) * numberOfUnaries**2)
        slackMult = np.array(range(newNofVariables, newNofVariables + nSlackMult))
        newNofVariables += nSlackMult

        # create convolution variables
        nConvolutions = numberOfBins * numberOfUnaries
        convolution = np.array(range(newNofVariables, newNofVariables + nConvolutions))
        newNofVariables += nConvolutions
        convolutionBins = np.reshape(
            convolution, (numberOfBins, numberOfUnaries)
        )  # reshape bins

        # append variables
        numberToCreate = nSlackMult + nConvolutions
        varsToCreateExtended = np.append(slackMult, convolution)
        task.appendvars(numberToCreate)
        task.putvartypelist(
            varsToCreateExtended, [mosek.variabletype.type_int] * numberToCreate
        )  # integer
        task.putvarboundlist(
            varsToCreateExtended,
            [mosek.boundkey.ra] * numberToCreate,
            [0.0] * numberToCreate,
            [1.0] * numberToCreate,
        )  # binary

        # convolution

        # allocate dict for sum of multiplications
        sumOfMultiplications = {}

        # append constraints
        if withSymmetryConstr:
            symN = (numberOfUnaries - 1) * numberOfBins
        else:
            symN = 0

        task.appendcons(numberOfBins + 3 * nSlackMult + symN)
        offset = curNofConstr + numberOfBins

        for z in range(0, numberOfBins):
            sumOfMultiplications[z] = np.array([]).astype(int)
            for k in range(0, z + 1):

                unary = np.array(range(0, numberOfUnaries))
                unary = np.transpose(np.tile(unary, (numberOfUnaries, 1)))
                unary2 = np.array(range(0, numberOfUnaries))
                unary2 = np.tile(unary2, (numberOfUnaries, 1))

                # constraints for sum of multiplications
                indexToSlack = (
                    (z / 2) * (1 + z) * numberOfUnaries**2
                    + k * numberOfUnaries**2
                    + unary * numberOfUnaries
                    + unary2
                ).astype(int)
                curSlackMult = np.concatenate(slackMult[indexToSlack])

                indexToSlack = np.concatenate(indexToSlack)
                sumOfMultiplications[z] = np.append(
                    sumOfMultiplications[z], np.array([curSlackMult])
                )

                # help constraints
                xIndex = np.concatenate(x1[k, unary])
                yIndex = np.concatenate(x2[z - k, unary2])

                numberOfElements = indexToSlack.size

                # slackMult <= x
                task.putaijlist(
                    offset + 3 * indexToSlack, xIndex, [-1] * numberOfElements
                )
                task.putaijlist(
                    offset + 3 * indexToSlack, curSlackMult, [1] * numberOfElements
                )

                task.putconboundlist(
                    offset + 3 * indexToSlack,
                    [mosek.boundkey.up] * numberOfElements,
                    [-self.inf] * numberOfElements,
                    [0] * numberOfElements,
                )

                # slackMult <= y
                task.putaijlist(
                    offset + 3 * indexToSlack + 1, yIndex, [-1] * numberOfElements
                )
                task.putaijlist(
                    offset + 3 * indexToSlack + 1, curSlackMult, [1] * numberOfElements
                )

                task.putconboundlist(
                    offset + 3 * indexToSlack + 1,
                    [mosek.boundkey.up] * numberOfElements,
                    [-self.inf] * numberOfElements,
                    [0] * numberOfElements,
                )

                # slackMult >= x + y - 1
                task.putaijlist(
                    offset + 3 * indexToSlack + 2, yIndex, [-1] * numberOfElements
                )
                task.putaijlist(
                    offset + 3 * indexToSlack + 2, xIndex, [-1] * numberOfElements
                )
                task.putaijlist(
                    offset + 3 * indexToSlack + 2, curSlackMult, [1] * numberOfElements
                )

                task.putconboundlist(
                    offset + 3 * indexToSlack + 2,
                    [mosek.boundkey.lo] * numberOfElements,
                    [-1] * numberOfElements,
                    [self.inf] * numberOfElements,
                )

        self.cutBins(self.edges, sumOfMultiplications)

        round = 0.5
        division = numberOfUnaries * numberOfBins / 30

        offset = curNofConstr

        for bin in range(0, numberOfBins):
            row = convolutionBins[bin, :]
            sumOfMultiplicationsRow = sumOfMultiplications[bin][:]

            nSums = sumOfMultiplicationsRow.size

            # sumOfNewVariables <= sumOfMultiplications[i] / divisor + ceil
            task.putaijlist(
                [bin + offset] * numberOfUnaries, row, [1] * numberOfUnaries
            )
            if sumOfMultiplicationsRow.size != 0:
                task.putaijlist(
                    [bin + offset] * nSums,
                    sumOfMultiplicationsRow,
                    [-1 / division] * nSums,
                )

            if asMin:
                task.putconbound(bin + offset, mosek.boundkey.lo, -round, -self.inf)
            else:
                task.putconbound(bin + offset, mosek.boundkey.up, -self.inf, round)

        newNofConstr = curNofConstr + numberOfBins + 3 * nSlackMult

        if withSymmetryConstr:

            for bin in range(0, numberOfBins):
                for unary in range(0, numberOfUnaries - 1):
                    offset = bin * (numberOfUnaries - 1) + unary

                    # (maximum[bin])[unary] >= (maximum[bin])[unary + 1])
                    task.putaij(newNofConstr + offset, convolutionBins[bin, unary], 1)
                    task.putaij(
                        newNofConstr + offset, convolutionBins[bin, unary + 1], -1
                    )

                    task.putconbound(
                        newNofConstr + offset, mosek.boundkey.lo, 0, self.inf
                    )

            newNofConstr += (numberOfUnaries - 1) * numberOfBins

        convolutionClass = RandomVariableMOSEK(convolutionBins, self.edges, task)

        return convolutionClass, newNofVariables, newNofConstr

    def maximum_UNARY_MAX_DIVIDE(
        self, secondVariable, curNofVariables, curNofConstr, withSymmetryConstr=False
    ):
        """
        Calculates maximum of 2 PDFs of variable. Works only for 2 identical edges. Is computed
        using the 'quadratic' algorithm and unary representation of bins - M 0/1-bins for each bin.
        Unarization is kept using the divison.
        Is in MOSEK environment.

        :param self: class RandomVariableMOSEK
        :param secondVariable: class RandomVariableMOSEK
        :param curNofVariables: current number of MOSEK variables - to know the indices
        :param curNofConstr: current number of MOSEK constraints - to know the indices
        :param withSymmetryConstr: boolean whether a symmetry constraints should be included

        :return maximumClass: class RandomVariableMOSEK with variables (1, 1)
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
        nSlackMult = int((numberOfBins / 2) * (1 + numberOfBins) * numberOfUnaries**2)
        slackMult = np.array(range(newNofVariables, newNofVariables + nSlackMult))
        newNofVariables += nSlackMult

        # create second slack multiplications
        nSlackMult2 = int(
            ((numberOfBins - 1) / 2) * (1 + (numberOfBins - 1)) * (numberOfUnaries) ** 2
        )
        slackMult2 = np.array(range(newNofVariables, newNofVariables + nSlackMult2))
        newNofVariables += nSlackMult2

        # create convolution variables
        nMaximums = numberOfBins * numberOfUnaries
        maximum = np.array(range(newNofVariables, newNofVariables + nMaximums))
        newNofVariables += nMaximums
        maximumBins = np.reshape(
            maximum, (numberOfBins, numberOfUnaries)
        )  # reshape bins

        # append variables
        numberToCreate = nSlackMult + nSlackMult2 + nMaximums
        varsToCreateExtended = np.append(slackMult, maximum)
        varsToCreateExtended = np.append(varsToCreateExtended, slackMult2)
        task.appendvars(numberToCreate)
        task.putvartypelist(
            varsToCreateExtended, [mosek.variabletype.type_int] * numberToCreate
        )  # integer
        task.putvarboundlist(
            varsToCreateExtended,
            [mosek.boundkey.ra] * numberToCreate,
            [0.0] * numberToCreate,
            [1.0] * numberToCreate,
        )  # binary

        # allocate dict for sum of multiplications
        sumOfMultiplications = {}

        # append constraints
        if withSymmetryConstr:
            symN = (numberOfUnaries - 1) * numberOfBins
        else:
            symN = 0
        task.appendcons(2 * numberOfBins + 3 * nSlackMult + 3 * nSlackMult2 + symN)

        # set offsets
        constrOffset = curNofConstr + 2 * numberOfBins
        constrOffset2 = constrOffset + 3 * nSlackMult

        for i in range(0, numberOfBins):
            sumOfMultiplications[i] = np.array([]).astype(int)
            for j in range(0, i + 1):

                for unary in range(0, numberOfUnaries):
                    for unary2 in range(0, numberOfUnaries):

                        # constraints for sum of multiplications
                        indexToSlack = int(
                            (i / 2) * (1 + i) * numberOfUnaries**2
                            + j * numberOfUnaries**2
                            + unary * numberOfUnaries
                            + unary2
                        )
                        curSlackMult = slackMult[indexToSlack]

                        sumOfMultiplications[i] = np.append(
                            sumOfMultiplications[i], np.array([curSlackMult])
                        )

                        # help constraints
                        xIndex = x1[i, unary]
                        yIndex = x2[j, unary2]

                        # slackMult <= x
                        task.putaij(constrOffset + 3 * indexToSlack, xIndex, -1)
                        task.putaij(constrOffset + 3 * indexToSlack, curSlackMult, 1)

                        task.putconbound(
                            constrOffset + 3 * indexToSlack,
                            mosek.boundkey.up,
                            -self.inf,
                            0,
                        )

                        # slackMult <= y
                        task.putaij(constrOffset + 3 * indexToSlack + 1, yIndex, -1)
                        task.putaij(
                            constrOffset + 3 * indexToSlack + 1, curSlackMult, 1
                        )

                        task.putconbound(
                            constrOffset + 3 * indexToSlack + 1,
                            mosek.boundkey.up,
                            -self.inf,
                            0,
                        )

                        # slackMult >= x + y - 1
                        task.putaij(constrOffset + 3 * indexToSlack + 2, yIndex, -1)
                        task.putaij(constrOffset + 3 * indexToSlack + 2, xIndex, -1)
                        task.putaij(
                            constrOffset + 3 * indexToSlack + 2, curSlackMult, 1
                        )

                        task.putconbound(
                            constrOffset + 3 * indexToSlack + 2,
                            mosek.boundkey.lo,
                            -1,
                            self.inf,
                        )

                        if i != j:
                            # constraints for sum of multiplications
                            indexToSlack = int(
                                ((i - 1) / 2) * (1 + i - 1) * numberOfUnaries**2
                                + j * numberOfUnaries**2
                                + unary * numberOfUnaries
                                + unary2
                            )
                            curSlackMult = slackMult2[indexToSlack]

                            sumOfMultiplications[i] = np.append(
                                sumOfMultiplications[i], np.array([curSlackMult])
                            )

                            # help constraints
                            xIndex = x1[j, unary]
                            yIndex = x2[i, unary2]

                            # slackMult <= x
                            task.putaij(constrOffset2 + 3 * indexToSlack, xIndex, -1)
                            task.putaij(
                                constrOffset2 + 3 * indexToSlack, curSlackMult, 1
                            )

                            task.putconbound(
                                constrOffset2 + 3 * indexToSlack,
                                mosek.boundkey.up,
                                -self.inf,
                                0,
                            )

                            # slackMult <= y
                            task.putaij(
                                constrOffset2 + 3 * indexToSlack + 1, yIndex, -1
                            )
                            task.putaij(
                                constrOffset2 + 3 * indexToSlack + 1, curSlackMult, 1
                            )

                            task.putconbound(
                                constrOffset2 + 3 * indexToSlack + 1,
                                mosek.boundkey.up,
                                -self.inf,
                                0,
                            )

                            # slackMult >= x + y - 1
                            task.putaij(
                                constrOffset2 + 3 * indexToSlack + 2, yIndex, -1
                            )
                            task.putaij(
                                constrOffset2 + 3 * indexToSlack + 2, xIndex, -1
                            )
                            task.putaij(
                                constrOffset2 + 3 * indexToSlack + 2, curSlackMult, 1
                            )

                            task.putconbound(
                                constrOffset2 + 3 * indexToSlack + 2,
                                mosek.boundkey.lo,
                                -1,
                                self.inf,
                            )

        division = numberOfBins * numberOfUnaries / 22
        round = 0.5

        offset = curNofConstr

        for bin in range(0, numberOfBins):
            row = maximumBins[bin, :]
            sumOfMultiplicationsRow = sumOfMultiplications[bin][:]

            nSums = sumOfMultiplicationsRow.size

            # sumOfNewVariables <= sumOfMultiplications[i] / divisor + ceil
            task.putaijlist(
                [bin + offset] * numberOfUnaries, row, [1] * numberOfUnaries
            )
            if nSums != 0:
                task.putaijlist(
                    [bin + offset] * nSums,
                    sumOfMultiplicationsRow,
                    [-1 / division] * nSums,
                )

            task.putconbound(bin + offset, mosek.boundkey.up, -self.inf, round)

            # sumOfNewVariables <= numberOfUnaries
            task.putaijlist(
                [bin + offset + numberOfBins] * numberOfUnaries,
                row,
                [1] * numberOfUnaries,
            )
            task.putconbound(
                bin + offset + numberOfBins,
                mosek.boundkey.up,
                -self.inf,
                numberOfUnaries,
            )

        newNofConstr = (
            curNofConstr + 2 * numberOfBins + 3 * nSlackMult + 3 * nSlackMult2
        )

        if withSymmetryConstr:

            for bin in range(0, numberOfBins):
                for unary in range(0, numberOfUnaries - 1):
                    offset = bin * (numberOfUnaries - 1) + unary

                    # (maximum[bin])[unary] >= (maximum[bin])[unary + 1])
                    task.putaij(newNofConstr + offset, maximumBins[bin, unary], 1)
                    task.putaij(newNofConstr + offset, maximumBins[bin, unary + 1], -1)

                    task.putconbound(
                        newNofConstr + offset, mosek.boundkey.lo, 0, self.inf
                    )

            newNofConstr += (numberOfUnaries - 1) * numberOfBins

        maximumClass = RandomVariableMOSEK(maximumBins, self.edges, task)

        return maximumClass, newNofVariables, newNofConstr

    def maximum_UNARY_MAX_DIVIDE_MEM_FREE(
        self,
        secondVariable,
        curNofVariables,
        curNofConstr,
        numberOfUnarie,
        withSymmetryConstr=False,
    ):
        """Calculates maximum of 2 PDFs of random variable. Works only for 2 identical edges. Is computed
        using the unary representation of bins - M 0/1-bins for each bin. Unarization is kept using the divison.
        Is in MOSEK environment. Memory free means we do not create new matrix - instead pass the indicies - too big scaling,
        and does not work well

        :param self: class RandomVariableMOSEK
        :param secondVariable: class RandomVariableMOSEK
        :param curNofVariables: current number of MOSEK variables - to know the indices
        :param curNofConstr: current number of MOSEK constraints - to know the indices
        :param withSymmetryConstr: boolean whether a symmetry constraints should be included
        :return maximumClass:  class RandomVariableMOSEK with variables
        :return newNofVariables: integer, new total number of MOSEK variables
        :return newNofConstr: integer, new total number of MOSEK constraints
        """

        # get data
        x1 = self.bins
        x2 = secondVariable.bins
        numberOfBins = len(x1)
        task = self.task
        newNofVariables = curNofVariables

        sum = 0
        for i in range(0, numberOfBins):
            for j in range(0, i + 1):
                sum += len(x1[i]) * len(x2[j])

        # create slack multiplications
        # nSlackMult = int((numberOfBins / 2) * (1 + numberOfBins) * numberOfUnaries ** 2)
        nSlackMult = sum
        slackMult = np.array(range(newNofVariables, newNofVariables + nSlackMult))
        newNofVariables += nSlackMult

        sum = 0
        for i in range(0, numberOfBins):
            for j in range(0, i):
                sum += len(x1[j]) * len(x2[i])

        # create second slack multiplications
        # nSlackMult2 = int(((numberOfBins - 1) / 2) * (1 + (numberOfBins - 1)) * (numberOfUnaries) ** 2)
        nSlackMult2 = sum
        slackMult2 = np.array(range(newNofVariables, newNofVariables + nSlackMult2))
        newNofVariables += nSlackMult2

        # create convolution variables
        nMaximums = numberOfBins * numberOfUnarie
        maximum = np.array(range(newNofVariables, newNofVariables + nMaximums))
        newNofVariables += nMaximums
        maximumBins = np.reshape(
            maximum, (numberOfBins, numberOfUnarie)
        )  # reshape bins

        # append variables
        numberToCreate = nSlackMult + nSlackMult2 + nMaximums
        varsToCreateExtended = np.append(slackMult, maximum)
        varsToCreateExtended = np.append(varsToCreateExtended, slackMult2)
        task.appendvars(numberToCreate)
        task.putvartypelist(
            varsToCreateExtended, [mosek.variabletype.type_int] * numberToCreate
        )  # integer
        task.putvarboundlist(
            varsToCreateExtended,
            [mosek.boundkey.ra] * numberToCreate,
            [0.0] * numberToCreate,
            [1.0] * numberToCreate,
        )  # binary

        # allocate dict for sum of multiplications
        sumOfMultiplications = {}

        # append constraints
        if withSymmetryConstr:
            symN = (numberOfUnarie - 1) * numberOfBins
        else:
            symN = 0
        task.appendcons(numberOfBins + nSlackMult + nSlackMult2 + symN)

        # set offsets
        constrOffset = curNofConstr + numberOfBins
        constrOffset2 = constrOffset + nSlackMult

        indexToSlack = 0
        blockIndex = 0
        for i in range(0, numberOfBins):
            sumOfMultiplications[i] = np.array([]).astype(int)
            for j in range(0, i + 1):

                size1 = len(x1[i])
                size2 = len(x2[j])

                unary = np.array(range(0, size1))
                unary = np.transpose(np.tile(unary, (size2, 1)))
                unary2 = np.array(range(0, size2))
                unary2 = np.tile(unary2, (size1, 1))

                index = blockIndex + unary * size2 + unary2

                curAuxMult = np.concatenate(slackMult[index])

                index = np.concatenate(index)
                sumOfMultiplications[i] = np.append(
                    sumOfMultiplications[i], np.array([curAuxMult])
                )

                # help constraints
                xIndex = np.concatenate(x1[i][unary])
                yIndex = np.concatenate(x2[j][unary2])

                numberOfElements = index.size
                # slackMult >= x + y + z - 2

                task.putaijlist(constrOffset + index, yIndex, [-1] * numberOfElements)
                task.putaijlist(constrOffset + index, xIndex, [-1] * numberOfElements)
                task.putaijlist(
                    constrOffset + index, curAuxMult, [1] * numberOfElements
                )

                task.putconboundlist(
                    constrOffset + index,
                    [mosek.boundkey.lo] * numberOfElements,
                    [-1] * numberOfElements,
                    [self.inf] * numberOfElements,
                )

                blockIndex += size1 * size2

        blockIndex = 0
        for i in range(0, numberOfBins):
            for j in range(0, i):
                size1 = len(x1[j])
                size2 = len(x2[i])

                unary = np.array(range(0, size1))
                unary = np.transpose(np.tile(unary, (size2, 1)))
                unary2 = np.array(range(0, size2))
                unary2 = np.tile(unary2, (size1, 1))

                index = blockIndex + unary * size2 + unary2

                curAuxMult = np.concatenate(slackMult2[index])

                index = np.concatenate(index)
                sumOfMultiplications[i] = np.append(
                    sumOfMultiplications[i], np.array([curAuxMult])
                )

                # help constraints
                xIndex = np.concatenate(x1[j][unary])
                yIndex = np.concatenate(x2[i][unary2])

                numberOfElements = index.size
                # slackMult >= x + y + z - 2

                task.putaijlist(constrOffset2 + index, yIndex, [-1] * numberOfElements)
                task.putaijlist(constrOffset2 + index, xIndex, [-1] * numberOfElements)
                task.putaijlist(
                    constrOffset2 + index, curAuxMult, [1] * numberOfElements
                )

                task.putconboundlist(
                    constrOffset2 + index,
                    [mosek.boundkey.lo] * numberOfElements,
                    [-1] * numberOfElements,
                    [self.inf] * numberOfElements,
                )

                blockIndex += size1 * size2

        # division = numberOfBins * numberOfUnarie / 22
        division = 100000000
        round = 0.5

        offset = curNofConstr

        for bin in range(0, numberOfBins):
            row = maximumBins[bin, :]
            sumOfMultiplicationsRow = sumOfMultiplications[bin][:]

            nSums = sumOfMultiplicationsRow.size

            # sumOfNewVariables <= sumOfMultiplications[i] / divisor + ceil
            task.putaijlist([bin + offset] * numberOfUnarie, row, [1] * numberOfUnarie)
            if nSums != 0:
                task.putaijlist(
                    [bin + offset] * nSums,
                    sumOfMultiplicationsRow,
                    [-1 / division] * nSums,
                )

            task.putconbound(bin + offset, mosek.boundkey.lo, -round, self.inf)

        newNofConstr = curNofConstr + numberOfBins + nSlackMult + nSlackMult2

        if withSymmetryConstr:

            for bin in range(0, numberOfBins):
                for unary in range(0, numberOfUnarie - 1):
                    offset = bin * (numberOfUnarie - 1) + unary

                    # (maximum[bin])[unary] >= (maximum[bin])[unary + 1])
                    task.putaij(newNofConstr + offset, maximumBins[bin, unary], 1)
                    task.putaij(newNofConstr + offset, maximumBins[bin, unary + 1], -1)

                    task.putconbound(
                        newNofConstr + offset, mosek.boundkey.lo, 0, self.inf
                    )

            newNofConstr += (numberOfUnarie - 1) * numberOfBins

        maximumClass = RandomVariableMOSEK(maximumBins, self.edges, task)

        return maximumClass, newNofVariables, newNofConstr

    def maximum_UNARY_MAX_DIVIDE_VECTORIZED(
        self,
        secondVariable,
        curNofVariables,
        curNofConstr,
        withSymmetryConstr=False,
        asMin=False,
    ):
        """Calculates maximum of 2 PDFs of random variable. Works only for 2 identical edges. Is computed
        using the unary representation of bins - M 0/1-bins for each bin. Unarization is kept using the divison.
        Is in MOSEK environment.

        :param self: class RandomVariableMOSEK
        :param secondVariable: class RandomVariableMOSEK
        :param curNofVariables: current number of MOSEK variables - to know the indices
        :param curNofConstr: current number of MOSEK constraints - to know the indices
        :param withSymmetryConstr: boolean whether a symmetry constraints should be included
        :return maximumClass:  class RandomVariableMOSEK with variables
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
        nSlackMult = int((numberOfBins / 2) * (1 + numberOfBins) * numberOfUnaries**2)
        slackMult = np.array(range(newNofVariables, newNofVariables + nSlackMult))
        newNofVariables += nSlackMult

        # create second slack multiplications
        nSlackMult2 = int(
            ((numberOfBins - 1) / 2) * (1 + (numberOfBins - 1)) * (numberOfUnaries) ** 2
        )
        slackMult2 = np.array(range(newNofVariables, newNofVariables + nSlackMult2))
        newNofVariables += nSlackMult2

        # create convolution variables
        nMaximums = numberOfBins * numberOfUnaries
        maximum = np.array(range(newNofVariables, newNofVariables + nMaximums))
        newNofVariables += nMaximums
        maximumBins = np.reshape(
            maximum, (numberOfBins, numberOfUnaries)
        )  # reshape bins

        # append variables
        numberToCreate = nSlackMult + nSlackMult2 + nMaximums
        varsToCreateExtended = np.append(slackMult, maximum)
        varsToCreateExtended = np.append(varsToCreateExtended, slackMult2)
        task.appendvars(numberToCreate)
        task.putvartypelist(
            varsToCreateExtended, [mosek.variabletype.type_int] * numberToCreate
        )  # integer
        task.putvarboundlist(
            varsToCreateExtended,
            [mosek.boundkey.ra] * numberToCreate,
            [0.0] * numberToCreate,
            [1.0] * numberToCreate,
        )  # binary

        # allocate dict for sum of multiplications
        sumOfMultiplications = {}

        # append sumOfMultiplications constraints
        if withSymmetryConstr:
            symN = (numberOfUnaries - 1) * numberOfBins
        else:
            symN = 0

        if asMin:
            task.appendcons(numberOfBins + nSlackMult + nSlackMult2 + symN)
            constrOffset = curNofConstr + numberOfBins
            constrOffset2 = constrOffset + nSlackMult

        else:
            task.appendcons(numberOfBins + 2 * nSlackMult + 2 * nSlackMult2 + symN)
            constrOffset = curNofConstr + numberOfBins
            constrOffset2 = constrOffset + 2 * nSlackMult

        # set offsets

        for i in range(0, numberOfBins):
            sumOfMultiplications[i] = np.array([]).astype(int)
            for j in range(0, i + 1):

                unary = np.array(range(0, numberOfUnaries))
                unary = np.transpose(np.tile(unary, (numberOfUnaries, 1)))
                unary2 = np.array(range(0, numberOfUnaries))
                unary2 = np.tile(unary2, (numberOfUnaries, 1))

                # constraints for sum of multiplications
                indexToSlack = (
                    (i / 2) * (1 + i) * numberOfUnaries**2
                    + j * numberOfUnaries**2
                    + unary * numberOfUnaries
                    + unary2
                ).astype(int)
                curSlackMult = np.concatenate(slackMult[indexToSlack])

                indexToSlack = np.concatenate(indexToSlack)
                sumOfMultiplications[i] = np.append(
                    sumOfMultiplications[i], np.array([curSlackMult])
                )

                # help constraints
                xIndex = np.concatenate(x1[i, unary])
                yIndex = np.concatenate(x2[j, unary2])

                numberOfElements = indexToSlack.size

                # slackMult <= x
                if not asMin:
                    task.putaijlist(
                        constrOffset + 2 * indexToSlack, xIndex, [-1] * numberOfElements
                    )
                    task.putaijlist(
                        constrOffset + 2 * indexToSlack,
                        curSlackMult,
                        [1] * numberOfElements,
                    )

                    task.putconboundlist(
                        constrOffset + 2 * indexToSlack,
                        [mosek.boundkey.up] * numberOfElements,
                        [-self.inf] * numberOfElements,
                        [0] * numberOfElements,
                    )

                    # slackMult <= y
                    task.putaijlist(
                        constrOffset + 2 * indexToSlack + 1,
                        yIndex,
                        [-1] * numberOfElements,
                    )
                    task.putaijlist(
                        constrOffset + 2 * indexToSlack + 1,
                        curSlackMult,
                        [1] * numberOfElements,
                    )

                    task.putconboundlist(
                        constrOffset + 2 * indexToSlack + 1,
                        [mosek.boundkey.up] * numberOfElements,
                        [-self.inf] * numberOfElements,
                        [0] * numberOfElements,
                    )

                if asMin:
                    # slackMult >= x + y - 1
                    task.putaijlist(
                        constrOffset + indexToSlack, yIndex, [-1] * numberOfElements
                    )
                    task.putaijlist(
                        constrOffset + indexToSlack, xIndex, [-1] * numberOfElements
                    )
                    task.putaijlist(
                        constrOffset + indexToSlack,
                        curSlackMult,
                        [1] * numberOfElements,
                    )

                    task.putconboundlist(
                        constrOffset + indexToSlack,
                        [mosek.boundkey.lo] * numberOfElements,
                        [-1] * numberOfElements,
                        [self.inf] * numberOfElements,
                    )

                if i != j:

                    # constraints for sum of multiplications
                    indexToSlack = (
                        ((i - 1) / 2) * (1 + i - 1) * numberOfUnaries**2
                        + j * numberOfUnaries**2
                        + unary * numberOfUnaries
                        + unary2
                    ).astype(int)
                    curSlackMult = slackMult2[indexToSlack]

                    sumOfMultiplications[i] = np.append(
                        sumOfMultiplications[i], np.array([curSlackMult])
                    )
                    curSlackMult = np.concatenate(slackMult2[indexToSlack])

                    indexToSlack = np.concatenate(indexToSlack)
                    sumOfMultiplications[i] = np.append(
                        sumOfMultiplications[i], np.array([curSlackMult])
                    )

                    # help constraints
                    xIndex = np.concatenate(x1[j, unary])
                    yIndex = np.concatenate(x2[i, unary2])

                    numberOfElements = indexToSlack.size

                    if not asMin:
                        # slackMult <= x
                        task.putaijlist(
                            constrOffset2 + 2 * indexToSlack,
                            xIndex,
                            [-1] * numberOfElements,
                        )
                        task.putaijlist(
                            constrOffset2 + 2 * indexToSlack,
                            curSlackMult,
                            [1] * numberOfElements,
                        )

                        task.putconboundlist(
                            constrOffset2 + 2 * indexToSlack,
                            [mosek.boundkey.up] * numberOfElements,
                            [-self.inf] * numberOfElements,
                            [0] * numberOfElements,
                        )

                        # slackMult <= y
                        task.putaijlist(
                            constrOffset2 + 2 * indexToSlack + 1,
                            yIndex,
                            [-1] * numberOfElements,
                        )
                        task.putaijlist(
                            constrOffset2 + 2 * indexToSlack + 1,
                            curSlackMult,
                            [1] * numberOfElements,
                        )

                        task.putconboundlist(
                            constrOffset2 + 2 * indexToSlack + 1,
                            [mosek.boundkey.up] * numberOfElements,
                            [-self.inf] * numberOfElements,
                            [0] * numberOfElements,
                        )

                    if asMin:
                        # slackMult >= x + y - 1
                        task.putaijlist(
                            constrOffset2 + indexToSlack,
                            yIndex,
                            [-1] * numberOfElements,
                        )
                        task.putaijlist(
                            constrOffset2 + indexToSlack,
                            xIndex,
                            [-1] * numberOfElements,
                        )
                        task.putaijlist(
                            constrOffset2 + indexToSlack,
                            curSlackMult,
                            [1] * numberOfElements,
                        )

                        task.putconboundlist(
                            constrOffset2 + indexToSlack,
                            [mosek.boundkey.lo] * numberOfElements,
                            [-1] * numberOfElements,
                            [self.inf] * numberOfElements,
                        )

        division = numberOfBins * numberOfUnaries / 22
        # division = 10
        round = 0.5

        offset = curNofConstr

        for bin in range(0, numberOfBins):
            row = maximumBins[bin, :]
            sumOfMultiplicationsRow = sumOfMultiplications[bin][:]

            nSums = sumOfMultiplicationsRow.size

            # sumOfNewVariables <= sumOfMultiplications[i] / divisor + ceil
            task.putaijlist(
                [bin + offset] * numberOfUnaries, row, [1] * numberOfUnaries
            )
            if nSums != 0:
                task.putaijlist(
                    [bin + offset] * nSums,
                    sumOfMultiplicationsRow,
                    [-1 / division] * nSums,
                )

            if asMin:
                task.putconbound(bin + offset, mosek.boundkey.lo, -round, self.inf)
            else:
                task.putconbound(bin + offset, mosek.boundkey.up, -self.inf, round)

        if asMin:
            newNofConstr = curNofConstr + numberOfBins + nSlackMult + nSlackMult2
        else:
            newNofConstr = (
                curNofConstr + numberOfBins + 2 * nSlackMult + 2 * nSlackMult2
            )

        if withSymmetryConstr:
            for bin in range(0, numberOfBins):
                for unary in range(0, numberOfUnaries - 1):
                    offset = bin * (numberOfUnaries - 1) + unary

                    # (maximum[bin])[unary] >= (maximum[bin])[unary + 1])
                    task.putaij(newNofConstr + offset, maximumBins[bin, unary], 1)
                    task.putaij(newNofConstr + offset, maximumBins[bin, unary + 1], -1)

                    task.putconbound(
                        newNofConstr + offset, mosek.boundkey.lo, 0, self.inf
                    )

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

        newBins = {}
        if edges[0] > 0:  # cut bins

            for i in range(numberOfBinsNeeded, numberOfBins):
                newBins[i] = bins[i - numberOfBinsNeeded]

            for i in range(numberOfBinsNeeded, numberOfBins):
                bins[i] = newBins[i]

            for i in range(0, numberOfBinsNeeded):
                bins[i] = np.array([]).astype(int)

        if edges[0] < 0:  # cut bins

            for i in range(numberOfBinsNeeded, numberOfBins):
                bins[i - numberOfBinsNeeded] = bins[i]

            # for i in range(numberOfBinsNeeded, numberOfBins):
            #     bins[i - numberOfBinsNeeded] = newBins[i - numberOfBinsNeeded]

            for i in range(numberOfBins - numberOfBinsNeeded, numberOfBins):
                bins[i] = np.array([]).astype(int)
