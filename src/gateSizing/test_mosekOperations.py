import mosek
import sys
import histogramGenerator
import numpy as np

from randomVariableHist_Numpy import RandomVariable
from mosekVariable import RandomVariableMOSEK

"""
  This module tests mosek maximum and convolution operations.
"""


# Define a stream printer to grab output from MOSEK
def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def testConvolution_MAX(dec = 3):
    mu1 = 7
    sigma1 = 2

    mu2 = 10
    sigma2 = 1


    numberOfGates = 2


    interval = (-10, 40)

    numberOfSamples = 2000000
    numberOfBins = 20
    numberOfUnaries = 20

    # DESIRED

    rv1 = histogramGenerator.get_gauss_bins_UNARY(mu1, sigma1, numberOfBins, numberOfSamples, interval, numberOfUnaries)
    rv2 = histogramGenerator.get_gauss_bins_UNARY(mu2, sigma2, numberOfBins, numberOfSamples, interval, numberOfUnaries)


    max1 = rv1.maxOfDistributionsQUAD_FORMULA_UNARY(rv2)
    # max1 = test1.convolutionOfTwoVarsShift(test2)
    desired = [max1.mean, max1.std]

    # print(desired)
    # print(max1.bins)

    # Make a MOSEK environment
    with mosek.Env() as env:
        # Attach a printer to the environment
        env.set_Stream(mosek.streamtype.log, streamprinter)

        # Create a task
        with env.Task(0, 0) as task:
            # Attach a printer to the task
            task.set_Stream(mosek.streamtype.log, streamprinter)

            numberVariablesRVs = numberOfGates*numberOfBins * numberOfUnaries

            # The constraints will initially have no bounds.
            # task.appendcons(numberVariablesRVs)
            # The variables will initially be fixed at zero (x=0).
            task.appendvars(numberVariablesRVs)

            # set variables to be boolean
            rvIndices = np.array(range(0, numberVariablesRVs))
            task.putvartypelist(rvIndices,
                                [mosek.variabletype.type_int] * numberVariablesRVs)



            bins1 = np.zeros((numberOfBins, numberOfUnaries)).astype(int)
            bins2 = np.zeros((numberOfBins, numberOfUnaries)).astype(int)

            gates = [rv1, rv2]
            bins = [bins1, bins2]

            # set objective function
            for gate in range(0, numberOfGates):
                currentBins = bins[gate]
                generatedRV = gates[gate]
                for bin in range(0, numberOfBins):
                    for unary in range(0, numberOfUnaries):

                        variableIndex = gate*numberOfBins*numberOfUnaries + bin*numberOfUnaries + unary
                        # task.putcj(variableIndex, 1)

                        # Set the bounds on variable
                        # 0 <= x_j <= 1

                        # task.putvarbound(variableIndex, mosek.boundkey.ra, 0.0, generatedRV.bins[bin, unary])
                        task.putvarbound(variableIndex, mosek.boundkey.ra, generatedRV.bins[bin, unary], 1)

                            # save index to the bins
                        currentBins[bin, unary] = variableIndex


            RV1 = RandomVariableMOSEK(bins1, rv1.edges, task)
            RV2 = RandomVariableMOSEK(bins2, rv1.edges, task)

            convolution, newNofVariables, newNofConstr = RV1.maximum_UNARY_MAX_DIVIDE_VECTORIZED(RV2, numberVariablesRVs, 0
                                                                                                     ,asMin=True, withSymmetryConstr=True)
            # print(newNofConstr)
            convolutionConCat = convolution.bins

                # create the objective function
            convolutionConCat = np.concatenate(convolutionConCat)
            task.putclist(convolutionConCat, [1]*convolutionConCat.shape[0])

                # solve problem

            # Input the objective sense (minimize/maximize)
            task.putobjsense(mosek.objsense.minimize)
            # task.putobjsense(mosek.objsense.maximize)

            # Solve the problem
            task.optimize()
            # Print a summary containing information
            # about the solution for debugging purposes
            task.solutionsummary(mosek.streamtype.msg)

            prosta = task.getprosta(mosek.soltype.itg)
            solsta = task.getsolsta(mosek.soltype.itg)

            # Output a solution
            xx = np.array([0.] * newNofVariables)
            task.getxx(mosek.soltype.itg, xx)

            if solsta in [mosek.solsta.integer_optimal]:
                pass
                # print("Optimal solution: %s" % xx)
            elif solsta == mosek.solsta.prim_feas:
                pass
                # print("Feasible solution: %s" % xx)
            elif mosek.solsta.unknown:
                if prosta == mosek.prosta.prim_infeas_or_unbounded:
                    print("Problem status Infeasible or unbounded.\n")
                elif prosta == mosek.prosta.prim_infeas:
                    print("Problem status Infeasible.\n")
                elif prosta == mosek.prosta.unkown:
                    print("Problem status unkown.\n")
                else:
                    print("Other problem status.\n")
            else:
                print("Other solution status")


            convolution.bins = xx[convolution.bins]
            print(convolution.bins)

            rv = RandomVariable(convolution.bins, edges=convolution.edges, unary=True)
            actual = [rv.mean, rv.std]

            print(desired)
            print(actual)

            np.testing.assert_almost_equal(desired, actual, decimal=dec)

def testMaximum_MAX(dec = 3):
    mu1 = 5.98553396
    sigma1 = 1

    # mu2 = 3.98483475
    # sigma2 = 1.802585
    mu2 = 5.98553396
    sigma2 = 1
    numberOfGates = 2


    interval = (-5, 15)

    numberOfSamples = 2000000
    numberOfBins = 16
    numberOfUnaries = 30

    # DESIRED

    rv1 = histogramGenerator.get_gauss_bins_UNARY(mu1, sigma1, numberOfBins, numberOfSamples, interval, numberOfUnaries)
    rv2 = histogramGenerator.get_gauss_bins_UNARY(mu2, sigma2, numberOfBins, numberOfSamples, interval, numberOfUnaries)


    max1 = rv1.maxOfDistributionsQUAD_FORMULA_UNARY(rv2)
    # max1 = test1.convolutionOfTwoVarsShift(test2)
    s = np.random.normal(mu1, sigma1, numberOfSamples)
    # s = np.random.normal(mu1, sigma1, numberOfSamples)
    max1 = np.maximum(s, s)
    # desired = [max1.mean, max1.std]
    desired = [np.mean(max1), np.std(max1)]

    # print(desired)
    # print(max1.bins)

    # Make a MOSEK environment
    with mosek.Env() as env:
        # Attach a printer to the environment
        env.set_Stream(mosek.streamtype.log, streamprinter)

        # Create a task
        with env.Task(0, 0) as task:
            # Attach a printer to the task
            task.set_Stream(mosek.streamtype.log, streamprinter)

            numberVariablesRVs = numberOfGates*numberOfBins * numberOfUnaries

            # The variables will initially be fixed at zero (x=0).
            task.appendvars(numberVariablesRVs)

            # set variables to be boolean
            rvIndices = np.array(range(0, numberVariablesRVs))
            task.putvartypelist(rvIndices,
                                [mosek.variabletype.type_int] * numberVariablesRVs)



            bins1 = np.zeros((numberOfBins, numberOfUnaries)).astype(int)
            bins2 = np.zeros((numberOfBins, numberOfUnaries)).astype(int)

            gates = [rv1, rv2]
            bins = [bins1, bins2]

            # set objective function
            for gate in range(0, numberOfGates):
                currentBins = bins[gate]
                generatedRV = gates[gate]
                for bin in range(0, numberOfBins):
                    for unary in range(0, numberOfUnaries):

                        variableIndex = gate*numberOfBins*numberOfUnaries + bin*numberOfUnaries + unary
                        # task.putcj(variableIndex, 1)

                        # Set the bounds on variable
                        # 0 <= x_j <= 1

                        task.putvarbound(variableIndex, mosek.boundkey.ra, 0.0, generatedRV.bins[bin, unary])

                            # save index to the bins
                        currentBins[bin, unary] = variableIndex




            RV1 = RandomVariableMOSEK(bins1, rv1.edges, task)
            RV2 = RandomVariableMOSEK(bins2, rv1.edges, task)

            maximum, newNofVariables, newNofConstr = RV1.maximum_UNARY_MAX_DIVIDE_VECTORIZED(RV2, numberVariablesRVs,
                                                                                             0, withSymmetryConstr=True)
            maximumConCat = maximum.bins

                # create the objective function
            maximumConCat = np.concatenate(maximumConCat)
            task.putclist(maximumConCat, [1]*maximumConCat.shape[0])

                # solve problem

            # Input the objective sense (minimize/maximize)
            task.putobjsense(mosek.objsense.maximize)

            # Solve the problem
            task.optimize()
            # Print a summary containing information
            # about the solution for debugging purposes
            task.solutionsummary(mosek.streamtype.msg)

            prosta = task.getprosta(mosek.soltype.itg)
            solsta = task.getsolsta(mosek.soltype.itg)

            # Output a solution
            xx = np.array([0.] * newNofVariables)
            task.getxx(mosek.soltype.itg, xx)

            if solsta in [mosek.solsta.integer_optimal]:
                pass
                # print("Optimal solution: %s" % xx)
            elif solsta == mosek.solsta.prim_feas:
                pass
                # print("Feasible solution: %s" % xx)
            elif mosek.solsta.unknown:
                if prosta == mosek.prosta.prim_infeas_or_unbounded:
                    print("Problem status Infeasible or unbounded.\n")
                elif prosta == mosek.prosta.prim_infeas:
                    print("Problem status Infeasible.\n")
                elif prosta == mosek.prosta.unkown:
                    print("Problem status unkown.\n")
                else:
                    print("Other problem status.\n")
            else:
                print("Other solution status")


            maximum.bins = xx[maximum.bins]
            # print(maximum.bins)

            rv = RandomVariable(maximum.bins, edges=maximum.edges, unary=True)
            actual = [rv.mean, rv.std]

            print(desired)
            print(actual)

            np.testing.assert_almost_equal(desired, actual, decimal=dec)

def testMaximum_MAX_CONV(dec = 3):
    mu1 = 5.98553396
    sigma1 = 1

    mu2 = 3
    sigma2 = 1.802585

    mu3 = 2
    sigma3 = 0.2

    numberOfGates = 3


    interval = (-1, 9)

    numberOfSamples = 2000000
    numberOfBins = 10
    numberOfUnaries = 10

    # DESIRED

    rv1 = histogramGenerator.get_gauss_bins_UNARY(mu1, sigma1, numberOfBins, numberOfSamples, interval, numberOfUnaries)
    rv2 = histogramGenerator.get_gauss_bins_UNARY(mu2, sigma2, numberOfBins, numberOfSamples, interval, numberOfUnaries)
    rv3 = histogramGenerator.get_gauss_bins_UNARY(mu3, sigma3, numberOfBins, numberOfSamples, interval, numberOfUnaries)

    # print(rv1.bins)
    # print(rv2.bins)
    # max1 = rv1.maxOfDistributionsQUAD_FORMULA_UNARY(rv2)
    # conv = max1.convolutionOfTwoVarsNaiveSAME_UNARY(rv3)
    conv = rv1.maximum_AND_Convolution_UNARY(rv2, rv3)
    # conv = conv.maxOfDistributionsQUAD_FORMULA_UNARY(rv4)
    # conv = conv.convolutionOfTwoVarsNaiveSAME_UNARY(rv3)
    # max1 = test1.convolutionOfTwoVarsShift(test2)


    desired = [conv.mean, conv.std]

    # print(desired)
    # print(max1.bins)

    # Make a MOSEK environment
    with mosek.Env() as env:
        # Attach a printer to the environment
        env.set_Stream(mosek.streamtype.log, streamprinter)

        # Create a task
        with env.Task(0, 0) as task:
            # Attach a printer to the task
            task.set_Stream(mosek.streamtype.log, streamprinter)

            numberVariablesRVs = numberOfGates*numberOfBins * numberOfUnaries

            # The variables will initially be fixed at zero (x=0).
            task.appendvars(numberVariablesRVs)

            # set variables to be boolean
            rvIndices = np.array(range(0, numberVariablesRVs))
            task.putvartypelist(rvIndices,
                                [mosek.variabletype.type_int] * numberVariablesRVs)



            bins1 = np.zeros((numberOfBins, numberOfUnaries)).astype(int)
            bins2 = np.zeros((numberOfBins, numberOfUnaries)).astype(int)
            bins3 = np.zeros((numberOfBins, numberOfUnaries)).astype(int)

            gates = [rv1, rv2, rv3]
            bins = [bins1, bins2, bins3]

            # set objective function
            for gate in range(0, numberOfGates):
                currentBins = bins[gate]
                generatedRV = gates[gate]
                for bin in range(0, numberOfBins):
                    for unary in range(0, numberOfUnaries):

                        variableIndex = gate*numberOfBins*numberOfUnaries + bin*numberOfUnaries + unary
                        # task.putcj(variableIndex, 1)

                        # Set the bounds on variable
                        # 0 <= x_j <= 1

                        task.putvarbound(variableIndex, mosek.boundkey.ra, 0.0, generatedRV.bins[bin, unary])

                            # save index to the bins
                        currentBins[bin, unary] = variableIndex




            RV1 = RandomVariableMOSEK(bins1, rv1.edges, task)
            RV2 = RandomVariableMOSEK(bins2, rv1.edges, task)
            RV3 = RandomVariableMOSEK(bins3, rv1.edges, task)

            maximum, newNofVariables, newNofConstr = RV1.maximum_AND_Convolution_VECTORIZED(RV2, RV3, numberVariablesRVs,
                                                                                             0)

            # maximum, newNofVariables, newNofConstr = RV1.maximum_UNARY_MAX_DIVIDE_VECTORIZED(RV2, numberVariablesRVs,
            #                                                                       0, withSymmetryConstr=True)

            # maximum, newNofVariables, newNofConstr = maximum.convolution_UNARY_MAX_DIVIDE_VECTORIZED(RV3, newNofVariables,
            #                                                                       newNofConstr, withSymmetryConstr=True)


            maximumConCat = maximum.bins

                # create the objective function
            maximumConCat = np.concatenate(maximumConCat)
            task.putclist(maximumConCat, [1]*maximumConCat.shape[0])

                # solve problem

            # Input the objective sense (minimize/maximize)
            task.putobjsense(mosek.objsense.maximize)

            # Solve the problem
            task.optimize()
            # Print a summary containing information
            # about the solution for debugging purposes
            task.solutionsummary(mosek.streamtype.msg)

            prosta = task.getprosta(mosek.soltype.itg)
            solsta = task.getsolsta(mosek.soltype.itg)

            # Output a solution
            xx = np.array([0.] * newNofVariables)
            task.getxx(mosek.soltype.itg, xx)

            if solsta in [mosek.solsta.integer_optimal]:
                pass
                # print("Optimal solution: %s" % xx)
            elif solsta == mosek.solsta.prim_feas:
                pass
                # print("Feasible solution: %s" % xx)
            elif mosek.solsta.unknown:
                if prosta == mosek.prosta.prim_infeas_or_unbounded:
                    print("Problem status Infeasible or unbounded.\n")
                elif prosta == mosek.prosta.prim_infeas:
                    print("Problem status Infeasible.\n")
                elif prosta == mosek.prosta.unkown:
                    print("Problem status unkown.\n")
                else:
                    print("Other problem status.\n")
            else:
                print("Other solution status")

            print(conv.bins)

            maximum.bins = xx[maximum.bins]
            print(maximum.bins)

            rv = RandomVariable(maximum.bins, edges=maximum.edges, unary=True)
            actual = [rv.mean, rv.std]

            print(desired)
            print(actual)

            np.testing.assert_almost_equal(desired, actual, decimal=dec)


def testMaximum_MAX_CONV_asmin(dec = 3):
    mu1 = 5.98553396
    sigma1 = 1

    mu2 = 3
    sigma2 = 1.802585

    mu3 = 2
    sigma3 = 0.2

    numberOfGates = 3


    interval = (0, 9)

    numberOfSamples = 2000000
    numberOfBins = 8
    numberOfUnaries = 8

    # DESIRED

    rv1 = histogramGenerator.get_gauss_bins_UNARY(mu1, sigma1, numberOfBins, numberOfSamples, interval, numberOfUnaries)
    rv2 = histogramGenerator.get_gauss_bins_UNARY(mu2, sigma2, numberOfBins, numberOfSamples, interval, numberOfUnaries)
    rv3 = histogramGenerator.get_gauss_bins_UNARY(mu3, sigma3, numberOfBins, numberOfSamples, interval, numberOfUnaries)

    # print(rv1.bins)
    # print(rv2.bins)
    # max1 = rv1.maxOfDistributionsQUAD_FORMULA_UNARY(rv2)
    # conv = max1.convolutionOfTwoVarsNaiveSAME_UNARY(rv3)
    conv = rv1.maximum_AND_Convolution_UNARY(rv2, rv3)
    # conv = conv.maxOfDistributionsQUAD_FORMULA_UNARY(rv4)
    # conv = conv.convolutionOfTwoVarsNaiveSAME_UNARY(rv3)
    # max1 = test1.convolutionOfTwoVarsShift(test2)


    desired = [conv.mean, conv.std]

    # print(desired)
    # print(max1.bins)

    # Make a MOSEK environment
    with mosek.Env() as env:
        # Attach a printer to the environment
        env.set_Stream(mosek.streamtype.log, streamprinter)

        # Create a task
        with env.Task(0, 0) as task:
            # Attach a printer to the task
            task.set_Stream(mosek.streamtype.log, streamprinter)

            numberVariablesRVs = numberOfGates*numberOfBins * numberOfUnaries

            # The variables will initially be fixed at zero (x=0).
            task.appendvars(numberVariablesRVs)

            # set variables to be boolean
            rvIndices = np.array(range(0, numberVariablesRVs))
            task.putvartypelist(rvIndices,
                                [mosek.variabletype.type_int] * numberVariablesRVs)



            bins1 = np.zeros((numberOfBins, numberOfUnaries)).astype(int)
            bins2 = np.zeros((numberOfBins, numberOfUnaries)).astype(int)
            bins3 = np.zeros((numberOfBins, numberOfUnaries)).astype(int)

            gates = [rv1, rv2, rv3]
            bins = [bins1, bins2, bins3]

            # set objective function
            for gate in range(0, numberOfGates):
                currentBins = bins[gate]
                generatedRV = gates[gate]
                for bin in range(0, numberOfBins):
                    for unary in range(0, numberOfUnaries):

                        variableIndex = gate*numberOfBins*numberOfUnaries + bin*numberOfUnaries + unary
                        # task.putcj(variableIndex, 1)

                        # Set the bounds on variable
                        # 0 <= x_j <= 1

                        # task.putvarbound(variableIndex, mosek.boundkey.ra, generatedRV.bins[bin, unary], 1)
                        task.putvarbound(variableIndex, mosek.boundkey.ra, 0, 1)

                            # save index to the bins
                        currentBins[bin, unary] = variableIndex




            RV1 = RandomVariableMOSEK(bins1, rv1.edges, task)
            RV2 = RandomVariableMOSEK(bins2, rv1.edges, task)
            RV3 = RandomVariableMOSEK(bins3, rv1.edges, task)

            maximum, newNofVariables, newNofConstr = RV1.maximum_AND_Convolution_VECTORIZED_MIN(RV2, RV3, numberVariablesRVs,
                                                                                             0)

            # maximum, newNofVariables, newNofConstr = RV1.maximum_UNARY_MAX_DIVIDE_VECTORIZED(RV2, numberVariablesRVs,
            #                                                                       0, withSymmetryConstr=True)

            # maximum, newNofVariables, newNofConstr = maximum.convolution_UNARY_MAX_DIVIDE_VECTORIZED(RV3, newNofVariables,
            #                                                                       newNofConstr, withSymmetryConstr=True)


            maximumConCat = maximum.bins

                # create the objective function
            maximumConCat = np.concatenate(maximumConCat)
            task.putclist(maximumConCat, [1]*maximumConCat.shape[0])

            task.appendcons(numberOfGates * numberOfBins)

            gateNodes = [bins1, bins2, bins3]

            for gate in range(0, numberOfGates):

                curNode = gateNodes[gate]
                generatedRV = gates[gate]

                for bin in range(0, numberOfBins):
                    generatedValues = np.sum(generatedRV.bins[bin, :])

                    # print(generatedValues)

                    row = curNode[bin, :]

                    task.putarow(newNofConstr + gate * numberOfBins + bin, row, [1] * row.size)
                    task.putconbound(newNofConstr + gate * numberOfBins + bin, mosek.boundkey.fx, generatedValues, generatedValues)


            newNofConstr = newNofConstr + numberOfGates * numberOfBins

            task.appendcons(numberOfGates * numberOfBins * (numberOfUnaries - 1))

            for gate in range(0, numberOfGates):
                curNode = gateNodes[gate]

                # symmetry constraints
                for bin in range(0, numberOfBins):
                    for unary in range(0, numberOfUnaries - 1):
                        offset = bin * (numberOfUnaries - 1) + unary

                        # (maximum[bin])[unary] >= (maximum[bin])[unary + 1])
                        task.putaij(newNofConstr + offset, curNode[bin, unary], 1)
                        task.putaij(newNofConstr + offset, curNode[bin, unary + 1], -1)

                        task.putconbound(newNofConstr + offset, mosek.boundkey.lo, 0, 0.0)

            newNofConstr += (numberOfUnaries - 1) * numberOfBins

                # solve problem

            # Input the objective sense (minimize/maximize)
            task.putobjsense(mosek.objsense.minimize)

            # Solve the problem
            task.optimize()
            # Print a summary containing information
            # about the solution for debugging purposes
            task.solutionsummary(mosek.streamtype.msg)

            prosta = task.getprosta(mosek.soltype.itg)
            solsta = task.getsolsta(mosek.soltype.itg)

            # Output a solution
            xx = np.array([0.] * newNofVariables)
            task.getxx(mosek.soltype.itg, xx)

            if solsta in [mosek.solsta.integer_optimal]:
                pass
                # print("Optimal solution: %s" % xx)
            elif solsta == mosek.solsta.prim_feas:
                pass
                # print("Feasible solution: %s" % xx)
            elif mosek.solsta.unknown:
                if prosta == mosek.prosta.prim_infeas_or_unbounded:
                    print("Problem status Infeasible or unbounded.\n")
                elif prosta == mosek.prosta.prim_infeas:
                    print("Problem status Infeasible.\n")
                elif prosta == mosek.prosta.unkown:
                    print("Problem status unkown.\n")
                else:
                    print("Other problem status.\n")
            else:
                print("Other solution status")

            print(conv.bins)

            maximum.bins = xx[maximum.bins]
            print(maximum.bins)

            rv = RandomVariable(maximum.bins, edges=maximum.edges, unary=True)
            actual = [rv.mean, rv.std]

            print(desired)
            print(actual)

            np.testing.assert_almost_equal(desired, actual, decimal=dec)


def test_setting(dec = 3):
    mu1 = 5.98553396
    sigma1 = 1

    mu2 = 3
    sigma2 = 1.802585

    mu3 = 2
    sigma3 = 0.2

    numberOfGates = 3


    interval = (-1, 9)

    numberOfSamples = 2000000
    numberOfBins = 10
    numberOfUnaries = 10

    # DESIRED

    rv1 = histogramGenerator.get_gauss_bins_UNARY(mu1, sigma1, numberOfBins, numberOfSamples, interval, numberOfUnaries)
    rv2 = histogramGenerator.get_gauss_bins_UNARY(mu2, sigma2, numberOfBins, numberOfSamples, interval, numberOfUnaries)
    rv3 = histogramGenerator.get_gauss_bins_UNARY(mu3, sigma3, numberOfBins, numberOfSamples, interval, numberOfUnaries)

    # print(rv1.bins)
    # print(rv2.bins)
    # max1 = rv1.maxOfDistributionsQUAD_FORMULA_UNARY(rv2)
    # conv = max1.convolutionOfTwoVarsNaiveSAME_UNARY(rv3)
    conv = rv1.maximum_AND_Convolution_UNARY(rv2, rv3)
    # conv = conv.maxOfDistributionsQUAD_FORMULA_UNARY(rv4)
    # conv = conv.convolutionOfTwoVarsNaiveSAME_UNARY(rv3)
    # max1 = test1.convolutionOfTwoVarsShift(test2)


    desired = [conv.mean, conv.std]

    # print(desired)
    # print(max1.bins)

    # Make a MOSEK environment
    with mosek.Env() as env:
        # Attach a printer to the environment
        env.set_Stream(mosek.streamtype.log, streamprinter)

        # Create a task
        with env.Task(0, 0) as task:
            # Attach a printer to the task
            task.set_Stream(mosek.streamtype.log, streamprinter)

            numberVariablesRVs = numberOfGates*numberOfBins * numberOfUnaries

            # The variables will initially be fixed at zero (x=0).
            task.appendvars(numberVariablesRVs)

            # set variables to be boolean
            rvIndices = np.array(range(0, numberVariablesRVs))
            task.putvartypelist(rvIndices,
                                [mosek.variabletype.type_int] * numberVariablesRVs)



            bins1 = np.zeros((numberOfBins, numberOfUnaries)).astype(int)
            bins2 = np.zeros((numberOfBins, numberOfUnaries)).astype(int)
            bins3 = np.zeros((numberOfBins, numberOfUnaries)).astype(int)

            gates = [rv1, rv2, rv3]
            bins = [bins1, bins2, bins3]


            # set objective function
            for gate in range(0, numberOfGates):
                currentBins = bins[gate]
                generatedRV = gates[gate]
                for bin in range(0, numberOfBins):
                    for unary in range(0, numberOfUnaries):

                        variableIndex = gate*numberOfBins*numberOfUnaries + bin*numberOfUnaries + unary
                        # task.putcj(variableIndex, 1)

                        # Set the bounds on variable
                        # 0 <= x_j <= 1

                        # task.putvarbound(variableIndex, mosek.boundkey.ra, generatedRV.bins[bin, unary], 1)
                        task.putvarbound(variableIndex, mosek.boundkey.ra, 0, 1)

                            # save index to the bins
                        currentBins[bin, unary] = variableIndex




            RV1 = RandomVariableMOSEK(bins1, rv1.edges, task)

                # create the objective function
            task.putclist(np.concatenate(bins1), [1]*np.concatenate(bins1).shape[0])

            task.appendcons(numberOfGates * numberOfBins)

            gateNodes = [bins1, bins2, bins3]

            for gate in range(0, numberOfGates):

                curNode = gateNodes[gate]
                generatedRV = gates[gate]

                for bin in range(0, numberOfBins):
                    generatedValues = np.sum(generatedRV.bins[bin, :])

                    # print(generatedValues)

                    row = curNode[bin, :]

                    task.putarow(gate * numberOfBins + bin, row, [1] * row.size)
                    task.putconbound(gate * numberOfBins + bin, mosek.boundkey.lo, generatedValues, 1)


            newNofConstr = numberOfGates * numberOfBins

            task.appendcons(numberOfGates * numberOfBins * (numberOfUnaries - 1))

            for gate in range(0, numberOfGates):
                curNode = gateNodes[gate]

                # symmetry constraints
                for bin in range(0, numberOfBins):
                    for unary in range(0, numberOfUnaries - 1):
                        offset = bin * (numberOfUnaries - 1) + unary

                        # (maximum[bin])[unary] >= (maximum[bin])[unary + 1])
                        task.putaij(newNofConstr + offset, curNode[bin, unary], 1)
                        task.putaij(newNofConstr + offset, curNode[bin, unary + 1], -1)

                        task.putconbound(newNofConstr + offset, mosek.boundkey.lo, 0, 0.0)

            newNofConstr += (numberOfUnaries - 1) * numberOfBins

                # solve problem

            # Input the objective sense (minimize/maximize)
            task.putobjsense(mosek.objsense.minimize)

            # Solve the problem
            task.optimize()
            # Print a summary containing information
            # about the solution for debugging purposes
            task.solutionsummary(mosek.streamtype.msg)

            prosta = task.getprosta(mosek.soltype.itg)
            solsta = task.getsolsta(mosek.soltype.itg)

            # Output a solution
            xx = np.array([0.] * numberVariablesRVs)
            task.getxx(mosek.soltype.itg, xx)

            if solsta in [mosek.solsta.integer_optimal]:
                pass
                # print("Optimal solution: %s" % xx)
            elif solsta == mosek.solsta.prim_feas:
                pass
                # print("Feasible solution: %s" % xx)
            elif mosek.solsta.unknown:
                if prosta == mosek.prosta.prim_infeas_or_unbounded:
                    print("Problem status Infeasible or unbounded.\n")
                elif prosta == mosek.prosta.prim_infeas:
                    print("Problem status Infeasible.\n")
                elif prosta == mosek.prosta.unkown:
                    print("Problem status unkown.\n")
                else:
                    print("Other problem status.\n")
            else:
                print("Other solution status")

            # print(conv.bins)
            # print()


            bins = xx[bins1]
            print(bins)
            print(rv1.bins)

            # rv = RandomVariable(maximum.bins, edges=maximum.edges, unary=True)
            # actual = [rv.mean, rv.std]

            # print(desired)
            # print(actual)

            # np.testing.assert_almost_equal(desired, actual, decimal=dec)

if __name__ == "__main__":
    # testConvolution_MAX(dec=8)
    # testMaximum_MAX(dec=8)

    # testMaximum_MAX_CONV(dec=8)
    testMaximum_MAX_CONV_asmin()

    # test_setting()


    print('All tests passed!')
