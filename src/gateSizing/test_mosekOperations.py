import mosek
import sys
import histogramGenerator
import numpy as np

from mosekVariable import RandomVariableMOSEK

"""
  This module tests mosek maximum and convolution operations.
"""


# Define a stream printer to grab output from MOSEK
def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def testConvolution_MAX():
    mu1 = 2.98553396
    sigma1 = 2.76804456

    mu2 = 3.98483475
    sigma2 = 1.802585
    numberOfGates = 2


    interval = (-5, 10)

    numberOfSamples = 2000000
    numberOfBins = 10
    numberOfUnaries = 17

    # DESIRED

    rv1 = histogramGenerator.get_gauss_bins_UNARY(mu1, sigma1, numberOfBins, numberOfSamples, interval, numberOfUnaries)
    rv2 = histogramGenerator.get_gauss_bins_UNARY(mu2, sigma2, numberOfBins, numberOfSamples, interval, numberOfUnaries)


    max1 = rv1.maxOfDistributionsQUAD_FORMULA_UNARY(rv2)
    # max1 = test1.convolutionOfTwoVarsShift(test2)
    desired = [max1.mean, max1.std]

    print(desired)
    print(max1.bins)

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



            bins1 = np.zeros((numberOfBins, numberOfUnaries))
            bins2 = np.zeros((numberOfBins, numberOfUnaries))

            gates = [rv1, rv2]
            bins = [bins1, bins2]

            # set objective function
            for gate in range(0, numberOfGates):
                currentBins = bins[gate]
                generatedRV = gates[gate]
                for bin in range(0, numberOfBins):
                    for unary in range(0, numberOfUnaries):

                        variableIndex = gate*numberOfBins*numberOfUnaries + bin*numberOfUnaries + unary
                        task.putcj(variableIndex, 1)

                        # Set the bounds on variable
                        # 0 <= x_j <= 1

                        task.putvarbound(variableIndex, mosek.boundkey.ra, 0.0, generatedRV.bins[bin, unary])

                            # save index to the bins
                        currentBins[bin, unary] = variableIndex




            # todo: set create 2 random variables and perform convolution

            RV1 = RandomVariableMOSEK(bins1, rv1.edges)
            RV2 = RandomVariableMOSEK(bins2, rv1.edges)

            convolution = RV1.convolution_UNARY_DIVIDE(RV2)
            # convolution = convolution.bins

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
            xx = [0.] * numberVariablesRVs
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




            # FORMULATE

            # RV1 = RandomVariableCVXPY(x1, rv1.edges)
            # RV2 = RandomVariableCVXPY(x2, rv1.edges)

            # GET obj. function and constr

            # convolution, constr = RV1.maximum_QUAD_UNARY_DIVIDE(RV2, asMin=False)
            # convolution = convolution.bins

            # objective function
            # sum = 0
            # for bin in range(0, numberOfBins):
            #     for unary in range(0, numberOfUnaries):
            #         sum += (convolution[bin])[unary]
            #
            # # other constraints
            #
            # for bin in range(0, numberOfBins):
            #     for unary in range(0, numberOfUnaries):
            #         constr.append((x1[bin])[unary] <= rv1.bins[bin, unary])  # set lower constr.
            #
            # for bin in range(0, numberOfBins):
            #     for unary in range(0, numberOfUnaries):
            #         constr.append((x2[bin])[unary] <= rv2.bins[bin, unary])  # set lower constr.




                # solve
            # objective = cp.Maximize(sum)
            # prob = cp.Problem(objective, constr)
            # prob.solve(verbose=True, solver=cp.MOSEK)

            # PRINT OUT THE VALUES
            # print("Problem value: " + str(prob.value))
            #
            # convBins = np.zeros((numberOfBins, numberOfUnaries))
            # for bin in range(0, numberOfBins):
            #     for unary in range(0, numberOfUnaries):
            #         convBins[bin, unary] = (convolution[bin])[unary].value
            #
            # edges = np.linspace(interval[0], interval[1], numberOfBins + 1)
            # convRV = RandomVariable(convBins, edges, unary=True)
            #
            # print(convRV.bins)
            #
            # actual = [convRV.mean, convRV.std]
            # print(actual)
            #
            # # TESTING
            #
            # np.testing.assert_almost_equal(desired, actual, decimal=dec)

    return None


    return None



if __name__ == "__main__":
    testConvolution_MAX()
