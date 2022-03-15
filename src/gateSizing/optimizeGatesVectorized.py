import numpy as np
import cvxpy as cp


# Aout = np.where(A<=0, 0, A)
# Ain  = np.where(A>=0, 0, 1)

# # optimization variables
# x = cp.Variable(m, pos=True)         # sizes
# t = cp.Variable(m, pos=True)         # arrival times

# input capacitance is an affine function of sizes
# cin = alpha + beta*x


def computeIncidenceMatrices(A: np.array):
    """
    Get standard cell-edge incidence matrix of the circuit

    :param A: (m, n) numpy array
    :return tuple(Ain, Aout) of edge-outgoing and edge-incoming part
    """

    Aout = np.where(A > 0, 1, 0)
    Ain = np.where(A < 0, 1, 0)



    return Aout, Ain

    # AoutParam = cp.Variable(Aout.shape, pos=True)
    # AoutParam = cp.Parameter(Aout.shape, pos=True)
    # AoutParam.value = Aout
    # AoutParam = cp.Constant(Aout)

    # AinParam = cp.expressions.constants.Constant(cp.bmat(Ain))
    # AinParam.value = Ain
    # AinParam = cp.Parameter(shape=Ain.shape, pos=True)
    # AinParam.value = Ain
    # AinParam = cp.Constant(Ain)


    # AinParam = cp.Variable(Ain.shape, pos=True)

    return AoutParam, AinParam

    # return cp.bmat(Aout), cp.bmat(Ain)



def computeInputCapacitance(alphas: np.array, betas: np.array, x: cp.Variable) -> cp.Expression:
    """
    Calculate input capacitance as affine function ... alpha + beta * x

    :param alphas: array of alphas
    :param betas: array of betas
    :param x: cvxpy variable (1, n)
    :return inputCap: (1, n) array of input capacitances
    """

    inputCap = alphas + cp.multiply(x, betas)
    return inputCap


#
# given by Fout = Aout*Ain'
# cload = (Aout @ Ain.T) * cin



def computeLoadCapacitance(Aout, Ain, inputCapacitance: cp.Expression) -> cp.Expression:
    """
    Load capacitance is computed as a sum of a fanout.

    :param Aout: (m, n) output matrix
    :param Ain: (m, n) input matrix
    :param inputCapacitance: (1, n) cvxpy variable of input capacitance
    :return cloadMat: (m, n) load capacitance matrix
    """

    Fout = Aout @ Ain.T
    cloadMat = Fout @ inputCapacitance

    # print(cloadMat)
    return cloadMat

    cload = np.array([None] * 7)

    cload[0] = cloadMat[0]
    cload[1] = cloadMat[1]
    cload[2] = cloadMat[2]
    cload[3] = cloadMat[3]
    cload[4] = cloadMat[4]

    cload[5] = Cout6
    cload[6] = Cout7

    #return cload
    # return inputCapacitance
    return cp.hstack(cload)
    # return cp.bmat(cload)

# Create an array of constraints
# constr_cload = []
# # Make your requirement a constraint
# constr_cload.append(cload[5]==Cout6)
# constr_cload.append(cload[6]==Cout7)


# delay is the product of its driving resistance R = gamma/x and cload
# d = cload * gamma/x


def computeGateDelays(capLoad: cp.Expression, gammas: np.array, x: cp.Variable) -> cp.Expression:
    """
    Load capacitance is computed as a sum of a fanout.

    :param Aout: (m, n) output matrix
    :param Ain: (m, n) input matrix
    :param inputCapacitance: (1, n) cvxpy variable of input capacitance
    :return cloadMat: (m, n) load capacitance matrix
    """
    cloadTimesGamma = cp.multiply(capLoad, gammas)
    gateDelays = cp.multiply(cloadTimesGamma, 1 / x)

    return gateDelays


# total power
# power = (f*e) * x
#
# # total area
# area = a * x
#
# # constraints
# constr_x = x >= 1 # all sizes greater than 1 (normalized)
#
# # create timing constraints
# # these constraints enforce t_j + d_j <= t_i over all gates j that drive gate i
# constr_timing = Aout.T*t + Ain.T*d <= Ain.T*t
# # for gates with inputs not connected to other gates we enforce d_i <= t_i
# input_timing  = d[0:2] <= t[0:2]
#
#
#
# # objective is the upper bound on the overall delay
# # and that is the max of arrival times for output gates 6 and 7
# D = cp.atoms.elementwise.maximum.maximum(t[5],t[6])
#
# # collect all the constraints
# constraints = [power <= Pmax, area <= Amax, constr_timing, input_timing, constr_x] + constr_cload


# formulate the GP problem and solve it
# objective = cp.Minimize(D)
#
# problem = cp.Problem(objective, constraints)
# problem.solve(gp=True)


def getConstrCload(capLoad, loadCapacitances):
    """
    Function gets capacitance load constraint.
    Load capacitances of end gates are given, cannot set due
    to cvxpy expression - it is set as an constraint instead.

    :param capLoad: (1, n) cvxpy variable of capacitance load
    :param loadCapacitances: (1, m) array of load capacitances of output gates
    :return constr_cload: (1, m) load cap. constr.
    """
    constr_cload = []
    constr_cload.append(capLoad[5] == loadCapacitances[0])
    constr_cload.append(capLoad[6] == loadCapacitances[1])

    return constr_cload



def getConstrTiming(Aout, Ain, gateDelays, t: cp.Variable, x):
    """
    Timing constraints.
        Aout' * t + Ain' * gateDelays <= Ain' * t
    These constraints enforce t_j + d_j <= t_i
    over all gates j that drive gate i.

    For gates with inputs not connected to other gates we enforce d_i <= t_i

    :param Aout: (n, m) cvxpy variable of capacitance load
    :param Ain: (n, m) array of load capacitances of output gates
    :param gateDelays: (1, m) array of load capacitances of output gates
    :param t: (1, m) of cvxpy variable of timing
    :return constr_cload: (1, m) load cap. constr.
    """
    constr_timing = Aout.T @ t + Ain.T @ gateDelays <= Ain.T @ t

    input_timing = getInputTimingConstr(gateDelays, t)

    return constr_timing , input_timing



def getInputTimingConstr(gateDelays: cp.Expression, t: cp.Variable) -> [bool]:
    """
    Get timing constraints for gate delays

    :param gateDelays: (1, n) cvxpy variable of gate delays
    :param t: (1, n) cvxpy variable of timing
    :return timing_constr: (1, n) cvxpy expression of timing constr.
    """

    constr1 = gateDelays[0] <= t[0]
    constr2 = gateDelays[1] <= t[1]

    timing_constr = [constr1, constr2]

    return timing_constr


def getMaximumDelay(t: cp.Variable) -> cp.Expression:
    """
    Get maximum delay - that is the upper bound
    on the overall delay and that is the max of
    arrival times for output gates

    :param t: (1, n) cvxpy variable of timing
    :return circuitDelay: (1, n) cvxpy expression of circuit delay
    """

    varsToMin = cp.hstack([t[5], t[6]])
    circuitDelay = cp.max(varsToMin)

    return circuitDelay


def computeTotalPower(frequencies: np.array, energyLoss: np.array, x: cp.Variable) -> cp.Expression:
    """
    Compute total power as sum_i ( f_i * e_i * x_i )

    :param frequencies: (1, n) numpy array of gate frequencies
    :param energyLoss: (1, n) numpy array of energy loss of each gate
    :param x: (1, n) cvxpy variable
    :return circuitDelay: (1, n) cvxpy expression of circuit delay
    """

    power = cp.sum(cp.multiply((cp.multiply(frequencies, x)), energyLoss))
    return power


def computeTotalArea(gateScales: np.array, x: cp.Variable) -> cp.Expression:
    """
    Compute total area as sum_i ( gateScale_i * x_i )

    :param gateScales: (1, n) numpy array of unit scaling factor of gate i
    :param x: (1, n) cvxpy variable
    :return area: Double, total area
    """

    area = cp.sum(cp.multiply(gateScales, x))
    return area




def optimizeGates(frequencies, energyLoss, gateScales, alphas, betas, gammas, maxArea, maxPower,
                  loadCapacitances, A, numberOfCells):
    # optimization variables
    x = cp.Variable(numberOfCells, pos=True)  # sizes
    t = cp.Variable(numberOfCells, pos=True)  # arrival times

    Aout, Ain = computeIncidenceMatrices(A)

    # alphasParam = cp.Parameter(alphas.shape, pos=True)
    # alphasParam.value = alphas
    alphasParam = cp.Constant(alphas)

    # betasParam = cp.Parameter(betas.shape, pos=True)
    # betasParam.value = betas
    betasParam = cp.Constant(betas)

    # gammasParam = cp.Parameter(gammas.shape, pos=True)
    # gammasParam.value = gammas
    gammasParam = cp.Constant(gammas)

    inputCap = computeInputCapacitance(alphasParam, betasParam, x)
    computedLCapacitance = computeLoadCapacitance(Aout, Ain, inputCap)

    # computedLCapacitance = x
    gateDelays = computeGateDelays(computedLCapacitance, gammasParam, x)
    maxDelay = getMaximumDelay(t)

    totalPower = computeTotalPower(frequencies, energyLoss, x)
    totalArea = computeTotalArea(gateScales, x)

    # get constraints

    cloadConstr = getConstrCload(computedLCapacitance, loadCapacitances)
    timingConstr, inputTimingConstr = getConstrTiming(Aout, Ain, gateDelays, t, x   )

    posX = x >= 1  # all sizes greater than 1 (normalized)expr


    # formulate the GGP


    # constraints = [totalPower <= maxPower, totalArea <= maxArea, timingConstr, posX] + inputTimingConstr
    constraints = [totalPower <= maxPower, totalArea <= maxArea, posX, timingConstr] + inputTimingConstr

    objective = cp.Minimize(maxDelay)

    prob = cp.Problem(objective, constraints)

    prob.solve(gp=True, verbose=True, solver=cp.MOSEK)

    print("sizing params: ", x.value)

    return prob.value


if __name__ == "__main__":
    # IMPORTANT FOR JAKUB: commented code outside of functions is code from dmytro (see google collab)

    numberOfCells = 7  # number of cells
    numberOfEdges = 8  # number of edges
    A = np.zeros((numberOfCells, numberOfEdges))

    A[0][0] = 1
    A[1][1] = 1
    A[1][2] = 1
    A[2][3] = 1
    A[2][7] = 1
    A[3][0] = -1
    A[3][1] = -1
    A[3][4] = 1
    A[3][5] = 1
    A[4][2] = -1
    A[4][3] = -1
    A[4][6] = 1
    A[5][4] = -1
    A[6][5] = -1
    A[6][6] = -1
    A[6][7] = -1

    # problem constants
    f = np.array([1, 0.8, 1, 0.7, 0.7, 0.5, 0.5])
    e = np.array([1, 2, 1, 1.5, 1.5, 1, 2])
    Cout6 = 10
    Cout7 = 10

    a = np.ones(numberOfCells)
    alpha = np.ones(numberOfCells)
    beta = np.ones(numberOfCells)
    gamma = np.ones(numberOfCells)

    # maximum area and power specification
    Amax = 25
    Pmax = 50

    print(optimizeGates(f, e, a, alpha, beta, gamma, Amax, Pmax, [Cout6, Cout7], A, numberOfCells))
