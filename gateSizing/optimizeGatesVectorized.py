import numpy as np
import cvxpy as cp
import mosek

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

# Aout = np.where(A<=0, 0, A)
# Ain  = np.where(A>=0, 0, 1)

# # optimization variables
# x = cp.Variable(m, pos=True)         # sizes
# t = cp.Variable(m, pos=True)         # arrival times

# input capacitance is an affine function of sizes
# cin = alpha + beta*x


""" Calculate input capacitance as affine function ... alpha + beta * x

      Params:
        A: [MxN] standard cell-edge incidence matrix of the circuit
      Return: tuple(Ain, Aout) of edge-outgoing and edge-incoming part """


def computeIncidenceMatrices(A):
    Aout = np.where(A > 0, 1, 0)
    Ain = np.where(A < 0, 1, 0)

    AoutParam = cp.Parameter(Aout.shape, pos=True)
    AoutParam.value = Aout

    AinParam = cp.Parameter(Ain.shape, pos=True)
    AinParam.value = Ain

    return AoutParam, AinParam

    # return ( cp.bmat( Aout ), cp.bmat( Ain ))


""" Calculate input capacitance as affine function ... alpha + beta * x

      Params:
        alphas: array [1xN]
        betas: array [1xN]
        x: variable [1xN]
      Return: array [1xN] of load capacitances """


def computeInputCapacitance(alphas, betas, x):
    inputCap = alphas + cp.multiply(x, betas)
    return inputCap


#
# given by Fout = Aout*Ain'
# cload = (Aout @ Ain.T) * cin

""" Load capacitance is the input capacitance times the fan-out matrix.
    Given by Fout = Aout*Ain'
    (sum of a fanout)

        Params:
          inputCapacitance : array variable [1xN] of input capacitances
          loadCapacitances : array [1xM] of load capacitances of output gates
          numberOfGates : Integer, total number of gates
        Return: array [1xN] of load capacitances
    """


def computeLoadCapacitance(Aout, Ain, inputCapacitance):
    Fout = cp.bmat( Aout @ Ain.T )
    cloadMat = Fout @ inputCapacitance

    cload = [None] * 7

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


# Create an array of constraints
# constr_cload = []
# # Make your requirement a constraint
# constr_cload.append(cload[5]==Cout6)
# constr_cload.append(cload[6]==Cout7)


# delay is the product of its driving resistance R = gamma/x and cload
# d = cload * gamma/x

""" Delay on each gate is computed as (load capacitance * gamma) / resistance 

        Params:
          capLoad : [1xN] of load capacitances
          gammas : array [1xN] of  gamma constants
          x : [1xN] variable
        Return: array [1xN] of gate delays        """


def computeGateDelays(capLoad, gammas, x):
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

""" Function gets capacitance load constraint.
    Load capacitances of end gates are given, cannot set due 
    to cvxpy expression - it is set as an constraint instead.

        Params:
          capLoad : [1xN] of computed load capacitances
          loadCapacitances: [1xK] given array of load capacitances
        Return: array [1xK] of constraints        """


def getConstrCload(capLoad, loadCapacitances):
    # TODO: generalize
    constr_cload = []
    constr_cload.append(capLoad[5] == loadCapacitances[0])
    constr_cload.append(capLoad[6] == loadCapacitances[1])

    return constr_cload


""" Timing constraints. 
        Aout' * t + Ain' * gateDelays <= Ain' * t
    These constraints enforce t_j + d_j <= t_i 
    over all gates j that drive gate i.
    
    For gates with inputs not connected to other gates we enforce d_i <= t_i

        Params:
          Aout : [1xN] of computed load capacitances
          Ain: [1xK] given array of load capacitances
          gateDelays: array [1xN] of gate delays
          t: variable [1xN]
        Return: tuple (constr_timing, input_timing) of all timing constraints        """


def getConstrTiming(Aout, Ain, gateDelays, t):

    print(gateDelays)

    constr_timing = Aout.T @ t + Ain.T @ cp.hstack(gateDelays) <= Ain.T @ t

    input_timing = getInputTimingConstr(gateDelays, t)

    return constr_timing , input_timing

    # TODO: generalize


def getInputTimingConstr(gateDelays, t):
    constr = gateDelays[0:2] <= t[0:2]

    return constr


""" Get maximum delay - that is the upper bound 
    on the overall delay and that is the max of 
    arrival times for output gates

        Params:
          t : array [1xK] arrival times variable
        Return: [1x1] maximum of all path delays       """


def getMaximumDelay(t):  # TODO: generalize

    varsToMin = cp.hstack( [t[5], t[6]] )
    circuitDelay = cp.max(varsToMin)

    return circuitDelay


""" Compute total power 

        as sum_i ( f_i * e_i * x_i ) ;    
        Params:
          frequencies : array [1xN] of gate frequencies
          energyLoss : array [1xN] of energy loss of each gate
          x : [1xN] variable
        Return: Double, total power        """


def computeTotalPower(frequencies, energyLoss, x):
    power = cp.sum(cp.multiply((cp.multiply(frequencies, x)), energyLoss))
    return power

    """ Compute total area

        as sum_i ( gateScale_i * x_i ) ;                       
        Params:
          gateScales : array [1xN] of unit scaling factor of gate i
          x : [1x1] variable scale of gate i 
        Return: Double, total area        """


def computeTotalArea(gateScales, x):
    area = cp.sum(cp.multiply(gateScales, x))
    return area




def optimizeGates(frequencies, energyLoss, gateScales, alphas, betas, gammas, maxArea, maxPower,
                  loadCapacitances, A, numberOfCells):
    # optimization variables
    x = cp.Variable(numberOfCells, pos=True)  # sizes
    t = cp.Variable(numberOfCells, pos=True)  # arrival times

    Aout, Ain = computeIncidenceMatrices(A)

    alphasParam = cp.Parameter(alphas.shape, pos=True)
    alphasParam.value = alphas

    betasParam = cp.Parameter(betas.shape, pos=True)
    betasParam.value = betas

    gammasParam = cp.Parameter(gammas.shape, pos=True)
    gammasParam.value = gammas

    inputCap = computeInputCapacitance(alphasParam, betasParam, x)
    computedLCapacitance = computeLoadCapacitance(Aout, Ain, inputCap)

    gateDelays = computeGateDelays(computedLCapacitance, gammasParam, x)
    maxDelay = getMaximumDelay(t)

    totalPower = computeTotalPower(frequencies, energyLoss, x)
    totalArea = computeTotalArea(gateScales, x)

    # get constraints

    cloadConstr = getConstrCload(computedLCapacitance, loadCapacitances)
    timingConstr, inputTimingConstr = getConstrTiming(Aout, Ain, gateDelays, t)

    posX = x >= 1  # all sizes greater than 1 (normalized)expr

    # formulate the GGP

    # constraints = [totalPower <= maxPower, totalArea <= maxArea, timingConstr, inputTimingConstr, posX]
    constraints = [totalPower <= maxPower, totalArea <= maxArea, timingConstr, posX]

    objective = cp.Minimize(maxDelay)

    prob = cp.Problem(objective, constraints)
    prob.solve(gp=True, verbose=False, solver=cp.MOSEK)

    print("sizing params: ", x.value)

    return prob.value


print(optimizeGates(f, e, a, alpha, beta, gamma, Amax, Pmax, [Cout6, Cout7], A, numberOfCells))
