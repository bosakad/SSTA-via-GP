import numpy as np
import cvxpy as cp



def computeInputCapacitance(alphas, betas, x):
    """
    Calculate input capacitance as affine function ... alpha + beta * x

    :param alphas: array of alphas
    :param betas: array of betas
    :param x: cvxpy variable (1, n)
    :return inputCap: (1, n) array of input capacitances
    """

    inputCap = alphas + cp.multiply(x, betas)
    return inputCap


def computeLoadCapacitance(inputCapacitance, loadCapacitances, numberOfGates):
    """
    Load capacitance is computed as a sum of a fanout.

    :param inputCapacitance: (1, n) cvxpy var of input capacitanes
    :param loadCapacitances: (1, m) of load capacitances of output gates
    :param numberOfGates: int, total number of gates
    :return cload: (1, n) array of load capacitances
    """

    cload = [None] * numberOfGates

    cload[0] = inputCapacitance[3]
    cload[1] = inputCapacitance[3] + inputCapacitance[4]
    cload[2] = inputCapacitance[4] + inputCapacitance[6]
    cload[3] = inputCapacitance[5] + inputCapacitance[6]
    cload[4] = inputCapacitance[6]

    cload[5] = loadCapacitances[0]
    cload[6] = loadCapacitances[1]

    return cp.hstack(cload)  # concatenation


def computeGateDelays(capLoad, gammas, x):
    """
    Delay on each gate is computed as (load capacitance * gamma) / resistance

    :param capLoad: (1, n) cvxpy var of load capacitanes
    :param gammas: (1, n) gammas
    :param x: (1, n) cvxpy variable
    :return cload: (1, n) array of load capacitances
    """

    cloadTimesGamma = cp.multiply ( capLoad , gammas )
    gateDelays = cp.multiply (cloadTimesGamma , 1 / x)

    return gateDelays


def getPathDelays(gateDelays):
    """
    Delay on each gate is computed as (load capacitance * gamma) / resistance

    :param gateDelays: (1, n) cvxpy variable of delays
    :return cload: (1, m) cvxpy array of all delay paths
    """
    delays = [gateDelays[0] + gateDelays[3] + gateDelays[5],
              gateDelays[0] + gateDelays[3] + gateDelays[6],
              gateDelays[1] + gateDelays[3] + gateDelays[5],
              gateDelays[1] + gateDelays[3] + gateDelays[6],
              gateDelays[1] + gateDelays[4] + gateDelays[6],
              gateDelays[2] + gateDelays[4] + gateDelays[5],
              gateDelays[2] + gateDelays[6]]

    return cp.hstack(delays)


def getMaximumDelay(pathDelays):
    """
    Delay on each gate is computed as (load capacitance * gamma) / resistance

   :param pathDelays: (1, m) cvxpy variable of path delays
   :return circuitDelay: cvxpy variable - max delay
   """

    circuitDelay = cp.max(pathDelays)
    return circuitDelay


def computeTotalPower(frequencies, energyLoss, x):
    """
    Compute total power as sum_i ( f_i * e_i * x_i )

    :param frequencies: (1, n) cvxpy variable of gate frequencies
    :param energyLoss: (1, n) cvxpy variable of energy loss of each gate
    :param x: (1, n) cvxpy variable
    :return circuitDelay: cvxpy variable - max delay
    """

    power = cp.sum(cp.multiply((cp.multiply(frequencies, x)), energyLoss))
    return power


def computeTotalArea(gateScales, x):
    """
    Compute total area as sum_i ( gateScale_i * x_i )

    :param gateScales: (1, n) array of unit scaling factor of gate i
    :param x: (1, n) cvxpy variable
    :return area: Double, total area
    """

    area = cp.sum(cp.multiply(gateScales, x))
    return area


def getDelaySSTA():
    """
    Just a test function
    :return delay: array of delays for each gate
    :return constr: constraints for the delays
    """

    numberOfBins = 10
    numberOfUnaries = 10

    xs = {}

    # create a variable as a dict.
    for gate in range(0, numberOfGates):
        xs[gate] = {}
        for bin in range(0, numberOfBins):
            (xs[gate])[bin] = {}
            for unary in range(0, numberOfUnaries):
                ((xs[gate])[bin])[unary] = cp.Variable(boolean=True)


    sum = 0
    for gate in range(0, numberOfGates):
        for bin in range(0, numberOfBins):
            for unary in range(0, numberOfUnaries):
               sum += ((xs[gate])[bin])[unary]

    print(sum)
    obj = cp.Variable(pos=True)
    constr = [ obj >= sum ]

    return obj, constr


def optimizeGates(frequencies, energyLoss, gateScales, alphas, betas, gammas, maxArea, maxPower, loadCapacitances,
                  numberOfGates, delaysRVs=None):
    # defining variable
    x = cp.Variable(numberOfGates, pos=True)

    # computing the objective function

    inputCapacitance = computeInputCapacitance(alphas, betas, x)
    loadCapacitance = computeLoadCapacitance(inputCapacitance, loadCapacitances, numberOfGates)
    gateDelays = computeGateDelays(loadCapacitance, gammas, x)

    pathDelays = getPathDelays(gateDelays)
    circuitDelay = getMaximumDelay(pathDelays)

    # circuitDelay, otherConstr = getDelaySSTA() # test


    # computing the constraints

    totalPower = computeTotalPower(frequencies, energyLoss, x)
    totalArea = computeTotalArea(gateScales, x)

    # formulating GGP

    constraints = [totalPower <= maxPower, totalArea <= maxArea]
    # constraints.extend(otherConstr)   # test
    objective = cp.Minimize(circuitDelay)

    prob = cp.Problem(objective, constraints)
    prob.solve(gp=True, verbose=True, solver=cp.MOSEK)

    print("sizing params: ", x.value)

    return prob.value




# hard coded example

numberOfGates = 7
# numberOfGates = 5

f = np.array([4, 0.8, 1, 0.8, 1.7, 0.5, 2.5])
e = np.array([1, 2, 1, 1.5, 1.5, 1, 0.2])
Cout6 = 7
Cout7 = 5

a = np.ones(numberOfGates)
alpha = np.ones(numberOfGates)
beta = np.ones(numberOfGates)
gamma = np.ones(numberOfGates)

Amax = 30
Pmax = 55


optimizeGates(f, e, a, alpha, beta, gamma, Amax, Pmax, [Cout6, Cout7], numberOfGates)
