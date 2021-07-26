import numpy as np
import cvxpy as cp
import mosek

# hard coded example

numberOfGates = 7

f = np.array([1, 0.8, 1, 0.7, 0.7, 0.5, 0.5])
e = np.array([1, 2, 1, 1.5, 1.5, 1, 2])
Cout6 = 10
Cout7 = 10

a = np.ones(numberOfGates)
alpha = np.ones(numberOfGates)
beta = np.ones(numberOfGates)
gamma = np.ones(numberOfGates)

Amax = 25
Pmax = 50

""" Calculate input capacitance as affine function ... alpha + beta * x

      Params:
        alphas: array [1xN] of ???
        betas: array [1xN] of ???
        x: variable [1xN]
      Return: array [1xN] of load capacitances """


def computeInputCapacitance(alphas, betas, x):
    inputCap = alphas + cp.multiply(x, betas)
    return inputCap

    # TODO - generalize
    """ Load capacitance is computed as a sum of a fanout.
  
        Params:
          inputCapacitance : array variable [1xN] of input capacitances
          loadCapacitances : array [1xM] of load capacitances of output gates
          numberOfGates : Integer, total number of gates
        Return: array [1xN] of load capacitances
    """


def computeLoadCapacitance(inputCapacitance, loadCapacitances, numberOfGates):
    cload = [None] * numberOfGates

    cload[0] = inputCapacitance[3]
    cload[1] = inputCapacitance[3] + inputCapacitance[4]
    cload[2] = inputCapacitance[4] + inputCapacitance[6]
    cload[3] = inputCapacitance[5] + inputCapacitance[6]
    cload[4] = inputCapacitance[6]

    cload[5] = loadCapacitances[0]
    cload[6] = loadCapacitances[1]

    return cp.hstack(cload)  # concatenation

    """ Delay on each gate is computed as (load capacitance * gamma) / resistance 
  
        Params:
          capLoad : [1xN] of load capacitances
          gammas : array [1xN] of  gamma constants
          x : [1xN] variable
          numberOfGates : Integer, total number of gates
        Return: array [1xN] of gate delays        """


def computeGateDelays(capLoad, gammas, x, numberOfGates):
    cloadTimesGamma = cp.multiply ( capLoad , gamma )
    gateDelays = cp.multiply (cloadTimesGamma , 1 / x)

    return gateDelays

    # TODO: generalize, use SSTA
    """ Get all possible path delays
  
        Params:
          gateDelays : array [1xN] of gate delays
        Return: array [1xK] of all possible delay paths        """


def getPathDelays(gateDelays):
    delays = [gateDelays[0] + gateDelays[3] + gateDelays[5],
              gateDelays[0] + gateDelays[3] + gateDelays[6],
              gateDelays[1] + gateDelays[3] + gateDelays[5],
              gateDelays[1] + gateDelays[3] + gateDelays[6],
              gateDelays[1] + gateDelays[4] + gateDelays[6],
              gateDelays[2] + gateDelays[4] + gateDelays[5],
              gateDelays[2] + gateDelays[6]]

    return cp.hstack(delays)

    """ Get maximum delay
  
        Params:
          pathDelays : array [1xK] of all possible delay paths
        Return: [1x1] maximum of all path delays       """


def getMaximumDelay(pathDelays):
    circuitDelay = cp.max(pathDelays)
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




def optimizeGates(frequencies, energyLoss, gateScales, alphas, betas, gammas, maxArea, maxPower, loadCapacitances,
                  numberOfGates):
    # defining variable
    x = cp.Variable(numberOfGates, pos=True)

    inputCapacitance = computeInputCapacitance(alphas, betas, x)
    loadCapacitance = computeLoadCapacitance(inputCapacitance, loadCapacitances, numberOfGates)

    gateDelays = computeGateDelays(loadCapacitance, gammas, x, numberOfGates)

    pathDelays = getPathDelays(gateDelays)
    circuitDelay = getMaximumDelay(pathDelays)

    totalPower = computeTotalPower(frequencies, energyLoss, x)
    totalArea = computeTotalArea(gateScales, x)

    # formulating GGP

    constraints = [totalPower <= maxPower, totalArea <= maxArea]
    objective = cp.Minimize(circuitDelay)

    prob = cp.Problem(objective, constraints)
    prob.solve(gp=True, verbose=True, solver=cp.MOSEK)

    print("sizing params: ", x.value)

    return prob.value;


# hard coded example
optimizeGates(f, e, a, alpha, beta, gamma, Amax, Pmax, [Cout6, Cout7], numberOfGates)
