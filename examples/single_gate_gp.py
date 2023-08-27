import numpy as np
from src.timing.cvxpyVariable import RandomVariableCVXPY
import cvxpy as cp
from src.timing.randomVariableHist_Numpy import RandomVariable
import src.utility_tools.histogramGenerator as histogramGenerator
from src.utility_tools.node import Node
import src.timing.SSTA as SSTA
import matplotlib.pyplot as plt


# set parameters
numberOfGates=1
numberOfBins=25
interval=(-4, 25)

n_samples = 2000000
seed = 0

gateParams = [0.0, 1.0]

# fix a random seed seed exists
if seed != None:
    seed = seed
    np.random.seed(seed)

####################################
####### Generate Input data ########
####################################

# list with inputs' mean values
input_means = [np.random.randint(20, 70) / 10 for _ in range(numberOfGates + 1)]
# list with inputs' stds
input_stds = [np.random.randint(20, 130) / 100 for _ in range(numberOfGates + 1)]

# CVXPY

constraints = []

# generate inputs
startingNodes = []
xs_starting = {}

# generate input 1
I1 = histogramGenerator.get_gauss_bins(
    input_means[0], input_stds[0], numberOfBins, n_samples, interval, forGP=True
)

# create variables
xs_starting[0] = {}
for bin in range(0, numberOfBins):
    xs_starting[0][bin] = cp.Variable(pos=True)

    constraints.append(xs_starting[0][bin] >= I1.bins[bin])

node = Node(RandomVariableCVXPY(xs_starting[0], I1.edges))
startingNodes.append(node)

# generate input 2
I2 = histogramGenerator.get_gauss_bins(
    input_means[1], input_stds[1], numberOfBins, n_samples, interval, forGP=True
)

# create variables
xs_starting[1] = {}
for bin in range(0, numberOfBins):
    xs_starting[1][bin] = cp.Variable(pos=True)

    # add S <= G
    constraints.append(xs_starting[1][bin] >= I2.bins[bin])

node = Node(RandomVariableCVXPY(xs_starting[1], I2.edges))
startingNodes.append(node)


# generetate nodes
generatedNodes = []
xs_generated = {}
g = histogramGenerator.get_gauss_bins(
    gateParams[0], gateParams[1], numberOfBins, n_samples, interval, forGP=True
)
xs_generated[0] = {}

# create variables
for bin in range(0, numberOfBins):
    xs_generated[0][bin] = cp.Variable(pos=True)

    # add S <= G
    constraints.append(xs_generated[0][bin] >= g.bins[bin])

node = Node(RandomVariableCVXPY(xs_generated[0], g.edges))
generatedNodes.append(node)

####################################
####### set circuit design ########
####################################

startingNodes[0].setNextNodes([generatedNodes[0]])
startingNodes[1].setNextNodes([generatedNodes[0]])


delays, newConstr = SSTA.calculateCircuitDelay(startingNodes, cvxpy=True, GP=True)
delays = delays[numberOfGates + 1 :]

# add constraints calcuated during SSTA
constraints.extend(newConstr)

# setting objective
sum = 0
for bin in range(0, numberOfBins):
    sum += delays[-1].bins[bin]
    
####################################
####### solve ########
####################################

objective = cp.Minimize(sum) # this is same as maximize with different inequalities in traversal and S >= G instead of S <= G
prob = cp.Problem(objective, constraints)

prob.solve(
    verbose=True,
    solver=cp.MOSEK,
    gp=True,
    mosek_params={
        "MSK_DPAR_INTPNT_CO_TOL_MU_RED": 0.1,
        "MSK_DPAR_OPTIMIZER_MAX_TIME": 1200,
    }
    
)

time = prob.solver_stats.solve_time


####################################
####### get result ########
####################################

rvs = []

for gate in range(0, len(delays)):  # construct RVs

    finalBins = np.zeros(numberOfBins)
    for bin in range(0, numberOfBins):
        
        finalBins[bin] = ((delays[gate].bins)[bin]).value

    rvs.append(RandomVariable(finalBins, generatedNodes[0].randVar.edges))

print("Final sink distribution:")
print(rvs[-1].bins)

lastGate = (rvs[-1].mean, rvs[-1].std)

print("Mean and std: ")
print(lastGate)


