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
numberOfUnaries=8
numberOfBins=8
interval=(-2, 10)
withSymmetryConstr=True

# set number of samples for the distributions
n_samples = 2000000
seed = 0


# set gate parameters
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

# CVXPY part starts here

constraints = []

# generate inputs
startingNodes = []
xs_starting = {}

I1 = histogramGenerator.get_gauss_bins_UNARY(
    input_means[0],
    input_stds[0],
    numberOfBins,
    n_samples,
    interval,
    numberOfUnaries,
)

# uncomment to see the input distribution
# print(I1.bins)

xs_starting[0] = {}

# create variables
for bin in range(0, numberOfBins):
    xs_starting[0][bin] = {}
    for unary in range(0, numberOfUnaries):
        xs_starting[0][bin][unary] = cp.Variable(boolean=True)
        
        # add the S <= G constraint from paper
        constraints.append(xs_starting[0][bin][unary] <= I1.bins[bin][unary])

node = Node(RandomVariableCVXPY(xs_starting[0], I1.edges))
startingNodes.append(node)

I2 = histogramGenerator.get_gauss_bins_UNARY(
    input_means[1],
    input_stds[1],
    numberOfBins,
    n_samples,
    interval,
    numberOfUnaries,
)

# uncomment to see the input distribution
# print(I2.bins)

xs_starting[1] = {}

# create variables
for bin in range(0, numberOfBins):
    xs_starting[1][bin] = {}
    for unary in range(0, numberOfUnaries):
        xs_starting[1][bin][unary] = cp.Variable(boolean=True)
        
        # add the S <= G constraint from paper
        constraints.append(xs_starting[1][bin][unary] <= I2.bins[bin][unary])

node = Node(RandomVariableCVXPY(xs_starting[1], I2.edges))
startingNodes.append(node)


    # generetate gate distribution
generatedNodes = []
xs_generated = {}
g = histogramGenerator.get_gauss_bins_UNARY(
    gateParams[0],
    gateParams[1],
    numberOfBins,
    n_samples,
    interval,
    numberOfUnaries,
)
xs_generated[0] = {}

for bin in range(0, numberOfBins):
    xs_generated[0][bin] = {}
    for unary in range(0, numberOfUnaries):
        xs_generated[0][bin][unary] = cp.Variable(boolean=True)

        # add the S <= G constraint from paper
        constraints.append(xs_generated[0][bin][unary] <= g.bins[bin][unary])

node = Node(RandomVariableCVXPY(xs_generated[0], g.edges))
generatedNodes.append(node)

 
####################################
####### set circuit design ########
####################################

# set inputs
startingNodes[0].setNextNodes([generatedNodes[0]])
startingNodes[1].setNextNodes([generatedNodes[0]])

# calculate all algorithms
delays, newConstr = SSTA.calculateCircuitDelay(
    startingNodes, cvxpy=True, unary=True, withSymmetryConstr=withSymmetryConstr
)
delays = delays[numberOfGates + 1 :]

# add new constraints generated during the traverse
constraints.extend(newConstr)

# setting objective (1^T @ H @ 1)
sum = 0
for bin in range(0, numberOfBins):
    for unary in range(0, numberOfUnaries):
        sum += delays[-1].bins[bin][unary]

# solve
objective = cp.Maximize(sum)
prob = cp.Problem(objective, constraints)

prob.solve(
    verbose=True,
    solver=cp.MOSEK
)


####################################
####### get result ########
####################################

rvs = []


finalBins = np.zeros((numberOfBins, numberOfUnaries))
for bin in range(0, numberOfBins):
    for unary in range(0, numberOfUnaries):
        finalBins[bin, unary] = ((delays[0].bins)[bin])[unary].value

rvs.append(
    RandomVariable(finalBins, generatedNodes[0].randVar.edges, unary=True)
)

print("Final sink distribution:")
print(rvs[-1].bins)

lastGate = (rvs[-1].mean, rvs[-1].std)

print("Mean and std: ")
print(lastGate)
