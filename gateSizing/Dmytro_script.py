import cvxpy as cp
import numpy as np

import histogramGenerator
import matplotlib.pyplot as plt
from randomVariableHist_Numpy import RandomVariable

def main():

    sigma = 1
    mu = 2
    distr = "Gauss"
    # distr = "LogNormal"

    numberOfBins = 20
    numberOfUnaries = 100
    numberOfSamples = 1000000
    interval = (-2, 10)

    constraints = []

    # generate inputs
    xs = {}

    g = histogramGenerator.get_gauss_bins_UNARY(mu, sigma, numberOfBins, numberOfSamples,
                                                    interval, numberOfUnaries, distr)

    for bin in range(0, numberOfBins):
        xs[bin] = {}
        for unary in range(0, numberOfUnaries):
            xs[bin][unary] = cp.Variable(boolean=True)

            constraints.append(xs[bin][unary] >= g.bins[bin][unary])

    # objective function
    sum = 0
    for bin in range(0, numberOfBins):
        for unary in range(0, numberOfUnaries):
            sum += xs[bin][unary]

    # solve

    objective = cp.Minimize(sum)
    prob = cp.Problem(objective, constraints)

    prob.solve(verbose=True, solver=cp.GUROBI,
               MIPGAP=0.01,  # relative gap
               TimeLimit=1200,  # 'MSK_DPAR_OPTIMIZER_MAX_TIME': 1200}  # max time
               )

    # get the values
    newBins = np.zeros((numberOfBins, numberOfUnaries))
    for bin in range(0, numberOfBins):
        for unary in range(0, numberOfUnaries):
            newBins[bin, unary] = xs[bin][unary].value

    rv = RandomVariable(newBins, g.edges, unary=True)
    Aprx_normalHist = histogramGenerator.get_Histogram_from_UNARY(rv)

        # get generated values
    normalGeneratedHist = histogramGenerator.get_Histogram_from_UNARY(g)

    gauss = lambda x: 1/np.sqrt(2*np.pi)/sigma * np.exp(-(x-mu)**2/2/sigma**2)
    xdata = np.linspace(interval[0],interval[1],200)
    ydata = gauss(xdata)

    # plot the results
    # plt.plot(xdata,ydata)
    plt.hist(normalGeneratedHist.edges[:-1], normalGeneratedHist.edges, weights=normalGeneratedHist.bins, color='blue')
    plt.hist(Aprx_normalHist.edges[:-1], Aprx_normalHist.edges, weights=Aprx_normalHist.bins, color='orange')

    plt.show()


if __name__ == "__main__":
    main()
