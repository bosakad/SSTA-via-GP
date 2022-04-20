import histogramGenerator
import numpy as np
import matplotlib.pyplot as plt
from randomVariableHist_Numpy import RandomVariable
import cvxpy as cp

"""
This module has funtions that generate distribution with a given parameters (such as area of the circuit or max power)
and fits the curve (linear reg.)

"""

def generateDistr(area: [], power: [], interval: tuple, numberOfBins: int, shouldSave: bool):
    """
    Generates a distributions with parameters and saves it into numpy file.

    :param area: array of area constraints
    :param power: array of power constraints
    :param interval: tuple - interval
    :param numberOfBins: int - number of bins
    :param shouldSave: boolean - true if should save into file, false otherwise

    :return distros: (numberOfDistros, NumberOfBins) np matrix of generated values
    :return edges: (1, numberOfBins + 1) np array of edges
    """



        # some random parameters: todo: make them real
    # mus = [1, 1.3, 1.4, 1.5, 2.6]
    # stds = [0.17, 0.16, 0.15, 0.13, 0.12]
    mus = [2, 1.95, 1.85, 1.9, 1, 0.9, 0, 0]
    stds = [0.165,0.164, 0.163, 0.161, 0.16, 0.14, 0.13, 0.1]


        # generate distr
    numberOfDistr = area.shape[0]

    edges = None
    distros = np.zeros((numberOfDistr, numberOfBins))
    for d in range(0, numberOfDistr):
        rv = histogramGenerator.get_gauss_bins(mus[d], stds[d], numberOfBins, numberOfSamples=1000000,
                                                                binsInterval=interval, distr="LogNormal")
        distros[d, :] = rv.bins[:]

        edges = rv.edges



    if shouldSave:

            # save data
        outfile = "Inputs.outputs/generatedDistros"

        np.savez(outfile, distr=distros, edges=edges, area=area, power=power, interval=interval, numberBins=numberOfBins)

    return distros, edges


def plotDistros(distros, edges):
    """
    Plot distros
    :param distros: (numberOfDistros, NumberOfBins) np matrix of generated values
    :param edges: (1, numberOfBins + 1) np array of edges
    """

    numberOfDistros = distros.shape[0]

    for d in range(0, numberOfDistros):
        bins = distros[d, :]
        plt.hist(edges[:-1], edges, weights=bins, density="PDF")


    # plt.savefig("Inputs.outputs/generatedDistros.jpeg", dpi=500)
    plt.show()

def plotLinesForBin(distros, area, power, coef, bin):
    """
    Plot distros and lines
    :param distros: (numberOfDistros, NumberOfBins) np matrix of generated values
    :param area: array of area constraints
    :param power: array of power constraints
    :param coef: (numberOfBins, 3) - 3 coeficients for each bins - 1. = b, 2. = area, 3. = power
                                                                  - reg([a, p]) = b + a* area + p* power
    """
    coef = coef[bin, :]
    ax = plt.axes(projection='3d')

    # Data for a three-dimensional line
    # zline = np.linspace(0, 15, 1000)
    # xline = np.sin(zline)
    # yline = np.cos(zline)
    # ax.plot3D(xline, yline, zline, 'gray')

    ax.scatter3D(area[:], power[:], distros[:, bin], c='red',marker='^')

    # print(area)
    # print(power)
    # print(distros[:, bin])

    minA = np.min(area)
    maxA = np.max(area)
    minP = np.min(power)
    maxP = np.max(power)

    def f (a, p): return coef[0]*a + coef[1]*p + coef[2]* np.power(a, -1) + coef[3]* np.power(p, -1)
    # def f (a, p): return coef[0] + coef[1]*a + coef[2]*p

    # print(f(5, 20))
    # print(f(7, 30))
    # print(f(8, 40))


    precision = 10
    areas = np.linspace(minA, maxA, precision)
    powers = np.linspace(minP, maxP, precision)

    values = np.zeros((precision, precision))
    areas = np.outer(areas, np.ones(precision)).T
    powers = np.outer(powers, np.ones(precision))


    for i in range(0, precision):
        for j in range(0, precision):
            values[i, j] = f(areas[i, j], powers[i, j])


    # print(areas.shape)
    # print(powers.shape)
    # print(values.shape)


    ax.plot_surface(areas, powers, values.T, cmap='viridis', edgecolor='none')
    # print(area[:])

    plt.show()
    # plt.savefig("Inputs.outputs/fitting.jpeg", dpi=500)


def linearRegression(distros, area, power, asQP=False):
    """
    Performs linear regression on the generated data

    :param distros: (numberOfDistros, NumberOfBins) np matrix of generated values
    :param area: array of area constraints
    :param power: array of power constraints
    :return coef: (numberOfBins, 3) - 3 coeficients for each bins - 1. = b, 2. = area, 3. = power
                                                                  - reg([a, p]) = b + a* area + p* power
    """

    numberOfBins = distros.shape[1]
    numberOfDistros = distros.shape[0]
    coef = np.zeros((numberOfBins, 4))

    for bin in range(0, numberOfBins):

        A = np.zeros((numberOfDistros, 4))

        # A[:, 0] = 1     # b
        A[:, 0] = area[:]   # area
        A[:, 1] = power[:]  # power
        A[:, 2] = np.power(area[:], -1)  # area
        A[:, 3] = np.power(power[:], -1)  # power

        print(area)
        print(power)

        b = distros[:, bin]


        # if asQP:
        #     pass
        # else:

        x = cp.Variable(4, pos=True)
        # constr = [x >= 0]
        cost = cp.sum_squares(A @ x - b)
        prob = cp.Problem(cp.Minimize(cost), [])
        prob.solve()

        # print(x.value)

        # x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        coef[bin, :] = np.array(x.value)

        print(x.value)

        # print(x)

        # exit(-1)

    coef[coef <= 0] += 0.000000000000000000000000001

    print(coef)
    return coef

def saveModel(coef):
    """
    Saves model
    :param coef: coeficients of the model
    :return: None
    """

    outfile = "Inputs.outputs/model"

    np.savez(outfile, coef=coef)

    return None

def plotDistrosForInputs(a, f, e):

    interval = (0, 20)

    coef = np.load("Inputs.outputs/model.npz")
    model = coef['coef']
    print(model)
    numberOfBins = model.shape[0]

    gate = 0

    a_i = a[gate]
    f_i = f[gate]
    e_i = e[gate]

    x = 10

    # for x in [1, 1.5, 2, 2.5, 3, 3.5, 4]:
    for x in [1, 2, 4, 6, 10, 20, 100]:
    # for x in [20, 100]:

        # x = 1 + 2*iter

        distr = np.zeros(numberOfBins)
        for bin in range(0, numberOfBins):
            a1 = model[bin, 0]
            p1 = model[bin, 1]
            a2 = model[bin, 2]
            p2 = model[bin, 3]

            prob = a1*a_i*x + p1*f_i*e_i*x + a2* (1/ (a_i*x)) + p2* (1/(f_i*e_i*x))
            distr[bin] = prob

        print(distr)
        STATIC_BINS = np.linspace(interval[0], interval[1], numberOfBins+1)

        plt.hist(STATIC_BINS[:-1], STATIC_BINS[:], weights=distr, density="PDF", alpha=0.4)

        rv = RandomVariable(distr, edges=STATIC_BINS)

        print(rv.mean, rv.std)

        plt.legend([str(x)])
        plt.show()

    # plt.legend(["1", '2', '3', '4'])


    # plt.show()


if __name__ == "__main__":

    # parameters
    # area = np.array([1, 4, 5, 15, 20, 25, 35])
    # power = np.array([1, 4, 5, 40, 80, 100, 140])

    area = np.array([1, 2, 3, 4, 5., 6, 7]) ** 1.6
    power = np.array([1, 2, 3, 4, 5, 6, 7]) ** 1.6
    # e = np.array([1, 2, 1, 1.5, 1.5, 1])

    # area = np.array([10, 20, 30, 35, 50, 55, 70])
    # power = np.array([10, 20, 30, 40, 45, 60, 65])

    # area = np.array([5, 7, 8])
    # power = np.array([20, 30, 40])

    interval = (0, 28)
    numberOfBins = 6

    distros, edges = generateDistr(area, power, interval, numberOfBins, shouldSave=False)
    # plotDistros(distros, edges)
    coef = linearRegression(distros, area, power)

    plotLinesForBin(distros, area, power, coef, 5)

    saveModel(coef)

    f = np.array([4, 0.8, 1, 0.8, 1.7, 0.5])
    e = np.array([1, 2, 1, 1.5, 1.5, 1])
    a = np.ones(6)
    plotDistrosForInputs(a, f, e)
