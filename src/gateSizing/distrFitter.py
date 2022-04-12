import histogramGenerator
import numpy as np
import matplotlib.pyplot as plt

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
    mus = [1.9,1.9, 1.9, 1.9, 1, 0.9, 0.8, 0.7]
    stds = [0.165,0.164, 0.163, 0.161, 0.16, 0.14, 0.13, 0.11]


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


    plt.savefig("Inputs.outputs/generatedDistros.jpeg", dpi=500)
    # plt.show()

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

    # def f (a, p): return coef[0] + coef[1]*a + coef[2]*p + coef[3]* np.square(a) + coef[4]* np.square(p)
    def f (a, p): return coef[0] + coef[1]*a + coef[2]*p

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


def linearRegression(distros, area, power):
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
    coef = np.zeros((numberOfBins, 3))

    for bin in range(0, numberOfBins):

        A = np.zeros((numberOfDistros, 3))

        A[:, 0] = 1     # b
        A[:, 1] = area[:]   # area
        A[:, 2] = power[:]  # power
        # A[:, 3] = np.square(area[:])  # area
        # A[:, 4] = np.square(power[:])  # area

        b = distros[:, bin]


            # Rx = Q^T b
        Q, R = np.linalg.qr(A)
        x = np.linalg.solve(R, (Q.T@b).T)
        coef[bin, :] = x[:]

        #
        # print(coef[bin, 0] + coef[bin, 1] * 5 + coef[bin, 2] * 20)
        # print(coef[bin, 0] + coef[bin, 1] * 7 + coef[bin, 2] * 30)
        # print(coef[bin, 0] + coef[bin, 1] * 8 + coef[bin, 2] * 40)
        # print(distros[:, bin])
        # print()

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

if __name__ == "__main__":

    # parameters
    # area = np.array([1, 4, 5, 15, 20, 25, 35])
    # power = np.array([1, 4, 5, 40, 80, 100, 140])
    area = np.array([10, 20, 30, 35, 50, 55, 70])
    power = np.array([10, 20, 30, 40, 45, 60, 65])

    # area = np.array([5, 7, 8])
    # power = np.array([20, 30, 40])

    interval = (0, 10)
    numberOfBins = 100

    distros, edges = generateDistr(area, power, interval, numberOfBins, shouldSave=False)
    plotDistros(distros, edges)
    coef = linearRegression(distros, area, power)

    plotLinesForBin(distros, area, power, coef, 24)

    saveModel(coef)

    print(coef)

