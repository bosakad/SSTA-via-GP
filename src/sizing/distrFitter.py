import src.other.histogramGenerator as histogramGenerator
import numpy as np
import matplotlib.pyplot as plt
from src.timing.randomVariableHist_Numpy import RandomVariable
import cvxpy as cp
import matplotlib.transforms as mtransforms

"""
This module has funtions that generate distribution with a given parameters (such as area of the circuit or max power)
and fits the curve (linear reg.)

"""


def generateDistr(
    area: [],
    power: [],
    interval: tuple,
    numberOfBins: int,
    shouldSave: bool,
    GP=False,
    Normal=False,
):
    """
    Generates a distributions with parameters and saves it into numpy file.

    :param area: array of area constraints
    :param power: array of power constraints
    :param interval: tuple - interval
    :param numberOfBins: int - number of bins
    :param shouldSave: boolean - true if should save into file, false otherwise
    :param Normal: boolean - true if normal, false if lognormal

    :return distros: (numberOfDistros, NumberOfBins) np matrix of generated values
    :return edges: (1, numberOfBins + 1) np array of edges
    """

    if GP and not Normal:
        mus = [1.97, 1.95, 1.85, 1.3, 0.8, 0.5, 0.3, 0.15]
        stds = [0.165, 0.163, 0.162, 0.161, 0.16, 0.14, 0.14, 0.13]
    elif GP and Normal:
        mus = np.array([10.8, 4.8, 4.8, 4.5, 4, 3.5, 3.2, 0.3]) / 2.5
        stds = np.array([1, 0.8, 0.72, 0.71, 0.66, 0.44, 0.31, 0.21]) * 1.5
    else:
        mus = [1.1, 1.1, 1.05, 1, 1, 0.9, 0, 0]
        stds = [0.165, 0.164, 0.163, 0.161, 0.16, 0.14, 0.13, 0.1]

        # generate distr
    numberOfDistr = area.shape[0]

    edges = None
    distros = np.zeros((numberOfDistr, numberOfBins))
    for d in range(0, numberOfDistr):
        if Normal:
            rv = histogramGenerator.get_gauss_bins(
                mus[d],
                stds[d],
                numberOfBins,
                numberOfSamples=1000000,
                binsInterval=interval,
            )
        else:
            rv = histogramGenerator.get_gauss_bins(
                mus[d],
                stds[d],
                numberOfBins,
                numberOfSamples=1000000,
                binsInterval=interval,
                distr="LogNormal",
            )
        distros[d, :] = rv.bins[:]

        edges = rv.edges

    if shouldSave:

        # save data
        outfile = "../../inputs_outputs/generatedDistros"

        np.savez(
            outfile,
            distr=distros,
            edges=edges,
            area=area,
            power=power,
            interval=interval,
            numberBins=numberOfBins,
        )

    return distros, edges


def plotDistros(distros, edges):
    """
    Plot distributions

    :param distros: (numberOfDistros, NumberOfBins) np matrix of generated values
    :param edges: (1, numberOfBins + 1) np array of edges
    """

    numberOfDistros = distros.shape[0]

    for d in range(0, numberOfDistros):
        bins = distros[d, :]
        plt.hist(edges[:-1], edges, weights=bins, density="PDF")

    plt.show()


def plotLinesForBin(distros, area, power, coef, bin, GP=False):
    """
    Plot distros and lines

    :param distros: (numberOfDistros, NumberOfBins) np matrix of generated values
    :param area: array of area constraints
    :param power: array of power constraints
    :param coef: (numberOfBins, 3) - 3 coeficients for each bins - 1. = b, 2. = area, 3. = power
                                                                  - reg([a, p]) = b + a* area + p* power
    """
    coef = coef[bin, :]
    ax = plt.axes(projection="3d")

    # Data for a three-dimensional line
    # zline = np.linspace(0, 15, 1000)
    # xline = np.sin(zline)
    # yline = np.cos(zline)
    # ax.plot3D(xline, yline, zline, 'gray')

    ax.scatter3D(area[:], power[:], distros[:, bin], c="red", marker="^")


    minA = np.min(area)
    maxA = np.max(area)
    minP = np.min(power)
    maxP = np.max(power)

    if GP:

        def f(a, p):
            return (
                coef[0] * a
                + coef[1] * p
                + coef[2] * np.power(a, -1)
                + coef[3] * np.power(p, -1)
            )

    else:

        def f(a, p):
            return coef[0] + coef[1] * a + coef[2] * p


    precision = 10
    areas = np.linspace(minA, maxA, precision)
    powers = np.linspace(minP, maxP, precision)

    values = np.zeros((precision, precision))
    areas = np.outer(areas, np.ones(precision)).T
    powers = np.outer(powers, np.ones(precision))

    for i in range(0, precision):
        for j in range(0, precision):
            values[i, j] = f(areas[i, j], powers[i, j])


    ax.plot_surface(areas, powers, values.T, cmap="viridis", edgecolor="none")
    # print(area[:])

    plt.show()


def linearRegression(distros, area, power, GP=False):
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

    if GP:
        numberOfCoefs = 4
    else:
        numberOfCoefs = 3

    coef = np.zeros((numberOfBins, numberOfCoefs))

    for bin in range(0, numberOfBins):

        A = np.zeros((numberOfDistros, numberOfCoefs))

        if GP:
            A[:, 0] = area[:]  # area
            A[:, 1] = power[:]  # power
            A[:, 2] = np.power(area[:], -1)  # area
            A[:, 3] = np.power(power[:], -1)  # power
        else:
            A[:, 0] = 1  # b
            A[:, 1] = area[:]  # area
            A[:, 2] = power[:]  # power

        b = distros[:, bin]

        if GP:
            x = cp.Variable((4,), pos=True)
        else:
            x = cp.Variable((3,))

        cost = cp.sum_squares(A @ x - b)
        prob = cp.Problem(cp.Minimize(cost), [])
        prob.solve()

        # x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        coef[bin, :] = np.array(x.value)

        # print(x.value)

    if GP:
        coef[coef <= 0] += 0.000000000000000000000000001

    return coef


def saveModel(coef, GP=False, Normal=False):
    """
    Saves model

    :param coef: coeficients of the model
    :param Normal: true if normal, false if lognormal
    :return: None
    """

    if GP:
        if Normal:
            outfile = "../../inputs_outputs/model_Normal"
        else:
            outfile = "../../inputs_outputs/model"
    else:
        outfile = "../../inputs_outputs/model_MIXED_INT"

    np.savez(outfile, coef=coef)

    return None


def plotDistrosForInputs(a, f, e, GP=False):

    interval = (0, 35)

    if not GP:
        coef = np.load("../../inputs_outputs/model_MIXED_INT.npz")
        coefs = [coef]
    else:
        coef1 = np.load("../../inputs_outputs/model.npz")
        coef2 = np.load("../../inputs_outputs/model_Normal.npz")
        coefs = [coef1, coef2]

    gate = 5

    a_i = a[gate]
    f_i = f[gate]
    e_i = e[gate]

    if GP:
        numIter = 2
    else:
        numIter = 1

    fig, axs = plt.subplots(
        6, 2, gridspec_kw={"wspace": 0.5, "hspace": 0.5}, sharex=True
    )
    data = [1, 2, 5, 10, 15, 25]

    for i in range(0, numIter):
        model = coefs[i]["coef"]
        numberOfBins = model.shape[0]

        STATIC_BINS = np.linspace(
            interval[0] / 1e11, interval[1] / 1e11, numberOfBins + 1
        )

        for j in range(0, len(data)):
            x = data[j]
            distr = np.zeros(numberOfBins)
            for bin in range(0, numberOfBins):
                if GP:
                    a1 = model[bin, 0]
                    p1 = model[bin, 1]
                    a2 = model[bin, 2]
                    p2 = model[bin, 3]

                    prob = (
                        a1 * a_i * x
                        + p1 * f_i * e_i * x
                        + a2 * (1 / np.power((a_i * x), 1))
                        + p2 * (1 / np.power((f_i * e_i * x), 1))
                    )
                else:
                    shift = model[bin, 0]
                    ac = model[bin, 1]
                    pc = model[bin, 2]

                    prob = shift + ac * a_i * x + pc * f_i * e_i * x

                distr[bin] = prob

            first = 28

            axs[j, i].hist(
                STATIC_BINS[: -(1 + first)],
                STATIC_BINS[:-first],
                weights=distr[:-first],
                density="PDF",
                alpha=0.4,
            )

            rv = RandomVariable(distr, edges=STATIC_BINS)

            # plt.legend(["Scaling factor: " + str(x)])

    axs[5, 0].set_xlabel("Delay(sec.)")
    axs[2, 0].set_ylabel("PDF")
    axs[5, 1].set_xlabel("Delay(sec.)", fontsize=10)
    axs[2, 1].set_ylabel("PDF")

    axs[0, 0].set_title("LogNormal")
    axs[0, 1].set_title("Guassian")

    labels = ["a", "b", "c", "d", "e", "f", "a'", "b'", "c'", "d'", "e'", "f'"]
    j = 0
    for ax in axs[:, 0]:

        # print(ax.xlabel)
        trans = mtransforms.ScaledTranslation(120 / 72, -5 / 72, fig.dpi_scale_trans)
        ax.text(
            0.0,
            1.0,
            labels[j],
            transform=ax.transAxes + trans,
            fontsize="medium",
            verticalalignment="top",
            fontfamily="DejaVu Sans",
            weight="bold",
            bbox=dict(facecolor="1", edgecolor="none", pad=3.0),
        )
        j += 1

    for ax in axs[:, 1]:
        # print(ax.xlabel)
        trans = mtransforms.ScaledTranslation(120 / 72, -5 / 72, fig.dpi_scale_trans)
        ax.text(
            0.0,
            1.0,
            labels[j],
            transform=ax.transAxes + trans,
            fontsize="medium",
            verticalalignment="top",
            fontfamily="DejaVu Sans",
            weight="bold",
            bbox=dict(facecolor="1", edgecolor="none", pad=3.0),
        )
        j += 1

    # plt.show()
    plt.savefig("../../inputs_outputs/distributionFactors.jpeg", dpi=800, bbox_inches="tight")


if __name__ == "__main__":

    # parameters

    area = np.array([1, 2, 3, 4, 5.0, 6, 7]) ** 1.9
    power = np.array([1, 2, 3, 4, 5, 6, 7]) ** 1.9

    interval = (0, 35)
    numberOfBins = 35
    # numberOfBins = 50
    asGp = True
    Normal = True

    distros, edges = generateDistr(
        area, power, interval, numberOfBins, shouldSave=False, GP=asGp, Normal=Normal
    )
    # plotDistros(distros, edges)
    coef = linearRegression(distros, area, power, GP=asGp)

    # plotLinesForBin(distros, area, power, coef, 0, GP=asGp)

    # saveModel(coef, GP=asGp, Normal=Normal)

    f = np.array([4, 0.8, 1, 0.8, 1.7, 0.5])
    e = np.array([1, 2, 1, 1.5, 1.5, 1])
    a = np.ones(6)
    plotDistrosForInputs(a, f, e, GP=asGp)
