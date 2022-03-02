import matplotlib.pyplot as plt
import numpy as np
import math

"""
    This module includes plotting functions for numberOfGates/ number of non zeros graphs.
"""


def plotNonzeros():
    file = open("../Inputs/testParsing.txt")
    line = file.readline()

    results = eval(line)    # load dictionary

    Gates = np.array([], dtype=int)
    numZerosPrecise = np.array([])
    numZerosNonPrecise = np.array([])
    errorPrecise = np.array([])
    errorNonPrecise = np.array([])

        # store data from dictionary into numpy array
    for gateNum, precise in results:

        if precise:
            numZerosPrecise = np.append(numZerosPrecise, results[(gateNum, precise)][0])
            errorPrecise = np.append(errorPrecise, results[(gateNum, precise)][2])
        else:
            numZerosNonPrecise = np.append(numZerosNonPrecise, results[(gateNum, precise)][0])
            errorNonPrecise = np.append(errorNonPrecise, results[(gateNum, precise)][2])

        if gateNum not in Gates:
            Gates = np.append(Gates, gateNum)


        # set histograms
    fig, axs = plt.subplots(2, 1)
    axs[0].hist(Gates, Gates, weights=numZerosPrecise, color='blue')
    # axs[0].set_title('Precise')
    axs[0].hist(Gates, Gates, weights=numZerosNonPrecise, color='orange')

    axs[1].hist(Gates, Gates, weights=errorPrecise, color='blue')
    axs[1].hist(Gates, Gates, weights=errorNonPrecise, color='orange')
    # axs[0].set_title('Less-Precise')
    # axs[1, 0].hist(Gates, Gates, weights=numZerosPrecise, color='red')
    # axs[1, 1].hist(Gates, Gates, weights=numZerosPrecise, color='orange')

    for ax in axs.flat:
        ax.set(xlabel='Number Of Gates')

    axs.flat[0].set(ylabel='Nonzeros')
    axs.flat[1].set(ylabel='Error')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    # set legend
    axs[0].legend(["Precise", "Less-Precise"])
    axs[1].legend(["Precise", "Less-Precise"])

    plt.show()


if __name__ == "__main__":
    plotNonzeros()
