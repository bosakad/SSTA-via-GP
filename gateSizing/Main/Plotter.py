import matplotlib.pyplot as plt
import numpy as np
import math

"""
    This module includes plotting functions for numberOfGates/ number of non zeros graphs.
"""

def parseVerbose(verboseFile):
    """
    :return nonzeros: number of nonzeros after presolving
    """

    verbose = open(verboseFile)


    nonZeros = [[], []]

    AlgIndex = -1
    iterMet = False
    while True:
        line = verbose.readline()
        if not line:
            break

        if iterMet == False and ". iteration" in line:
            if "0. iteration:" in line:
                print('new')
                AlgIndex += 1
            iterMet = True


        if iterMet and "Presolved" in line:
            nonZ = [int(s) for s in line.split() if s.isdigit()]
            nonZeros[AlgIndex].append(nonZ[-1])
            iterMet = False

    return np.array(nonZeros)

def plotNonzeros():
    file = open("../Inputs/testParsing.txt")    # file with 1 line - dictionary
    verbose = "../Inputs/verbose.stdout"        # file with verbose text, should be complex problem - so there is 'Presolved'
    readFromVerbose = True

    line = file.readline()

    results = eval(line)    # load dictionary

    Gates = np.array([], dtype=int)
    zerosWithConstr = np.array([])
    zerosNoConstr = np.array([])
    errorWithConstr = np.array([])
    errorNoConstr = np.array([])


    if readFromVerbose:
        nonZ = parseVerbose(verbose)
        print(nonZ.shape)
        zerosNoConstr = nonZ[0][:]
        zerosWithConstr = nonZ[1][:]


    # store data from dictionary into numpy array
    for gateNum, WithConstr in results:

        if WithConstr:
            if not readFromVerbose:
                zerosWithConstr = np.append(zerosWithConstr, results[(gateNum, WithConstr)][0])
            errorWithConstr = np.append(errorWithConstr, results[(gateNum, WithConstr)][2])
            errorWithConstr = np.append(errorWithConstr, results[(gateNum, WithConstr)][3])
        else:
            if not readFromVerbose:
                zerosNoConstr = np.append(zerosNoConstr, results[(gateNum, WithConstr)][0])
            errorNoConstr = np.append(errorNoConstr, results[(gateNum, WithConstr)][2])
            errorNoConstr = np.append(errorNoConstr, results[(gateNum, WithConstr)][3])

        if gateNum not in Gates:
            Gates = np.append(Gates, gateNum)


    errorNoConstr = errorNoConstr.reshape((len(errorNoConstr)//2, 2))
    errorWithConstr = errorWithConstr.reshape((len(errorWithConstr)//2, 2))

    print(errorNoConstr)
    print(zerosNoConstr)
    print(Gates)

    # set histograms
    fig, axs = plt.subplots(3, 1)
    # axs[0].hist(zerosWithConstr, bins=Gates, color='blue', edgecolor='black')
    # axs[0].hist(Gates, Gates, weights=zerosNoConstr, color='orange', edgecolor='black')
    # axs[0].plot(Gates, zerosWithConstr, color='blue')
    axs[0].plot(Gates, zerosNoConstr, color='orange')
    # axs[0].scatter(Gates, zerosWithConstr, color='blue')
    axs[0].scatter(Gates, zerosNoConstr, color='orange')

    # axs[1].plot(Gates, errorWithConstr[:, 0], color='blue')
    axs[1].plot(Gates, errorNoConstr[:, 0], color='orange')
    # axs[1].scatter(Gates, errorWithConstr[:, 0], color='blue')
    axs[1].scatter(Gates, errorNoConstr[:, 0], color='orange')
    # axs[1].hist(Gates, Gates, weights=errorWithConstr[:, 0], color='blue', edgecolor='black')
    # axs[1].hist(Gates, Gates, weights=errorNoConstr[:, 0], color='orange', edgecolor='black')
    # axs[2].hist(Gates, Gates, weights=errorWithConstr[:, 1], color='blue')
    # axs[2].hist(Gates, Gates, weights=errorNoConstr[:, 1], color='orange')
    # axs[2].scatter(Gates, errorWithConstr[:, 1], color='blue')
    axs[2].scatter(Gates, errorNoConstr[:, 1], color='orange')
    # axs[2].plot(Gates, errorWithConstr[:, 1], color='blue')
    axs[2].plot(Gates, errorNoConstr[:, 1], color='orange')

    for ax in axs.flat:
        ax.set(xlabel='Number Of Gates')

    axs.flat[0].set(ylabel='Nonzeros')
    axs.flat[1].set(ylabel='Mean Error')
    axs.flat[2].set(ylabel='Std Error')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    # set legend
    axs[0].legend(["With constraints", "Without constraints"])
    axs[1].legend(["With constraints", "Without constraints"])
    axs[2].legend(["With constraints", "Without constraints"])

    # plt.show()
    plt.savefig("../Inputs/scaling.jpeg", dpi=500)

if __name__ == "__main__":
    plotNonzeros()
