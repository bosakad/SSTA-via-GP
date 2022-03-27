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
    file = open("Inputs.outputs/testParsing.txt")    # file with 1 line - dictionary
    verbose = "../Inputs/verbose.stdout"        # file with verbose text, should be complex problem - so there is 'Presolved'
    readFromVerbose = False
    plotUnder = True
    only1 = True


    line = file.readline()

    results = eval(line)    # load dictionary

    Gates = np.array([], dtype=int)
    zerosWithConstr = np.array([])
    zerosNoConstr = np.array([])
    errorWithConstr = np.array([])
    errorNoConstr = np.array([])
    timeWithConstr = np.array([])
    timeNoConstr = np.array([])

    varsWithConstr = np.array([])
    varsNoConstr = np.array([])

    constrWithConstr = np.array([])
    constrNoConstr = np.array([])

    if readFromVerbose:
        nonZ = parseVerbose(verbose)
        zerosNoConstr = nonZ[0][:]
        zerosWithConstr = nonZ[1][:]


    # store data from dictionary into numpy array
    for gateNum, WithConstr in results:

        if WithConstr:
            if not readFromVerbose:
                zerosWithConstr = np.append(zerosWithConstr, results[(gateNum, WithConstr)][0])
            timeWithConstr = np.append(timeWithConstr, results[(gateNum, WithConstr)][2])
            errorWithConstr = np.append(errorWithConstr, results[(gateNum, WithConstr)][3])
            errorWithConstr = np.append(errorWithConstr, results[(gateNum, WithConstr)][4])
            varsWithConstr = np.append(varsWithConstr, results[(gateNum, WithConstr)][5])
            constrWithConstr = np.append(constrWithConstr, results[(gateNum, WithConstr)][6])
        else:
            if not readFromVerbose:
                zerosNoConstr = np.append(zerosNoConstr, results[(gateNum, WithConstr)][0])

            timeNoConstr = np.append(timeNoConstr, results[(gateNum, WithConstr)][2])
            errorNoConstr = np.append(errorNoConstr, results[(gateNum, WithConstr)][3])
            errorNoConstr = np.append(errorNoConstr, results[(gateNum, WithConstr)][4])
            varsNoConstr = np.append(varsNoConstr, results[(gateNum, WithConstr)][5])
            constrNoConstr = np.append(constrNoConstr, results[(gateNum, WithConstr)][6])

        if gateNum not in Gates:
            Gates = np.append(Gates, gateNum)


    errorNoConstr = errorNoConstr.reshape((len(errorNoConstr)//2, 2))
    errorWithConstr = errorWithConstr.reshape((len(errorWithConstr)//2, 2))


    if not plotUnder:
        # set histograms
        fig, axs = plt.subplots(3, 2)

        # nonzeros
        axs[0, 0].plot(Gates, zerosWithConstr, color='blue')
        axs[0, 0].plot(Gates, zerosNoConstr, color='orange')
        axs[0, 0].scatter(Gates, zerosWithConstr, color='blue')
        axs[0, 0].scatter(Gates, zerosNoConstr, color='orange')

        # variables
        axs[1, 0].scatter(Gates, varsWithConstr, color='blue')
        axs[1, 0].scatter(Gates, varsNoConstr, color='orange')
        axs[1, 0].plot(Gates, varsWithConstr, color='blue')
        axs[1, 0].plot(Gates, varsNoConstr, color='orange')

        # constraints
        axs[2, 0].scatter(Gates, constrWithConstr, color='blue')
        axs[2, 0].scatter(Gates, constrNoConstr, color='orange')
        axs[2, 0].plot(Gates, constrWithConstr, color='blue')
        axs[2, 0].plot(Gates, constrNoConstr, color='orange')

        # time
        axs[0, 1].scatter(Gates, timeWithConstr, color='blue')
        axs[0, 1].scatter(Gates, timeNoConstr, color='orange')
        axs[0, 1].plot(Gates, timeWithConstr, color='blue')
        axs[0, 1].plot(Gates, timeNoConstr, color='orange')

        # error mean
        axs[1, 1].plot(Gates, errorWithConstr[:, 0], color='blue')
        axs[1, 1].plot(Gates, errorNoConstr[:, 0], color='orange')
        axs[1, 1].scatter(Gates, errorWithConstr[:, 0], color='blue')
        axs[1, 1].scatter(Gates, errorNoConstr[:, 0], color='orange')

        # error std
        axs[2, 1].scatter(Gates, errorWithConstr[:, 1], color='blue')
        axs[2, 1].scatter(Gates, errorNoConstr[:, 1], color='orange')
        axs[2, 1].plot(Gates, errorWithConstr[:, 1], color='blue')
        axs[2, 1].plot(Gates, errorNoConstr[:, 1], color='orange')

        for ax in axs.flat:
            ax.set(xlabel='Number Of Gates')

        axs.flat[0].set(ylabel='Nonzeros')
        axs.flat[1].set(ylabel='MAPE Mean')
        axs.flat[2].set(ylabel='MAPE std')
        axs.flat[3].set(ylabel='Time')
        axs.flat[4].set(ylabel='Variables')
        axs.flat[5].set(ylabel='Constraints')

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()

        # set legend
        axs[0, 0].legend(["With constraints", "Without constraints"])
        axs[1, 0].legend(["With constraints", "Without constraints"])
        axs[2, 0].legend(["With constraints", "Without constraints"])
        axs[0, 1].legend(["With constraints", "Without constraints"])
        axs[1, 1].legend(["With constraints", "Without constraints"])
        axs[2, 1].legend(["With constraints", "Without constraints"])
    else:
        # set histograms
        fig, axs = plt.subplots(3, 1)
        i = 0


        # nonzeros
        axs[i].plot(Gates, zerosWithConstr, color='blue')
        axs[i].scatter(Gates, zerosWithConstr, color='blue')

        if not only1:
            axs[i].plot(Gates, zerosNoConstr, color='orange')
            axs[i].scatter(Gates, zerosNoConstr, color='orange')
        axs.flat[i].set(ylabel='Nonzeros')
        # axs[i].legend(["With constraints", "Without constraints"])
        i += 1

        # variables
        axs[i].scatter(Gates, varsWithConstr, color='blue')
        axs[i].plot(Gates, varsWithConstr, color='blue')
        if not only1:
            axs[i].scatter(Gates, varsNoConstr, color='orange')
            axs[i].plot(Gates, varsNoConstr, color='orange')
        axs.flat[i].set(ylabel='Variables')
        # axs[i].legend(["With constraints", "Without constraints"])
        i += 1

        # constraints
        axs[i].scatter(Gates, constrWithConstr, color='blue')
        axs[i].plot(Gates, constrWithConstr, color='blue')
        if not only1:
            axs[i].scatter(Gates, constrNoConstr, color='orange')
            axs[i].plot(Gates, constrNoConstr, color='orange')
        axs.flat[i].set(ylabel='Constraints')
        # axs[i].legend(["With constraints", "Without constraints"])
        i += 1

        #
        # # time
        # axs[i].scatter(Gates, timeWithConstr, color='blue')
        # axs[i].scatter(Gates, timeNoConstr, color='orange')
        # axs[i].plot(Gates, timeWithConstr, color='blue')
        # axs[i].plot(Gates, timeNoConstr, color='orange')
        # axs.flat[i].set(ylabel='Time')
        # axs[i].legend(["With constraints", "Without constraints"])
        # i += 1
        #
        # # error mean
        # axs[i].plot(Gates, errorWithConstr[:, 0], color='blue')
        # axs[i].plot(Gates, errorNoConstr[:, 0], color='orange')
        # axs[i].scatter(Gates, errorWithConstr[:, 0], color='blue')
        # axs[i].scatter(Gates, errorNoConstr[:, 0], color='orange')
        # axs.flat[i].set(ylabel='MAPE Mean')
        # axs[i].legend(["With constraints", "Without constraints"])
        # i += 1
        #
        # # error std
        # axs[i].scatter(Gates, errorWithConstr[:, 1], color='blue')
        # axs[i].scatter(Gates, errorNoConstr[:, 1], color='orange')
        # axs[i].plot(Gates, errorWithConstr[:, 1], color='blue')
        # axs[i].plot(Gates, errorNoConstr[:, 1], color='orange')
        # axs.flat[i].set(ylabel='MAPE std')
        # axs[i].legend(["With constraints", "Without constraints"])
        # i += 1


        for ax in axs.flat:
            ax.set(xlabel='Number Of Gates')


        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()


    # plt.show()
    plt.savefig("Inputs.outputs/scaling.jpeg", dpi=500)


def plotPresolve():
    file = open("Inputs.outputs/testParsing.txt")    # file with 1 line - dictionary
    verbose = "../Inputs/verbose.stdout"        # file with verbose text, should be complex problem - so there is 'Presolved'
    readFromVerbose = False
    plotUnder = True


    line = file.readline()

    results = eval(line)    # load dictionary

    Gates = np.array([], dtype=int)

    if readFromVerbose:
        nonZ = parseVerbose(verbose)
        zerosNoConstr = nonZ[0][:]
        zerosWithConstr = nonZ[1][:]

    nonZeros = {}
    vars = {}
    constraints = {}

    # store data from dictionary into numpy array
    for gateNum, passes in results:

        if passes not in nonZeros:
            nonZeros[passes] = np.array([])
            vars[passes] = np.array([])
            constraints[passes] = np.array([])

        nonZeros[passes] = np.append(nonZeros[passes], results[(gateNum, passes)][0])
        vars[passes] = np.append(vars[passes], results[(gateNum, passes)][1])
        constraints[passes] = np.append(constraints[passes], results[(gateNum, passes)][2])

        if gateNum not in Gates:
            Gates = np.append(Gates, gateNum)



    # set histograms
    fig, axs = plt.subplots(3, 1)
    i = 0

        # non zeros
    colors = ['blue', 'orange', 'green']
    legend = [0, 0, 0]
    for passes in nonZeros:
        # nonzeros
        axs[0].plot(Gates, nonZeros[passes], color=colors[i])
        axs[0].scatter(Gates, nonZeros[passes], color=colors[i])
        legend[i] = passes
        i += 1

    axs.flat[0].set(ylabel='Nonzeros')
    axs[0].legend(legend)


        # number of variables
    colors = ['blue', 'orange', 'green']
    legend = [0, 0, 0]
    i = 0
    for passes in vars:
        # nonzeros
        axs[1].plot(Gates, vars[passes], color=colors[i])
        axs[1].scatter(Gates, vars[passes], color=colors[i])
        legend[i] = passes
        i += 1

    axs.flat[1].set(ylabel='Variables')
    axs[1].legend(legend)

        # number of constraints
    colors = ['blue', 'orange', 'green']
    legend = [0, 0, 0]
    i = 0
    for passes in constraints:
        # nonzeros
        axs[2].plot(Gates, constraints[passes], color=colors[i])
        axs[2].scatter(Gates, constraints[passes], color=colors[i])
        legend[i] = passes
        i += 1

    axs.flat[2].set(ylabel='Constraints')
    axs[2].legend(legend)


    for ax in axs.flat:
        ax.set(xlabel='Number Of Gates')


    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()


    # plt.show()
    plt.savefig("Inputs.outputs/scaling.jpeg", dpi=500)

def plotForThesis():
    from randomVariableHist_Numpy import RandomVariable
    bins = np.array([0, 0.25, 0.5, 0.25, 0])
    edges = np.array([0, 1, 2, 3, 4, 5])


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim(0, 0.75)

    rv = RandomVariable(bins, edges)
    plt.hist(rv.edges[:-1], rv.edges, weights=rv.bins, density="PDF", color='orange')
    # plt.show()


    for i, j in zip(edges[2:-1], bins[1:-1]):
        ax.annotate(str(j), xy=(i - 0.6, j + 0.005))

    plt.savefig("Inputs.outputs/exampleHist.jpeg", dpi=500)

    # plt.show()

if __name__ == "__main__":
    plotNonzeros()
    # plotPresolve()
    # plotForThesis()
