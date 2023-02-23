import matplotlib.pyplot as plt
import numpy as np
import matplotlib.transforms as mtransforms
import src.sizing.optimizeGates as optimizeGates
from src.timing.randomVariableHist_Numpy import RandomVariable

"""
    This module includes plotting functions for thesis.
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
                print("new")
                AlgIndex += 1
            iterMet = True

        if iterMet and "Presolved" in line:
            nonZ = [int(s) for s in line.split() if s.isdigit()]
            nonZeros[AlgIndex].append(nonZ[-1])
            iterMet = False

    return np.array(nonZeros)


def parseVerboseGP(verboseFile):
    """
    :return nonzeros: number of nonzeros after presolving
    """

    verbose = open(verboseFile)

    variables = []
    constrs = []
    cones = []

    AlgIndex = -1
    iterMet = False
    opt=False
    while True:
        line = verbose.readline()
        if not line:
            break

        if iterMet == False and ". iteration" in line:
            if "0. iteration:" in line:
                AlgIndex += 1
            iterMet = True

        if iterMet == True and "Presolve started" in line:
            opt = True

        if iterMet and opt and "Constraints" in line:
            nonZ = [int(s) for s in line.split() if s.isdigit()]
            constrs.append(nonZ[-1])

        if iterMet and opt and "Cones" in line:
            nonZ = [int(s) for s in line.split() if s.isdigit()]
            cones.append(nonZ[-1])

        if iterMet and opt and "Scalar variables" in line:
            nonZ = [int(s) for s in line.split() if s.isdigit()]
            variables.append(nonZ[-1])
            iterMet = False
            opt = False

        # parse dict
        if "{" in line:
            dict = eval(line)


    return constrs, cones, variables, dict

def plotComparison(res1=None, res2=None, res3=None):
    # file = open("Inputs.outputs/testParsing.txt")  # file with 1 line - dictionary

    # line = file.readline()

    # results = eval(line)  # load dictionary

    Gates = np.array([], dtype=int)
    mip1 = np.array([])
    mip2 = np.array([])
    mip3 = np.array([])
    time1 = np.array([])
    time2 = np.array([])
    time3 = np.array([])

    # store data from dictionary into numpy array
    for gateNum, WithConstr in res1:

        time1 = np.append(time1, res1[(gateNum, WithConstr)][2])
        mip1 = np.append(mip1, res1[(gateNum, WithConstr)][7])

        if gateNum not in Gates:
            Gates = np.append(Gates, gateNum)

    # store data from dictionary into numpy array
    for gateNum, WithConstr in res2:

        time2 = np.append(time2, res2[(gateNum, WithConstr)][2])
        mip2 = np.append(mip2, res2[(gateNum, WithConstr)][7])

        if gateNum not in Gates:
            Gates = np.append(Gates, gateNum)

    # store data from dictionary into numpy array
    for gateNum, WithConstr in res3:

        time3 = np.append(time3, res3[(gateNum, WithConstr)][2])
        mip3 = np.append(mip3, res3[(gateNum, WithConstr)][7])

        if gateNum not in Gates:
            Gates = np.append(Gates, gateNum)

    fig, axs = plt.subplots(
        2, 1, gridspec_kw={"wspace": 0.3, "hspace": 0.3}, sharex=True
    )
    i = 0

    # nonzeros

    p0 = axs[i].plot(
        Gates,
        mip1,  # data
        marker="o",  # each marker will be rendered as a circle
        markersize=6,  # marker size
        markerfacecolor="red",  # marker facecolor
        markeredgecolor="black",  # marker edgecolor
        markeredgewidth=1,  # marker edge width
        linestyle="-",  # line style will be dash line
        linewidth=3,
        zorder=3,
    )  # line width

    p1 = axs[i].plot(
        Gates,
        mip2,  # data
        marker="o",  # each marker will be rendered as a circle
        markersize=6,  # marker size
        markerfacecolor="red",  # marker facecolor
        markeredgecolor="black",  # marker edgecolor
        markeredgewidth=1,  # marker edge width
        linestyle="-",  # line style will be dash line
        linewidth=4,
        zorder=2,
    )  # line width

    p2 = axs[i].plot(
        Gates,
        mip3,  # data
        marker="o",  # each marker will be rendered as a circle
        markersize=6,  # marker size
        markerfacecolor="red",  # marker facecolor
        markeredgecolor="black",  # marker edgecolor
        markeredgewidth=1,  # marker edge width
        linestyle="-",  # line style will be dash line
        linewidth=4,
        zorder=1,
    )  # line width

    axs.flat[i].set(ylabel="Mip gap at root (%)")
    i += 1

    p0 = axs[i].plot(
        Gates,
        time1,  # data
        marker="o",  # each marker will be rendered as a circle
        markersize=6,  # marker size
        markerfacecolor="red",  # marker facecolor
        markeredgecolor="black",  # marker edgecolor
        markeredgewidth=1,  # marker edge width
        linestyle="-",  # line style will be dash line
        linewidth=4,
        zorder=3,
    )  # line width

    p1 = axs[i].plot(
        Gates,
        time2,  # data
        marker="o",  # each marker will be rendered as a circle
        markersize=6,  # marker size
        markerfacecolor="red",  # marker facecolor
        markeredgecolor="black",  # marker edgecolor
        markeredgewidth=1,  # marker edge width
        linestyle="-",  # line style will be dash line
        linewidth=4,
        zorder=2,
    )  # line width

    p2 = axs[i].plot(
        Gates,
        time3,  # data
        marker="o",  # each marker will be rendered as a circle
        markersize=6,  # marker size
        markerfacecolor="red",  #    marker facecolor
        markeredgecolor="black",  # marker edgecolor
        markeredgewidth=1,  # marker edge width
        linestyle="-",  # line style will be dash line
        linewidth=4,
        zorder=1,
    )  # line width

    axs.flat[i].set(ylabel="Time (seconds)")
    i += 1

    # plt.savefig("scalingComparison.png", dpi=1000)
    plt.show()


def plotNonzeros(results, only1=True):
    # file = open("Inputs.outputs/testParsing.txt")  # file with 1 line - dictionary
    # verbose = "../Inputs/verbose.stdout"  # file with verbose text, should be complex problem - so there is 'Presolved'
    readFromVerbose = False
    plotUnder = True
    forPaper = True


    # line = file.readline()

    # results = eval(line)  # load dictionary

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

    mipGapWithConstr = np.array([])
    mipGapNoConstr = np.array([])

    # if readFromVerbose:
    #     nonZ = parseVerbose(verbose)
    #     zerosNoConstr = nonZ[0][:]
    #     zerosWithConstr = nonZ[1][:]

    # store data from dictionary into numpy array
    for gateNum, WithConstr in results:

        if WithConstr:
            if not readFromVerbose:
                zerosWithConstr = np.append(
                    zerosWithConstr, results[(gateNum, WithConstr)][0]
                )
            timeWithConstr = np.append(
                timeWithConstr, results[(gateNum, WithConstr)][2]
            )
            errorWithConstr = np.append(
                errorWithConstr, results[(gateNum, WithConstr)][3]
            )
            errorWithConstr = np.append(
                errorWithConstr, results[(gateNum, WithConstr)][4]
            )
            varsWithConstr = np.append(
                varsWithConstr, results[(gateNum, WithConstr)][5]
            )
            constrWithConstr = np.append(
                constrWithConstr, results[(gateNum, WithConstr)][6]
            )
            mipGapWithConstr = np.append(
                mipGapWithConstr, results[(gateNum, WithConstr)][7]
            )
        else:
            if not readFromVerbose:
                zerosNoConstr = np.append(
                    zerosNoConstr, results[(gateNum, WithConstr)][0]
                )

            timeNoConstr = np.append(timeNoConstr, results[(gateNum, WithConstr)][2])
            errorNoConstr = np.append(errorNoConstr, results[(gateNum, WithConstr)][3])
            errorNoConstr = np.append(errorNoConstr, results[(gateNum, WithConstr)][4])
            varsNoConstr = np.append(varsNoConstr, results[(gateNum, WithConstr)][5])
            constrNoConstr = np.append(
                constrNoConstr, results[(gateNum, WithConstr)][6]
            )
            mipGapNoConstr = np.append(
                mipGapNoConstr, results[(gateNum, WithConstr)][7]
            )

        if gateNum not in Gates:
            Gates = np.append(Gates, gateNum)

    errorNoConstr = errorNoConstr.reshape((len(errorNoConstr) // 2, 2))
    errorWithConstr = errorWithConstr.reshape((len(errorWithConstr) // 2, 2))

    # print(errorWithConstr)

    if forPaper:

        if only1:
            # set histograms
            fig, axs = plt.subplots(4, 1, gridspec_kw={"wspace": 0.5, "hspace": 0.5})
        else:
            fig, axs = plt.subplots(3, 1, gridspec_kw={"wspace": 0.5, "hspace": 0.5})

        i = 0

        # nonzeros

        p0 = axs[i].plot(
            Gates,
            zerosWithConstr,  # data
            marker="o",  # each marker will be rendered as a circle
            markersize=4.5,  # marker size
            markerfacecolor="red",  # marker facecolor
            markeredgecolor="black",  # marker edgecolor
            markeredgewidth=1,  # marker edge width
            linestyle="-",  # line style will be dash line
            linewidth=3,
            zorder=3,
        )  # line width

        p1 = axs[i].plot(
            Gates,
            varsWithConstr,  # data
            marker="o",  # each marker will be rendered as a circle
            markersize=4.5,  # marker size
            markerfacecolor="red",  # marker facecolor
            markeredgecolor="black",  # marker edgecolor
            markeredgewidth=1,  # marker edge width
            linestyle="-",  # line style will be dash line
            linewidth=3,
            zorder=2,
        )  # line width

        p2 = axs[i].plot(
            Gates,
            constrWithConstr,  # data
            marker="o",  # each marker will be rendered as a circle
            markersize=4.5,  # marker size
            markerfacecolor="red",  # marker facecolor
            markeredgecolor="black",  # marker edgecolor
            markeredgewidth=1,  # marker edge width
            linestyle="-",  # line style will be dash line
            linewidth=3,
            zorder=1,
        )  # line width

        # axs[i].scatter(Gates, varsWithConstr, color='orange')
        #
        # axs[i].scatter(Gates, constrWithConstr, color='green')
        #
        # # axs[i].plot(Gates, zerosWithConstr, color='blue')
        # axs[i].plot(Gates, varsWithConstr, color='orange')
        # axs[i].plot(Gates, constrWithConstr, color='green')

        axs.flat[i].set(ylabel="Count")
        # axs[i].legend(["Nonzeros", "Variables", "Constraints"])
        i += 1

        # constraints

        if only1:
            p2 = axs[i].plot(Gates, mipGapWithConstr,  # data
                             marker='o',  # each marker will be rendered as a circle
                             markersize=5,  # marker size
                             markerfacecolor='red',  # marker facecolor
                             markeredgecolor='black',  # marker edgecolor
                             markeredgewidth=1,  # marker edge width
                             linestyle='-',  # line style will be dash line
                             linewidth=3.5, zorder=1)  # line width
            axs.flat[i].set(ylabel='Mip gap \nat root(%)')
            i += 1

        # time
        # axs[i].scatter(Gates, timeWithConstr, color='blue')
        # axs[i].plot(Gates, timeWithConstr, color='blue')
        p2 = axs[i].plot(
            Gates,
            timeWithConstr,  # data
            marker="o",  # each marker will be rendered as a circle
            markersize=5,  # marker size
            markerfacecolor="red",  # marker facecolor
            markeredgecolor="black",  # marker edgecolor
            markeredgewidth=1,  # marker edge width
            linestyle="-",  # line style will be dash line
            linewidth=3.5,
            zorder=1,
        )  # line width
        axs.flat[i].set(ylabel="Time\n(seconds)")
        # axs[i].legend(["With constraints", "Without constraints"])
        i += 1

        # error mean
        p2 = axs[i].plot(
            Gates,
            errorWithConstr[:, 0],  # data
            marker="o",  # each marker will be rendered as a circle
            markersize=5,  # marker size
            markerfacecolor="red",  # marker facecolor
            markeredgecolor="black",  # marker edgecolor
            markeredgewidth=1,  # marker edge width
            linestyle="-",  # line style will be dash line
            linewidth=3.5,
            zorder=1,
        )  # line width
        p2 = axs[i].plot(
            Gates,
            errorWithConstr[:, 1],  # data
            marker="o",  # each marker will be rendered as a circle
            markersize=5,  # marker size
            markerfacecolor="red",  # marker facecolor
            markeredgecolor="black",  # marker edgecolor
            markeredgewidth=1,  # marker edge width
            linestyle="-",  # line style will be dash line
            linewidth=3.5,
            zorder=1,
        )  # line width

        # plt.fill_between(Gates, errorWithConstr[:, 1], y2=errorWithConstr[:, 0], alpha=0.3, color='orange')
        # plt.fill_between(Gates, errorWithConstr[:, 0], alpha=0.3, color='blue')
        plt.fill_between(Gates, errorWithConstr[:, 1], alpha=0.3, color="orange")
        plt.fill_between(
            Gates, errorWithConstr[:, 0], errorWithConstr[:, 1], alpha=0.3, color="blue"
        )

        axs.flat[i].set(ylabel="Relative Error(%)")
        i += 1

        for ax in axs.flat:
            ax.set(xlabel="Number of gates")

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()

        labels = ["a", "b", "c", "d"]
        j = 0
        for ax in axs:
            # label physical distance in and down:
            trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
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

    else:

        if not plotUnder:
            # set histograms
            fig, axs = plt.subplots(3, 2)

            # nonzeros
            axs[0, 0].plot(Gates, zerosWithConstr, color="blue")
            axs[0, 0].plot(Gates, zerosNoConstr, color="orange")
            axs[0, 0].scatter(Gates, zerosWithConstr, color="blue")
            axs[0, 0].scatter(Gates, zerosNoConstr, color="orange")

            # variables
            axs[1, 0].scatter(Gates, varsWithConstr, color="blue")
            axs[1, 0].scatter(Gates, varsNoConstr, color="orange")
            axs[1, 0].plot(Gates, varsWithConstr, color="blue")
            axs[1, 0].plot(Gates, varsNoConstr, color="orange")

            # constraints
            axs[2, 0].scatter(Gates, constrWithConstr, color="blue")
            axs[2, 0].scatter(Gates, constrNoConstr, color="orange")
            axs[2, 0].plot(Gates, constrWithConstr, color="blue")
            axs[2, 0].plot(Gates, constrNoConstr, color="orange")

            # time
            axs[0, 1].scatter(Gates, timeWithConstr, color="blue")
            axs[0, 1].scatter(Gates, timeNoConstr, color="orange")
            axs[0, 1].plot(Gates, timeWithConstr, color="blue")
            axs[0, 1].plot(Gates, timeNoConstr, color="orange")

            # error mean
            axs[1, 1].plot(Gates, errorWithConstr[:, 0], color="blue")
            axs[1, 1].plot(Gates, errorNoConstr[:, 0], color="orange")
            axs[1, 1].scatter(Gates, errorWithConstr[:, 0], color="blue")
            axs[1, 1].scatter(Gates, errorNoConstr[:, 0], color="orange")

            # error std
            axs[2, 1].scatter(Gates, errorWithConstr[:, 1], color="blue")
            axs[2, 1].scatter(Gates, errorNoConstr[:, 1], color="orange")
            axs[2, 1].plot(Gates, errorWithConstr[:, 1], color="blue")
            axs[2, 1].plot(Gates, errorNoConstr[:, 1], color="orange")

            for ax in axs.flat:
                ax.set(xlabel="Number Of Gates")

            axs.flat[0].set(ylabel="Nonzeros")
            axs.flat[1].set(ylabel="MAPE Mean")
            axs.flat[2].set(ylabel="MAPE std")
            axs.flat[3].set(ylabel="Time")
            axs.flat[4].set(ylabel="Variables")
            axs.flat[5].set(ylabel="Constraints")

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
            fig, axs = plt.subplots(7, 1)
            i = 0

            # nonzeros
            axs[i].plot(Gates, zerosWithConstr, color="blue")
            axs[i].scatter(Gates, zerosWithConstr, color="blue")

            if not only1:
                axs[i].plot(Gates, zerosNoConstr, color="orange")
                axs[i].scatter(Gates, zerosNoConstr, color="orange")
            axs.flat[i].set(ylabel="Nonzeros")
            # axs[i].legend(["With constraints", "Without constraints"])
            i += 1

            # variables
            axs[i].scatter(Gates, varsWithConstr, color="blue")
            axs[i].plot(Gates, varsWithConstr, color="blue")
            if not only1:
                axs[i].scatter(Gates, varsNoConstr, color="orange")
                axs[i].plot(Gates, varsNoConstr, color="orange")
            # axs.flat[i].set(ylabel='Variables')
            axs.flat[i].set(ylabel="Vars")
            # axs[i].legend(["With constraints", "Without constraints"])
            i += 1

            # constraints
            axs[i].scatter(Gates, constrWithConstr, color="blue")
            axs[i].plot(Gates, constrWithConstr, color="blue")
            if not only1:
                axs[i].scatter(Gates, constrNoConstr, color="orange")
                axs[i].plot(Gates, constrNoConstr, color="orange")
            # axs.flat[i].set(ylabel='Constraints')
            axs.flat[i].set(ylabel="Constr")
            # axs[i].legend(["With constraints", "Without constraints"])
            i += 1

            # constraints
            axs[i].scatter(Gates, mipGapWithConstr, color="blue")
            axs[i].plot(Gates, mipGapWithConstr, color="blue")
            if not only1:
                axs[i].scatter(Gates, mipGapNoConstr, color="orange")
                axs[i].plot(Gates, mipGapNoConstr, color="orange")
            axs.flat[i].set(ylabel='MIP gap at root')
            # axs.flat[i].set(ylabel="MIP")
            # axs[i].legend(["With constraints", "Without constraints"])
            i += 1

            # time
            axs[i].scatter(Gates, timeWithConstr, color="blue")
            if not only1:
                axs[i].scatter(Gates, timeNoConstr, color="orange")
                axs[i].plot(Gates, timeNoConstr, color="orange")

            axs[i].plot(Gates, timeWithConstr, color="blue")
            axs.flat[i].set(ylabel="Time")
            # axs[i].legend(["With constraints", "Without constraints"])
            i += 1

            # error mean
            axs[i].plot(Gates, errorWithConstr[:, 0], color="blue")
            if not only1:
                axs[i].plot(Gates, errorNoConstr[:, 0], color="orange")
                axs[i].scatter(Gates, errorNoConstr[:, 0], color="orange")

            axs[i].scatter(Gates, errorWithConstr[:, 0], color="blue")
            # axs.flat[i].set(ylabel='MAPE Mean')
            axs.flat[i].set(ylabel="Mean")
            # axs[i].legend(["With constraints", "Without constraints"])
            i += 1

            # error std
            axs[i].scatter(Gates, errorWithConstr[:, 1], color="blue")
            if not only1:
                axs[i].scatter(Gates, errorNoConstr[:, 1], color="orange")
                axs[i].plot(Gates, errorNoConstr[:, 1], color="orange")

            axs[i].plot(Gates, errorWithConstr[:, 1], color="blue")
            # axs.flat[i].set(ylabel='MAPE std')
            axs.flat[i].set(ylabel="std")
            # axs[i].legend(["With constraints", "Without constraints"])
            i += 1

            for ax in axs.flat:
                ax.set(xlabel="N")

            # Hide x labels and tick labels for top plots and y ticks for right plots.
            for ax in axs.flat:
                ax.label_outer()

    # plt.savefig("scalingMIP.png", dpi=1000, bbox_inches='tight')
    plt.show()


def plotScalingOptimization():
    file = open("Inputs.outputs/testParsing.txt")  # file with 1 line - dictionary
    verbose = "../Inputs/verbose.stdout"  # file with verbose text, should be complex problem - so there is 'Presolved'
    readFromVerbose = False
    plotUnder = True
    only1 = True
    forPaper = True

    line = file.readline()

    results = eval(line)  # load dictionary

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

    mipGapWithConstr = np.array([])
    mipGapNoConstr = np.array([])

    if readFromVerbose:
        nonZ = parseVerbose(verbose)
        zerosNoConstr = nonZ[0][:]
        zerosWithConstr = nonZ[1][:]

    # store data from dictionary into numpy array
    for gateNum, WithConstr in results:

        if WithConstr:
            if not readFromVerbose:
                zerosWithConstr = np.append(
                    zerosWithConstr, results[(gateNum, WithConstr)][0]
                )
            timeWithConstr = np.append(
                timeWithConstr, results[(gateNum, WithConstr)][2]
            )
            errorWithConstr = np.append(
                errorWithConstr, results[(gateNum, WithConstr)][3]
            )
            errorWithConstr = np.append(
                errorWithConstr, results[(gateNum, WithConstr)][4]
            )
            varsWithConstr = np.append(
                varsWithConstr, results[(gateNum, WithConstr)][5]
            )
            constrWithConstr = np.append(
                constrWithConstr, results[(gateNum, WithConstr)][6]
            )
            mipGapWithConstr = np.append(
                mipGapWithConstr, results[(gateNum, WithConstr)][7]
            )
        else:
            if not readFromVerbose:
                zerosNoConstr = np.append(
                    zerosNoConstr, results[(gateNum, WithConstr)][0]
                )

            timeNoConstr = np.append(timeNoConstr, results[(gateNum, WithConstr)][2])
            errorNoConstr = np.append(errorNoConstr, results[(gateNum, WithConstr)][3])
            errorNoConstr = np.append(errorNoConstr, results[(gateNum, WithConstr)][4])
            varsNoConstr = np.append(varsNoConstr, results[(gateNum, WithConstr)][5])
            constrNoConstr = np.append(
                constrNoConstr, results[(gateNum, WithConstr)][6]
            )
            mipGapNoConstr = np.append(
                mipGapNoConstr, results[(gateNum, WithConstr)][7]
            )

        if gateNum not in Gates:
            Gates = np.append(Gates, gateNum)

    # set histograms
    fig, axs = plt.subplots(3, 1, gridspec_kw={"wspace": 0.5, "hspace": 0.5})
    i = 0

    p0 = axs[i].plot(
        Gates,
        zerosWithConstr,  # data
        marker="o",  # each marker will be rendered as a circle
        markersize=4.5,  # marker size
        markerfacecolor="red",  # marker facecolor
        markeredgecolor="black",  # marker edgecolor
        markeredgewidth=1,  # marker edge width
        linestyle="-",  # line style will be dash line
        linewidth=3,
        zorder=3,
    )  # line width

    p0 = axs[i].plot(
        Gates,
        zerosNoConstr,  # data
        marker="o",  # each marker will be rendered as a circle
        markersize=4.5,  # marker size
        markerfacecolor="red",  # marker facecolor
        markeredgecolor="black",  # marker edgecolor
        markeredgewidth=1,  # marker edge width
        linestyle="-",  # line style will be dash line
        linewidth=3,
        zorder=3,
    )  # line width

    axs.flat[i].set(ylabel="Cones")
    i += 1

    p1 = axs[i].plot(
        Gates,
        varsWithConstr,  # data
        marker="o",  # each marker will be rendered as a circle
        markersize=4.5,  # marker size
        markerfacecolor="red",  # marker facecolor
        markeredgecolor="black",  # marker edgecolor
        markeredgewidth=1,  # marker edge width
        linestyle="-",  # line style will be dash line
        linewidth=3,
        zorder=2,
    )  # line width

    p1 = axs[i].plot(
        Gates,
        varsNoConstr,  # data
        marker="o",  # each marker will be rendered as a circle
        markersize=4.5,  # marker size
        markerfacecolor="red",  # marker facecolor
        markeredgecolor="black",  # marker edgecolor
        markeredgewidth=1,  # marker edge width
        linestyle="-",  # line style will be dash line
        linewidth=3,
        zorder=2,
    )  # line width

    axs.flat[i].set(ylabel="Variables")
    i += 1

    p2 = axs[i].plot(
        Gates,
        constrWithConstr,  # data
        marker="o",  # each marker will be rendered as a circle
        markersize=4.5,  # marker size
        markerfacecolor="red",  # marker facecolor
        markeredgecolor="black",  # marker edgecolor
        markeredgewidth=1,  # marker edge width
        linestyle="-",  # line style will be dash line
        linewidth=3,
        zorder=1,
    )  # line width

    p2 = axs[i].plot(
        Gates,
        constrNoConstr,  # data
        marker="o",  # each marker will be rendered as a circle
        markersize=4.5,  # marker size
        markerfacecolor="red",  # marker facecolor
        markeredgecolor="black",  # marker edgecolor
        markeredgewidth=1,  # marker edge width
        linestyle="-",  # line style will be dash line
        linewidth=3,
        zorder=1,
    )  # line width

    axs.flat[i].set(ylabel="Constraints")
    i += 1

    # constraints

    # p2 = axs[i].plot(Gates, mipGapWithConstr,  # data
    #                  marker='o',  # each marker will be rendered as a circle
    #                  markersize=5,  # marker size
    #                  markerfacecolor='red',  # marker facecolor
    #                  markeredgecolor='black',  # marker edgecolor
    #                  markeredgewidth=1,  # marker edge width
    #                  linestyle='-',  # line style will be dash line
    #                  linewidth=3.5, zorder=1)  # line width
    # axs.flat[i].set(ylabel='Mip gap \nat root(%)')
    # i += 1

    for ax in axs.flat:
        ax.set(xlabel="Number of gates")

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    labels = ["a", "b", "c", "d"]
    j = 0
    for ax in axs:
        # label physical distance in and down:
        trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
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
    # plt.savefig("Inputs.outputs/scaling.jpeg", dpi=800, bbox_inches="tight")


def plotPresolve(verbose, bins=False):
    # file = open("Inputs.outputs/testParsing.txt")  # file with 1 line - dictionary
    readFromVerbose = True
    plotUnder = True


    Gates = np.array([], dtype=int)

    if readFromVerbose:
        constrWithConstr, zerosWithConstr, varsWithConstr, results = parseVerboseGP(verbose)

    timeWithConstr = np.array([])
    errorWithConstrM = np.array([])
    errorWithConstrS = np.array([])

    # store data from dictionary into numpy array
    for gateNum, passes in results:

        # if passes not in zerosWithConstr:
        #     zerosWithConstr[passes] = np.array([])
        #     vars[passes] = np.array([])
        #     constraints[passes] = np.array([])

        timeWithConstr = np.append(timeWithConstr, results[(gateNum, passes)][0])
        errorWithConstrM = np.append(errorWithConstrM, results[(gateNum, passes)][1])
        errorWithConstrS = np.append(errorWithConstrS, results[(gateNum, passes)][2])

        if gateNum not in Gates:
            Gates = np.append(Gates, gateNum)


# set histograms
    fig, axs = plt.subplots(3, 1, gridspec_kw={"wspace": 0.5, "hspace": 0.5})
    i = 0

    # nonzeros

    p0 = axs[i].plot(
        Gates,
        zerosWithConstr,  # data
        marker="o",  # each marker will be rendered as a circle
        markersize=4.5,  # marker size
        markerfacecolor="red",  # marker facecolor
        markeredgecolor="black",  # marker edgecolor
        markeredgewidth=1,  # marker edge width
        linestyle="-",  # line style will be dash line
        linewidth=3,
        zorder=3,
    )  # line width

    p1 = axs[i].plot(
        Gates,
        varsWithConstr,  # data
        marker="o",  # each marker will be rendered as a circle
        markersize=4.5,  # marker size
        markerfacecolor="red",  # marker facecolor
        markeredgecolor="black",  # marker edgecolor
        markeredgewidth=1,  # marker edge width
        linestyle="-",  # line style will be dash line
        linewidth=3,
        zorder=2,
    )  # line width

    p2 = axs[i].plot(
        Gates,
        constrWithConstr,  # data
        marker="o",  # each marker will be rendered as a circle
        markersize=4.5,  # marker size
        markerfacecolor="red",  # marker facecolor
        markeredgecolor="black",  # marker edgecolor
        markeredgewidth=1,  # marker edge width
        linestyle="-",  # line style will be dash line
        linewidth=3,
        zorder=1,
    )  # line width

    axs.flat[i].set(ylabel="Count")
    # axs[i].legend(["Nonzeros", "Variables", "Constraints"])
    i += 1

    # constraints
    p2 = axs[i].plot(
        Gates,
        timeWithConstr,  # data
        marker="o",  # each marker will be rendered as a circle
        markersize=5,  # marker size
        markerfacecolor="red",  # marker facecolor
        markeredgecolor="black",  # marker edgecolor
        markeredgewidth=1,  # marker edge width
        linestyle="-",  # line style will be dash line
        linewidth=3.5,
        zorder=1,
    )  # line width
    axs.flat[i].set(ylabel="Time\n(seconds)")
    # axs[i].legend(["With constraints", "Without constraints"])
    i += 1

    # error mean
    p2 = axs[i].plot(
        Gates,
        errorWithConstrM,  # data
        marker="o",  # each marker will be rendered as a circle
        markersize=5,  # marker size
        markerfacecolor="red",  # marker facecolor
        markeredgecolor="black",  # marker edgecolor
        markeredgewidth=1,  # marker edge width
        linestyle="-",  # line style will be dash line
        linewidth=3.5,
        zorder=1,
    )  # line width
    p2 = axs[i].plot(
        Gates,
        errorWithConstrS,  # data
        marker="o",  # each marker will be rendered as a circle
        markersize=5,  # marker size
        markerfacecolor="red",  # marker facecolor
        markeredgecolor="black",  # marker edgecolor
        markeredgewidth=1,  # marker edge width
        linestyle="-",  # line style will be dash line
        linewidth=3.5,
        zorder=1,
    )  # line width

    # plt.fill_between(Gates, errorWithConstr[:, 1], y2=errorWithConstr[:, 0], alpha=0.3, color='orange')
    # plt.fill_between(Gates, errorWithConstr[:, 0], alpha=0.3, color='blue')
    plt.fill_between(Gates, errorWithConstrM, alpha=0.3, color="orange")
    plt.fill_between(
        Gates, errorWithConstrM, errorWithConstrS, alpha=0.3, color="blue"
    )

    if bins:
        axs.flat[i].set(ylabel="Relative Error(%)")
    else:
        axs.flat[i].set(ylabel="Relative Error(%)")

    i += 1

    for ax in axs.flat:
        if bins:
            ax.set(xlabel="Number of bins")
        else:
            ax.set(xlabel="Number of gates")

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    labels = ["a", "b", "c", "d"]
    j = 0
    for ax in axs:
        # label physical distance in and down:
        trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
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


    # plt.savefig("ScalingGatesGP.png", dpi=1000)
    plt.show()

def plotForThesis():

    bins = np.array([0, 0.25, 0.5, 0.25, 0])
    edges = np.array([0, 1, 2, 3, 4, 5])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim(0, 0.75)

    rv = RandomVariable(bins, edges)
    plt.hist(rv.edges[:-1], rv.edges, weights=rv.bins, density="PDF", color="orange")
    # plt.show()

    for i, j in zip(edges[2:-1], bins[1:-1]):
        ax.annotate(str(j), xy=(i - 0.6, j + 0.005))

    # plt.savefig("Inputs.outputs/exampleHist.jpeg", dpi=500)

    # plt.show()


def plotDelays():

    # plot
    fig, axs = plt.subplots(2, 1, gridspec_kw={"wspace": 0.5, "hspace": 0.5})
    models = ["Gauss", "LogNormal"]
    for i in range(0, 2):
        Sizing = [
            [2.88992964, 5.3002286, 6.39230527, 6.04825413, 4.13713269, 10.23214329],
            [2.68734118, 6.10676087, 5.98733808, 6.39787868, 4.1761852, 9.64449589],
        ]
        delay, mc = optimizeGates.optimizeCVXPY_GP(models[i], Sizing[i])

        # plt.hist(delays[-1].edges[:-1], delays[-1].edges, weights=delays[-1].bins, alpha=0.2, color='orange')
        axs[i].hist(
            delay.edges[:-1],
            delay.edges,
            weights=delay.bins,
            density="PDF",
            color="blue",
        )
        _ = axs[i].hist(
            mc / 1e11,
            bins=len(delay.edges[:-1]) - 1,
            density="PDF",
            alpha=0.8,
            color="orange",
        )
        axs[i].set_xlabel("Delay(seconds)")
        axs[i].set_ylabel("PDF")

    labels = ["a", "b"]
    j = 0
    for ax in axs:
        # label physical distance in and down:
        trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
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
    # plt.savefig("Inputs.outputs/delayComparison2.jpeg", dpi=800, bbox_inches="tight")


if __name__ == "__main__":
    # plotNonzeros()
    # plotScalingOptimization()
    # plotDelays()
    print(parseVerboseGP("testParse"))
    # plotPresolve()
    # plotForThesis()
    # plotForThesis2()
