import numpy as np

from src.utility_tools.node import Node
import src.utility_tools.histogramGenerator as histogramGenerator
import src.timing.SSTA as SSTA
import networkx as nx

# import matplotlib.pyplot as plt
from src.timing.infiniteLadder import putTuplesIntoArray
from src.timing.examples_monteCarlo.montecarlo import (
    get_inputs,
    get_unknown_nodes,
    simulation,
    preprocess,
)


def testSSTA_1(dec=0):

    numberOfSamples = int(1000000)
    numberOfBins = 70
    distribution = "Normal"
    binsInterval = (0, 25)

    # DESIRED - monte carlo

    adjacency = np.array(
        [
            [0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )

    G = nx.from_numpy_matrix(adjacency, create_using=nx.DiGraph())

    list_of_inputs = get_inputs(adjacency)

    unknown_nodes = get_unknown_nodes(G, list_of_inputs)
    gate = [3, 0.5]

    input_means = [0, 8, 12]
    input_stds = [0, 0.45, 0.3]

    inputs_simulation = preprocess(
        list_of_inputs,
        input_means,
        input_stds,
        unknown_nodes,
        gate,
        numberOfSamples,
        distribution,
    )

    mc = simulation(G, inputs_simulation, unknown_nodes, gate, numberOfSamples)

    desired = putTuplesIntoArray(numbers=mc)

    # ACTUAL - ssta

    g1 = histogramGenerator.get_gauss_bins(
        8, 0.45, numberOfBins, numberOfSamples, binsInterval
    )  # g1, g2 INPUT gates, g3 middle
    g2 = histogramGenerator.get_gauss_bins(
        12, 0.3, numberOfBins, numberOfSamples, binsInterval
    )  # g4 output - inputs: g3 g1
    g3 = histogramGenerator.get_gauss_bins(
        3, 0.5, numberOfBins, numberOfSamples, binsInterval
    )  # g5 output - inputs: g3, g2
    g4 = histogramGenerator.get_gauss_bins(
        3, 0.5, numberOfBins, numberOfSamples, binsInterval
    )
    g5 = histogramGenerator.get_gauss_bins(
        3, 0.5, numberOfBins, numberOfSamples, binsInterval
    )

    n1 = Node(g1)
    n2 = Node(g2)
    n3 = Node(g3)
    n4 = Node(g4)
    n5 = Node(g5)

    # set circuit design
    n1.setNextNodes([n3, n4])
    n2.setNextNodes([n3, n5])
    n3.setNextNodes([n4, n5])

    delays = SSTA.calculateCircuitDelay([n1, n2])

    actual = putTuplesIntoArray(rvs=delays)

    # TESTING

    np.testing.assert_almost_equal(
        desired,
        actual,
        decimal=dec,
        err_msg="Monte Carlo: \n" + str(desired) + "\n\n" + "SSTA: \n" + str(actual),
    )

    return None


def testSSTA_2(dec=0):

    numberOfSamples = 5000000
    numberOfBins = 50
    distribution = "Normal"
    binsInterval = (-2, 30)

    # DESIRED - monte carlo

    adjacency = np.array(
        [
            [0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )

    G = nx.from_numpy_matrix(adjacency, create_using=nx.DiGraph())

    list_of_inputs = get_inputs(adjacency)

    unknown_nodes = get_unknown_nodes(G, list_of_inputs)
    gate = [5, 0.5]

    input_means = [0, 10, 12]
    input_stds = [0, 0.45, 0.3]

    inputs_simulation = preprocess(
        list_of_inputs,
        input_means,
        input_stds,
        unknown_nodes,
        gate,
        numberOfSamples,
        distribution,
    )

    mc = simulation(G, inputs_simulation, unknown_nodes, gate, numberOfSamples)

    desired = putTuplesIntoArray(numbers=mc)
    # print()
    # ACTUAL - ssta

    g1 = histogramGenerator.get_gauss_bins(
        10, 0.45, numberOfBins, numberOfSamples, binsInterval
    )  # g1, g2 INPUT gates, g3 middle
    g2 = histogramGenerator.get_gauss_bins(
        12, 0.3, numberOfBins, numberOfSamples, binsInterval
    )  # g4 output - inputs: g3 g1
    g3 = histogramGenerator.get_gauss_bins(
        5, 0.5, numberOfBins, numberOfSamples, binsInterval
    )  # g5 output - inputs: g3, g2
    g4 = histogramGenerator.get_gauss_bins(
        5, 0.5, numberOfBins, numberOfSamples, binsInterval
    )
    g5 = histogramGenerator.get_gauss_bins(
        5, 0.5, numberOfBins, numberOfSamples, binsInterval
    )

    n1 = Node(g1)
    n2 = Node(g2)
    n3 = Node(g3)
    n4 = Node(g4)
    n5 = Node(g5)

    # set circuit design
    n1.setNextNodes([n3, n4])
    n2.setNextNodes([n3, n5])
    n3.setNextNodes([n4, n5])

    delays = SSTA.calculateCircuitDelay([n1, n2])

    actual = putTuplesIntoArray(rvs=delays)

    # TESTING

    # test whole
    np.testing.assert_almost_equal(
        desired,
        actual,
        decimal=dec,
        err_msg="Monte Carlo: \n" + str(desired) + "\n\n" + "SSTA: \n" + str(actual),
    )

    return None
