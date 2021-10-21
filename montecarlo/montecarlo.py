import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


#####################
## Reading a Graph ##
#####################

def get_inputs(adjacency):
    inputs = []
    for i in range(len(adjacency)):
        if i * adjacency[0][i] != 0:
            inputs.append(i * adjacency[0][i])
    return inputs


def get_unknown_nodes(G, inputs):
    unknown_nodes = []
    for node in range(1, G.number_of_nodes()):
        if node not in inputs:
            unknown_nodes.append(node)
    return unknown_nodes


def get_ordered_paths(G, input_nodes, disordered_nodes):
    predecessors = input_nodes.copy()
    unknown_nodes = disordered_nodes.copy()
    temp = []
    ordered_list = []
    while (len(ordered_list) != len(disordered_nodes)):
        for node in unknown_nodes:
            if (list(G.predecessors(node))[0] in predecessors) and (list(G.predecessors(node))[1] in predecessors):
                predecessors.append(node)
                ordered_list.append(node)
            else:
                temp.append(node)
        unknown_nodes = temp

    return ordered_list


#####################
#### Monte Carlo ####
#####################

def preprocess(input_nodes, input_means, input_stds, unknown_nodes, gate, n_samples, distribution):
    m0 = gate[0]
    s0 = gate[1]

    # create an empty list of lists to store the simulation data
    montecarlo = [[] for _ in range(len(input_nodes + unknown_nodes) + 1)]

    if distribution == 'Normal':
        # get the data for input nodes
        for i in input_nodes:
            montecarlo[i] = np.random.normal(input_means[i], input_stds[i], n_samples)

    if distribution == 'Gamma':
        # 
        """
        mean = shape * scale
        var  = shape * scale**2
        
        The input stds are used as scale parameters here.
        """
        # get the data for input nodes
        for i in input_nodes:
            scale = input_stds[i]
            shape = input_means[i] / scale
            montecarlo[i] = np.random.gamma(shape, scale, n_samples)

    if distribution == 'LogNormal':
        # get the data for input nodes
        for i in input_nodes:
            # get corresponding mu and sigma for the logrnomal pdf
            sigma = np.sqrt(np.log(input_stds[i] ** 2 / (input_means[i] ** 2) + 1))
            mu = np.log(input_means[i]) - sigma ** 2 / 2
            # generate lognormal samples
            montecarlo[i] = np.random.lognormal(mu, sigma, n_samples)

    return montecarlo


def simulation(G, input_simulation_data, unknown_nodes, gate, n_samples):
    m0 = gate[0]
    s0 = gate[1]


    # list that contains simulation data for inputs
    montecarlo = input_simulation_data

    sink = unknown_nodes[-1]

    for node in unknown_nodes:
        a = list(G.predecessors(node))[0]
        b = list(G.predecessors(node))[1]

        # print(np.mean(montecarlo[a]))
        # print(np.mean(montecarlo[b]))

        max = np.maximum(montecarlo[a], montecarlo[b])
        # print(np.mean(max))
        # print(np.mean(np.random.normal(m0, s0, n_samples)))

        montecarlo[node] = max + np.random.normal(m0, s0, n_samples)


        # print(np.mean(montecarlo[node]))


    return montecarlo


def main():
    # number of sample for MC
    n_samples = int(100000)
    # n_samples = int(5)
    distribution = 'Normal'  # try 'LogNormal' and 'Gamma'
    #

    # adjacency = np.array([[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    #                       [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #                       [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    #                       [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    #                       [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    #                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    #                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    #                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    #                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    #                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    #                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    #                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        # test case
    adjacency = np.array([  [0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 0, 0],
                            [0, 0, 0, 1, 0, 1, 0],
                            [0, 0, 0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0]])

    G = nx.from_numpy_matrix(adjacency, create_using=nx.DiGraph())

    list_of_inputs = get_inputs(adjacency)

    # print(list_of_inputs)
    unknown_nodes = get_unknown_nodes(G, list_of_inputs)
    # print(f'The circuit consist of {len(unknown_nodes)} nodes.\n')

    # gates are assumed to have the same delays
    gate = [0.5, 0.5]  # mean and std for the gates

    # list of means and stds of input arrival times
    # input_means = [0, 1, 0.5, 1.4, 1, 0.5, 0.75]
    # input_stds = [0, 0.45, 0.3, 0.6, 0.3, 0.3, 0.35]

    input_means = [0, 1, 0.5]
    input_stds = [0, 0.45, 0.3]


    inputs_simulation = preprocess(list_of_inputs, input_means, input_stds, unknown_nodes, gate, n_samples,
                                   distribution)

    mc = simulation(G, inputs_simulation, unknown_nodes, gate, n_samples)
    maxdelay = mc[-1]


    # print out the results

    for i in range(1, len(mc)):

        delay = mc[i]
        print('Mean of ' + str(i) + 'th delay is: ' + str(np.mean(delay)) + ', std: ' + str(np.std(delay)) )


    # print(f'The mean delay is {np.mean(maxdelay)}')
    # print(f'The std of a delay is {np.std(maxdelay)}')

    _ = plt.hist(maxdelay, bins=50, density='PDF', alpha=0.7)
    plt.ylabel('PDF of delay', size=14)
    plt.xlabel('time', size=14)
    plt.title('Histogram of the MAX delay', size=16)
    plt.show()


if __name__ == "__main__":
    main()
