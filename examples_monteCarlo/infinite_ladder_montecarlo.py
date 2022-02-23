import numpy as np
import argparse
from tabulate import tabulate

parser = argparse.ArgumentParser(description='The script performs Monte Carlo simulation for a ladder of gates.')
parser.add_argument('-n','--ngates', type=int, default = 10,
					help='a number of gates in a ladder')
parser.add_argument('-m','--nsamples', type=int, default = int(1e5),
					help='a number of samples in a simulation')
parser.add_argument('-s', '--seed', type=int, default=None,
					help='random seed')
parser.add_argument('-g', '--gate', nargs="*", type=float, default=[0,1.0],
					help='a [mean, std] pair for a gate self delay (all gates are assumed the same).')

def MonteCarlo_inputs(input_means, input_stds, n_samples, distribution):
	'''Generates random samples for all the inputs.

	Args:
		input_means  -- list of floats, inputs' mean values of delays
		input_stds   -- list of floats, inputs' standard deviations of delays
		n_samples    -- int, number of samples used for each simulation
		distribution -- str, defines a distribution to draw samples from; can take one of two values:
		                'Normal' or 'LogNormal'

	Returns:
		list of lists with random samples. Each sublist contains samples for the corresponding input.
	'''

	# create an empty list of lists to store the simulation data
	montecarlo = [[] for _ in range(len(input_means)) ]

	if distribution == 'Normal':
		# get the data for input nodes
		for i in range(len(input_means)):
			montecarlo[i] = np.random.normal(input_means[i], input_stds[i], n_samples)
	        
	if distribution == 'LogNormal':
	    # get the data for input nodes
	    for i in range(len(input_means)):
	        # get corresponding mu and sigma for the logrnomal pdf
	        sigma = np.sqrt( np.log( input_stds[i]**2/(input_means[i]**2) + 1 ) )
	        mu = np.log(input_means[i]) - sigma**2/2
	        # generate lognormal samples
	        montecarlo[i] = np.random.lognormal(mu, sigma, n_samples)
	        
	return montecarlo

def MonteCarlo_nodes(input1, input2, gate, n_samples):
    '''Performs simuation of a logic gate operation.
    
    Args:
		input1, input2 -- arrays of floats, simulation samples for two inputs
		gate           -- list of floats, determines gate's operation time in the following format:
		                  [mean value, std];
		                  gate's operation time is assumed to have a Gaussian distribution
		n_samples      -- int, number of samples used

    Returns:
    	array with samples for the total gate delay.
    '''
    m0 = gate[0]
    s0 = gate[1]
    
    # gate operation time is assumed to have Gaussian distribution
    montecarlo = np.maximum(input1,input2) + np.random.normal(m0, s0, n_samples)
    
    return montecarlo

def get_moments_from_simulations(simulations):
	'''Calculates mean and std for MC simulation data.

	Args:
		simulations -- a list with arrays of MC samples.
			Each array corresponds to its own gate.

	Returns:
		A list withh pairs [mean, std]

	'''
	result = []
	for data in simulations:
		result.append([np.mean(data),np.std(data)])

	return result

def main():

	# parse command line arguments
	args = parser.parse_args()
	number_of_nodes = args.ngates
	n_samples = args.nsamples

	gate = [0,1]

	# fix a random seed seed exists
	if args.seed:
		seed = args.seed
		np.random.seed(seed)
	

	####################################
	####### Generate Input data ########
	####################################

	# list with inputs' mean values
	input_means = [ np.random.randint(20,70)/10 for _ in range(number_of_nodes+1) ]
	# list with inputs' stds
	input_stds = [ np.random.randint(20,130)/100 for _ in range(number_of_nodes+1) ]

	####################################
	######## Perform Simulation ########
	####################################

	# simulate inputs
	nodes_simulation = [0 for _ in range(number_of_nodes)]
	inputs_simulation = MonteCarlo_inputs(input_means, input_stds, n_samples, 'Normal')

	# traverse the circuit
	nodes_simulation[0] = MonteCarlo_nodes(inputs_simulation[0], inputs_simulation[1], gate, n_samples)
	for i in range(1,number_of_nodes):
		nodes_simulation[i] = MonteCarlo_nodes(nodes_simulation[i-1], inputs_simulation[i+1], gate, n_samples)

	results = get_moments_from_simulations(nodes_simulation)

	# print(
	# 	tabulate(results, headers=["Mean", "std"]
	# 	)
	# 	)




if __name__ == '__main__':
	main()