import numpy as np


# it characterizes how surprising or predictable the choice of the next character will be.
def reweight_distribution(original_distribution, temperature=0.5):
	'''
	original_distribution: original_distribution is a 1D Numpy array of probability values 
	that must sum to 1.
	temperature: temperature is a factor quantifying the entropy of the output distribution.
	return: Returns a reweighted version of the original distribution. 

	The sum of the distribution may no longer be 1, 
	so you divide it by its sum to obtain the new distribution.
	'''
	distribution = np.log(original_distribution) / temperature
	distribution = np.exp(distribution)
	return distribution / np.sum(distribution)


# Higher temperatures result in sampling distributions of higher entropy 
# that will generate more surprising and unstructured generated data
