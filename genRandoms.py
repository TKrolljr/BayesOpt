from numpy.random import random

def genRandomInRange(min, max):
	'''
	generates a uniform random value between min and max
	
	@params
	-------
	min - float
		the minimum of the random range
	max - float
		the maximum of the random range
		
		
	@returns - float
		the random value generated
	'''
	return (random()*(max - min)+min)
	
def RandomInRange_Tuple(rangeTuple):
	'''
	generates a uniform random value between the range denoted by rangeTuple
	
	@params
	-------
	rangeTuple - (float,float)
		contains the bounds within which to generate: (min,max)
		
	
	@returns - float
		the random value generated
	'''
	return (random()*(rangeTuple[1] - rangeTuple[0])+rangeTuple[0])