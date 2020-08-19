#!/usr/bin/env python
"""This file contains everything necessary to perform domain-agnostic Bayes Optimization

The algorithm here is for MAXIMIZATION

This is the algorithm for Bayes Optimization
        1. Coarsely sample the domain of the objective function
        
        LOOP until convergence
        2. Approximate a surrogate function
        3. Minimize/Maximize the acquisition function
        4. Sample the objective function at this point
        5. Update sample list/best parameter set
        End LOOP

        6. Output best parameter set


Bayes_Optimization_Algorithm.py
author: Tobias Kroll
created: 6/11/2020
py version: 3.7
"""


from sklearn.gaussian_process import GaussianProcessRegressor
from numpy import asarray
from numpy import argmax
from numpy import argmin
from numpy import vstack
import numpy
from scipy.stats import norm
from numpy.random import normal
from skopt.learning.gaussian_process.kernels import Matern

from src import genRandoms as rand
import scipy
import warnings

def normalizeArray(sampleArray):
	'''
	normalizes the passed array into mean=0,std=1

	:param sampleArray: numpy array
		the array to be normalized
	:return: numpy array
		the normalized array
	'''
	mean = numpy.mean(sampleArray,axis=0)
	std = numpy.std(sampleArray,axis=0)
	sampleArray = (sampleArray - mean) / std
	return sampleArray

def normalizeArray(sampleArray,mean,std):
	'''
	normalizes the passed array assuming using the passed values for mean and std

	:param sampleArray: numpy array
		the array to normalize
	:param mean: float
		mean to normalize to
	:param std: float
		standard dev to normalize to
	:return: numpy array
		the normalized array
	'''
	sampleArray = (sampleArray - mean) / std
	return sampleArray

def descaleNum0to1(normNum, range):
	'''
	scale a number to the passed range. 0 scales to range[0] and 1 to range[1]

	:param normNum: float
		the number to be rescaled
	:param range: tuple - (float,float)
		the range to descale to; format of (min,max)
	:return: float
		the rescaled number
	'''
	return (normNum * (range[1]-range[0]))+range[0]

def descaleSamples0to1(sampleArray, paramRanges):
	'''
	scale an array to the passed range. 0 scales to range[0] and 1 to range[1]

	:param sampleArray: numpy array
		the array to be scaled
	:param paramRanges: tuple (float,float)
		the range to descale to; format of (min,max)
	:return: 2d numpy array
		the scaled array
	'''
	sampleArray = numpy.atleast_2d(sampleArray)
	if paramRanges is None:
		return sampleArray
	denormalizedArr = numpy.asarray([[descaleNum0to1(paramSet[i], paramRanges[i]) for i in range(len(paramSet))] \
								   for paramSet in sampleArray])
	return denormalizedArr

def scaleNum0to1(num, range):
	'''
	normalization of one number in the range in the sense of feature scaling all to [0,1]
	:param num:
		the number to b e normalized
	:param range: tuple (a,b)
		a is the lower bound, b is the upper bound
	:return: float
		the normalized value for num
	'''
	return ((num-range[0])/(range[1]-range[0]))

def rescaleSamples0to1(sampleArray, paramRanges):
	'''
	normalization of sampleArray in the range in the sense of feature scaling all to [0,1]

	:param sampleArray: 2d numpy array
		contains the samples to be n ormalized
	:param paramRanges: list of tuples
		the ranges for each parameter in sampleArray
	:return: 2d numpy array
		a new array in shape of sampleArray, with normalized values
	'''
	if paramRanges is None:
		return sampleArray
	normalizedArr = numpy.asarray([[scaleNum0to1(paramSet[i], paramRanges[i]) for i in range(len(paramSet))]\
					 for paramSet in sampleArray])
	return normalizedArr

class normalizationWrapper:
	'''
	The wrapper handles scaling of parameters to (0,1) for the desired BayesOptAlf object

	:fields
		self.paramRanges: list of tuples; [(float,float),(float,float)...]
			the nonscaled parameter ranges
		self.obj: function with signature foo(2darray)
			the post-normalization objective function
		self.bOpt: BayesOptAlg object
			the wrapped BayesOptAlg object
	'''
	def __init__(self, paramRanges, objectiveFunction, genNewSamples=True, paramFile=None,\
				 startingSamples=2):
		'''
		initialization of the normalizationWrapper. The wrapper handles scaling of parameters to (0,1) range

		:param paramRanges: list of tuples; [(float,float),(float,float)...]
			the initial, nonscaled parameter ranges
		:param objectiveFunction: function with signature foo(2darray)
			the initial, nonscaled objective function
		:param genNewSamples: bool - default=True
			whether the BayesOptAlg should generate its own new samples
		:param paramFile: String - default=None
			if genNewSamples is false, the file from which to read the initial samples
		:param startingSamples: int - default=2
			the number of samples to generate, if applicable
		'''
		self.paramRanges = paramRanges
		def ObjNormWrapper(normalizedParameters):
			return objectiveFunction(descaleSamples0to1(normalizedParameters, self.paramRanges))
		normParamRanges = [(0,1) for range in paramRanges]
		self.bOpt = BayesOptAlg(normParamRanges,ObjNormWrapper,genNewSamples=genNewSamples,\
							   paramFile=paramFile,startingSamples=startingSamples)
		self.obj = ObjNormWrapper

	def learn(self,iterNum):
		'''
		wrapping function for bOpt.learn()

		:param iterNum: int
			the number of iterations to learn for
		:return: None
		'''

		for i in range(iterNum):
			self.bOpt.learn(1)

	def getSampleArray(self):
		'''
		gets the descaled sample array

		:return: 2d array
			the descaled sample array; format [[float,float...][float,float...]...]
		'''
		return descaleSamples0to1(self.bOpt.sampleArray, self.paramRanges)

class BayesOptAlg:
	'''
	This class represents the Bayes Optimization Algorithm

	Fields:
	---------
	objectiveFunction - function
		the function the algorithm will optimize
	model - GaussianProcessRegressor
		the maintained surrogate model for the objective function
	paramRanges - list of tuples, list of 2-length lists
		the ranges in which the algorithm should search for best values
	sampleArray - numpy array of arrays
		an array containing all samples (arrays of parameters) drawn from the objective 
		function; it matches 1:1 with sampleResults
	sampleResults - numpy array of arrays
		an array containing length-1 arrays of the results of the samples in sampleArray;
		it matches 1:1 with sample Arrays
	normalizedSampleArray - numpy array of arrays
		sampleArray normalized
	'''
	def __init__(self, paramRanges, objectiveFunction, genNewSamples=True, paramFile=None,\
				 startingSamples=2):
		'''
		Constructor
		
		@params
		-------
		paramRanges - list of tuples, list of 2-length lists of floats
			the ranges in which the algorithm should search for best values; passed to the field
		objectiveFunction - function
			the function the algorithm will optimize; passed to the field
		genNewSamples - bool - default:True
			flag on whether to gen new samples (True) or not (False)
		paramFile - String - default:None
			the path from which new samples should be read if genNewSamples is False
		'''
		self.objectiveFunction = objectiveFunction
		kernel = Matern(length_scale=1.0, nu=2.5)
		self.model = GaussianProcessRegressor(normalize_y=True)
		self.paramRanges = paramRanges
		self.sampleArray = None
		self.sampleResults = None
		if(genNewSamples):
			self.sampleArray = asarray(\
			[[rand.RandomInRange_Tuple(paramRanges[i]) for i in range(len(paramRanges))] \
			for temp in range(startingSamples)])
		
			self.sampleResults = asarray(\
			self.pollObjectiveFunction(self.sampleArray))
		else:
			self.readSamplesFromFile(paramFile)
		self.model.fit(self.sampleArray, self.sampleResults.ravel())

	#TODO read from file
	def readSamplesFromFile(self, paramFile):
		sampleFile = open(paramFile)
		lines = sampleFile.readlines()
		

	def pollObjectiveFunction(self, paramSetList):
		'''
		evaluates the objectiveFunction field with the passed paramList
		
		@params
		-------
		paramSetList - list of lists/2d array of floats
			the list of parameters Sets to evaluate the obj function for
			
			
		@returns - numpy 2d array
			the value of the objective function at paramSetList, where ret[0] = objFunc(paramSetList[0])
		'''
		return self.objectiveFunction(paramSetList)

	def pollSurrogate(self, paramSet, return_std=False):
		'''
		evaluates the surrogate function/model field for the passed paramSet
		
		@params
		-------
		paramSet - list of lists of floats/2d array of floats
			the set of parameter sets to evaluate
		return_std - bool
			function returns only means when false; returns a tuple of means, standardDev otherwise
			
		@returns - list of floats /  (list of floats,list of floats)
			returns either the means of the surrogate at the paramSets, or means and standard dev
		'''
		warnings.simplefilter("ignore", UserWarning)
		with warnings.catch_warnings():
			return self.model.predict(paramSet, return_std=return_std)

	def acquisitionFunc(self,candidateParamSets, expected_improvement=False, probOfImprovement=False):
		'''
		evaluates an acquisition function at the passed candidateParams

		GUARANTEE: the 'best' next parameter set is located at the minimum of the returned acquisition func
		
		@params
		-------
		candidateParamSets - list of lists of floats/2d array of floats
			the set of parameter sets where the acquisition function is to be evaluated
		expected_improvement - bool - default:False
			when true the function will choose the expected Improvement function to evaluate
		probOfImprovement - bool - default:False
			when true the function will choose the probability of Improvement function to evaluate
		
		@returns - list of floats/1d array of floats
			a list matching the size of candidateParams; containing the values of the acquisition func
		'''
		if candidateParamSets.ndim == 1:
			candidateParamSets = numpy.atleast_2d(candidateParamSets)

		if(expected_improvement):
			return -self.expectedImprovement(candidateParamSets)
		if(probOfImprovement):
			return -self.probabilityOfImprovement(candidateParamSets)
		return -self.expectedImprovement(candidateParamSets)
		pass
	
	#from an online tutorial
	def probabilityOfImprovement(self, candidateSamples):
		'''
		the probability of Improvement (PoI) acquisition function
		
		PoI(x) = CDF(zScore(x))
		cite: https://machinelearningmastery.com/what-is-bayesian-optimization/
		
		@params
		-------
		candidateSamples - list of lists of floats/2d array of floats
			the set of parameter sets where the PoI should be evaluated
			
			
		@returns - list of floats/1darray of floats
			a list matching the size of candidateParams; containing the values of the PoI func
		'''
		warnings.simplefilter("ignore", UserWarning)
		with warnings.catch_warnings():
			# calculate the best surrogate score found so far
			yhat= self.pollSurrogate(self.sampleArray)
			best = max(yhat)

			# calculate mean and stdev via surrogate function
			mu, std = self.pollSurrogate(candidateSamples,True)

			# calculate the probability of improvement
			probs = norm.cdf((mu - best) / (std+1E-9))
		return probs


	def expectedImprovement(self, candidateParamSets,tradeoff=0.01):
		'''
		the expected Improvement (EI) acquisition function

		EI(x) = (f(best)-f(x)) * CDF(zScore(x)) + (std(x)*PDF(zScore(x))
		
		@params
		-------
		candidateParamSets - list of lists of floats/2d array of floats
			the set of parameter sets where the PoI should be evaluated
		tradeoff - float - default:0.01
			the constant tradeoff value t, trading off between exploitation and exploration
			
		@returns - list of floats/1darray of floats
			a list matching the size of candidateParams; containing the values of the EI func
		'''
		candidateMeans, candidateSTDs = self.pollSurrogate(candidateParamSets,return_std=True)
		knownMeans = self.model.predict(self.sampleArray)
		bestMean = max(knownMeans)

		candidateMeans = candidateMeans.reshape(-1,1)
		candidateSTDs = candidateSTDs.reshape(-1, 1)

		with numpy.errstate(divide='ignore'):
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				zScore = (candidateMeans - bestMean - tradeoff) / candidateSTDs
				explorationVal = candidateSTDs * norm.pdf(zScore)
				exploitationVal = (candidateMeans - bestMean - tradeoff) * norm.cdf(zScore)
				ret = explorationVal + exploitationVal
				ret[candidateSTDs==0.0] = 0.0

		return ret

	def optimize_acquisitionFunc(self,candidateNum=10):
		'''
		finds the maximum of an acquisition function
		
		@returns - list of floats/1darray of floats
			the parameter set that maximizes the acquisition Function
		'''

		#generate a list of random paramSets
		candidateParams = asarray([[rand.RandomInRange_Tuple(range) for range in self.paramRanges] \
						   for j in range(candidateNum)])

		#bfgs starting from our candidate param Sets
		bfgs_results = [scipy.optimize.minimize(self.acquisitionFunc,candidate,method='L-BFGS-B',bounds=self.paramRanges)\
			 for candidate in candidateParams]

		#find the argmin paramSet; i.e. the (hopeful) minimum of the acquisition Function, then return it
		bfgs_mins = [result.fun for result in bfgs_results]
		bfgs_candidates = [result.x for result in bfgs_results]
		index = argmin(bfgs_mins)
		x = bfgs_candidates[index]
		return x


	def learn(self, numIterations,immediatelyFit=True):
		'''
		causes the object to actually explore the objective Function and execute the BayesOpt algorithm

		LoopStart
		optimize acq Function
		poll objective at this location
		add to other pollings
		fit the Gaussian Process Regressor
		LoopEnd

		@params
		-------
		numIterations - int
			the number of iterations the algorithm is to explore for
		immediatelyFit - bool - default:True
			when true, the GP is immediately fit to the new data; when false, it must be manually done
		'''
		for i in range(numIterations):
			candidate = asarray([self.optimize_acquisitionFunc()])
			if type(candidate) != 'array':
				candidate = asarray(candidate)
			candidateY = numpy.atleast_2d(self.pollObjectiveFunction(candidate))
			self.sampleArray = vstack((self.sampleArray,candidate))
			self.sampleResults = vstack((self.sampleResults,candidateY))
			if(immediatelyFit):
				self.model.fit(self.sampleArray,self.sampleResults.ravel())

	def optimalParams(self):
		'''
		This function takes the current best param set and uses BFGS to find thw minimum around it
		
		
		@returns - list/1darray
			the found best set of parameters
		'''
		bestIndex = argmax(self.sampleResults)
		def objFuncWrap(paramSetList):
			return -self.objectiveFunction(paramSetList)
		res = scipy.optimize.minimize(objFuncWrap, self.sampleArray[bestIndex],
								method='L-BFGS-B',
								bounds=self.paramRanges)
		return res.x


def __testObjectiveFunction(paramList):
	'''
	test objective function #1
	
	@params
	-------
	paramList - list/1d array
		the parameters passed into the function
		
		
	@returns - float
		the value of the function at paramList
	'''
	noise = normal(loc=0, scale=0.04)
	return (-(paramList[0]-0.5)**2) + (-(paramList[1]-0.7)**2) * (1+noise)



def __testObjectiveFunction2(paramList):
	'''
	test objective function #2
	
	@params
	-------
	paramList - list/1d array
		the parameters passed into the function
		
		
	@returns - float
		the value of the function at paramList
	'''
	a = paramList[0]
	b = paramList[1]
	term1 = ((a - 0.25)**4)
	term2 = (5 * ((a - 0.125)**2))
	term3 = ((b - 0.4)**4)
	term4 = (6*(b - 0.2)**2)
	return - ((term1 - term2) + (term3 - term4))

if __name__ == '__main__':
	from matplotlib import pyplot
	#print(scipy.optimize.rosen(numpy.asarray([1.0,1.0])))
	paramList = [(0,3),(0,3)]
	def inverseRose(x):
		ret = [[(-(scipy.optimize.rosen(x[i])))] for i  in range(len(x))]
		return ret

	bOpt = BayesOptAlg(paramList,(inverseRose),startingSamples=100)
	#print(bOpt.sampleArray)
	#print(bOpt.sampleResults)
	#print(bOpt.expectedImprovement(bOpt.sampleArray))
	bOpt.learn(100)
	print(bOpt.sampleArray)
	print(bOpt.sampleResults)
	temp = argmax(bOpt.sampleResults)
	print(bOpt.sampleArray[temp])
	fig = pyplot.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(bOpt.sampleArray[:,0],bOpt.sampleArray[:,1],bOpt.sampleResults)
	pyplot.xlim((0,3))
	pyplot.ylim((0,3))
	ax.set_zlim((-100,0))
	pyplot.show()
        
        	#print("a=%.3f;\tb=%.3f;\tfunc()=%.3f" % (a,b,bOpt.pollObjectiveFunction([a,b])))




