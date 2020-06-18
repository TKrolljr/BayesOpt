#!/usr/bin/env python
"""This file contains everything necessary to perform domain-agnostic Bayes Optimization

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
from numpy.random import random
from numpy import asarray
from numpy import argmax
from numpy import vstack
from scipy.stats import norm
from matplotlib import pyplot
from numpy.random import normal
import warnings

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
	'''
	def __init__(self, paramRanges, objectiveFunction, genNewSamples=True, paramFile=None):
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
		self.model = GaussianProcessRegressor()
		self.paramRanges = paramRanges
		self.sampleArray = None
		self.sampleResults = None
		if(genNewSamples):
			self.sampleArray = asarray(\
			[[RandomInRange_Tuple(paramRanges[i]) for i in range(len(paramRanges))] \
			for temp in range(2)])
		
			self.sampleResults = asarray(\
			[[self.pollObjectiveFunction(paramSet)] for paramSet in self.sampleArray])
			
		else:
			self.readSamplesFromFile(paramFile)
		
		self.model.fit(self.sampleArray, self.sampleResults)

	def readSamplesFromFile(self, paramFile):
		sampleFile = open(paramFile)
		lines = sampleFile.readlines()
		

	def pollObjectiveFunction(self, paramList):
		'''
		evaluates the objectiveFunction field with the passed paramList
		
		@params
		-------
		paramList - list/1d array of floats
			the list of parameters to evaluate the obj function for
			
			
		@returns - float
			the value of the objective function at paramList
		'''
		return self.objectiveFunction(paramList)

	def pollSurrogate(self, paramSet, return_std=False):
		'''
		evaluates the surrogate function/model field for the passed paramSet
		
		@params
		-------
		paramSet - list of lists of floats/2d array of floats
			the set of parameter sets to evaluate
		return_std - bool
			function returns only means when false; returns a tuple of means, standardDev otherwise
			
		@returns - list of floats / list of (float,float)
			returns either the means of the surrogate at the paramSets, or means and standard dev
		'''
		return self.model.predict(paramSet, return_std)

	def acquisitionFunc(self,candidateParams, expected_improvement=False, probOfImprovement=False):
		'''
		evaluates an acquisition function at the passed candidateParams
		
		@params
		-------
		candidateParams - list of lists of floats/2d array of floats
			the set of parameter sets where the acquisition function is to be evaluated
		expected_improvement - bool - default:False
			when true the function will choose the expected Improvement function to evaluate
		probOfImprovement - bool - default:False
			when true the function will choose the probability of Improvement function to evaluate
		
		@returns - list of floats/1d array of floats
			a list matching the size of candidateParams; containing the values of the acquisition func
		'''
		if(expected_improvement):
			return self.expectedImprovement(candidateParams)
		if(probOfImprovement):
			return self.probabilityOfImprovement(candidateParams)
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
		with warnings.catch_warnings():
			# calculate the best surrogate score found so far
			yhat= self.pollSurrogate(self.sampleArray)
			best = max(yhat)
			# calculate mean and stdev via surrogate function
			mu, std = self.pollSurrogate(candidateSamples,True)
			mu = mu[:, 0]
			# calculate the probability of improvement
			probs = norm.cdf((mu - best) / (std+1E-9))
		return probs


	def expectedImprovement(self, candidateParams):
		'''
		the expected Improvement (EI) acquisition function
		
		EI(x) = (f(best)-f(x)) * CDF(zScore(x)) + std(x)*PDF(zScore(x))
		
		@params
		-------
		candidateSamples - list of lists of floats/2d array of floats
			the set of parameter sets where the PoI should be evaluated
			
			
		@returns - list of floats/1darray of floats
			a list matching the size of candidateParams; containing the values of the EI func
		'''
		with warnings.catch_warnings():
			#find f(best); the result of the current best parameter set
			bestCurrentResult = max(self.pollSurrogate(self.sampleArray))
		
			#find means and stds in regards to the candidate Params
			candidateMeans, candidateSTDs = self.pollSurrogate(candidateParams,return_std=True)
			candidateMeans = candidateMeans[:,0]
			
			#find zScores
			zScores = (candidateMeans-bestCurrentResult)/(candidateSTDs)
			
			exploitationVal = -1*((bestCurrentResult-candidateMeans)*norm.cdf(zScores))
			explorationVal = (candidateSTDs*norm.pdf(zScores))
		return  ( exploitationVal + explorationVal)
		
	def optimize_acquisitionFunc(self):
		'''
		finds the maximum of an acquisition function
		
		@returns - list of floats/1darray of floats
			the parameter set that maximizes the acquisition Function
		'''
		candidateNum = 100
	
	
		candidateParams = [[RandomInRange_Tuple(range) for range in self.paramRanges] \
		 for j in range(candidateNum)] 
		#print(candidateParams)
		 #done one candidateNum times TODO: better number here
		#print(self.acquisitionFunc(candidateParams))
		optimalIndex = argmax(self.acquisitionFunc(candidateParams,probOfImprovement=True))
		return candidateParams[optimalIndex]
				
	def learn(self, numIterations):
		'''
		causes the object to actually explore the objective Function and execute the BayesOpt algorithm
		
		@params
		-------
		numIterations - int
			the number of iterations the algorithm is to explore for
		'''
		for i in range(numIterations):
			candidate = self.optimize_acquisitionFunc()
			candidateY = self.pollObjectiveFunction(candidate)
			self.sampleArray = vstack((self.sampleArray,[candidate]))
			self.sampleResults = vstack((self.sampleResults,[[candidateY]]))
			self.model.fit(self.sampleArray,self.sampleResults)
			
	def optimalParams(self):
		'''
		function that returns the parameter Set that is currently the best
		
		
		@returns - list/1darray
			the current optimal set of parameters
		'''
		bestIndex = argmax(self.sampleResults)
		return self.sampleArray(bestIndex)
			

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
	paramList = [(-3,3),(-3,3)]
	bOpt = BayesOptAlg(paramList,__testObjectiveFunction2)
	#print(bOpt.sampleArray)
	#print(bOpt.sampleResults)
	#print(bOpt.expectedImprovement(bOpt.sampleArray))
	bOpt.learn(100)
	temp = argmax(bOpt.sampleResults)
	#print(bOpt.sampleResults[temp])
	print(bOpt.expectedImprovement([[0.5,0.7]]))
	fig = pyplot.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(bOpt.sampleArray[:,0],bOpt.sampleArray[:,1],bOpt.sampleResults)
	pyplot.show()
        
        	#print("a=%.3f;\tb=%.3f;\tfunc()=%.3f" % (a,b,bOpt.pollObjectiveFunction([a,b])))




