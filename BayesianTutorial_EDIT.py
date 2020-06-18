# example of bayesian optimization for a 1d function from scratch
from math import sin
from math import pi
from numpy import arange
from numpy import vstack
from numpy import argmax
from numpy import asarray
from numpy.random import normal
from numpy.random import random
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
from matplotlib import pyplot

# objective function
def objective(X, noise=0.1):
	noise = normal(loc=0, scale=noise)
	return (-(X[0]-0.5)**2) + (-(X[1]-0.7)**2)# + noise

# surrogate or approximation for the objective function
def surrogate(model, X):
	# catch any warning generated when making a prediction
	with catch_warnings():
		# ignore generated warnings
		simplefilter("ignore")
		return model.predict(X, return_std=True)

# probability of improvement acquisition function
def acquisition(X, Xsamples, model):
	# calculate the best surrogate score found so far
	yhat, _ = surrogate(model, X)
	best = max(yhat)
	# calculate mean and stdev via surrogate function
	mu, std = surrogate(model, Xsamples)
	print(std)
	mu = mu[:, 0]
	# calculate the probability of improvement
	probs = norm.cdf((mu - best) / (std+1E-9))
	return probs

# optimize the acquisition function
def opt_acquisition(X, y, model):
	# random search, generate random samples
	Xsamples = [[0,0]]
	for i in range(100):
		x = random()
		y = random()
		Xsamples = vstack((Xsamples,[x,y]))
		
		
	Xsamples = Xsamples.reshape(len(Xsamples), 2)
	# calculate the acquisition function for each sample
	scores = acquisition(X, Xsamples, model)
	#print(scores)
	# locate the index of the largest scores
	ix = argmax(scores)
	return [Xsamples[ix, 0],Xsamples[ix, 1]]

''''
# plot real observations vs surrogate function
def plot(X, y, model):
	# scatter plot of inputs and real objective function
	pyplot.scatter(X, y)
	# line plot of surrogate function across domain
	Xsamples = asarray(arange(0, 1, 0.001))
	Xsamples = Xsamples.reshape(len(Xsamples), 1)
	ysamples, _ = surrogate(model, Xsamples)
	pyplot.plot(Xsamples, ysamples)
	# show the plot
	pyplot.show()
'''

#X = None #seriously, this can't be right; it's a goddamn 2-d array

X = asarray([[random(),random()] for i in range(100)])

# sample the domain sparsely with noise
#for i in range(100):
#	x = random()
#	y = random()
#	X = [[x, y]] if X is None else vstack((X,[x,y]))

y = asarray([objective(x) for x in X])
# reshape into rows and cols
X = X.reshape(len(X), 2)
y = y.reshape(len(y), 1)
# define the model
model = GaussianProcessRegressor()
# fit the model
model.fit(X, y)
#print(X)
# plot before hand
#plot(X, y, model)
# perform the optimization process
for i in range(100):
	# select the next point to sample
	x = opt_acquisition(X, y, model)
	# sample the point
	actual = objective(x)
	# summarize the finding
	est, _ = surrogate(model, [x])
	print('>a=%.3f, b=%.3f, f()=%3f, actual=%.3f' % (x[0], x[1], est, actual))
	# add the data to the dataset
	X = vstack((X, [x]))
	y = vstack((y, [[actual]]))
	# update the model
	model.fit(X, y)

# plot all samples and the final surrogate function
#plot(X, y, model)
# best result
ix = argmax(y)
print('Best Result: a=%.3f, b=%.3f, y=%.3f' % (X[ix,0],X[ix,1], y[ix]))