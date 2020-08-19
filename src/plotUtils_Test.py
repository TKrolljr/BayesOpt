import numpy
from matplotlib import pyplot as plt

from src.Bayes_Optimization_Algorithm import BayesOptAlg
from src.Bayes_Optimization_Algorithm import normalizationWrapper
from src import Bayes_Optimization_Algorithm as BoA, plotUtils


def sinObj1(x, noise=0.1):
    '''
    sinusoidal function no.1

    Max~=1.9 when x=[-1,2]
    This one is HARD!

    :param x: float, 1d numpy array
        location/s to sample
    :param noise: float - default=0.1
        the standard dev of the noise to ass to the function
    :return:
        float, 1d numpy array
    '''
    #noise = numpy.random.normal(loc=0, scale=noise)
    return (x**2 * numpy.sin(5 * numpy.pi * x)**6.0)# + noise


def sinObj2(x):
    '''
    sinusoidal function no.2

    Maximum is at x=0.0 when x=[-1,2]


    :param x: float, 1d numpy array
        location/s to sample
    :return:
        float, 1d numpy array
    '''
    return (-1)*(x * numpy.sin(x))

def sinObj3(x):
    '''
        sinusoidal function no.3

        Max~=-0.3  when x=[-1,2]

        :param x: float, 1d numpy array
            location/s to sample
        :return:
            float, 1d numpy array
    '''
    return -numpy.sin(3*x) - x**2 + 0.7*x

def BayesOptTest():
    fig = plt.figure()
    bGraph = fig.add_subplot(211)
    bOpt = BayesOptAlg([(-1.0,2.0)],sinObj1)
    acGraph = fig.add_subplot(212)
    for  i in range(100):
        plotUtils.plotBayesOpt(bOpt, bGraph, acGraph)
        bOpt.learn(1)
        plt.pause(3.0)

    plt.show()

def normalizationTest():
    fig = plt.figure()
    bGraph = fig.add_subplot(211)
    normBOpt = normalizationWrapper([(-1.0, 2.0)], sinObj3)
    bOpt = normBOpt.bOpt
    acGraph = fig.add_subplot(212)
    for i in range(10):
        plotUtils.plotNormalizedBOpt(normBOpt, bGraph, acGraph)
        normBOpt.learn(1)
        plt.pause(3.0)

    bestIndex = numpy.argmax(bOpt.sampleResults.ravel())
    print("corresponding paramSet:", end='')
    print(BoA.descaleSamples0to1(bOpt.sampleArray[bestIndex], normBOpt.paramRanges))
    x = bOpt.optimalParams()
    x = BoA.descaleSamples0to1(x, normBOpt.paramRanges)
    print(x)
    plt.show()



if __name__ == "__main__":
    normalizationTest()