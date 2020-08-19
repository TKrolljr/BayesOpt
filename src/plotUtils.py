import numpy
from matplotlib import pyplot as plt
from src import Bayes_Optimization_Algorithm as BoA


def plotNormalizedBOpt(normBOpt,bGraph,acGraph):
    '''
    plots objective, surrogate, surrogate standard devs, sampled locations, and expected improvement
    for the passed in normalization wrapper

    :param normBOpt: NormalizationWrapper
        the wrapper for which graphs should be generated
    :param bGraph: matplotlib.figure
        the 'bayesOpt' figure. will contain bjective, surrogate,
        surrogate standard devs, sampled locations
    :param acGraph: matplotlib.figure
        the 'acquisition' figure. will contain expected improvement function
    :return: None
    '''
    #housekeeping and setup declarations
    bOpt = normBOpt.bOpt
    bGraph.clear()
    acGraph.clear()

    #XParams is an array uniformly covering the 'x' parameter we are optimizing
    XParams = numpy.atleast_2d(numpy.linspace(bOpt.paramRanges[0][0], bOpt.paramRanges[0][1], 100)).T

    #surrogateY holds the target values of our trained model; STDs the standard deviations. Both at
    #the points in XParams
    surrogateY, STDs = bOpt.model.predict(XParams, return_std=True)

    #realY is to hold the results of XParams passed into our objective function.
    realY = numpy.asarray(normBOpt.obj(XParams))
    realY = realY.reshape(len(realY), 1)
    objMean = numpy.mean(realY,axis=0)
    objSTD = numpy.std(realY,axis=0)
    def norm(arr):
        return BoA.normalizeArray(arr,objMean,objSTD)

    #normalize arrays
    realY = norm(realY)
    surrogateY = norm(surrogateY)

    #plot objective, surrogate, and surrogate standard devs
    bGraph.plot(XParams, surrogateY, color='green', label='Prediction')
    bGraph.plot(XParams, realY, color="red", label='Real ObjFunc')
    bGraph.fill(numpy.concatenate([XParams, XParams[::-1]]),
                numpy.concatenate([surrogateY - 1.9600 * STDs,
                                   (surrogateY + 1.9600 * STDs)[::-1]]),
                color='#FFE4E1')

    #plot the already sampled locations
    bGraph.scatter(bOpt.sampleArray.ravel(), norm(bOpt.sampleResults), marker='x', color='b', zorder=10, s=200,
                   label='Observations')
    bGraph.legend(loc='lower right')

    #plot expected improvement graph
    eiY = bOpt.expectedImprovement(XParams)
    acGraph.plot(XParams, eiY, color='r', label='EI')
    acGraph.legend(loc='lower right')



    plt.show(block=False)

def plotBayesOpt(bOpt,bGraph,acGraph):
    '''
    plots objective, surrogate, surrogate standard devs, sampled locations, and expected improvement
    for the passed in BayesOpt object

    :param bOpt: Bayes_Optimization_Algorithm.BayesOptAlg
        the Bayes Opt object for which graphs should be generated
    :param bGraph: matplotlib.figure
        the 'bayesOpt' figure. will contain bjective, surrogate,
        surrogate standard devs, sampled locations
    :param acGraph: matplotlib.figure
        the 'acquisition' figure. will contain expected improvement function
    :return: None
    '''
    bGraph.clear()
    acGraph.clear()
    surrogateX = numpy.atleast_2d(numpy.linspace(bOpt.paramRanges[0][1], bOpt.paramRanges[0][0], 100)).T
    surrogateY, STDs = bOpt.model.predict(surrogateX,return_std=True)
    realY = numpy.asarray(bOpt.pollObjectiveFunction(surrogateX))
    realY = realY.reshape(len(realY), 1)
    bGraph.plot(surrogateX,surrogateY,color='green',label='Prediction')
    bGraph.plot(surrogateX,realY,color="red",label='Real ObjFunc')
    bGraph.fill(numpy.concatenate([surrogateX, surrogateX[::-1]]),
                      numpy.concatenate([surrogateY - 1.9600 * STDs,
                                     (surrogateY + 1.9600 * STDs)[::-1]]),
                      color='#FFE4E1')

    bGraph.scatter(bOpt.sampleArray.ravel(), bOpt.sampleResults,marker='x',color='b',zorder=10,s=200,label='Observations')
    bGraph.axvline(surrogateX[numpy.argmax(realY)],label=('bestGridSearch=' + str(surrogateX[numpy.argmax(realY)])))
    bGraph.legend(loc='lower right')
    eiY = bOpt.expectedImprovement(surrogateX)
    acGraph.plot(surrogateX, eiY, color='r', label='EI')
    acGraph.legend(loc='lower right')
    plt.show(block=False)
