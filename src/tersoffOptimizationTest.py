#!/usr/bin/env python3
"""
tersoffOptimizationTest.py
author: Tobias Kroll
created: 6/18/2020
py version: 3.7
"""
from bayes_opt import BayesianOptimization
from src.ForceFieldParams import ForceFieldParam
from src.ForceField_Tersoff import ForceField_Tersoff
from src.LammpsFile import LammpsFile
from mpi4py import MPI
from lammps import lammps
from src.Bayes_Optimization_Algorithm import BayesOptAlg
from src.Bayes_Optimization_Algorithm import normalizationWrapper
from numpy import argmax
import numpy
from matplotlib import pyplot as plt
import subprocess
import os

from src import Bayes_Optimization_Algorithm as BoA, plotUtils
from src.FileReading import KnownValuesReading as KVR
import time

from src.errorsUtil import singleMeanSquaredError

from src.FileReading.TrainsetReading import readTrainsetFile
import src.FileReading.ParamReading as ParamReading

accumulatedLammpsTime = 0

def objectiveFunctionFromPartials(simpleParamList,ffieldParams,ffield,partialObjList):
    simpleParamList = numpy.atleast_2d(simpleParamList)
    ret = []
    sum = 0
    for paramSet in simpleParamList:
        sum = 0
        for i in range(len(paramSet)):
            ffieldParams[i].currValue = paramSet[i]

        ffield.updateFFieldfromParams(ffieldParams)
        ffield.writeForceField("ffield.tersoff")
        for partial in partialObjList:
            sum += partial.calcPartial()

        ret.append(sum)

    ret = numpy.atleast_2d(ret)
    ret = ret.reshape(-1,1)
    return ret

def objectiveFunction_test1(simpleParamList, ffieldParams=None,ffield=None):
    for i in range(len(simpleParamList)):
        ffieldParams[i].currValue = simpleParamList[i]

    ffield.updateFFieldfromParams(ffieldParams)
    ffield.writeForceField("ffield.tersoff")
    args = "-screen tersoffOptimization.log"
    words = args.split()
    myLMP = lammps(cmdargs=words)
    myLMP.file("CompFile.lmp")
    x = myLMP.extract_compute("etersoff",0,0)
    myLMP.close()
    return -((-303.46972-x )**2)

def partialError(lmpFile,structurePath,energyValue):
    lmpFile.replacementDict["%structure"] = structurePath
    lmpFile.replacementDict["%compute"] = "compute etersoff all pair tersoff"
    lmpFile.writeLammpsFile_withReplacement("partialError_Calc.lmp")
    args = "-screen tersoffOptimization.log"
    words = args.split()
    start = time.time()
    myLMP = lammps(cmdargs=words)
    myLMP.file("partialError_Calc.lmp")
    x = myLMP.extract_compute("etersoff", 0, 0)
    myLMP.close()
    end = time.time()
    global accumulatedLammpsTime
    accumulatedLammpsTime += (end - start)
    energyValue = float(energyValue)
    return -((energyValue-x)**2)

def objectiveFunction_test2(simpleParamList, printPartials=False,ffieldParams=None,ffield=None,lmpFile=None,structToEnergyMap=None):
    #update the ffieldParam objects to reflect the passed parameter vector
    simpleParamList = numpy.atleast_2d(simpleParamList)
    results = []
    for paramSet in simpleParamList:
        for i in range(len(paramSet)):
            ffieldParams[i].currValue = paramSet[i]
        #update and write the force field matching the passed param vector
        ffield.updateFFieldfromParams(ffieldParams)
        ffield.writeForceField("ffield.tersoff")

        #totalError is the sum of all partial errors
        totalError = 0.0
        for structure in structToEnergyMap.keys():
            temp = partialError(lmpFile,structure,structToEnergyMap[structure])
            totalError += temp
            if(printPartials==True):
                print("The partial error of the structure ", structure, " is ",temp)
        results.append(totalError)
    #print(accumulatedLammpsTime)
    results = numpy.asarray(results)
    results = numpy.atleast_2d(results)
    results = results.reshape(-1,1)
    #print(results)
    return results

def test1():
    # print("hello World",flush=True)
    comm = MPI.COMM_WORLD
    ffield_ = ForceField_Tersoff("Si3N4_DATA/ffield.original")
    print(ffield_.data[0][3][11])
    ffieldParams_ = [ForceFieldParam((0, 3, 11), (0.5, 6.0), linkedParams=[[0, 4, 0]])]
    paramRanges = [val.paramRange for val in ffieldParams_]

    def ObjFunc(paramList):
        return objectiveFunction_test1(paramList, ffieldParams=ffieldParams_, ffield=ffield_)

    bOpt = BayesOptAlg(paramRanges, ObjFunc)
    bOpt.learn(10)
    print(max(bOpt.sampleResults))
    x = argmax(bOpt.sampleResults)
    print(bOpt.sampleArray[x])

def test2():
    # print("hello World",flush=True)
    ffieldParams_ = []
    comm = MPI.COMM_WORLD
    ffield_ = ForceField_Tersoff("Si3N4_DATA/ffield.original")
    print(ffield_.data[0][3][11])
    #print(ffield_.data[0][3][12])
    #print(ffield_.data[0][3][15])
    #print(ffield_.data[0][3][16])
    ffieldParams_ = [ForceFieldParam((0, 3, 11), (1.5, 6.0), linkedParams=[[0, 4, 0]])]
    #ffieldParams_.append(ForceFieldParam((0, 3, 12), (25, 600), linkedParams=[[0, 4, 0]]))
    #ffieldParams_.append(ForceFieldParam((0, 3, 15), (1, 8), linkedParams=[[0, 4, 0]]))
    #ffieldParams_.append(ForceFieldParam((0, 3, 16), (1000, 8000), linkedParams=[[0, 4, 0]]))
    paramRanges = [val.paramRange for val in ffieldParams_]

    lmpFile = LammpsFile("lammpsTest.lmp")

    structToEnergyMap = {
        "data.alpha_Si3N4_1.0" : -303.46972,
        "data.alpha_Si3N4_0.98" : -301.88812,
        "data.alpha_Si3N4_1.02" : -301.83053,
        "data.alpha_Si3N4_0.9" : -264.85529,
        "data.alpha_Si3N4_1.1" : -274.56516
    }
    def ObjFunc(paramSetList):
        return objectiveFunction_test2(paramSetList, ffieldParams=ffieldParams_, ffield=ffield_,\
                                       lmpFile=lmpFile,structToEnergyMap=structToEnergyMap)

    fig = plt.figure()
    bOpt = BayesOptAlg(paramRanges, ObjFunc,startingSamples=2)
    bGraph = fig.add_subplot(211)
    acGraph = fig.add_subplot(212)
    for i in range(10):
        plotUtils.plotBayesOpt(bOpt, bGraph, acGraph)
        bOpt.learn(1)
        print("best error:", end="")
        print(max(bOpt.sampleResults.ravel()))
        plt.pause(0.5)
    print(max(bOpt.sampleResults))
    x = argmax(bOpt.sampleResults)
    print(bOpt.sampleArray[x])
    plotUtils.plotBayesOpt(bOpt, bGraph, acGraph)
    plt.show()

def normalizationTest():
    # print("hello World",flush=True)
    ffieldParams_ = []
    comm = MPI.COMM_WORLD
    ffield_ = ForceField_Tersoff("Si3N4_DATA/ffield.original")
    ffieldParams_ = [ForceFieldParam((0, 3, 11), (1.5, 6.0), linkedParams=[[0, 4, 0]])]
    ffieldParams_.append(ForceFieldParam((0, 3, 12), (25.0, 600.0), linkedParams=[[0, 4, 0]]))
    ffieldParams_.append(ForceFieldParam((0, 3, 15), (1.0, 8.0), linkedParams=[[0, 4, 0]]))
    ffieldParams_.append(ForceFieldParam((0, 3, 16), (1000.0, 8000.0), linkedParams=[[0, 4, 0]]))
    ffieldParams_.append(ForceFieldParam((0, 7, 8), (-1.0, 1.0)))#, linkedParams=[[0, 4, 0]]))
    ffieldParams_.append(ForceFieldParam((0, 7, 6), (1000.0, 1000000.0)))  # , linkedParams=[[0, 4, 0]]))
    ffieldParams_.append(ForceFieldParam((0, 7, 7), (1.0, 1000.0)))  # , linkedParams=[[0, 4, 0]]))
    paramRanges = [val.paramRange for val in ffieldParams_]

    originalValues = [ffield_.data[0][param.loc[1]][param.loc[2]] for param in ffieldParams_]
    print(originalValues)
    lmpFile = LammpsFile("lammpsTest.lmp")

    structToEnergyMap = KVR.readStructureEnergyValuesToDict("Si3N4_DATA/README.071820")

    def ObjFunc(paramSetList):
        return objectiveFunction_test2(paramSetList, ffieldParams=ffieldParams_, ffield=ffield_, \
                                       lmpFile=lmpFile, structToEnergyMap=structToEnergyMap)

    normBOpt = normalizationWrapper(paramRanges, ObjFunc, startingSamples=50)
    normBOpt.learn(50)
    bOpt = normBOpt.bOpt
    print("best error prior to GD:")
    print(max(normBOpt.bOpt.sampleResults.ravel()))
    bestIndex = argmax(bOpt.sampleResults.ravel())
    print("corresponding paramSet:")
    print(BoA.descaleSamples0to1(bOpt.sampleArray[bestIndex], normBOpt.paramRanges))
    x = bOpt.optimalParams()
    x = BoA.descaleSamples0to1(x,normBOpt.paramRanges)
    print("\n------------\n\nbest error after GD:")
    print(ObjFunc(x))
    print("corresponding paramSet:")
    print(x)
    print("\n------------\n")
    print("The partial errors: ", objectiveFunction_test2(x,printPartials=True, ffieldParams=ffieldParams_, ffield=ffield_, \
                                       lmpFile=lmpFile, structToEnergyMap=structToEnergyMap))

def trainsetReadingTest():
    # print("hello World",flush=True)
    ffieldParams_ = []
    comm = MPI.COMM_WORLD
    ffield_ = ForceField_Tersoff("Si3N4_DATA/ffield.original")
    ffieldParams_ = [ForceFieldParam((0, 3, 11), (1.5, 6.0), linkedParams=[[0, 4, 0]])]
    ffieldParams_.append(ForceFieldParam((0, 3, 12), (25.0, 600.0), linkedParams=[[0, 4, 0]]))
    #ffieldParams_.append(ForceFieldParam((0, 3, 15), (1.0, 8.0), linkedParams=[[0, 4, 0]]))
    #ffieldParams_.append(ForceFieldParam((0, 3, 16), (1000.0, 8000.0), linkedParams=[[0, 4, 0]]))
    #ffieldParams_.append(ForceFieldParam((0, 7, 8), (-1.0, 1.0)))  # , linkedParams=[[0, 4, 0]]))
    #ffieldParams_.append(ForceFieldParam((0, 7, 6), (1000.0, 1000000.0)))  # , linkedParams=[[0, 4, 0]]))
    #ffieldParams_.append(ForceFieldParam((0, 7, 7), (1.0, 1000.0)))  # , linkedParams=[[0, 4, 0]]))

    paramRanges = [val.paramRange for val in ffieldParams_]

    originalValues = [ffield_.data[0][param.loc[1]][param.loc[2]] for param in ffieldParams_]
    print(originalValues)

    objList = readTrainsetFile("Si3N4_DATA/difftrainset.test",singleMeanSquaredError,"Si3N4_DATA/")

    def objWrapper(simpleParamList):
        return -objectiveFunctionFromPartials(simpleParamList, ffieldParams_, ffield_, objList)

    normBOpt = normalizationWrapper(paramRanges, objWrapper, startingSamples=50)
    normBOpt.learn(50)
    bOpt = normBOpt.bOpt
    print("best error prior to GD:")
    print(max(normBOpt.bOpt.sampleResults.ravel()))
    bestIndex = argmax(bOpt.sampleResults.ravel())
    print("corresponding paramSet:")
    print(BoA.descaleSamples0to1(bOpt.sampleArray[bestIndex], normBOpt.paramRanges))
    x = bOpt.optimalParams()
    x = BoA.descaleSamples0to1(x, normBOpt.paramRanges)
    print("\n------------\n\nbest error after GD:")
    print(objWrapper(x))
    print("corresponding paramSet:")
    print(x)
    print("THE KNOWN BEST PARAMS:")
    print(objWrapper(originalValues))

def geoFileReadingTest():
    # print("hello World",flush=True)
    ffieldParams_ = []
    comm = MPI.COMM_WORLD
    ffield_ = ForceField_Tersoff("Si3N4_DATA/ffield.original")

    #ffieldParams_ = [ForceFieldParam((0, 3, 11), (0.01, 15.0), linkedParams=[[0, 4, 0]])]
    #ffieldParams_.append(ForceFieldParam((0, 3, 12), (25.0, 600.0), linkedParams=[[0, 4, 0]]))
    # ffieldParams_.append(ForceFieldParam((0, 3, 15), (1.0, 8.0), linkedParams=[[0, 4, 0]]))
    # ffieldParams_.append(ForceFieldParam((0, 3, 16), (1000.0, 8000.0), linkedParams=[[0, 4, 0]]))
    # ffieldParams_.append(ForceFieldParam((0, 7, 8), (-1.0, 1.0)))  # , linkedParams=[[0, 4, 0]]))
    # ffieldParams_.append(ForceFieldParam((0, 7, 6), (1000.0, 1000000.0)))  # , linkedParams=[[0, 4, 0]]))
    # ffieldParams_.append(ForceFieldParam((0, 7, 7), (1.0, 1000.0)))  # , linkedParams=[[0, 4, 0]]))'''

    ffieldParams_ = ParamReading.readParams("DATA2/params")
    paramRanges = [val.paramRange for val in ffieldParams_]

    originalValues = [ffield_.data[0][param.loc[1]][param.loc[2]] for param in ffieldParams_]
    print(originalValues)
    subprocess.call(["cp", "src/GeoFileReading/geo2data.exe", "DATA2"])
    subprocess.call(["./DATA2/geo2data.exe", "DATA2/geo", "DATA2/ffield"])
    os.remove("DATA2/geo2data.exe")


    objList = readTrainsetFile("DATA2/trainset.in", singleMeanSquaredError, "")

    def objWrapper(simpleParamList):
        return -objectiveFunctionFromPartials(simpleParamList, ffieldParams_, ffield_, objList)

    normBOpt = normalizationWrapper(paramRanges, objWrapper, startingSamples=400)
    normBOpt.learn(100)
    bOpt = normBOpt.bOpt
    print("best error prior to GD:")
    print(max(normBOpt.bOpt.sampleResults.ravel()))
    bestIndex = argmax(bOpt.sampleResults.ravel())
    print("corresponding paramSet:")
    print(BoA.descaleSamples0to1(bOpt.sampleArray[bestIndex], normBOpt.paramRanges))
    x = bOpt.optimalParams()
    x = BoA.descaleSamples0to1(x, normBOpt.paramRanges)
    print("\n------------\n\nbest error after GD:")
    print(objWrapper(x))
    print("corresponding paramSet:")
    print(x)
    print("THE STARTING PARAMS:")
    print(objWrapper(x))

def skoptTest():
    # print("hello World",flush=True)
    comm = MPI.COMM_WORLD
    ffield_ = ForceField_Tersoff("Si3N4_DATA/ffield.original")
    print(ffield_.data[0][3][11])
    print(ffield_.data[0][3][12])
    print(ffield_.data[0][3][15])
    print(ffield_.data[0][3][16])

    ffieldParams_ = [ForceFieldParam((0, 3, 11), (0.5, 6.0), linkedParams=[[0, 4, 0]])]
    ffieldParams_.append(ForceFieldParam((0, 3, 12), (25.0, 600.0), linkedParams=[[0, 4, 0]]))
    ffieldParams_.append(ForceFieldParam((0, 3, 15), (1.0, 8.0), linkedParams=[[0, 4, 0]]))
    ffieldParams_.append(ForceFieldParam((0, 3, 16), (1000.0, 8000.0), linkedParams=[[0, 4, 0]]))
    paramRanges = [val.paramRange for val in ffieldParams_]

    lmpFile = LammpsFile("lammpsTest.lmp")

    structToEnergyMap = {
        "data.alpha_Si3N4_1.0": -303.46972,
        "data.alpha_Si3N4_0.98": -301.88812,
        "data.alpha_Si3N4_1.02": -301.83053,
        "data.alpha_Si3N4_0.9": -264.85529,
        "data.alpha_Si3N4_1.1": -274.56516
    }

    def ObjFunc(paramList):
        return -(objectiveFunction_test2(paramList, ffieldParams=ffieldParams_, ffield=ffield_, \
                                       lmpFile=lmpFile, structToEnergyMap=structToEnergyMap)[0][0])

    from skopt import gp_minimize

    res = gp_minimize(ObjFunc,  # the function to minimize
                      paramRanges,  # the bounds on each dimension of x
                      acq_func="EI",  # the acquisition function
                      n_calls=100,  # the number of evaluations of f
                      n_random_starts=100,  # the number of random initialization points
                      #noise=0.1 ** 2,  # the noise level (optional)
                      n_restarts_optimizer=9) #number of restarts for the acquisition optimizer
                      #random_state=1234)  # the random seed

    print(res.x)
    print("min="+str(res.fun))
    print("or does it equal="+str(ObjFunc(res.x)))

def bayes_opt_test():
    # print("hello World",flush=True)
    comm = MPI.COMM_WORLD
    ffield_ = ForceField_Tersoff("Si3N4_DATA/ffield.original")
    print(ffield_.data[0][3][11])
    print(ffield_.data[0][3][12])
    print(ffield_.data[0][3][15])
    print(ffield_.data[0][3][16])

    ffieldParams_ = [ForceFieldParam((0, 3, 11), (0.5, 6.0), linkedParams=[[0, 4, 0]])]
    ffieldParams_.append(ForceFieldParam((0, 3, 12), (25.0, 600.0), linkedParams=[[0, 4, 0]]))
    ffieldParams_.append(ForceFieldParam((0, 3, 15), (1.0, 8.0), linkedParams=[[0, 4, 0]]))
    ffieldParams_.append(ForceFieldParam((0, 3, 16), (1000.0, 8000.0), linkedParams=[[0, 4, 0]]))
    paramRanges = [val.paramRange for val in ffieldParams_]

    lmpFile = LammpsFile("lammpsTest.lmp")

    structToEnergyMap = {
        "data.alpha_Si3N4_1.0": -303.46972,
        "data.alpha_Si3N4_0.98": -301.88812,
        "data.alpha_Si3N4_1.02": -301.83053,
        "data.alpha_Si3N4_0.9": -264.85529,
        "data.alpha_Si3N4_1.1": -274.56516
    }

    def ObjFunc(a,b,c,d):
        paramList = [a,b,c,d]
        return (objectiveFunction_test2(paramList, ffieldParams=ffieldParams_, ffield=ffield_, \
                                       lmpFile=lmpFile, structToEnergyMap=structToEnergyMap)[0][0])
    pbounds = { 'a' : ffieldParams_[0].paramRange,
               'b' : ffieldParams_[1].paramRange,
               'c' : ffieldParams_[2].paramRange,
               'd' : ffieldParams_[3].paramRange}
    from bayes_opt import SequentialDomainReductionTransformer
    boundsTransformer = SequentialDomainReductionTransformer()
    optimizer = BayesianOptimization(
        f=ObjFunc,
        pbounds=pbounds
    )
    optimizer.maximize(
        init_points=1000,
        n_iter=100,
    )
    print(optimizer.max)
    print("or does it equal="+str(ObjFunc(optimizer.max['params']['a'],optimizer.max['params']['b'],\
                                          optimizer.max['params']['c'],optimizer.max['params']['d'])))
    print("at random point" + str(ObjFunc(2.057,293.8,4.088,4512.0)))

def quickTest():
    # print("hello World",flush=True)
    ffieldParams_ = []
    comm = MPI.COMM_WORLD
    ffield_ = ForceField_Tersoff("Si3N4_DATA/ffield.original")
    print(ffield_.data[0][3][11])
    print(ffield_.data[0][3][12])
    print(ffield_.data[0][3][15])
    print(ffield_.data[0][3][16])
    ffieldParams_ = [ForceFieldParam((0, 3, 11), (0.5, 6.0), linkedParams=[[0, 4, 0]])]
    ffieldParams_.append(ForceFieldParam((0, 3, 12), (25.0, 600.0), linkedParams=[[0, 4, 0]]))
    ffieldParams_.append(ForceFieldParam((0, 3, 15), (1.0, 8.0), linkedParams=[[0, 4, 0]]))
    ffieldParams_.append(ForceFieldParam((0, 3, 16), (1000.0, 8000.0), linkedParams=[[0, 4, 0]]))
    paramRanges = [val.paramRange for val in ffieldParams_]

    lmpFile = LammpsFile("lammpsTest.lmp")

    structToEnergyMap = {
        "data.alpha_Si3N4_1.0": -303.46972,
        "data.alpha_Si3N4_0.98": -301.88812,
        "data.alpha_Si3N4_1.02": -301.83053,
        "data.alpha_Si3N4_0.9": -264.85529,
        "data.alpha_Si3N4_1.1": -274.56516
    }

    def ObjFunc(paramSetList):
        return -objectiveFunction_test2(paramSetList, ffieldParams=ffieldParams_, ffield=ffield_, \
                                       lmpFile=lmpFile, structToEnergyMap=structToEnergyMap)

    x = [1.25744366, 35.77402639, 4.51140332, 1233.33779981]
    import scipy.optimize
    result = scipy.optimize.minimize(ObjFunc,x,method='L-BFGS-B',bounds=paramRanges)
    print(result.x)
    print(result.fun)

if __name__ == "__main__":
    geoFileReadingTest()



