#!/usr/bin/env python
"""
tersoffOptimizationTest.py
author: Tobias Kroll
created: 6/18/2020
py version: 3.7
"""

from ForceFieldParams import ForceFieldParam
from ForceField_Tersoff import ForceField_Tersoff
from operator import add
import os
from mpi4py import MPI
from lammps import lammps
from Bayes_Optimization_Algorithm import BayesOptAlg
from numpy import argmax

def MD_objectiveFunction(simpleParamList, ffieldParams=None,ffield=None):
    for i in range(len(simpleParamList)):
        ffieldParams[i].currValue = simpleParamList[i]

    updateFFieldfromParams(ffield,ffieldParams)
    ffield.writeForceField("ffield.tersoff")
    args = "-screen tersoffOptimization.log"
    words = args.split()
    myLMP = lammps(cmdargs=words)
    myLMP.file("CompFile.lmp")
    x = myLMP.extract_compute("etersoff",0,0)
    myLMP.close()
    return -((-303.46972-x )**2)


def updateFFieldfromParams(ffield, ffieldParams):
    for param in ffieldParams:
        loc = param.loc
        ffield.data[loc[0]][loc[1]][loc[2]] = param.currValue
        if param.linkedParams is None:
            pass
        else:
            for link in param.linkedParams:
                ffield.data[loc[0]+link[0]][loc[1]+link[1]][loc[2]+link[2]] = param.currValue

if __name__ == "__main__":
    #print("hello World",flush=True)
    comm = MPI.COMM_WORLD
    ffield_ = ForceField_Tersoff("ffield.original")
    ffieldParams_ = [ForceFieldParam((0,3,11),(0.5,6.0),linkedParams=[[0,4,0]])]
    paramRanges = [val.paramRange for val in ffieldParams_]
    def ObjFunc(paramList):
        return MD_objectiveFunction(paramList,ffieldParams=ffieldParams_,ffield=ffield_)
    bOpt = BayesOptAlg(paramRanges,ObjFunc)
    bOpt.learn(100)
    print(max(bOpt.sampleResults))
    print(bOpt.sampleResults)
    print(bOpt.sampleArray)
    x = argmax(bOpt.sampleResults)
    print(bOpt.sampleArray[x])



