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

def MD_objectiveFunction(simpleParamList, ffieldParams=None,ffield=None):
    for i in range(len(simpleParamList)):
        ffieldParams[i].currValue = simpleParamList[i]

    updateFFieldfromParams(ffield,ffieldParams)
    ffield.writeForceField("ffield_SiN.tersoff")
    myLMP = lammps()
    myLMP.file("CompFile.lmp")


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
    ffield = ForceField_Tersoff("ffield.original")
    ffieldParams = [ForceFieldParam((0,0,16),(1000.0,3000.0),linkedParams=[[0,1,0]])]