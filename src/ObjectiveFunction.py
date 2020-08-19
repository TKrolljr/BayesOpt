#!/usr/bin/env python
"""
ObjectiveFunction.py
author: Tobias Kroll
created: 7/20/2020
py version: 3.7
"""
from abc import ABCMeta, abstractmethod

from lammps import lammps

from src.LammpsFile import LammpsFile


def lammpsOperation(lmpFile, replacementDict)->lammps:
    '''
    Does a lammps operation using the passed in LammsFile's write function,

    :param lmpFile: LammpFile object
        the LammpsFile object to use in generating lammps input files
    :param replacementDict:
        the custom replacementDict that is used by lmpFile to write a readable lammps input file
    :return: lammps object
        the lammps object after reading the created input file
    '''
    #HARDCODED!! structure replacement only TODO: change this behavior to be less hardcoded
    lmpFile.replacementDict = replacementDict

    lmpFile.writeLammpsFile_withReplacement("tempLammpsCalc.lmp")

    #set up lammps commmand line arguments
    args = "-screen tersoffOptimization.log"
    words = args.split()
    myLMP = lammps(cmdargs=words)

    #run the lammps file
    myLMP.file("tempLammpsCalc.lmp")

    return myLMP

class partialObjFunc(metaclass=ABCMeta):
    '''
    abstract class which represents a single partial objective function component.
    Effectively one line in the trainset file

    This should probably be an interface but Python is silly that way eh.
    '''

    def __init__(self,errorFunc,structureDirectoryPath):
        self.errorFunc = errorFunc
        self.structureDirectoryPath = structureDirectoryPath

    @abstractmethod
    def calcPartial(self):
        pass

class energyPartialObj(partialObjFunc):
    '''
    partialObjFunc implementation for objects pertaining to lines in the ENERGY section

    responsible for handling its own partial Obj Function:
    weight * errorFunc(((operation1) * struc1energy / factor1) - ((operation2) * struc2energy / factor2)

    fields:
        weight: float
            the weight associated with this aprtial obj func (a multiplicative coeff for optimization)
        strucTuple1: 3tuple (String,String,float)
            (operation,structureName,factor) first structure-related values for the partial obj
        strucTuple2: 3tuple (String,String,float)
            (operation,structureName,factor) second structure-related values for the partial obj
        energyValue: float
            the 'known' energy Value of energy difference Value that is to be errored against
        lmpFile: LammpsFile
            LammpsFile object needed to make a call to lammps for calculations
    '''
    def __init__(self, weight, strucTuple1, strucTuple2, energyValue,lmpFile,errorFunc,structureDirectoryPath):
        '''
        simple 1 to 1 object constructor

        :param weight: float
            the weight associated with this aprtial obj func (a multiplicative coeff for optimization)
        :param strucTuple1: 3tuple
            (operation,structureName,factor) first structure-related values for the partial obj
        :param strucTuple2: 3tuple
            strucTuple2: 3tuple (String,String,float)
        :param energyValue: float
            the 'known' energy Value of energy difference Value that is to be errored against
        :param structureDirectoryPath: String
            the directory where necessary structure files are located
        :param lmpFile: LammpsFile
            LammpsFile object needed to make a call to lammps for calculations
        '''
        super().__init__(errorFunc,structureDirectoryPath)
        self.weight = weight
        self.strucTuple1 = strucTuple1
        self.strucTuple2 = strucTuple2
        self.energyValue = energyValue
        self.lmpFile = lmpFile

    def calcPartial(self):
        '''
        calculates the partial error for this partial objective function

        :return: float
            the calculated error
        '''
        replacementDict = { "%structure" : self.structureDirectoryPath + "data." + self.strucTuple1[1],
                            "%compute" : "compute energy all pair tersoff"}
        lmp = lammpsOperation(self.lmpFile,replacementDict=replacementDict)
        energy1 = lmp.extract_compute("energy",0,0)
        lmp.close()

        #account for the operator set in the trainfile
        coeff1 = 1
        if (self.strucTuple1[0] == '-'):
            coeff1 = -1

        #account for the factors set in the trainfile
        coeff1 /= self.strucTuple1[2]
        predictedVal = (coeff1*energy1)

        #steps as above, but account for there not being a second structure sometimes
        if(not(self.strucTuple2 is None)):
            replacementDict["%structure"]= self.structureDirectoryPath + "data." + self.strucTuple2[1]
            lmp = lammpsOperation(self.lmpFile, replacementDict=replacementDict)
            energy2 = lmp.extract_compute("energy", 0, 0)
            lmp.close()
            coeff2 = 1
            if (self.strucTuple2[0] == '-'):
                coeff2 = -1
            coeff2 /= self.strucTuple2[2]

            predictedVal = (coeff1 * energy1) + (coeff2 * energy2)

        return self.weight * self.errorFunc(predictedVal,self.energyValue)





if __name__ == "__main__":
    lmpFile = LammpsFile("lammpsTest.lmp")
    myO = energyPartialObj(1.0,('+',"alpha_Si3N4_1.0",1),None,  -303.46972,lmpFile)
    print(myO.calcPartial())