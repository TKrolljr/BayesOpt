#!/usr/bin/env python
"""
TrainsetReading.py
author: Tobias Kroll
created: 7/19/2020
py version: 3.7
"""
from src import ObjectiveFunction as of
from src.FileReading import FileReadingUtils as fr
from src.LammpsFile import LammpsFile


def readEnergies(data,errorFunc,structureDirectoryPath):
    '''
    reads a passed in ENERGY section of the trainsetfile

    :param data: list of list of Strings
        the parsed and split lines of the energy section
    :param structureDirectoryPath: String
        the directory that contains the structures in the trainset file
    :return: list of energyPartialObj
        the list of created partial objective function objects, one per line
    '''
    lmpFile = LammpsFile("src/lammpsTest.lmp")

    retList = []
    for i in range(len(data)):
        weight = float(data[i][0])
        op1 = (data[i][1])
        struc1 = data[i][2]
        fac1 = int(data[i][3])
        tuple1=(op1,struc1,fac1)
        tuple2=None
        energy = data[i][4]
        if(len(data[i]) > 5):
            op2 = (data[i][4])
            struc2 = data[i][5]
            fac2 = int(data[i][6])
            tuple2=(op2,struc2,fac2)
            energy = data[i][7]
        energy = float(energy)
        retList.append(of.energyPartialObj(weight,tuple1,tuple2,energy,lmpFile,errorFunc,structureDirectoryPath))

    print(retList)
    return retList


def readTrainsetFile(pathname, errorFunc, structureDirectoryPath):
    '''
    reads in a trainset file at the passed location, and returns with a list of partial objective function
    objects, one per line

    :param pathname: String
        path to the trainset file
    :param structureDirectoryPath: String
        directory containing the structures in the trainset files
    :return: list of partialObjFunc
        the created list of partialObjFunc objects
    '''
    partialObjList = []
    lines = fr.readFileAsLines(pathname)
    for i in range(len(lines)):
        lines[i] = lines[i].replace('/',' ')
    lines = fr.removeWhiteSpaceAndComments(lines)
    data = [line.split() for line in lines]
    index = 0
    length = len(data)
    while(index < length):
        SectionIdentifier = data[index][0]
        print(SectionIdentifier)
        endIndex = -1
        #Why in the actual does python not have a switch statement???????
        if(SectionIdentifier == "CHARGE"):
            endIndex = __findIndexOfSectionEnd(data,"ENDCHARGE")
            index = endIndex + 1
        elif(SectionIdentifier == "STRESS"):
            endIndex = __findIndexOfSectionEnd(data, "ENDSTRESS")
            index = endIndex + 1
        elif(SectionIdentifier == "PRESSURE"):
            endIndex = __findIndexOfSectionEnd(data, "ENDPRESSURE")
            index = endIndex + 1
        elif(SectionIdentifier == "STSTRAIN"):
            endIndex = __findIndexOfSectionEnd(data, "ENDSTSTRAIN")
            index = endIndex + 1
        elif(SectionIdentifier == "ATOM_FORCE"):
            endIndex = __findIndexOfSectionEnd(data, "ENDATOM_FORCE")
            index = endIndex + 1
        elif(SectionIdentifier == "CELL"):
            endIndex = __findIndexOfSectionEnd(data, "ENDCELL")
            index = endIndex + 1
        elif(SectionIdentifier == "FREQUENCIES"):
            endIndex = __findIndexOfSectionEnd(data, "ENDFREQUENCIES")
            index = endIndex + 1
        elif(SectionIdentifier == "HEATFO"):
            endIndex = __findIndexOfSectionEnd(data, "ENDHEATFO")
            index = endIndex + 1
        elif(SectionIdentifier == "GEOMETRY"):
            endIndex = __findIndexOfSectionEnd(data, "ENDGEOMETRY")
            index = endIndex + 1
        elif(SectionIdentifier == "STRUCTURE"):
            endIndex = __findIndexOfSectionEnd(data, "ENDSTRUCTURE")
            index = endIndex + 1
        elif(SectionIdentifier == "ENERGY"):
            endIndex = __findIndexOfSectionEnd(data, "ENDENERGY")
            partialObjList += readEnergies(data[index+1:endIndex-1],errorFunc,structureDirectoryPath)
            index = endIndex + 1
        else:
            print("ERROR YOU ABSOLUTE BUFOON CHECK THE TRAINSETFILE")
            break

    return partialObjList

def __findIndexOfSectionEnd(data, endString):
    '''
    finds the index of lines (really lists pertaining to a line) starting with endString
    This is used to find where trainset file sections end

    :param data: list of list of Strings
        the parsed trainsetfile data
    :param endString: String
        the String that we are searching for
    :return: int
        the index at which we found endString
    '''
    index = -1
    for i in range(len(data)):
        if(data[i][0] == endString):
            index = i
    return index

if __name__ == "__main__":
    readTrainsetFile("../trainset.in")