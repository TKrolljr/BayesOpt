#!/usr/bin/env python
'''
This module encompasses everything needed to parse a Tersoff Force Field File.

ForceField_Tersoff.py
author: Tobias Kroll
created: 6/8/2020
py version: 3.7
'''
from src.FileReading import FileReadingUtils as fr


class ForceField_Tersoff:
    """
    This class represents the data of a Tersoff Forcefield.

    The class is responsible for reading in a force field file, and parsing it to obtain
    and store its data. It can change this data. It is also capable of writing its data to
    a new forcefield file.

    Fields:
    -------
    ffield_path - String
        the path to the object's force field file
    lines - list
        a list where each element is a line of the force field file (parsed of comments, etc.)
    data - list
        a list that is generalized to a structure all forcefields fit into. It holds actual data entries
    -------
    """
	
    def __init__(self, ffield_path):
        """
        Constructor of the object. Reads and parses the file at ffield_path.
        
        @params
        -------
        ffield_path - String
            the path to the force field file
        """
        self.ffield_path = ffield_path
        self.lines = fr.readFileAsLines(ffield_path)
        lines = fr.removeWhiteSpaceAndComments(self.lines)
        lines = fr.removeFromLines(self.lines,"Tersoff")
        self.data = self.parseAsData()

    def parseAsData(self):
        """
        parses the input in lines into the 3d grid expected for entry access.
        
        @returns - 3darray of floats/list of lists of lists of floats
            list containing the data accessible to be modified: data[section][line][entry]
        """
        data = [line.split() for line in self.lines]
        data = [data]
        return data
		
    def writeForceField(self,ffield_path):
        """
        writes the Objects data into a file created at ffield_path.
        
        @params
        -------
        ffield_path - String
            the path to write the Force field to
        """
        file = open(ffield_path, "w")
        file.write("#Tersoff\n")
        for section in self.data:
            for line in section:
                for entry in line:
                    file.write(str(entry) + " ")

                file.write("\n")

            file.write("#end of ffield")

    def updateFFieldfromParams(self, ffieldParams):
        '''
        updates the force fields data with the values of the passed parameters

        :param ffieldParams: list of ForceFieldParams
            the parameters from which to update the forcefield
        :return: None
        '''
        for param in ffieldParams:
            loc = param.loc
            self.data[loc[0]][loc[1]][loc[2]] = param.currValue
            if param.linkedParams is None:
                pass
            else:
                for link in param.linkedParams:
                    self.data[loc[0] + link[0]][loc[1] + link[1]][loc[2] + link[2]] = param.currValue
