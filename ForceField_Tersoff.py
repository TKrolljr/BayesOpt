#!/usr/bin/env python
'''
This module encompasses everything needed to parse a Tersoff Force Field File.

ForceField_Tersoff.py
author: Tobias Kroll
created: 6/8/2020
py version: 3.7
'''
import os
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
        self.lines = self.readForceField(ffield_path)
        self.removeCommentsAndWhitespace()
        self.data = self.parseAsData()
		
    def readForceField(self):
        """
        opens the force field file at ffield_path and reads it into lines.
    
        @returns - list of Strings
            a list where each element is a line of the forcefield file
        """
        ffield_file = open(self.ffield_path)
        lines = ffield_file.readlines()
        return lines
		
    def removeCommentsAndWhitespace(self):
        """
        removes lines that are comments or only whitespace in self.lines
        """
        
        i = 0
        linesLength = len(self.lines)
        #Doing it like this, because for line in self.lines[:] is really doing the same thing but
        #also copying the entire list to iterate through... python upsets me
        while (i < linesLength):
            line = self.lines[i]
            if(line[0] == "#"): #line is a comment
	            self.lines.remove(self.lines[i])
	            linesLength-=1
            elif(line == "Tersoff\n"): #the line is 'Tersoff'; the force field identifier
	            self.lines.remove(self.lines[i])
	            linesLength-=1
            elif(line.isspace()): #the line is only whitespace
	            self.lines.remove(self.lines[i])
	            linesLength-=1
            else:
                i+=1
	    
	
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
        try:
            file = open(ffield_path,"x")
            file.write("Tersoff\n")
            for section in self.data:
                for line in section:
                    for entry in line:
                        file.write(entry+" ")
                
                    file.write("\n")
            
                file.write("#end of ffield")
        
        except:
            print("There was an error opening the file to write at path: " + ffield_path)

	
if __name__ == "__main__":
    testObj = ForceField_Tersoff("ffield.tersoff")
    print(testObj.data[0][0][6])
    os.remove("testffield.tersoff")
    testObj.writeForceField("testffield.tersoff")