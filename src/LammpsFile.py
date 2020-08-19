#!/usr/bin/env python
"""
ObjectiveFunction.py
author: Tobias Kroll
created: 6/17/2020
py version: 3.7
"""

from src.FileReading import FileReadingUtils as fr


class LammpsFile:
    '''
    class is to handle reading and maintaining a singular lamps file

    Fields:
        pathname - String
            the path to the handled file
        data - list of list of Strings
            2d list containing each line of the passed file, further split into words\
        replacementDict - dict; String to String
            dictionary maintaining the list of replacements to perform on the lammps file during write
    '''
    def __init__(self,pathname):
        '''

        :param pathname: String
            path to the handled file
        '''
        self.pathname = pathname
        lines = fr.readFileAsLines(pathname)
        self.data = [line.split() for line in lines]
        self.replacementDict = {}

    def writeLammpsFile_withReplacement(self,lmp_path):
        '''
        writes the handled file, with replacement values, to the passed location

        :param lmp_path: String
            location to write to
        '''
        file = open(lmp_path,'w')

        for line in self.data:
            for entry in line:
                if entry[0] == '%':
                    try:
                        file.write(str(self.replacementDict[entry])+" ")
                    except:
                        file.write(entry + " ")
                else:
                    file.write(entry+" ")

            file.write("\n")

if __name__ == "__main__":
    x = LammpsFile("lammpsTest.lmp")
    x.replacementDict["%structure"] = "HELLO.DATA"
    x.writeLammpsFile_withReplacement("test.lmp")