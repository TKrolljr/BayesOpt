#!/usr/bin/env python
'''
TODO: Possibly get rid off down the line
'''

'''
Module to handle the reading of known values saved in files.
Particularly handles the reading of structure files and their corresponding desired/calculated target values

KnownValuesReading.py
author: Tobias Kroll
created: 7/13/2020
py version: 3.7
'''
from src.FileReading import FileReadingUtils as fr


def readStructureEnergyValuesToDict(pathname):
    '''
    reads the structure-energy pairins required to calculate error between a lammps MD and known values

    :param pathname: String
        the path to the text file holding the pairings
    :return: Dictionary
        a dictionary which holds the structure file name as a key, and energy values as value
    '''
    lines = fr.readFileAsLines(pathname)
    lines = fr.removeWhiteSpaceAndComments(lines)
    data = [line.split() for line in lines]
    dict = {}
    for pair in data:
        dict[pair[0]] = pair[1]
    return dict