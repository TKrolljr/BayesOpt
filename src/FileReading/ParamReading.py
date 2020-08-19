#!/usr/bin/env python3
"""
tersoffOptimizationTest.py
author: Tobias Kroll
created: 8/12/2020
py version: 3.7
"""
import src.FileReading.FileReadingUtils as fr
from src.ForceFieldParams import ForceFieldParam


def readParams(pathname):
    '''
    reads in the parameter file at pathname, and returns a split-line by line list (one line per
    inner list, one word per inner list member)

    :param pathname: String
        the location of the param list
    :return: 2d array/list of lists
        the list containing the param files lines
    '''
    lines = fr.readFileAsLines(pathname)
    lines = fr.removeWhiteSpaceAndComments(lines)
    data = [line.split() for line in lines]
    ffieldParams_ = []
    for line in data:
        ffieldParams_.append(ForceFieldParam((int(line[0])-1, int(line[1])-1, int(line[2])+2)\
                                             , (float(line[4]), float(line[5])), linkedParams=[]))
        temp = ffieldParams_[len(ffieldParams_)-1]
        print(temp.loc)
    return ffieldParams_
