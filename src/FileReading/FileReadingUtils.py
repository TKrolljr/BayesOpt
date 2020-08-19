#!/usr/bin/env python
"""
FileReadingUtils.py
author: Tobias Kroll
created: 7/20/2020
py version: 3.7
"""


def readFileAsLines(pathname):
    '''
    populates a list containing the lines of the file

    :param pathname: String
        the path to the file to be read
    :return: list of Strings
        list containing the lines of the file at pathname
    '''
    file = open(pathname)
    lines = file.readlines()
    return lines

def removeWhiteSpaceAndComments(lines):
    '''
    removes entries that are whitespace or comments from the passed list

    :param lines: list of Strings
        the lines to be parsed
    :return: list of Strings
        the passed in list without whitespace and comments
    '''
    i = 0
    linesLength = len(lines)
    while (i < linesLength):
        line = lines[i]
        if (line[0] == "#"):  # line is a comment
            lines.remove(lines[i])
            linesLength -= 1
        elif (line.isspace()):  # the line is only whitespace
            lines.remove(lines[i])
            linesLength -= 1
        else:
            i += 1
    return lines

def removeFromLines(lines, removalString):
    '''
    removes lines that are only the passed in String
    PASS WITHOUT \n !!!

    :param lines: list of Strings
        the list to be parsed
    :param removalString: String
        the string of which lines are to be removed
    :return: list of Strings
        the passed list without lines of the passed String
    '''
    i = 0
    linesLength = len(lines)
    while (i < linesLength):
        line = lines[i]
        if (line[0] == (removalString+"\n")):  # line is a comment
            lines.remove(lines[i])
            linesLength -= 1
        else:
            i += 1
    return lines
