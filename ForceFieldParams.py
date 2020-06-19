#!/usr/bin/env python
'''
module for conjoining a parameter's location in the force field, their possible range, and their current Value

ForceFieldParams.py
author: Tobias Kroll
created: 6/18/2020
py version: 3.7
'''
import genRandoms as rand

class ForceFieldParam:
    '''
    class containing a parameter's location in the force field, their possible range, and their current Value
    '''
    def __init__(self, loc, paramRange, currValue=None, linkedParams=None):

        self.loc = loc
        self.paramRange = paramRange
        if currValue is None:
            currValue = rand.RandomInRange_Tuple(paramRange)
        self.currValue = currValue
        self.linkedParams = linkedParams

