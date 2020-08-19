#!/usr/bin/env python
"""
errorsUtil.py
author: Tobias Kroll
created: 7/22/2020
py version: 3.7
"""

def singleMeanSquaredError(predictedNum, realNum):
    '''
    mean squared error for single values

    :param predictedNum: float
        the predicted value
    :param realNum: float
        the 'known' real value
    :return: float
        the mean squared error between the two passed value
    '''
    return ((predictedNum - realNum)**2)