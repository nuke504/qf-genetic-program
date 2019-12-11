# This file contains the population generation functions

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from copy import deepcopy

# Chromosome Initialisation Functions
def smaGen(periodLimit = 40):
    periods = np.random.randint(low = 1, high = periodLimit, size=2)
    return {'fast_period':periods.min(),'slow_period':periods.max(),'data':'price'}

def rsiGen(periodLimit = 40, lower_limit_max = 30, upper_limit_min = 70):
    lower_limit = np.random.randint(low = 1, high = lower_limit_max)
    upper_limit = np.random.randint(low = upper_limit_min, high = 100)
    return {'period':np.random.randint(low = 1, high = periodLimit),'upperLimit':upper_limit,'lowerLimit':lower_limit,'data':'price'}

def macdGen(periodLimit = 40, signal_period_limit = 40):
    periods = np.random.randint(low = 1, high = periodLimit, size=2)
    return {'fast_period':periods.min(),'slow_period':periods.max(),'signal_period':np.random.randint(low = 1, high = signal_period_limit),'data':'price'}

def bollingerGen(periodLimit = 40, max_no_std = 3):
    return {'period':np.random.randint(low = 2, high = periodLimit),'no_of_sd':np.random.rand()*max_no_std,'data':'price'}

def ofGen(periodLimit = 40, upper_limit = 40):
    return {'flow_sum_period':np.random.randint(low = 1, high = periodLimit),'flow_overbought_limit':np.random.randint(low = 1, high = upper_limit),'flow_oversold_limit':np.random.randint(low = 1, high = upper_limit),'data':'orderflow'}