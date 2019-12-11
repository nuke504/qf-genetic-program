import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from copy import deepcopy

# Indicator Utils
def simple_moving_avg(x, n):
    weights = np.ones(n)
    weights /= weights.sum()
    sma = np.convolve(x, weights, mode='full')[:len(x)]
    sma[:n] = sma[n]
    return sma

def sliding_window_view(arr, window_shape, steps):
    in_shape = np.array(arr.shape[-len(steps):])  # [x, (...), z]
    window_shape = np.array(window_shape)  # [Wx, (...), Wz]
    steps = np.array(steps)  # [Sx, (...), Sz]
    nbytes = arr.strides[-1]  # size (bytes) of an element in `arr`

    # number of per-byte steps to take to fill window
    window_strides = tuple(np.cumprod(arr.shape[:0:-1])[::-1]) + (1,)
    # number of per-byte steps to take to place window
    step_strides = tuple(window_strides[-len(steps):] * steps)
    # number of bytes to step to populate sliding window view
    strides = tuple(int(i) * nbytes for i in step_strides + window_strides)

    outshape = tuple((in_shape - window_shape) // steps + 1)
    # outshape: ([X, (...), Z], ..., [Wx, (...), Wz])
    outshape = outshape + arr.shape[:-len(steps)] + tuple(window_shape)
    return np.lib.stride_tricks.as_strided(arr, shape=outshape, strides=strides, writeable=False)

def smaCross(arr):
    #param: arr. Arr should be a 2*2 array
    #type: arr
    
    # If golden cross, sma_fast(t-1) [0,0] < sma_slow(t-1) [1,0] AND sma_fast(t) > sma_slow(t)
    # If dead cross, sma_fast(t-1) > sma_slow(t-1) AND sma_fast(t) < sma_slow(t)
    
    # return: list of whether golden cross or dead cross occurs at time t
    golden_cross = False
    dead_cross = False
    if arr[0,0] < arr[1,0] and arr[0,1] > arr[1,1]:
        golden_cross = True
    if arr[0,0] > arr[1,0] and arr[0,1] < arr[1,1]:
        dead_cross = True
    
    return [golden_cross,dead_cross]

def RSI(x, n):
    windowView = sliding_window_view(deepcopy(np.atleast_2d(x[1:]-x[:-1])), window_shape = (1,n), steps = (1,1))[0]
    sumUp = np.sum(np.where(windowView > 0, windowView, 0), axis = 2)
    sumDown = np.sum(np.where(windowView < 0, -1*windowView, 0), axis = 2)
    sumDown = np.where(sumDown == 0,np.inf,sumDown)
    rsi = 100-100/(1+sumUp/sumDown).flatten()
    return np.concatenate([np.repeat(rsi[0],n),rsi])

def max_drawdown(array, verbose = False):
    cum_return = array.cumsum()

    # Max Drawdown
    lowest_point = np.argmax(np.maximum.accumulate(cum_return) - cum_return) # at which point is the running maximum - current value is largest
    if lowest_point == 0:
        highest_point = 0
    else:
        highest_point = np.argmax(cum_return[:lowest_point]) # start of period
    max_drawdown = cum_return[highest_point] - cum_return[lowest_point]

    # Max Drawdown duration
    duration = []
    for i in range(len(cum_return)):
        cur_val = cum_return[i]
        remaining_vals = cum_return[i+1:]
        is_lower = np.array(remaining_vals <= cur_val)
        # If there are no more remaining values to compare with
        if is_lower.size == 0:
            duration.append(-1)
        # If the first value of the remaining array is higher, we don't need to continue anymore
        elif is_lower[0] == False:
            duration.append(-1)
        # Else we count the number of consecutive trues
        else:
            index = 0
            is_lower = np.insert(is_lower, obj = -1 , values = [False]) # If all are true, the last false will stop the while loop.
            while is_lower[index] == True:
                index += 1
            if all(is_lower[:-2]): # If all values in is_lower except the last value is true, we append index + 1
                duration.append(index+ 1) # Whole array is consecutively true  
            else:
                duration.append(index)
    max_duration_of_drawdown = max(duration)
    max_duration_start = np.argmax(np.array(duration))

    # Visualization
    if verbose:
        plt.plot(cum_return)
        plt.plot([lowest_point, highest_point], [cum_return[lowest_point], cum_return[highest_point]], 'o', color='Red', markersize=10)
        plt.plot([max_duration_start, max_duration_start + max_duration_of_drawdown], [cum_return[max_duration_start], cum_return[max_duration_start + max_duration_of_drawdown]], 'o', color='Black', markersize=10)
        plt.show()
    return max_drawdown, max_duration_of_drawdown

def EWM(arr, n):
    return pd.DataFrame({'y' : arr}).ewm(span = n, adjust = False).mean()

def BollingerBand(arr, n):
    mean = pd.DataFrame({'y' : arr}).rolling(window = n).mean()
    std = pd.DataFrame({'y' : arr}).rolling(window = n).std()
    return mean, std
