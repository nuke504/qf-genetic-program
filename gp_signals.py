import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from copy import deepcopy
from indicator_utils import simple_moving_avg, RSI, max_drawdown, BollingerBand, EWM, sliding_window_view, smaCross

# Signal Generating Functions
def SMASignal(input_data, fast_period, slow_period, signal_to_return = 'buy', **kwargs):
    # Return: Bool array buy signal, Bool array sell signal
    sma_data = np.concatenate([np.atleast_2d(simple_moving_avg(input_data, fast_period)),np.atleast_2d(simple_moving_avg(input_data, slow_period))], axis = 0)
    sma_window = sliding_window_view(deepcopy(sma_data), window_shape = (2,2), steps = (1,1))[0]
    # This array when to buy and when to sell
    # Since we are restricted to a long position and we want to close our position at the end, remove
    # 1) The first sell signal if the sell signal is before the buy signal
    # 1) The last buy signal if the buy signal is after the last sell signal
    positions = np.transpose(np.array(list(map(smaCross, sma_window))))
    if signal_to_return == 'buy':
        return np.concatenate([[False],positions[0]])
    elif signal_to_return == 'sell':
        return np.concatenate([[False],positions[1]])

def RSISignal(input_data, period, upperLimit, lowerLimit, signal_to_return = 'buy', **kwargs):
    # sellLimit represents overbought price. I.e. sell signal
    # buyLimit represents oversold price. I.e. buy signal
    rsi = RSI(input_data, period)
    if signal_to_return == 'buy':
        return rsi <= lowerLimit
    elif signal_to_return == 'sell':
        return rsi >= upperLimit

def MACDSignal(input_data, fast_period, slow_period, signal_period, signal_to_return = 'buy', verbose = False, **kwargs):
    exp1 = EWM(input_data, n = fast_period)
    exp2 = EWM(input_data, n = slow_period)
    
    macd = exp1 - exp2
    signal_line = macd.ewm(span= signal_period, adjust=False).mean().values
    macd = macd.values
    prev_macd = np.roll(macd, -1)
    prev_macd = prev_macd[:-1]
    prev_signal_line = np.roll(signal_line, -1)
    prev_signal_line = prev_signal_line[:-1]
    
    if signal_to_return == 'buy':
        # If the MACD crosses the signal line upward
        buy = np.logical_and(macd[:-1] > signal_line[:-1], prev_macd <= prev_signal_line)
        buy = np.append(buy, False)
        return buy.flatten()
    elif signal_to_return == 'sell':
        # The other way around
        sell = np.logical_and(macd[:-1] < signal_line[:-1], prev_macd >= prev_signal_line)
        sell = np.append(sell, False)
        return sell.flatten()
    # Viz
    if verbose:
        buy = np.logical_and(macd[:-1] > signal_line[:-1], prev_macd <= prev_signal_line)
        buy = np.append(buy, False)
        sell = np.logical_and(macd[:-1] < signal_line[:-1], prev_macd >= prev_signal_line)
        sell = np.append(sell, False)
        plt.plot(macd)
        plt.plot(signal_line)
        buy_sig = signal_line* buy
        buy_sig[buy_sig == 0] = np.nan
        sell_sig = signal_line* sell
        sell_sig[sell_sig == 0] = np.nan
        plt.plot(sell_sig, 'o', color='Red', markersize=4)
        plt.plot(buy_sig, 'o', color='Black', markersize=4)

def BollingerSignal(input_data, period, no_of_sd = 2, signal_to_return = 'buy', verbose = False, **kwargs):
    mean, std = BollingerBand(input_data, period)
    input_data = input_data.reshape(-1, 1)
    avail_price = input_data[period:]
    
    if signal_to_return == 'buy':
        lower_bollinger = (mean - no_of_sd * std).values
        avail_lower_bollinger = lower_bollinger[period:]
        buy = avail_price < avail_lower_bollinger
        buy = np.append([False] * period, buy).reshape(-1, 1)
        return buy.flatten()
    elif signal_to_return == 'sell':
        higher_bollinger = (mean + no_of_sd * std).values
        avail_higher_bollinger = higher_bollinger[period:]
        sell = avail_price > avail_higher_bollinger
        sell = np.append([False] * period, sell).reshape(-1, 1)
        return sell.flatten()

    if verbose:
        lower_bollinger = (mean - no_of_sd * std).values
        avail_lower_bollinger = lower_bollinger[period:]
        buy = avail_price < avail_lower_bollinger
        buy = np.append([False] * period, buy).reshape(-1, 1)
        
        higher_bollinger = (mean + no_of_sd * std).values
        avail_higher_bollinger = higher_bollinger[period:]
        sell = avail_price > avail_higher_bollinger
        sell = np.append([False] * period, sell).reshape(-1, 1)
        
        higher_bollinger = higher_bollinger.reshape(-1, 1)
        lower_bollinger = lower_bollinger.reshape(-1, 1)
        plt.plot(price)
        plt.plot(higher_bollinger)
        plt.plot(lower_bollinger)
        buy_sig = price* buy
        buy_sig[buy_sig == 0] = np.nan
        sell_sig = price* sell
        sell_sig[sell_sig == 0] = np.nan
        plt.plot(sell_sig, 'o', color='Red', markersize=4)
        plt.plot(buy_sig, 'o', color='Black', markersize=4)
        plt.show()

def OFSignal(input_data, flow_sum_period, flow_overbought_limit, flow_oversold_limit,  signal_to_return = 'buy', **kwargs):
    """
    futures: Dataframe of the futures data as is
    flow_sum_period: period of lookback for orderflow strength determinant
    flow_overbought_limit: limit to produce sell signal when flow_count crosses this figure
    flow_oversold_limit: limit to produce buy signal when flow_count crosses this figure
    """

    # Gets the direction of flow, 1 inflow (more buy than sell), vice versa
    flow_direction = np.where(input_data > 0, 1, np.where(input_data < 0, -1, np.nan))
    # find the sum of the past orderflow direction, high +ve number means persistent upwards buying
    futures= pd.DataFrame({'flow_direction' : flow_direction})
    futures["flow_count"] = futures["flow_direction"].rolling(flow_sum_period).sum()
    flow_count = futures["flow_count"].values
    flow_count = np.nan_to_num(flow_count)

    # sell signal when flow_count exit flow_overbought_limit
    if signal_to_return == 'buy':
        buy = np.where(flow_count < -flow_oversold_limit, True, False)
        return buy
    elif signal_to_return == 'sell':
        sell = np.where(flow_count > flow_overbought_limit, True, False)
        return sell