import plotly.graph_objects as go
import plotly as ply
import numpy as np
import pandas as pd
from gp_signals import SMASignal, RSISignal, MACDSignal, BollingerSignal, OFSignal
from gp_utils import fitness
from indicator_utils import max_drawdown
import os 
import pickle
import sys
import matplotlib.pyplot as plt
from pprint import pprint

best_chromosome_name = 'Sat_Nov_9_100647_2019'

# Load Dataset

file_name = 'NO'

def loadData():
    priceArr_list = []
    for i in os.listdir('genetic_program\\{}'.format(file_name)):
        current_date = i[7:17]
        df_price = pd.read_csv('genetic_program\\{}\\{}'.format(file_name, i), parse_dates = ['DateTime'])
        df_price = df_price.set_index('DateTime')
        df_cst = df_price
        df_cst = df_cst.resample('D').ohlc()['Price']
        df_price['MeanTP'] = (df_price['Bbid'] + df_price['Bask'])/2
        df_price['OF'] = np.where(df_price['Price'] > df_price['MeanTP'], df_price['Volume'], np.where(df_price['Price'] < df_price['MeanTP'], - df_price['Volume'],0))
        priceDF = df_price.groupby(pd.Grouper(freq='1Min')).tail(1)['Price']
        priceDF.index = priceDF.index.map(lambda x: x.replace(second=0))
        necessaryTimes = pd.date_range(priceDF.index[0], priceDF.index[-1], freq='T').tolist()
        missingTimes = [i for i in necessaryTimes if i not in priceDF.index]
        # Assuming 1st minute will always have a trade
        for time in missingTimes:
            priceDF[time] = priceDF[time - pd.Timedelta(minutes=1)]
        priceArr = priceDF.values
        ofArr = df_price.resample('T').sum()['OF'].values
        priceArr = priceDF.values
        priceArr_list.append({'price':priceArr, 'orderflow': ofArr,'date':current_date,'candlestick':df_cst, 'datetime' : priceDF.index})

    priceArr_list = sorted(priceArr_list,key = lambda x: x['date'])
    priceArr_list = np.array(priceArr_list)

    with open('genetic_program\\processed_data_{}.pk'.format(file_name), 'wb') as outfile:
        pickle.dump(priceArr_list, outfile)
    return 

if __name__ == "__main__":
    signalGenFunctions = {'sma':SMASignal,'rsi':RSISignal,'macd':MACDSignal,'bollinger':BollingerSignal,'of':OFSignal}

    with open(f"genetic_program\\best_chromosomes\\{best_chromosome_name}_chromosome.pk", 'rb') as infile:
        print('Best Chromosome loaded')
        bestChromosome = pickle.load(infile)

    if not os.path.exists('genetic_program\\processed_data_{}.pk'.format(file_name)):
        print('Processing data')
        loadData()
        
    with open('genetic_program\\processed_data_{}.pk'.format(file_name), 'rb') as infile:
        print('Cached data found, loading cached data')
        priceArr_list = pickle.load(infile)

    # Collect all buy, sell signals, print out results
    total_pnl = []
    buy_periods = []
    sell_periods = []
    buy_prices_all = []
    sell_prices_all = []
    for dayPrice in list(priceArr_list):
        buy_period, sell_period, buy_prices, sell_prices, daily_pnl = fitness(dayPrice, signalGenFunctions,bestChromosome, return_results = True)
        buy_periods.append(dayPrice['datetime'][buy_period])
        sell_periods.append(dayPrice['datetime'][sell_period])
        buy_prices_all.append(buy_prices)
        sell_prices_all.append(sell_prices)
        total_pnl.append(daily_pnl)

    buy_periods = np.concatenate(np.array(buy_periods))
    sell_periods = np.concatenate(np.array(sell_periods))
    buy_prices_all = np.concatenate(np.array(buy_prices_all))
    sell_prices_all = np.concatenate(np.array(sell_prices_all))

    buy_sell_signals = pd.DataFrame({'buy_periods' : buy_periods, 
                                    'sell_periods' : sell_periods, 
                                    'buy_prices_all' : buy_prices_all, 
                                    'sell_prices_all' : sell_prices_all})

    buy_sell_signals.to_csv('results.csv')
    print("Buy and sell signals saved to '{}\\results.csv'".format(os.getcwd()))

    pnl = np.array([day_pnl.sum() for day_pnl in total_pnl])
    maxDrawDown, maxDrawDownP = max_drawdown(pnl)

    results = dict(
        profit = pnl.sum(),
        revenue = pnl[pnl > 0].sum(),
        losses = pnl[pnl < 0].sum(),
        trade_number = np.concatenate(total_pnl).shape[0],
        win_p = sum(pnl > 0)/len(pnl),
        reward_risk = np.mean(pnl[pnl > 0])/abs(np.mean(pnl[pnl < 0])),
        t_stat = np.sqrt(pnl.shape[0])*np.mean(pnl)/np.std(pnl, ddof = 1),
        max_drawdown = maxDrawDown,
        max_drawdown_period = maxDrawDownP
    )
    print("Performance on {}".format(file_name))
    pprint(results)