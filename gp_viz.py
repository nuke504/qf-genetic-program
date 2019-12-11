import plotly.graph_objects as go
import numpy as np
import pandas as pd
from gp_signals import SMASignal, RSISignal, MACDSignal, BollingerSignal, OFSignal
from gp_utils import fitness
import os 
import pickle
import sys
import matplotlib.pyplot as plt

# Load Best Chromosome
best_chromosome_name = 'Mon_Nov_11_151624_2019'

def loadData():
    for i in os.listdir('NO'):
        df_price = pd.read_csv('NO/{}'.format(i), parse_dates = ['DateTime'])
        df_price = df_price.set_index('DateTime')
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
        priceArr_list.append({'price':priceArr, 'orderflow': ofArr})

    priceArr_list = np.array(priceArr_list)

    with open('processed_data.pk', 'wb') as outfile:
        pickle.dump(priceArr_list, outfile)

# Define Function Packages
signalGenFunctions = {'sma':SMASignal,'rsi':RSISignal,'macd':MACDSignal,'bollinger':BollingerSignal,'of':OFSignal}

with open(f'/Users/starlight/Desktop/QF Project/qf206/genetic_program/best_chromosomes/Mon_Nov_11_151624_2019_chromosome.pk', 'rb') as infile:
    bestChromosome = pickle.load(infile)

if not os.path.exists(os.getcwd()+'/processed_data.pk'):
    loadData()
    
with open('processed_data.pk', 'rb') as infile:
    priceArr_list = pickle.load(infile)

prices = np.concatenate([day_data['price'] for day_data in priceArr_list])
orderflow = np.concatenate([day_data['orderflow'] for day_data in priceArr_list])

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(len(prices)), y = prices, mode='lines',name='Price'))
# fig.add_trace(go.Scatter(x=np.arange(len(orderflow)), y = orderflow, mode='lines',name='Orderflow'))
fig.update_layout(height = 600,
                  width = 1000,
                  title_text = 'Subplots', xaxis_rangeslider_visible=True)
fig.show()