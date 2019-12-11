import os
import numpy as np
import pandas as pd
import logging
import time
import pickle
import sys
import ast
import getopt
import re
import matplotlib.pyplot as plt

from gp import GeneticProgram
from gp_population_utils import smaGen, rsiGen, macdGen, bollingerGen, ofGen
from gp_signals import SMASignal, RSISignal, MACDSignal, BollingerSignal, OFSignal
from gp_utils import fitness, selection
from indicator_utils import max_drawdown
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

parsed_arguments = sys.argv[1:]

cwd = os.getcwd()
if not os.path.exists(cwd+'/logs/'):
    os.makedirs(cwd+'/logs/')

log_formatter = '%(asctime)s - %(levelname)s - %(message)s'
current_time = str(time.ctime()).replace(':','').replace('  ',' ').replace(' ','_')
logging.basicConfig(level=logging.INFO, format=log_formatter, filename="logs/gp_log_" + current_time + ".log")
logger = logging.getLogger("Main Logger")

# Initialisation
priceArr_list = []

logger.info(f'Reading Data')
for i in os.listdir('data'):
    current_date = i[7:17]
    df_price = pd.read_csv('data/{}'.format(i), parse_dates = ['DateTime'])
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
    priceArr_list.append({'price':priceArr, 'orderflow': ofArr,'date':current_date})

priceArr_list = sorted(priceArr_list,key = lambda x: x['date'])
priceArr_list = np.array(priceArr_list)

# Define Function Packages
popGenFunctions = {'sma':smaGen,'rsi':rsiGen,'macd':macdGen,'bollinger':bollingerGen,'of':ofGen}
signalGenFunctions = {'sma':SMASignal,'rsi':RSISignal,'macd':MACDSignal,'bollinger':BollingerSignal,'of':OFSignal}

# Generate random test set

testIdx = np.random.randint(0,len(priceArr_list))
testArray = priceArr_list[testIdx]
trainArray = priceArr_list[np.arange(len(priceArr_list))[np.argwhere(np.arange(len(priceArr_list)) != testIdx).flatten()]]
logger.info(f'Test Array Idx: {testIdx}')
logger.info(f'Train Array Numbers: {len(trainArray)}')
logger.info(f'Train Array Shape: {len(trainArray[0])}')
# trainArray = priceArr_list[0:-1]
# fullArray = {}
# fullArray['price'] = []
# fullArray['orderflow'] = []
# for arr in trainArray:
#     fullArray['price'].append(arr['price'])
#     fullArray['orderflow'].append(arr['orderflow'])

# fullArray['price'] = np.concatenate(fullArray['price'])
# fullArray['orderflow'] = np.concatenate(fullArray['orderflow'])
# fullArray = [fullArray]
# testArray = priceArr_list[-1]

# Enter key params here
number_of_training_arrays = 30
init_arguments = dict(
    indicator_list = ['sma','rsi','macd','bollinger','of'],
    selection_number = 2,
    init_population = 200,
    crossover_p = 0.8,
    mutation_p = 0.2,
    operator_list = ['or','and'],
    fitness_metric = 'accuracy',
    variance_penalty = 2,
    max_levels = 4,
    max_iter = 30,
    train_as_whole = False
)

# Update default arguments with arguments from the command line
all_arguments = [str(key)+"=" for key in init_arguments.keys()] + ['number_of_training_arrays=']
opts, args = getopt.getopt(parsed_arguments,'x',all_arguments)

updated_arguments = {}
for key, value in opts:
    if key != '--number_of_training_arrays':
        if re.search(r'[^[:alnum:]]',value) is not None:
            updated_arguments[key.replace('--','')] = ast.literal_eval(value)
        elif value.isdigit():
            updated_arguments[key.replace('--','')] = int(value)
        else:
            try:
                updated_arguments[key.replace('--','')] = float(value)
            except ValueError:
                updated_arguments[key.replace('--','')] = value
    elif key  == '--number_of_training_arrays':
        number_of_training_arrays = int(value)

init_arguments.update(updated_arguments)

logger.info(f'Start Genetic Program')
gp = GeneticProgram(input_data = trainArray[0:number_of_training_arrays], #np.random.choice(trainArray,number_of_training_arrays),
                    test_data = testArray,
                    fitness_function = fitness,
                    pop_generator_functions = popGenFunctions,
                    sig_generator_functions = signalGenFunctions,
                    selection_function = selection,
                    logger = logger,
                    **init_arguments)
init_arguments['number_of_training_arrays'] = number_of_training_arrays
init_arguments['current_time'] = current_time

gp.optimise()
logger.info(f'Genetic Program Completed')

# Store in Pickle
cwd = os.getcwd()
if not os.path.exists(cwd+'/best_chromosomes/'):
    os.makedirs(cwd+'/best_chromosomes/')
with open('best_chromosomes/' + current_time + '_chromosome.pk', 'wb') as outfile:
    pickle.dump(gp.best_chromosomes[-1], outfile)

logger.info(f'Testing Optimal Result with the entire dataset')

pnl = []
for dayPrice in list(priceArr_list):
    pnl.append(fitness(dayPrice, signalGenFunctions,gp.best_chromosomes[-1], verbose = True, metric = 'tstat', return_pnl=True))

# Add rolling 30 day t-statistic
rolling_tstat = []
for idx, pnl_array in enumerate(pnl):
    if idx < len(pnl)-30:
        t_stat = np.concatenate(pnl[idx:idx+30])
        t_stat = np.sqrt(t_stat.shape[0])*np.mean(t_stat)/np.std(t_stat, ddof = 1)
        rolling_tstat.append(t_stat)

plt.figure(figsize=(15, 8))
ax = plt.gca()
ax.plot(rolling_tstat)
plt.xlabel('Day')
plt.legend()
plt.savefig("pngs/rolling_tstat_" + current_time + ".png")
plt.show()

pnl = np.concatenate(pnl)
logger.info(f'Total Profit: {pnl.sum()}')
logger.info(f'Total Revenue: {pnl[pnl > 0].sum()}')
logger.info(f'Total Losses: {pnl[pnl < 0].sum()}')
logger.info(f'Number of Trades: {pnl.shape[0]}')
logger.info(f'Probability of win: {sum(pnl > 0)/len(pnl)}')
logger.info(f'Reward to Risk: {np.mean(pnl[pnl > 0])/abs(np.mean(pnl[pnl < 0]))}')
logger.info(f'T Statistic: {np.sqrt(pnl.shape[0])*np.mean(pnl)/np.std(pnl, ddof = 1)}')