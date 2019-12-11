import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from copy import deepcopy
from indicator_utils import max_drawdown
from multiprocessing import Process, Queue

# Metric evaluation
def getFitness(pnl, metric = 'accuracy'):
    if len(pnl) <= 1:
        return -np.inf
    if metric == 'accuracy':
        return sum(pnl > 0)/len(pnl)
    elif metric == 'tstat':
        if np.std(pnl, ddof = 1) == 0:
            return -np.inf
        else:
            return np.sqrt(pnl.shape[0])*np.mean(pnl)/np.std(pnl, ddof = 1)
    elif metric == 'risk_reward_ratio':
        if sum(pnl > 0)/len(pnl) == 1: # If its too good to be true then just return negative infinity
            return -np.inf
        else:
            return np.mean(pnl[pnl > 0])/abs(np.mean(pnl[pnl < 0]))
    elif metric == 'profit':
        return pnl.sum()
    elif metric == 'profit_limit_accuracy':
        if sum(pnl > 0)/len(pnl) <= 0.5: # If accuracy is less than 65%, the strategy is nullified
            return -np.inf
        else:
            return pnl.sum()
    elif metric == 'accuracy_limit_risk_reward':
        if sum(pnl > 0)/len(pnl) == 1:
            return 1
        elif np.mean(pnl[pnl > 0])/abs(np.mean(pnl[pnl < 0])) <= 0.7:
            return -np.inf
        else:
            return sum(pnl > 0)/len(pnl)
    else:
        raise NotImplementedError(f'{metric} not yet implemented')

# Fitness function
def fitness(data, signalGenerators, chromosome, stockLimit = 1, verbose = False, maxDrawDownPeriodsLimit = np.inf, maxDrawDownLimit = np.inf, minTrades = 3, metric = 'accuracy', logger = None, return_pnl = False, return_results = False):
    def getBuySellSignal(tree_list, data, signalGenerators, signal_to_return = 'buy'):
        def left_child(k):
            return 2*(k+1)-1

        def right_child(k):
            return 2*(k+1)

        def operator(arr1, arr2, operation):
            if operation == 'xor':
                return np.logical_xor(arr1,arr2)
            elif operation == 'or':
                return arr1 | arr2
            elif operation == 'and':
                return arr1 & arr2
            else:
                raise Exception('Operator not yet implemented')

        # Account for tree_lists of length 1
        if len(tree_list) == 1:
            return signalGenerators[tree_list[0]](input_data = data[tree_list[0]['data']], signal_to_return = signal_to_return, **tree_list[0])
        else:
            signalStore = list(np.empty(int(len(tree_list)/2)))
            for node_idx in range(int(len(tree_list)/2)-1,-1,-1):
                if tree_list[node_idx] is not None:
                    if type(tree_list[node_idx]) == str:
                        leftChildIdx = left_child(node_idx)
                        rightChildIdx = right_child(node_idx)
                        leftChild = tree_list[leftChildIdx]
                        rightChild = tree_list[rightChildIdx]
                        if type(leftChild) == str:
                            arrLeft = signalStore[leftChildIdx]
                        else:
                            arrLeft = signalGenerators[leftChild[0]](input_data = data[leftChild[1]['data']], signal_to_return = signal_to_return, **leftChild[1])
                        if type(rightChild) == str:
                            arrRight = signalStore[rightChildIdx]
                        else:
                            arrRight = signalGenerators[rightChild[0]](input_data = data[rightChild[1]['data']], signal_to_return = signal_to_return, **rightChild[1])
                        buy_signal = operator(arrLeft,arrRight,tree_list[node_idx])
                        signalStore[node_idx] = deepcopy(buy_signal)
                    elif type(tree_list[node_idx]) == tuple:
                        buy_signal = signalGenerators[tree_list[node_idx][0]](input_data = data[tree_list[node_idx][1]['data']], signal_to_return = signal_to_return, **tree_list[node_idx][1])
                        signalStore[node_idx] = deepcopy(buy_signal)
            return signalStore[0]

    # Generate buy and sell signals
    buySignal = getBuySellSignal(chromosome['buy']['tree_list'],data,signalGenerators,signal_to_return = 'buy')
    sellSignal = getBuySellSignal(chromosome['sell']['tree_list'],data,signalGenerators,signal_to_return = 'sell')

    stock = 0
    buyPrices = []
    sellPrices = []

    # If the buy/sell signal is in the last period, then it is already too late to buy/sell the stock
    # if it is in the second last period and there is still stock available, close position
    
    periodsWithBuySignals = np.argwhere(buySignal).flatten()
    periodsWithSellSignals = np.argwhere(sellSignal).flatten()
    periodsWithBuySellSignals = np.unique(np.concatenate([periodsWithBuySignals,periodsWithSellSignals]))
    periodsWithBuySignals = set(periodsWithBuySignals)
    periodsWithSellSignals = set(periodsWithSellSignals)

    if return_results:
        actual_buy_period = []
        actual_sell_period = []

    for period in periodsWithBuySellSignals:
        if period in periodsWithBuySignals and stock < stockLimit:
            stock += 1
            buyPrices.append(data['price'][period]) # Buy and sell 5 at every time. Each index point is 100 yen
            if return_results:
                actual_buy_period.append(period)
        elif period in periodsWithSellSignals and stock > 0:
            stock -= 1
            sellPrices.append(data['price'][period])
            if return_results:
                actual_sell_period.append(period)
    # force the position to close
    if stock > 0:
        sellPrices += [data['price'][data['price'].shape[0]-1]]*stock
        stock = 0
        if return_results:
            actual_sell_period.append(data['price'].shape[0]-1)
    assert stock == 0
    
    # Calculate P&L
    buyPrices = np.array(buyPrices)
    sellPrices = np.array(sellPrices)
    pnl = 500*(sellPrices-buyPrices)-400 # Account for transaction cost

    if return_pnl:
        return pnl

    if return_results:
        assert len(actual_buy_period) == len(buyPrices)
        assert len(actual_sell_period) == len(sellPrices)
        assert len(actual_buy_period) == len(actual_sell_period)
        return np.array(actual_buy_period), np.array(actual_sell_period), buyPrices, sellPrices, pnl
    
    # Return T-statistic
    if pnl.shape[0] == 1 or pnl.shape[0] < minTrades or len(pnl) == 0:
        return -np.inf
    
    # Check Max Drawdown Period for the strategy
    maxDrawDown, maxDrawDownP = max_drawdown(pnl)
    if maxDrawDown > maxDrawDownLimit or maxDrawDownP > maxDrawDownPeriodsLimit:
        return -np.inf
    
    if verbose:
        if logger is None:
            print('Total Profit',pnl.sum())
            print('Total Revenue',pnl[pnl > 0].sum())
            print('Total Losses',pnl[pnl < 0].sum())
            print('Number of Trades:',pnl.shape[0])
            print('Probability of win: ', sum(pnl > 0)/len(pnl))
            print('Reward to Risk: ', np.mean(pnl[pnl > 0])/abs(np.mean(pnl[pnl < 0])))
            print('T Stat:',np.sqrt(pnl.shape[0])*np.mean(pnl)/np.std(pnl, ddof = 1))
            print('Max Drawdown: ', maxDrawDown)
            print('Max Drawdown Period: ', maxDrawDownP)
        else:
            logger.info(f'Total Profit: {pnl.sum()}')
            logger.info(f'Total Revenue: {pnl[pnl > 0].sum()}')
            logger.info(f'Total Losses: {pnl[pnl < 0].sum()}')
            logger.info(f'Number of Trades: {pnl.shape[0]}')
            logger.info(f'Probability of win: {sum(pnl > 0)/len(pnl)}')
            logger.info(f'Reward to Risk: {np.mean(pnl[pnl > 0])/abs(np.mean(pnl[pnl < 0]))}')
            logger.info(f'T Statistic: {np.sqrt(pnl.shape[0])*np.mean(pnl)/np.std(pnl, ddof = 1)}')
            logger.info(f'Max Drawdown: {maxDrawDown}')
            logger.info(f'Max Drawdown Period: {maxDrawDownP}')
        
    return getFitness(pnl,metric)


# Selection Function
def selectionOriginal(population, data, fitness_function, signal_generators, variance_penalty = 4, validation_set_idx = None, selection_number = 2, p = 0.2, verbose = False, fitness_metric = 'accuracy', train_as_whole = False):
    def utility(fitness_arr, A = 4):
        mean_util = np.mean(fitness_arr)
        if mean_util == -np.inf:
            return -np.inf
        if len(fitness_arr) == 1:
            return mean_util
        else:
            return mean_util - A*np.std(fitness_arr, ddof = 1)
    # Use tournament selection method
    n = len(population)
    if p is not None:
        p_values = [p*(1-p)**i for i in range(n)] #np.ones(n)/n
    else:
        p_values = np.ones(n)/n
    
    folds = len(data)
    if validation_set_idx is not None:
        train_data = np.array(data)[np.arange(folds)[np.argwhere(np.arange(folds) != validation_set_idx).flatten()]]
    else:
        train_data = np.array(data)
    
    if not train_as_whole:
        utilityValues = []
        for idx, chromosome in enumerate(population):
            fitness = [fitness_function(data = data_part, signalGenerators = signal_generators, chromosome = chromosome, metric = fitness_metric) for data_part in train_data]
            row = [idx,utility(fitness, A = variance_penalty),np.mean(fitness)]
            utilityValues.append(row)
    if train_as_whole:
        utilityValues = []
        for idx, chromosome in enumerate(population):
            pnl = [fitness_function(data = data_part, signalGenerators = signal_generators, chromosome = chromosome, return_pnl = True) for data_part in train_data]
            pnl = np.concatenate(pnl)
            chromosome_fitness = getFitness(pnl, metric = fitness_metric)
            row = [idx,chromosome_fitness,chromosome_fitness]
            utilityValues.append(row)
    utilityValues = np.array(utilityValues)
    utilityValues = np.array(sorted(utilityValues, key = lambda x:-x[1]))
    
    selected_chromosome = deepcopy(population[int(utilityValues[int(0.0*n)][0])]) # Select the chromosome with the top utility value
    if verbose:
        print(utilityValues)
    selected_idx = utilityValues[:,0][list(map(lambda x:x.min(),np.random.choice(np.arange(utilityValues.shape[0]), size = (n,selection_number), p = p_values)))].astype(int)
    return [deepcopy(chromosome) for chromosome in np.array(population)[selected_idx]], utilityValues[:,1], utilityValues[:,2], selected_chromosome

def selection(population, data, fitness_function, signal_generators, variance_penalty = 4, validation_set_idx = None, selection_number = 2, p = 0.2, verbose = False, fitness_metric = 'accuracy', train_as_whole = False):

    def utility(fitness_arr, A = 4):
        mean_util = np.mean(fitness_arr)
        if mean_util == -np.inf:
            return -np.inf
        if len(fitness_arr) == 1:
            return mean_util
        else:
            return mean_util - A*np.std(fitness_arr, ddof = 1)
        
    def fitnessMultiprocess(queue, fitness_function, data, signalGenerators, chromosome, metric):
        queue.put(fitness_function(data = data, signalGenerators = signalGenerators, chromosome = chromosome, metric = metric))

    # Use tournament selection method
    n = len(population)
    if p is not None:
        p_values = [p*(1-p)**i for i in range(n)] #np.ones(n)/n
    else:
        p_values = np.ones(n)/n
    
    folds = len(data)
    if validation_set_idx is not None:
        train_data = np.array(data)[np.arange(folds)[np.argwhere(np.arange(folds) != validation_set_idx).flatten()]]
    else:
        train_data = np.array(data)
    
    if not train_as_whole:
        utilityValues = []
        for idx, chromosome in enumerate(population):
            indices = np.concatenate([np.arange(0,len(data),150),[len(data)]])
            fitness_values = []
            for i in range(0,len(indices)):
                if i < len(indices)-1:
                    fitness_queue = Queue()
                    processes = []
                    for data_part in train_data[indices[i]:indices[i+1]]:
                        p = Process(target = fitnessMultiprocess, args = (fitness_queue, fitness_function, data_part, signal_generators, chromosome, fitness_metric))
                        processes.append(p)
                        p.start()
                    for p in processes:
                        fitness_values.append(fitness_queue.get())
                    for p in processes:
                        p.join()
            fitness_values = np.array(fitness_values)
            row = [idx,utility(fitness_values, A = variance_penalty),np.mean(fitness_values)]
            utilityValues.append(row)
    if train_as_whole:
        utilityValues = []
        for idx, chromosome in enumerate(population):
            pnl = [fitness_function(data = data_part, signalGenerators = signal_generators, chromosome = chromosome, return_pnl = True) for data_part in train_data]
            pnl = np.concatenate(pnl)
            chromosome_fitness = getFitness(pnl, metric = fitness_metric)
            row = [idx,chromosome_fitness,chromosome_fitness]
            utilityValues.append(row)
    utilityValues = np.array(utilityValues)
    utilityValues = np.array(sorted(utilityValues, key = lambda x:-x[1]))
    
    selected_chromosome = deepcopy(population[int(utilityValues[int(0.0*n)][0])]) # Select the chromosome with the top utility value
    if verbose:
        print(utilityValues)
    selected_idx = utilityValues[:,0][list(map(lambda x:x.min(),np.random.choice(np.arange(utilityValues.shape[0]), size = (n,selection_number), p = p_values)))].astype(int)
    return [deepcopy(chromosome) for chromosome in np.array(population)[selected_idx]], utilityValues[:,1], utilityValues[:,2], selected_chromosome