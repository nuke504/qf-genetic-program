import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import datetime as dt
import logging
import time
import os

from copy import deepcopy, copy
from binarytree import Node

class GeneticProgram(object):
    def __init__(self, input_data, fitness_function, pop_generator_functions, sig_generator_functions, selection_function, indicator_list, logger, operator_list = ['xor','or','and'],
                 test_data = None, crossover_p = 0.75, mutation_p = 0.1, init_population = 20, selection_number = 2, max_iter = 100, train_percentage = 0.75, variance_penalty = 4,
                 max_levels = 4, fitness_metric = 'accuracy', preserve_max = True, train_as_whole = False, population_params = {}, selection_params = {}):
        
        # Store Functions
        self.fitness_function = fitness_function
        self.pop_generator_functions = pop_generator_functions
        self.sig_generator_functions = sig_generator_functions
        self.selection_function = selection_function
        
        # Store Values
        self.input_data = input_data
        self.test_data = test_data
        self.folds = len(input_data)
        self.validation_iter = itertools.cycle(np.arange(len(input_data)))
        self.init_population = init_population
        self.selection_number = selection_number
        self.crossover_p = crossover_p
        self.mutation_p = mutation_p
        self.max_iter = max_iter
        self.indicator_list = indicator_list
        self.operator_list = operator_list
        self.preserve = preserve_max
        self.max_levels = max_levels
        self.variance_penalty = variance_penalty
        self.fitness_metric = fitness_metric
        self.train_as_whole = train_as_whole
        
        self.logger = logger
        
        # Results
        self.fitness_max = []
        self.fitness_mean = []
        self.test_fitness = []
        self.validation_fitness_mean = []
        self.best_chromosomes = []
                
        # Create and store legend
        legend={idx:value for idx, value in enumerate(self.operator_list+self.indicator_list)}
        legend[None] = None
        self.legend = legend
        
        # Create initial population
        population = []
        for i in range(init_population):
            chromosome = {}
            for group in ['buy','sell']:
                chromosome[group] = {}
                if self.max_levels > 1:
                    newTree = Node(np.random.choice(np.arange(len(self.operator_list))))
                    self.addNode(newTree,operators = self.operator_list, indicators = self.indicator_list, max_level = self.max_levels, progression_p = 0.5)
                    chromosome[group]['tree'] = deepcopy(newTree)
                    chromosome[group]['tree_list'] = self.getTreeList(newTree)
                elif self.max_levels == 1:
                    newTree = Node(np.random.choice(np.arange(len(self.operator_list),len(self.indicator_list+self.operator_list))))
                    chromosome[group]['tree'] = deepcopy(newTree)
                    chromosome[group]['tree_list'] = self.getTreeList(newTree)
            population.append(chromosome)
        self.population = deepcopy(population)
    
    def crossover(self):
        for chromosome in self.population:
            if np.random.rand() <= self.crossover_p:
                for group in ['buy','sell']:
                    another_chromosome = np.random.choice(self.population)

                    treeL1 = deepcopy(chromosome[group]['tree_list'])
                    treeL2 = deepcopy(another_chromosome[group]['tree_list'])

                    choice1 = int(np.random.choice([idx for idx, value in enumerate(chromosome[group]['tree'].values) if value is not None]))
                    choice2 = int(np.random.choice([idx for idx, value in enumerate(another_chromosome[group]['tree'].values) if value is not None]))

                    allowed = self.checkLegality(choice1,choice2,chromosome[group]['tree'],another_chromosome[group]['tree'],max_level = self.max_levels)
                    while not allowed:
                        choice1 = int(np.random.choice([idx for idx, value in enumerate(chromosome[group]['tree'].values) if value is not None]))
                        choice2 = int(np.random.choice([idx for idx, value in enumerate(another_chromosome[group]['tree'].values) if value is not None]))
                        allowed = self.checkLegality(choice1,choice2,chromosome[group]['tree'],another_chromosome[group]['tree'],max_level = self.max_levels)

                    if choice1 == 0 and choice2 == 0:
                        chromosome[group]['tree'], another_chromosome[group]['tree'] = another_chromosome[group]['tree'], chromosome[group]['tree']
                    else:
                        chromosome[group]['tree'][choice1],another_chromosome[group]['tree'][choice2] = another_chromosome[group]['tree'][choice2],chromosome[group]['tree'][choice1]

                    # Reform tree_list.
                    treeL1New = self.getTreeList(chromosome[group]['tree'])
                    treeL2New = self.getTreeList(another_chromosome[group]['tree'])

                    self.treeListCopier(0,treeL1,treeL1New,treeL2,choice1,choice2)
                    self.treeListCopier(0,treeL2,treeL2New,treeL1,choice2,choice1)

                    chromosome[group]['tree_list'] = deepcopy(treeL1New)
                    another_chromosome[group]['tree_list'] = deepcopy(treeL2New)

                self.population = [deepcopy(chromosome) for chromosome in self.population]
    
    def mutation(self):
        for idx, chromosome in enumerate(self.population):
            if np.random.rand() <= self.mutation_p:
                for group in ['buy','sell']:
                    newTree = Node(np.random.choice(np.arange(len(self.operator_list))))
                    self.addNode(newTree,operators = self.operator_list, indicators = self.indicator_list, max_level = self.max_levels, progression_p = 0.5)
                    self.population[idx][group]['tree'] = deepcopy(newTree)
                    self.population[idx][group]['tree_list'] = self.getTreeList(newTree)

                self.population = [deepcopy(chromosome) for chromosome in self.population]
        
        # Keep the best chromosome from before
        if self.preserve:
            self.population = self.population[1:len(self.population)]+[deepcopy(self.best_chromosomes[-1])]
    
    def selection(self):
        validation_set_idx = next(self.validation_iter)
        self.population, utilityValues, fitnessValues, selected_chromosome = self.selection_function(self.population,
                                                                                                     data = self.input_data,
                                                                                                     fitness_function = self.fitness_function,
                                                                                                     signal_generators = self.sig_generator_functions,
                                                                                                     validation_set_idx = None,
                                                                                                     fitness_metric = self.fitness_metric,
                                                                                                     variance_penalty = self.variance_penalty,
                                                                                                     p = None,
                                                                                                     train_as_whole = self.train_as_whole,
                                                                                                     selection_number = self.selection_number)
        
        fitnessValuesTest = np.array([self.fitness_function(self.input_data[validation_set_idx],self.sig_generator_functions,chromosome, metric = self.fitness_metric) for chromosome in self.population])
        self.test_fitness.append(self.fitness_function(self.test_data, signalGenerators = self.sig_generator_functions, chromosome = selected_chromosome))
        if len(fitnessValuesTest[fitnessValuesTest != -np.inf]) == 0:
            self.validation_fitness_mean.append(-np.inf)
        else:
            self.validation_fitness_mean.append(np.nanmean(fitnessValuesTest[fitnessValuesTest != -np.inf]))

        self.best_chromosomes.append(deepcopy(selected_chromosome))
        self.fitness_max.append(np.nanmax(utilityValues))
        if len(utilityValues[utilityValues != -np.inf]) == 0:
            self.fitness_mean.append(-np.inf)
        else:
            self.fitness_mean.append(np.nanmean(utilityValues[utilityValues != -np.inf]))
    
    def plotResults(self):
        cwd = os.getcwd()
        if not os.path.exists(cwd+'/pngs/'):
            os.makedirs(cwd+'/pngs/')

        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        ax.plot(self.fitness_max, label = 'Top Utility')
        ax.plot(self.fitness_mean, label = 'Mean Utility')
        ax.plot(self.test_fitness, label = 'Selected Chromosome Test Fitness')
        ax.plot(self.validation_fitness_mean, label = 'Mean Validation Fitness')
        plt.xlabel('Iteration Number')
        plt.legend()
        plt.savefig("pngs/gp_results_" + str(time.ctime()).replace(':','').replace('  ',' ').replace(' ','_') + ".png")
        plt.show()
    
    def optimise(self, test = True, verbose = True):
        startTime = dt.datetime.now()
        for i in range(self.max_iter):
            self.selection()
            self.crossover()
            self.mutation()
            
            if verbose:
                self.trackResults(i)

        best_chromosome = (deepcopy(self.best_chromosomes[-1]),self.test_fitness[-1])

        self.logger.info(f'Optimal Result: {best_chromosome[0]}')
        self.logger.info(f'Fitness Metric: {self.fitness_metric}')
        self.logger.info(f'Time Taken: {dt.datetime.now()-startTime}')
        
        if test:
            self.logger.info(f'Test Fitness: {self.fitness_function(self.test_data, signalGenerators = self.sig_generator_functions, chromosome = best_chromosome[0])}')
        
        self.plotResults()
        
    def trackResults(self, iteration):
        self.logger.info(f'Iteration: {iteration}')
        self.logger.info(f'Top Utility: {self.fitness_max[-1]}')
        self.logger.info(f'Mean Utility: {self.fitness_mean[-1]}')
            
    def addNode(self, main_node, operators, indicators, current_level = 0, max_level = 3, progression_p = 0.5):
        # Choice Functions
        random_operator_choice = lambda :np.random.choice(np.arange(len(operators)))
        def random_indicator_choice(exclude = None):
            choices = np.arange(len(operators),len(indicators+operators))
            if exclude is None:
                return np.random.choice(choices)
            else:
                return np.random.choice(choices[np.argwhere(choices != exclude).flatten()])
        left_terminus = False
        if current_level < max_level-2:
            if np.random.rand() <= progression_p:
                main_node.left = Node(random_operator_choice())
                self.addNode(main_node = main_node.left, operators = operators, indicators = indicators, current_level = current_level + 1, max_level = max_level)
            else:
                left_terminus = True
                main_node.left = Node(random_indicator_choice())

            if np.random.rand() <= progression_p:
                main_node.right = Node(random_operator_choice())
                self.addNode(main_node = main_node.right, operators = operators, indicators = indicators, current_level = current_level + 1, max_level = max_level)
            else:
                if left_terminus:
                    main_node.right = Node(random_indicator_choice(exclude = main_node.left.value))
                else:
                    main_node.right = Node(random_indicator_choice())
        if current_level == max_level-2:
            main_node.left = Node(random_indicator_choice())
            main_node.right = Node(random_indicator_choice(exclude = main_node.left.value))
    
    def getTreeList(self,this_tree):
        treeList = [self.legend[value] for value in this_tree.values]
        for idx, value in enumerate(treeList):
            if value in self.indicator_list:
                treeList[idx] = (value,self.pop_generator_functions[value]())
        return deepcopy(treeList)
    
    @staticmethod
    def checkLegality(choice1,choice2,tree1,tree2, max_level = 3):
        def parent(k):
            return int((k+1)/2)-1

        def left_child(k):
            return 2*(k+1)-1

        def right_child(k):
            return 2*(k+1)

        # Not allowed to modify rootnode
        if choice1 == 0 and choice2 == 0:
            return True
        
        if choice1 == 0 or choice2 == 0:
            return False

        if choice1 >= int(len(tree1.values)/2) and choice2 >= int(len(tree2.values)/2):
            # If both child nodes, check that the child nodes on the same branch are not the same
            swap1 = None
            swap2 = None
            if left_child(parent(choice1)) == choice1:
                swap1 = 'left'
            else:
                swap1 = 'right'
            if left_child(parent(choice2)) == choice2:
                swap2 = 'left'
            else:
                swap2 = 'right'

            if swap1 == 'left':
                sibling1 = tree1.values[right_child(parent(choice1))]
            else:
                sibling1 = tree1.values[left_child(parent(choice1))]
            if swap2 == 'left':
                sibling2 = tree2.values[right_child(parent(choice2))]
            else:
                sibling2 = tree2.values[left_child(parent(choice2))]

            if tree1.values[choice1] == sibling2 or tree2.values[choice2] == sibling1:
                return False
            else:
                return True
        else:
            # If not, check to make sure that the height of the new tree does not exceed the height limit
            if tree1.height-tree1[choice1].height+1+tree2[choice2].height+1 <= max_level:
                return False
            if tree2.height-tree2[choice2].height+1+tree1[choice1].height+1 <= max_level:
                return False
            return True

    def treeListCopier(self, current_node_idx, tree_list_copy_from, tree_list_copy_to, tree_list_copy_from_other, choice_node_to_idx, choice_node_from_idx):
        left_child = lambda k:2*(k+1)-1
        right_child = lambda k:2*(k+1)
        tree_length = len(tree_list_copy_from)
        if current_node_idx != choice_node_to_idx: # If the current node is the choice node, copy the current node and all the subsequent values FROM the old tree
#             print('CURRENT TREE: Copying node',current_node_idx)
            tree_list_copy_to[current_node_idx] = copy(tree_list_copy_from[current_node_idx]) # Copy the values at the current position
            if current_node_idx < int(tree_length/2): # If current node is not a child node, traverse down the tree. Else, STOP
                if tree_list_copy_from[left_child(current_node_idx)] is not None:
                    self.treeListCopier(left_child(current_node_idx), tree_list_copy_from, tree_list_copy_to, tree_list_copy_from_other, choice_node_to_idx, choice_node_from_idx)
                if tree_list_copy_from[right_child(current_node_idx)] is not None:
                    self.treeListCopier(right_child(current_node_idx), tree_list_copy_from, tree_list_copy_to, tree_list_copy_from_other, choice_node_to_idx, choice_node_from_idx)
        elif current_node_idx == choice_node_to_idx: # If the current node is the choice node, copy the current node and all subsequent values FROM the other old tree
            self.treeListCopierOther(to_node_idx = choice_node_to_idx, from_node_idx = choice_node_from_idx, tree_list_copy_to = tree_list_copy_to, tree_list_copy_from = tree_list_copy_from_other)
            
    def treeListCopierOther(self, to_node_idx, from_node_idx, tree_list_copy_to, tree_list_copy_from):
        left_child = lambda k:2*(k+1)-1
        right_child = lambda k:2*(k+1)
        tree_length = len(tree_list_copy_from)
        # Copy the node value
#         print(f'OTHER -> CURRENT TREE: Copying node from {from_node_idx} to {to_node_idx}')
        tree_list_copy_to[to_node_idx] = copy(tree_list_copy_from[from_node_idx])
        # If its a parent node in the from tree, traverse down the from tree
        if from_node_idx < int(tree_length/2):
            if tree_list_copy_from[left_child(from_node_idx)] is not None:
                self.treeListCopierOther(to_node_idx = left_child(to_node_idx), from_node_idx = left_child(from_node_idx), tree_list_copy_to = tree_list_copy_to, tree_list_copy_from = tree_list_copy_from)
            if tree_list_copy_from[right_child(from_node_idx)] is not None:
                self.treeListCopierOther(to_node_idx = right_child(to_node_idx), from_node_idx = right_child(from_node_idx), tree_list_copy_to = tree_list_copy_to, tree_list_copy_from = tree_list_copy_from)

        