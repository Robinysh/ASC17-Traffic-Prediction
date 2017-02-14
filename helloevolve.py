"""
helloevolve.py implements a genetic algorithm that starts with a base
population of randomly generated strings, iterates over a certain number of
generations while implementing 'natural selection', and prints out the most fit
string.

The parameters of the simulation can be changed by modifying one of the many
global variables. To change the "most fit" string, modify OPTIMAL. POP_SIZE
controls the size of each generation, and GENERATIONS is the amount of 
generations that the simulation will loop through before returning the fittest
string.

This program subject to the terms of the BSD license listed below.

---

Copyright (c) 2011 Colin Drake

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import random
import train
import tensorflow as tf
#
# Global variables
# Setup optimal string and GA input variables.
#

DNA_SIZE    = 7
POP_SIZE    = 20
GENERATIONS = 5000
MUTATION_RATE = .2
CROSSOVER_RATE = .2
#
# Helper functions
# These are used as support, but aren't direct GA-specific functions.
#

logran = lambda start,end: lambda: 10**random.uniform(start,end)  
uniintran = lambda start,end: lambda: int(random.uniform(start,end))
listran = lambda start,end: lambda length: [int(random.uniform(start,end)) for _ in xrange(length)]
ran_flag = {0:logran(-6, -2), #Learning Rate
            1:logran(-4, -0.2), #Dropout
            2:logran(-4, 2), #weight_decay
            3:uniintran(1, 20), #number_of_features
            4:uniintran(1, 15), #batch_size
            5:uniintran(1, 20),  #number_of_hidden_layers
            6:listran(8, 256)} #HiddenUnits


def weighted_choice(items):
  """
  Chooses a random element from items, where items is a list of tuples in
  the form (item, weight). weight determines the probability of choosing its
  respective item. Note: this function is borrowed from ActiveState Recipes.
  """
  weight_total = sum((item[1] for item in items))
  n = random.uniform(0, weight_total)
  for item, weight in items:
    if n < weight:  #Linear Probability
      return item
    n = n - weight
  return item

def random_flags():
  """
  Return a random character between ASCII 32 and 126 (i.e. spaces, symbols,
  letters, and digits). All characters returned will be nicely printable.
  """
  flags = []
  for i in xrange(len(ran_flag)-1):
    flags.append(ran_flag[i]())
  flags.append(ran_flag[6](flags[5]))
  #flags.append(10**random.uniform(-6, -2)) #Initial learning rate
  #flags.append(10**random.uniform(-4, -0.2)) #ropout rate (1 - keep probability).')
  #flags.append(10**random.uniform(-4, 2)) #Weight for L2 loss on embedding matrix.')
  #flags.append(int(random.uniform(1,20))) #Number of features for graph convolution')
  #flags.append(int(random.uniform(1,20))) #Number of timesteps to feed each time.')
  
  #num_of_hidden_layer = int(random.uniform(1,10))
  #flags.append(num_of_hidden_layer) #Number of Hidden Layers for the graph convolution.') 
  #flags.append([int(random.uniform(8,256)) for _ in xrange(num_of_hidden_layer)])
  
  return flags

def random_population():
  """
  Return a list of POP_SIZE individuals, each randomly generated via iterating
  """
  pop = []
  for i in xrange(POP_SIZE):
    pop.append(random_flags())
  return pop

#
# GA functions
# These make up the bulk of the actual GA algorithm.
#

def fitness(dna):
  """
  For each gene in the DNA, this function calculates the difference between
  it and the character in the same position in the OPTIMAL string. These values
  are summed and then returned.
  """
  
  hyperparameters = { 'learning_rate': dna[0],
                      'dropout': dna[1],
                      'weight_decay': dna[2],
                      'number_of_features': dna[3],
                      'number_of_hidden_layers': dna[5],
                      'batch_size': dna[4],
                      'print_interval': 10,
                      'epoch':2 ,
                      'amount_of_testing_data': 30 }
  hiddenUnits = dna[6]
  print hyperparameters
  print hiddenUnits
  """
  flags = tf.app.flags
  FLAGS = flags.FLAGS
  flags.DEFINE_float('learning_rate', dna[0], 'Initial learning rate.')
  flags.DEFINE_float('dropout', dna[1], 'Dropout rate (1 - keep probability).')
  flags.DEFINE_float('weight_decay',dna[2] , 'Weight for L2 loss on embedding matrix.')
  flags.DEFINE_integer('number_of_features', dna[3], 'Number of features for graph convolution')
  flags.DEFINE_integer('batch_size', dna[4], 'Number of timesteps to feed each time.')
  flags.DEFINE_integer('print_interval', 10, 'Number of runs per print.')
  flags.DEFINE_integer('epoch',1 , 'Number of epochs.')
  
  num_of_layer = random.uniform(2,10)
  flags.DEFINE_integer('number_of_layers', dna[5], 'Number of Layers for the graph convolution.')
  hiddenUnits = dna[6]
  """
  file = open("output.txt", "a")
  print>>file, "\n"
  for el in hyperparameters:
    print>>file, el, hyperparameters[el]
  print>>file, "HiddenUnits: ",hiddenUnits
  file.close()
  fitness = traffic_prediction(hyperparameters, hiddenUnits)
  return fitness

def mutate(dna):
  """
  For each gene in the DNA, there is a 1/mutation_chance chance that it will be
  switched out with a random character. This ensures diversity in the
  population, and ensures that is difficult to get stuck in local minima.
  """
  for i in xrange(DNA_SIZE-1):
    if random.random() < MUTATION_RATE:
      dna[i] = ran_flag[i]()
  if random.random() < MUTATION_RATE:
    dna[6] = ran_flag[6](dna[5])
      
  return dna

def crossover(dna1, dna2):
  """
  Slices both dna1 and dna2 into two parts at a random index within their
  length and merges them. Both keep their initial sublist up to the crossover
  index, but their ends are swapped.
  """
  for i in xrange(len(dna1)-2):
    if random.random() < CROSSOVER_RATE:
      dna1[i], dna2[i] = dna2[i], dna1[i]
  if random.random() < CROSSOVER_RATE:
    dna1[5], dna2[5] = dna2[5], dna1[5]
    dna1[6], dna2[6] = dna2[6], dna1[6]

  return dna1, dna2

#
# Main driver
# Generate a population and simulate GENERATIONS generations.
#

if __name__ == "__main__":
  # Generate initial population. This will create a list of POP_SIZE strings,
  # each initialized to a sequence of random characters.
  population = random_population()
  traffic_prediction=train.TrafficPrediction()
  # Simulate all of the generations.
  for generation in xrange(GENERATIONS):
    print "Generation %s... Random sample: '%s'" % (generation, population[0])
    file = open("output.txt", "a")
    print>>file, "\n\n\nGeneration: ",generation,"\n"
    file.close()

    weighted_population = []

    # Add individuals and their respective fitness levels to the weighted
    # population list. This will be used to pull out individuals via certain
    # probabilities during the selection phase. Then, reset the population list
    # so we can repopulate it after selection.
    for individual in population:
      fitness_val = fitness(individual)

      # Generate the (individual,fitness) pair, taking in account whether or
      # not we will accidently divide by zero.
      if fitness_val == 0:
        pair = (individual, 1.0)
      else:
        pair = (individual, 1.0/fitness_val)

      weighted_population.append(pair)

    population = []

    # Select two random individuals, based on their fitness probabilites, cross
    # their genes over at a random point, mutate them, and add them back to the
    # population for the next iteration.
    for _ in xrange(POP_SIZE/2):
      # Selection
      ind1 = weighted_choice(weighted_population)
      ind2 = weighted_choice(weighted_population)

      # Crossover
      ind1, ind2 = crossover(ind1, ind2)

      # Mutate and add back into the population.
      population.append(mutate(ind1))
      population.append(mutate(ind2))

  # Display the highest-ranked string after all generations have been iterated
  # over. This will be the closest string to the OPTIMAL string, meaning it
  # will have the smallest fitness value. Finally, exit the program.
  fittest_string = population[0]
  minimum_fitness = fitness(population[0])

  for individual in population:
    ind_fitness = fitness(individual)
    if ind_fitness <= minimum_fitness:
      fittest_string = individual
      minimum_fitness = ind_fitness

  print "Fittest String: %s" % fittest_string
  exit(0)
