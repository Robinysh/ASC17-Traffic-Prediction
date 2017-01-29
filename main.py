#LSTM RNN
#Architecture: LSTM RNN for each individual node with temporal and spatial state transfer
#Input: Speed of neighbour nodes to n depth, temporal and spatial state to n depth
#Output: Speed class at t = n+1
#Todo: Use multirnncell to customize rnn, Transfer Learning on nodes with identical input and output, input neighbour state as state instead of input, Implement stochastic gradient descent
#Food for Thoughts: One-hot encoding for normalization and distinct extreme classification?
#

import tensorflow as tf 
from tensorflow.python.ops import rnn, rnn_cell
import csv
import cPickle as cpk
import sys
csv.field_size_limit(sys.maxsize)

##Input data
graph_raw = cpk.load(open('graph.cpk','rb'))
speed_raw = csv.reader(open('data/speeds.csv','rb'),delimiter='\n')

##Data Processing
#Reverse graph from outward pointing to inward pointing set
graph = {}
for source in graph_raw:
  graph[source[0]] = []
for source in graph_raw:
  for dest in source[1]:
    graph[source[0]].append(dest)

##Convert speed.csv into dictionary
speed = {}
#skip first line of speed csv file
next(speed_raw)
for row in speed_raw:
  speed[int(row[0].split(',')[0])] = [int(el) for el in row[0].split(',')[1:]]
print len(speed.items()[0][1])

##Hyperparameters
learning_rate = 0.001
max_training_iters = 100000
batch_size = 128
display_step = 10
number_of_layers = 2
lstm_size = 12

##Input Training Data Parameter
num_classes = 5
timelength = 100

##Empty variables for storages
#node_state = [0]*len(graph)
output = {}

##RNN input
x = tf.placeholder(tf.int32, 1)
y = tf.placeholder(tf.int32, 1)

"""
#RNN Softmax Weights
softmax_weight = [tf.Variable(tf.random_normal([num_hidden_layers, num_classes]))]
softmax_biases = [tf.Variable(tf.random_normal([num_classes]))] * len(graph)
"""
lstm = rnn_cell.LSTMCell(lstm_size, state_is_tuple=True, activation=tf.nn.relu)
###TODO
graph_RNN_cell = {}
for el in graph:
  print lstm.state_size
  lstm.state_size = 1 + len(graph[el])
  graph_RNN_cell[el] = lstm

graph_RNN = dict((el, rnn_cell.MultiRNNCell([el]*number_of_layers, state_is_tuple=True)) for el in graph_RNN_cell)
initial_state = node_state = dict((el, stacked_lstm.zero_state(batch_size, tf.float32)) for el in graph_RNN)

# The value of state is updated after processing each batch of data. #TODO
#TODO
for node in graph_RNN:
  output[node], node_state[node] = graph_RNN[node](x, node_state[node])


"""
##Create RNN Structure
def RNN(x):
  lstm_cell = rnn_cell.BasicLSTMCell(num_hidden_layers)
  outputs, states = rnn.dynamic_rnn(
                      cell = lstm_cell,
                      inputs = x,
                      dtype=tf.int32
                      )

  #output, state = lstm_cell(x, state)
  return output
  #return tf.nn.softmax(tf.matmul(output, softmax_weights) + softmax_biases)

##Create Graph of RNN
graph_RNN = [RNN(el) for el in x]
"""
##Define loss and optimizer
cost = dict((el, tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output(el), y))) for el in graph_RNN)
optimizer = dict((el, tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost(el))) for el in cost)

##Initialize the variables
#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

##Train and run graph
with tf.Session() as sess:
  sess.run(init)
  print graph_RNN
  for time in xrange(len(speed.items()[0][1])):
    print time
    for node in graph_RNN:
      print "Node:", node, "Time:", time
      sess.run(optimizer[node], feed_dict={x: speed[node][time], y: speed[node][time+1]})

      """
      #Display when step reaches display step
      if step % display_step == 0:
        #Calculate batch accuracy and batch loss
        acc = sess.run(accuracy, feed_dict={x:batch_x, y:batch_y})
        loss = sess.run(cost, feed_dict={x:batch_x, y:batch_y})  
        print "IterNum", step*batch_size, ", Batch Loss", loss, "Training Accuracy=", acc
      """
      
      #Update spatial states
      #node_state = graph_RNN
      #print "Testing Accuracy:", sess.run(accuracy, feed_dict={x:test_data, y:test_label})
