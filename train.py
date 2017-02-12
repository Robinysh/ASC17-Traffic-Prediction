from __future__ import division
#from __future__ import print_function

import time
import tensorflow as tf
import math
from utils import *
from models import GCN

import sys
import cPickle as cpk
import csv
csv.field_size_limit(sys.maxsize)

##Input data
graph_raw = cpk.load(open('data/graph.cpk', 'rb'))
speed_raw = csv.reader(open('data/speeds.csv','rb'),delimiter='\n')
graph_list = cpk.load(open('data/graphlist.cpk','rb'))

#Record array
record = []
# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby',
flags.DEFINE_float('learning_rate', 4e-6, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.01, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 1e-1, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('number_of_features', 10, 'Number of features for graph convolution')
flags.DEFINE_integer('number_of_layers', 3, 'Number of Layers for the graph convolution.')
flags.DEFINE_integer('batch_size', 5, 'Number of timesteps to feed each time.')
flags.DEFINE_integer('early_stopping', 10, 'NOT YET IMPLEMENTED Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('print_interval', 10, 'Number of runs per print.')
flags.DEFINE_integer('epoch',10 , 'Number of epochs.')
flags.DEFINE_integer('amount_of_testing_data', 20, 'NOT YET IMPLEMENTED Amount of testing data for validationa')
hiddenUnits = [64,64]

#adjecency matrix
graph = [[ 0 for i in range(len(graph_list))] for j in range(len(graph_list))]

for node in graph_raw:
  #print "NODE1",node[1]
  for el in node[1]:
    graph[graph_list.index(node[0])][graph_list.index(el)] = 1
graph = np.array(graph)

speed = []
next(speed_raw)
for row in speed_raw:
  speed.append([int(el) for el in row[0].split(',')[1:-1]]) 
speed = np.array(speed).T
#speed = np.swapaxes(np.asarray(speed),0,1)
#graph = graph.tolist()
#print "GRAPH",graph
# Some preprocessing
#features = preprocess_features(np.asarray(inputs))
if FLAGS.model == 'gcn':
    support = [preprocess_adj(graph)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(graph, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    #'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'support': [tf.placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.placeholder(tf.float32, shape=(FLAGS.batch_size, speed.shape[1])),
    'labels': tf.placeholder(tf.float32, shape=(FLAGS.batch_size, speed.shape[1])),
    #'labels': tf.placeholder(tf.float32, shape=onehot.shape[2:0:-1]),
    'dropout': tf.placeholder_with_default(0., shape=())      
}

#Create Model
model = model_func(placeholders, input_dim=(FLAGS.batch_size, speed.shape[1]), hiddenUnits=hiddenUnits, logging=False)

# Initialize session
sess = tf.Session()

# Define model evaluation function
def evaluate(features, support, labels):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels,placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []
# Train model
number_of_runs = int(math.floor(len(speed[1])/FLAGS.batch_size)) -1
print number_of_runs
for epoch in xrange(FLAGS.epoch):
  for batch_position in xrange(0, number_of_runs):
      t = time.time()
      
      #batch = speed[batch_position*FLAGS.batch_size:(batch_position+1)*FLAGS.batch_size]
      batch = []
      for i in xrange(FLAGS.batch_size*2):
        batch.append(speed[batch_position*FLAGS.batch_size + i])
      
      # Construct feed dictionary
      feed_dict = construct_feed_dict(batch[:FLAGS.batch_size], support, batch[FLAGS.batch_size:], placeholders)
      feed_dict.update({placeholders['dropout']: FLAGS.dropout})

      # Training step
      outs = sess.run([model.opt_op, model.loss, model.accuracy, model.outputs, model.cross], feed_dict=feed_dict)
      #print "OUTS", outs

      # Validation
      cost, acc, duration = evaluate(batch[:FLAGS.batch_size], support, batch[FLAGS.batch_size:])
      cost_val.append(cost)
      
      # Print results
      if batch_position%FLAGS.print_interval==0:
        print "Epoch:", '%03d' % (epoch + 1),\
              "BatchPos:", batch_position,\
              "train_loss=%.5f"%sum(outs[1]),\
              "train_acc=%.5f"%outs[2],\
              "loss_diff=%.5f"%(sum(outs[1])-sum(cost)),\
              "acc_diff=%.5f" %(outs[2]-acc),\
              "time=%.5f" %(time.time() - t),\
            "\nOutputs", outs[3][0][0:5],\
            "\nLabels",batch[FLAGS.batch_size:][0][0:5],\
            "\nInput",batch[:FLAGS.batch_size][0][0:5],\
            "\nCross",outs[4][0:5], "\n"
      
        record.append(''.join(map(str,
            ("Epoch: " , '%d' % (epoch + 1),
            " BatchPos: ", batch_position,
            " train_loss: %.5f"%sum(outs[1]),
           "\nOutputs: ", outs[3][0][0:20],
           "\nLabels",batch[FLAGS.batch_size:][0][0:20],
           "\nInput", batch[:FLAGS.batch_size][0][0:20],
           "\nCross",outs[4][0:5],"\n\n"))))
      """
      if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
          print "Early stopping..."
          break
      """

# Testing
batch = speed[-2*FLAGS.batch_size:]

#for i in xrange(FLAGS.batch_size*2):
#  batch.append(speed[-i])
 
test_cost, test_acc, test_duration = evaluate(batch[:FLAGS.batch_size], support, batch[FLAGS.batch_size:])
print "Test set results:", "cost=", test_cost, "accuracy=", test_acc, "time=", test_duration
record.append(''.join(map(str, 
      ("Test set results:", "cost=", test_cost,
      "accuracy=", test_acc, "time=", test_duration))))
file = open("output.txt", "w+b")
for el in record:
  print>>file, el
