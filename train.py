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
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 1e-5, 'Initial learning rate.')
flags.DEFINE_integer('hidden1', 1024, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.01, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 1e-5, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('number_of_features', 10, 'Number of features for the graph convolution layer.')
flags.DEFINE_integer('batch_size', 5, 'Number of timesteps to feed each time.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('print_interval', 50, 'Number of runs per print.')
flags.DEFINE_integer('epoch', 1, 'Number of epoch.')
#adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

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
    print "SUPPORT"
    print support
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(graph, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
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
model = model_func(placeholders, input_dim=(FLAGS.batch_size, speed.shape[1]), logging=False)

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
#for epoch in range(FLAGS.epochs
for epoch in xrange(FLAGS.epoch):
  for batch_position in xrange(1, int(math.floor(len(speed[1]/FLAGS.batch_size)))):
      t = time.time()
      batch = []
      for i in xrange(FLAGS.batch_size*2):
        batch.append(speed[epoch*FLAGS.batch_size + i])
      
      # Construct feed dictionary
      feed_dict = construct_feed_dict(batch[:FLAGS.batch_size], support, batch[FLAGS.batch_size:], placeholders)
      #print '\n\n\nFEED', feed_dict,'\n\n\n\n\n\n'
      feed_dict.update({placeholders['dropout']: FLAGS.dropout})

      # Training step
      #sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
      outs = sess.run([model.opt_op, model.loss, model.accuracy, model.outputs, model.cross], feed_dict=feed_dict)
      #print "OUTS", outs

      # Validation
      cost, acc, duration = evaluate(batch[:FLAGS.batch_size], support, batch[FLAGS.batch_size:])
      cost_val.append(cost)
      
      # Print results
      if batch_position%FLAGS.print_interval==0:
        print "Epoch:", '%03d' % (epoch + 1), "BatchPos:", batch_position, "train_loss=%.5f"%sum(outs[1]),\
              "train_acc=%.5f"%outs[2], "loss_diff=%.5f"%(sum(outs[1])-sum(cost)),\
              "acc_diff=%.5f" %(outs[2]-acc), "time=%.5f" %(time.time() - t)
        #print "Outputs", outs[3],"\nLabels",batch[FLAGS.batch_size:][0:5],"\nInput",batch[:FLAGS.batch_size][0:5],"\nCross",outs[4][0:5]
        print "Outputs", outs[3][0][0:5],"\nLabels",batch[FLAGS.batch_size:][0][0:5],"\nInput",batch[:FLAGS.batch_size][0][0:5],"\nCross",outs[4][0:5]
      
        record.append(str("Epoch:" +  str('%03d' % (epoch + 1)) + "BatchPos:", str(batch_position), str("train_loss=%.5f"%sum(outs[1])) + "Outputs" +  str(outs[3][0][0:5]) +"\nLabels",str(batch[FLAGS.batch_size:][0][0:5])+"\nInput"+str(batch[:FLAGS.batch_size][0][0:5])+ "\nCross",str(outs[4][0:5])))
      """
      if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
          print "Early stopping..."
          break
      """
  print "Optimization Finished!"
"""
# Testing
test_cost, test_acc, test_duration = evaluate(inputs[-2], support, onehot[-2])
print "Test set results:", "cost=", test_cost,\
      "accuracy=", test_acc, "time=", test_duration
"""
cpk.dump(record, load("output.cpk", "rb"))
