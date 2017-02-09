from __future__ import division
#from __future__ import print_function

import time
import tensorflow as tf

from utils import *
from models import GCN, MLP

import sys
import cPickle as cpk
import csv
csv.field_size_limit(sys.maxsize)

##Input data
graph_raw = cpk.load(open('data/graph.cpk', 'rb'))
speed_raw = csv.reader(open('data/speeds.csv','rb'),delimiter='\n')
graph_list = cpk.load(open('data/graphlist.cpk','rb'))

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 5e-5,'Initial learning rate.')
flags.DEFINE_integer('hidden1', 1024, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 1e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
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
  speed.append([int(el) for el in row[0].split(',')[1:-1]]) #HACK:[1:] -> [2:]

speed = np.asarray(speed).T
onehot = np.zeros(speed.shape+(4,))

#Time*Number of Node*Onehot Label
for i in xrange(len(speed)):
  for j in xrange(len(speed[i])):
    onehot[i,j,speed[i,j]-1]=1

#onehot[np.arange(speed.shape[0]),np.arange(speed.shape[1]), speed] = 1

data_time = []
for month in xrange(3,4):
  for day in xrange(1,32):
    for hour in xrange(0,24):
      for min in xrange(0,60,5):
        data_time.append([month,day,hour,min])
  
for day in xrange(1,20):
  for hour in xrange(0,24):
    for min in xrange(0,60,5):
      data_time.append([4,day,hour,min])
    
for hour in xrange(0,8):
  for min in xrange(0,60,5):
    data_time.append([4,20,hour,min])
data_time.append([4,20,8,0])

#Time*Number of Node
inputs = np.swapaxes([data_time]*onehot.shape[1],0,1)
print inputs
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
    'features': tf.placeholder(tf.float32, shape=inputs.shape[1:]),
    'labels': tf.placeholder(tf.float32, shape=onehot.shape[1:]),
    #'labels': tf.placeholder(tf.float32, shape=onehot.shape[2:0:-1]),
    'dropout': tf.placeholder_with_default(0., shape=())
}

#Create Model
model = model_func(placeholders, input_dim=inputs.shape[1:], logging=False)

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
print "SHAPE"
print inputs[1].T
print onehot[1].T
print placeholders

# Train model
#for epoch in range(FLAGS.epochs)
print "TEST",onehot[1]
for epoch in range(inputs.shape[0]-1):
    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(inputs[epoch], support, onehot[epoch], placeholders)
    #print '\n\n\nFEED', feed_dict,'\n\n\n\n\n\n'
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    #sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
    outs = sess.run([model.opt_op, model.loss, model.accuracy, model.outputs], feed_dict=feed_dict)
    #print "OUTS", outs

    # Validation
    cost, acc, duration = evaluate(inputs[epoch], support, onehot[epoch])
    cost_val.append(cost)
    
    # Print results
    if epoch%50==0:
      print "Epoch:", '%04d' % (epoch + 1), "train_loss=",sum(outs[1]),\
            "train_acc=%.8f"%outs[2], "val_loss=",sum(cost),\
            "val_acc=%.8f" %acc, "time=%.5f" %(time.time() - t)
      print "Outputs", outs[3],"\nLabels",onehot[epoch],"\nInput",inputs[epoch]
    
    """
    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print "Early stopping..."
        break
    """
print "Optimization Finished!"

# Testing
test_cost, test_acc, test_duration = evaluate(inputs[-2], support, onehot[-d])
print "Test set results:", "cost=", test_cost,\
      "accuracy=", test_acc, "time=", test_duration
