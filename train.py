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
#Record array
record = []
# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

"""""""""""""""""""""
Hyperparameters
"""""""""""""""""""""
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby',
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.01, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 1e-1, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('number_of_features', 20, 'Number of features for graph convolution')
flags.DEFINE_integer('number_of_layers', 3, 'Number of Layers for the graph convolution.')
flags.DEFINE_integer('batch_size', 5, 'Number of timesteps to feed each time.')
flags.DEFINE_integer('early_stopping', 10, 'NOT YET IMPLEMENTED Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('print_interval', 10, 'Number of runs per print.')
flags.DEFINE_integer('epoch',10 , 'Number of epochs.')
flags.DEFINE_integer('amount_of_testing_data', 20, 'NOT YET IMPLEMENTED Amount of testing data for validationa')
hiddenUnits = [64, 64]

class TrafficPrediction(object):
  def __init__(self):
    """""""""""""""""""""
    Data Preprocessing
    """""""""""""""""""""
    csv.field_size_limit(sys.maxsize)

    ##Input data
    graph_raw  = cpk.load(open('data/graph.cpk', 'rb'))
    speed_raw  = csv.reader(open('data/speeds.csv','rb'),delimiter='\n')
    graph_list = cpk.load(open('data/graphlist.cpk','rb'))

    #adjecency matrix
    graph = [[ 0 for i in range(len(graph_list))] for j in range(len(graph_list))]

    for node in graph_raw:
      #print "NODE1",node[1]
      for el in node[1]:
        graph[graph_list.index(node[0])][graph_list.index(el)] = 1
    self.graph = np.array(graph)

    speed = []
    next(speed_raw)
    for row in speed_raw:
      speed.append([int(el) for el in row[0].split(',')[1:-1]]) 
    self.speed = np.array(speed).T

    self.data_time = []
    for month in xrange(3,4):
      for day in xrange(1,32):
        day_onehot        = [0]*7
        day_onehot[day//7] = 1 
        week              = day % 7
        for hour in xrange(0,24):
          for min in xrange(0,60,5):
            hour_onehot               = [0]*24
            hour_onehot[hour]         = hour - min/60
            hour_onehot[(hour+1)%24 ] = min/60
            self.data_time.append([week] + day_onehot + hour_onehot)
     
    for day in xrange(1,20):
      day_onehot        = [0]*7
      day_onehot[day//7] = 1 
      week              = day % 7
      for hour in xrange(0,24):
        for min in xrange(0,60,5):
          hour_onehot               = [0]*24
          hour_onehot[hour]         = hour - min/60
          hour_onehot[(hour+1)%24 ] = min/60
          self.data_time.append([week] + day_onehot + hour_onehot)

    day_onehot       = [0]*7
    day_onehot[21//7] = 1 
    week             = 21 % 7 
    for hour in xrange(0,8):
      for min in xrange(0,60,5):
        hour_onehot               = [0]*24
        hour_onehot[hour]         = hour - min/60
        hour_onehot[(hour+1)%24 ] = min/60
        self.data_time.append([week] + day_onehot + hour_onehot)

    hour_onehot    = [0]*24
    hour_onehot[8] = 1
    self.data_time.append([week] + day_onehot + hour_onehot)


    if FLAGS.model   == 'gcn':
        self.support      = [preprocess_adj(self.graph)]
        self.num_supports = 1
        self.model_func   = GCN
    elif FLAGS.model == 'gcn_cheby':
        self.support      = chebyshev_polynomials(self.graph, FLAGS.max_degree)
        self.num_supports = 1 + FLAGS.max_degree
        self.model_func   = GCN
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

  def __call__(self, FLAGS):
    # Define placeholders
    placeholders = {
        'support' : [tf.placeholder(tf.float32)],
        'GC_features': tf.placeholder(tf.float32, shape=(FLAGS.batch_size, self.speed.shape[1])),
        'FC_features': tf.placeholder(tf.float32, shape=(FLAGS.batch_size, len(self.data_time[0]))),
        'labels'  : tf.placeholder(tf.float32, shape=(FLAGS.batch_size, self.speed.shape[1])),
        'dropout' : tf.placeholder_with_default(0., shape=())      
    }

    #Create Model
    model = self.model_func(placeholders,
                       input_dim      = (FLAGS.batch_size, self.speed.shape[1]),
                       time_input_dim = (FLAGS.batch_size, len(self.data_time[0])),
                       hiddenUnits    = hiddenUnits,
                       logging        = False)

    # Initialize session
    sess = tf.Session()

    # Define model evaluation function
    def evaluate(features, support, time_features, labels):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(features, support, time_features, labels, placeholders)
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], (time.time() - t_test)


    # Init variables
    sess.run(tf.global_variables_initializer())

    cost_val = []

    # Train model
    number_of_runs = int(math.floor(len(self.speed[1])/FLAGS.batch_size)) - 1
    for epoch in xrange(FLAGS.epoch):
      for batch_position in xrange(0, number_of_runs):
          t = time.time()
          
          #batch = speed[batch_position*FLAGS.batch_size:(batch_position+1)*FLAGS.batch_size]
          #B*N
          speed_batch = []
          time_batch = []
          for i in xrange(FLAGS.batch_size + 1):
            speed_batch.append(self.speed[batch_position*FLAGS.batch_size + i])
          for i in xrange(FLAGS.batch_size):
            time_batch.append(self.data_time[batch_position*FLAGS.batch_size + i])
          
          # Construct feed dictionary
          feed_dict = construct_feed_dict(speed_batch[:-1], time_batch, self.support, speed_batch[1:], placeholders)
          feed_dict.update({placeholders['dropout']: FLAGS.dropout})

          # Training step
          outs = sess.run([model.opt_op, model.loss, model.accuracy, model.outputs, model.cross], feed_dict=feed_dict)

          #print "TEST", sess.run([model.outputs, model.placeholders['labels']], feed_dict=feed_dict)
          # Validation
          cost, acc, duration = evaluate(speed_batch[:-1], time_batch, self.support, speed_batch[1:])
          cost_val.append(cost)
          """ 
          # Print results
          if batch_position%FLAGS.print_interval==0:
            print "Epoch:", '%03d' % (epoch + 1),\
                  "BatchPos:", batch_position,\
                  "train_acc=%.5f"%outs[2],\
                  "acc_diff=%.5f" %(outs[2]-acc),\
                  "time=%.5f" %(time.time() - t),\
                  "\nOutputs", outs[3],\
                "\nLabels",speed_batch[1],\
                "\nInput",speed_batch[0][0:10],\
                "\nCross",outs[4], "\n"
            print "DIFF", outs[-1] 
            record.append(''.join(map(str,
                ("Epoch: " , '%d' % (epoch + 1),
                " BatchPos: ", batch_position,
               "\nOutputs: ", outs[3][0][0:20],
               "\nLabels",speed_batch[1][0:20],
               "\nInput", speed_batch[0][0:20],
               "\nCross",outs[4],"\n\n"))))
          """
          if batch_position%FLAGS.print_interval==0:
            print "Epoch:", '%03d' % (epoch + 1),\
                  "BatchPos:", batch_position,\
                  "train_loss=%.5f"%outs[1],\
                  "train_acc=%.5f"%outs[2],\
                  "loss_diff=%.5f"%(outs[1]-cost),\
                  "acc_diff=%.5f" %(outs[2]-acc),\
                  "time=%.5f" %(time.time() - t),\
                  "\nOutputs", outs[3][0][0:20],\
                  "\nLabels",speed_batch[1][0:20],\
                  "\nInput",speed_batch[0][0:20],"\n\n"
          
            record.append(''.join(map(str,
                ("Epoch: " , '%d' % (epoch + 1),
                " BatchPos: ", batch_position,
                " train_loss: %.5f"%outs[1],
               "\nOutputs: ", outs[3][0][0:20],
               "\nLabels",speed_batch[1][0:20],
               "\nInput", speed_batch[0][0:20],
               "\nCross",outs[4],"\n\n"))))
          """
          if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
              print "Early stopping..."
              break
          """

    # Testing
    speed_batch = self.speed[-2*FLAGS.batch_size:]
    time_batch = self.data_time[-2*FLAGS.batch_size:-FLAGS.batch_size]
    #for i in xrange(FLAGS.batch_size*2):
    #  speed_batch.append(speed[-i])
     
    test_cost, test_acc, test_duration = evaluate(speed_batch[:FLAGS.batch_size], time_batch, support, speed_batch[FLAGS.batch_size:])
    print "Test set results:", "cost=", test_cost, "accuracy=", test_acc, "time=", test_duration
    record.append(''.join(map(str, 
          ("Test set results:", "cost=", test_cost,
          "accuracy=", test_acc, "time=", test_duration))))
    file = open("output.txt", "w+b")
    for el in record:
      print>>file, el

    return ''.join(map(str,("Test set results:", "cost=", test_cost, "accuracy=", test_acc, "time=", test_duration))) 


if __name__ == "__main__":
      traffic_prediction = TrafficPrediction()
      traffic_prediction(FLAGS)
