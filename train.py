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
# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

class TrafficPrediction(object):
  def __init__(self):
    #Clear File
    open("output.txt", 'w').close() 
    """""""""""""""""""""
    Hyperparameters
    """""""""""""""""""""

    self.flags = tf.app.flags
    FLAGS = self.flags.FLAGS
    #flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby',
    self.flags.DEFINE_float('learning_rate', 0, 'Initial learning rate.')
    self.flags.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probability).')
    self.flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')
    self.flags.DEFINE_integer('number_of_features', 0, 'Number of features for graph convolution')
    self.flags.DEFINE_integer('number_of_hidden_layers',0, 'Number of Hidden Layers for the graph convolution.')
    self.flags.DEFINE_integer('batch_size', 0, 'Number of timesteps to feed each time.')
    self.flags.DEFINE_integer('early_stopping',1 , 'NOT YET IMPLEMENTED Tolerance for early stopping (# of epochs).')
    self.flags.DEFINE_integer('print_interval', 1, 'Number of runs per print.')
    self.flags.DEFINE_integer('epoch',2 , 'Number of epochs.')
    self.flags.DEFINE_integer('amount_of_testing_data', 1, 'NOT YET IMPLEMENTED Amount of testing data for validationa')

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
    #Time*Node
    self.speed = np.array(speed).T

    
    del_count = 0
    del_index = []
    self.data_time = []
    for month in xrange(3,4):
      week = 0
      for day in xrange(31):
        day_of_week = (day+2)%7
        if day_of_week != 6 and day_of_week != 0:
          day_onehot        = [0]*5
          day_onehot[day_of_week-1] = 1 
          if day_of_week==0: week += 1
          for hour in xrange(0,24):
            for min in xrange(0,60,5):
              hour_onehot               = [0]*24
              hour_onehot[hour]         = hour - min/60
              hour_onehot[(hour+1)%24 ] = min/60
              self.data_time.append([week] + day_onehot + hour_onehot)
              del_count += 1
        else:
          del_index = del_index + range(int(day*24*60/5), int((day+1)*24*60/5))
    week = 0 
    for day in xrange(19):
      #Skip Ching Ming Holiday
      if day != 3:
        day_of_week = (day+5)%7   
        if day_of_week != 6 and day_of_week != 0:
          day_onehot        = [0]*5
          day_onehot[day_of_week-1] = 1 
          if day_of_week==0: week += 1
          for hour in xrange(0,24):
            for min in xrange(0,60,5):
              hour_onehot               = [0]*24
              hour_onehot[hour]         = hour - min/60
              hour_onehot[(hour+1)%24 ] = min/60
              self.data_time.append([week] + day_onehot + hour_onehot)
        else:
          del_index = del_index + range(int((31+day)*24*60/5), int((31+day+1)*24*60/5))

      else:
        del_index = del_index + range(int((31+day)*24*60/5),int((31+day+1)*24*60/5))

    self.speed = np.delete(self.speed, del_index,axis=0) 
    day_onehot       = [0]*5
    day_onehot[(21+5)%7-1] = 1 
    week             = 21 // 7 
    for hour in xrange(0,8):
      for min in xrange(0,60,5):
        hour_onehot               = [0]*24
        hour_onehot[hour]         = hour - min/60
        hour_onehot[(hour+1)%24 ] = min/60
        self.data_time.append([week] + day_onehot + hour_onehot)
    
    hour_onehot    = [0]*24
    hour_onehot[8] = 1
    self.data_time.append([week] + day_onehot + hour_onehot)


    self.support      = [preprocess_adj(self.graph)]
    self.num_supports = 1
    self.model_func   = GCN

  def __call__(self, hyperparameters, hiddenUnits):
    self.FLAGS = self.flags.FLAGS
    FLAGS=self.flags.FLAGS #I have no idea how flags work.
    FLAGS.learning_rate = hyperparameters['learning_rate']
    FLAGS.dropout = hyperparameters['dropout']
    FLAGS.weight_decay = hyperparameters['weight_decay']
    FLAGS.number_of_features = hyperparameters['number_of_features']
    FLAGS.number_of_hidden_layers = hyperparameters['number_of_hidden_layers']
    FLAGS.batch_size = hyperparameters['batch_size']
    FLAGS.print_interval = hyperparameters['print_interval']
    FLAGS.epoch = hyperparameters['epoch']
    FLAGS.amount_of_testing_data = hyperparameters['amount_of_testing_data']
    
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
    number_of_runs = int(math.floor((len(self.speed)-FLAGS.amount_of_testing_data)/FLAGS.batch_size))
    for epoch in xrange(FLAGS.epoch):
      for batch_position in xrange(0, number_of_runs-1):
          t = time.time()
          
          #batch = speed[batch_position*FLAGS.batch_size:(batch_position+1)*FLAGS.batch_size]
          #B*N
          speed_batch = []
          time_batch = []
          speed_batch = self.speed[batch_position*FLAGS.batch_size:(batch_position+2)*FLAGS.batch_size]
          time_batch = self.data_time[batch_position*FLAGS.batch_size:(batch_position+1)*FLAGS.batch_size] 
          
          # Construct feed dictionary

          feed_dict = construct_feed_dict(speed_batch[:FLAGS.batch_size], time_batch, self.support, speed_batch[FLAGS.batch_size:], placeholders)
          feed_dict.update({placeholders['dropout']: FLAGS.dropout})

          # Training step
          outs = sess.run([model.opt_op, model.loss, model.accuracy, model.outputs, model.cross], feed_dict=feed_dict)

          #print "TEST", sess.run([model.outputs, model.placeholders['labels']], feed_dict=feed_dict)
          # Validation
          cost, acc, duration = evaluate(speed_batch[:FLAGS.batch_size], time_batch, self.support, speed_batch[FLAGS.batch_size:])
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
                  "Trained Data:", batch_position*FLAGS.batch_size,\
                  "train_loss=%.5f"%outs[1],\
                  "train_acc=%.5f"%outs[2],\
                  "loss_diff=%.5f"%(outs[1]-cost),\
                  "acc_diff=%.5f" %(outs[2]-acc),\
                  "time=%.5f" %(time.time() - t),\
                  "\nOutputs", outs[3][0][0:20],\
                  "\nLabels",speed_batch[1][0:20],\
                  "\nInput",speed_batch[0][0:20],"\n\n"
          
          
          """
          if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
              print "Early stopping..."
              break
          """

    test_cost = []
    test_acc = []
    test_batch_size = int(math.ceil(FLAGS.amount_of_testing_data/FLAGS.batch_size))
    for i in xrange(test_batch_size+1, 1,-1):
      speed_batch = self.speed[-FLAGS.batch_size*(i+1):-FLAGS.batch_size*(i-1)]
      time_batch = self.data_time[-FLAGS.batch_size*(i+1):-FLAGS.batch_size*i]
      #for i in xrange(FLAGS.batch_size*2):
      #  speed_batch.append(speed[-i])
      out = evaluate(speed_batch[:FLAGS.batch_size], time_batch, self.support, speed_batch[FLAGS.batch_size:])
      test_cost.append(out[0])
      test_acc.append(out[1])
      print "Test set results:", "cost=",out[0], "accuracy=", out[1], "time=", out[2]
    print "Summary: cost=", sum(test_cost)/len(test_cost), "accuracy=", sum(test_acc)/len(test_acc) 
    
            
    #Return loss for GA
    return sum(test_cost)/len(test_cost)

if __name__ == "__main__":
  """""""""""""""""""""
  Hyperparameters
  """""""""""""""""""""

  hyperparameters = { 'learning_rate': 5e-6, 
                      'dropout': 0.1,
                      'weight_decay': 1,
                      'number_of_features': 1,
                      'number_of_hidden_layers': 3,    
                      'batch_size': 20,
                      'early_stopping': 10,
                      'print_interval': 20,
                      'epoch': 10,
                      'amount_of_testing_data': 30 }
  hiddenUnits = [128, 128, 128]

      
  traffic_prediction = TrafficPrediction()
  traffic_prediction(hyperparameters, hiddenUnits)
