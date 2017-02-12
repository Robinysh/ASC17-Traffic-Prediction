import numpy as np

from layers import *
from metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        """
        hidden = [0]*self.number_of_features
        for i, layer in enumerate(self.layers[:-1]):
          hidden[i] = layer(self.activations[-1])
        """
        """
        for i in xrange(len(self.layers)):
          if self.layers[i].parallel:
            outputs=[]*self.layers[i].parallel_num
            for l in xrange(self.layers[i].parallel_num):
              outputs[l]=self.layers[i+l](self.activations[-1])
            i+=self.layers[i],parallel_num
            self.activation.append(outputs)
          else:  
            #self.activations.append(hidden)
            self.activations.append(self.layers[i](self.activations[-1]))  
        """
        #Unpack Batch inputs 
        self.inputs = tf.unstack(self.inputs, axis=0)
        for input in self.inputs: 
          # Build sequential layer model
          self.activations.append([])
          #Convert to vector: shape(N, 1)
          self.activations[-1].append(tf.expand_dims(input,1))
          for layer in self.layers:
              #If nested (i.e. GC)
              if isinstance(layer, list):
                outputs = []
                for sublayer in layer:
                  #Get last output of last trained batch
                  outputs.append(sublayer(self.activations[-1][-1]))
                #Output: N X F
                self.activations[-1].append(tf.concat(1, outputs))

              #If not nested (i.e. FC)
              else:
                output = layer(self.activations[-1][-1])
                self.activations[-1].append(output)
        #Output: B X N
        self.outputs = tf.transpose(tf.concat(1, np.array(self.activations)[:,-1].tolist()))
        
        #print "ACT",self.activations
        """
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        """
        #self.outputs = np.array(self.activations)[:,-1]
        
        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)
        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.number_of_features = FLAGS.number_of_features
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay losis
        
        for layer in self.layers:
            #If nested (i.e. GC)
            if isinstance(layer, list):
                continue
                #for sublayer in layer: 
                   #self.loss += FLAGS.weight_decay * tf.nn.l2_loss(sublayer.vars.values())
            #else:
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(layer.vars.values())
        """
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        """
        # Cross entropy error
        self.cross = tf.nn.softmax_cross_entropy_with_logits(tf.transpose(self.outputs), tf.transpose(self.placeholders['labels']))
        self.loss += tf.nn.softmax_cross_entropy_with_logits(tf.transpose(self.outputs), tf.transpose(self.placeholders['labels']))
        #print "SOFTMAX", tf.nn.softmax_cross_entropy_with_logits(self.outputs, self.placeholders['labels'])
    def _accuracy(self):
        correct_prediction = tf.equal(tf.round(self.outputs), self.placeholders['labels'])
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def _build(self):
        print "INOUT" , self.input_dim[1]
        self.layers.append([GraphConvolution(input_dim=self.input_dim[1],
                                             output_dim=self.input_dim[1],
                                             placeholders=self.placeholders,
                                             act=tf.nn.relu,
                                             dropout=True,
                                             bias=True,
                                             logging=self.logging,
                                             parallel=True,
                                             parallel_num=FLAGS.number_of_features) for _ in xrange(FLAGS.number_of_features)])
        self.layers.append(FullyConnected(input_dim=(self.input_dim[1], self.number_of_features),
                                        number_of_features=self.number_of_features,
                                        output_dim=self.output_dim[1],
                                        placeholders=self.placeholders,
                                        act=lambda x: x,
                                        dropout=True,
                                        bias=True,
                                        logging=self.logging))
        """ 
        for _ in FLAG.number_of_features:
          self.layers.append(GraphConvolution(input_dim=self.input_dim[1],
                                                output_dim=self.input_dim[1],
                                                placeholders=self.placeholders,
                                                act=tf.nn.relu,
                                                dropout=True,
                                                bias=True,
                                                logging=self.logging,
                                                parallel=True,
                                                parallel_num=FLAGS.number_of_features))

        self.layers.append(FullyConnected(input_dim=(self.number_of_features,)+ self.input_dim[1],
                                        number_of_features=self.number_of_features,
                                        output_dim=self.output_dim,
                                        placeholders=self.placeholders,
                                        act=lambda x: x,
                                        dropout=True,
                                        bias=True,
                                        logging=self.logging))
      
        """   
    def predict(self):
        return tf.nn.softmax(self.outputs)
