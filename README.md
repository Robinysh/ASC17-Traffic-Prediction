 Prediction question for ASC17

##Ideas###
* LSTM RNN Network at each node with temporal and spatial state:

    A. One LSTM RNN at each node, another LSTM RNN through all nodes, with activator function/NN to combine two outputs (Working On)

    B. Rewrite Tensorflow LSTM Implementaion
        
    - New LSTM Cell with two states (Details to be planned later)
    - Edit source code from tensorflow github:
    <https://github.com/tensorflow/tensorflow/blob/97f585d506cccc57dc98f234f4d5fcd824dd3c03/tensorflow/python/ops/rnn_cell.py#L353>


* LSTM RNN Network at each edge: 
    
    - Pass output of edges to another RNN for node output


* LSTM RNN Network at each node with only temporal state:

* Shared LSTM RNN Network for every node
  
    A. Shared state between nodes, input all node data as a vector
  
    B. Distinct state between nodes, input node data one by one
  
* Graph Convolution?

* Recursive Neural Network (TreeNet)
