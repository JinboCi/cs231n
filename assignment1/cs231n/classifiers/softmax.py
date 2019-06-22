import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  import math
  num_classes = W.shape[1]
  num_train = X.shape[0]
  # score = X.dot(W)
  for i in range(num_train):
      scores = X[i].dot(W)
      fyi = scores[y[i]]
      sum_efj = 0
      
      for j in range(num_classes):
            if j == y[i]:
                dW[:,j] -= X[i,:]
                
            sum_efj += math.exp(X[i].dot(W[:,j]))
            dW[:,j] += X[i,:]*np.exp(X[i].dot(W[:,j]))/np.sum(np.exp(X[i].dot(W)))
            # dW[:,j] += (X[i]*np.exp(X[i].dot(W[:,j]))/math.log(np.sum(np.exp(X[i].dot(W))))) / num_train 
            # dW[:, y[i]] -= X[i] / num_train 
      loss = loss - fyi + math.log(sum_efj)
    
                  
  loss = loss / num_train
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW += reg* W   



  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)
  exponential = np.exp(scores)
  train_indexes = np.arange(num_train)
  fyi = scores[train_indexes, y]
  loss += -np.sum(fyi) + np.sum(np.log(np.sum(exponential,axis = 1)))
  loss /= num_train
  loss += reg * np.sum(W * W)

  # import pdb
  # pdb.set_trace()  
  judge = np.zeros_like(scores)
  judge[train_indexes, y] = 1
  dfyi =  X.T.dot(judge)
  dexponential = X.T.dot(np.exp(scores)/(np.sum(np.exp(scores), axis =1).reshape(num_train,1)))
  dW += - dfyi + dexponential
  dW /= num_train
  dW += reg* W



  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

