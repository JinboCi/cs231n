import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i]/num_train
        dW[:,y[i]] -= X[i] /num_train
        

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  WjX_matrix = X.dot(W)
  WyiX_matrix = WjX_matrix[np.arange(num_train).reshape(num_train,1), y.reshape(num_train,1)]*np.ones((num_train, num_classes))
  delta_matrix = np.ones((num_train, num_classes))
  after_max_matrix = np.maximum(np.zeros((num_train, num_classes)), WjX_matrix - WyiX_matrix + delta_matrix)
  after_max_matrix[np.arange(num_train).reshape(num_train,1), y.reshape(num_train,1)] = 0
  # after_max_matrix_90 = np.rot90(after_max_matrix)
  # loss_2D_matrix = sum(after_max_matrix_90)
  loss = np.sum(after_max_matrix)
  loss /= num_train
  loss += reg * np.sum(W * W)

  
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # import pdb
  # pdb.set_trace()
  # basic_2d_01_matrix = (after_max_matrix > 0) * 1
  # basic_3d_Xi_matrix = np.repeat(X[ :, np.newaxis,:], num_classes, axis=1)
  # basic_3d_01_matrix = np.repeat(basic_2d_01_matrix[ :, :, np.newaxis], X.shape[1], axis=2)
  # Xi_matrix = basic_3d_Xi_matrix * basic_3d_01_matrix
  # Wj_matrix = np.rot90(np.sum(Xi_matrix, axis = 0))
  # # Xi_matrix_90 = np.rot90(Xi_matrix)
  # basic_2d_Wyi_matrix = np.sum(Xi_matrix, axis = 1)
  # # basic_2d_Wyi_matrix = np.rot90(np.rot90(np.rot90(basic_2d_Wyi_matrix_90)))
  # basic_3d_Wyi_matrix = np.repeat(basic_2d_Wyi_matrix[ :, np.newaxis,:], num_classes, axis=1)
  # basic_2d_arange_Wyi_matrix = np.arange(num_classes)
  # final_2d_arange_Wyi_matrix = np.repeat(basic_2d_arange_Wyi_matrix[np.newaxis, :], num_train, axis=0)
  # basic_2d_repeat_index_yi = np.repeat(y[ :, np.newaxis], num_classes, axis=1)
  # final_2d_01_matrix_yi = (basic_2d_repeat_index_yi == final_2d_arange_Wyi_matrix) *1
  # final_3d_01_matrix_yi = np.repeat(final_2d_01_matrix_yi[ :, :, np.newaxis], X.shape[1], axis=2)
  # final_3d_Wyi_matrix = basic_3d_Wyi_matrix * final_3d_01_matrix_yi
  # Wyi_matrix_270 = np.sum(final_3d_Wyi_matrix, axis = 0)
  # Wyi_matrix = np.rot90(Wyi_matrix_270)
  # dW = dW + (- Wyi_matrix + Wj_matrix) / num_train
  # dW = dW[::-1] + reg*W
  coeff_mat = np.zeros((num_train, num_classes))
  coeff_mat[after_max_matrix > 0] = 1
  coeff_mat[range(num_train), list(y)] = 0
  coeff_mat[range(num_train), list(y)] = -np.sum(coeff_mat, axis=1)

  dW = (X.T).dot(coeff_mat)
  dW = dW/num_train + reg*W
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
