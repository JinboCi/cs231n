import numpy as np

class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
        for j in range(num_train):
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################
            dists[i,j] = (sum((self.X_train[j]-X[i])**2))**0.5
        pass
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################
      diff = self.X_train - X[i]
      diff2 = diff**2
      rotation = np.rot90(diff2)
      broken_array_sum_array = sum(rotation)
      
      
      # broken_array_sum_map = [sum(l) for l in diff2]
      # broken_array_sum_list = list(broken_array_sum_map)
      # broken_array_sum_array = np.array(broken_array_sum_list)
      dists[i,:] = broken_array_sum_array**0.5
      
      pass
      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    # import pdb
    # pdb.set_trace()
    X_train_90 = np.rot90(self.X_train)
    X_90 = np.rot90(X)
    
    
    sum_test_train = np.dot(X, X_train_90[::-1])
    test_square_90 = (X_90)**2
    sum_test_square_90 = sum(test_square_90)*np.ones((num_train, num_test))
    sum_test_square_array_90 = sum_test_square_90
    sum_test_square_array_180 = np.rot90(sum_test_square_array_90)
    sum_test_square_array_270 = np.rot90(sum_test_square_array_180)
    sum_test_square_array = np.rot90(sum_test_square_array_270)
    train_square_90 = (X_train_90)**2
    # sum_train_square_90 = sum(train_square_90)
    # sum_train_square_array_90 = sum_train_square_90
    # sum_train_square_array_180 = np.rot90(sum_train_square_array_90)
    # sum_train_square_array_270 = np.rot90(sum_train_square_array_180)
    # sum_train_square_array_360 = np.rot90(sum_train_square_array_270)
    
    sum_train_square_array_90 = sum(train_square_90)
    sum_train_square_array = np.ones((num_test, num_train))*sum_train_square_array_90
    dists = (sum_train_square_array + sum_test_square_array - 2*sum_test_train)**0.5
    pass
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.y_train to find the labels of these       #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################
      index_sort = np.argsort(dists[i])
      k_nearest = index_sort[0:k]
      # import pdb
      # pdb.set_trace()
      
      
      closest_y = self.y_train[k_nearest]
      pass
      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
      count = {}
      for n in closest_y:
          count[n] = 0
      for j in closest_y:
          count[j] = count[j] + 1
      max_num = 0
      for l in count:
          if count[l] > max_num:
              y_pred[i] = l
              max_num = count[l]
              previous = l
          elif count[l] == max_num:
              y_pred[i] = min(previous, l)
              previous = l
      
      pass
      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################
    # print(y_pred)
    return y_pred

