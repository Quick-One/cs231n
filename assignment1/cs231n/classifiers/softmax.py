from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train, _ = X.shape
    _, num_classes = W.shape

    for i in range(num_train):
        score = X[i] @ W
        
        # numeric stability
        logC = -np.max(score)
        score += logC

        # softmax
        score = np.exp(score)
        score /= np.sum(score)
        loss += -np.log(score[y[i]])

        score[y[i]] -= 1
        
        dW += np.outer(X[i], score)
        
        # dW += np.reshape(X[i], (-1, 1)) @ np.reshape(score, (1, -1))

        # for j in range(num_classes):
        #     dW[:, j] += score[j] * X[i]

    loss /= num_train
    dW /= num_train
    loss += reg * np.sum(W * W) 
    dW += reg * 2 * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train, _ = X.shape
    _, num_classes = W.shape

    scores = X @ W
    logC = -np.max(scores, axis=1)
    scores += np.reshape(logC, (-1, 1))
    scores = np.exp(scores)
    scores /= np.reshape(np.sum(scores, axis=1), (-1, 1))

    correct_label_scores = scores[np.arange(num_train), y]
    
    loss = np.sum(-np.log(correct_label_scores))
    loss /= num_train
    loss += reg * np.sum(W * W)

    scores[np.arange(num_train), y] -= 1
    dW += X.T @ scores
    dW /= num_train
    dW += reg * 2 * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
