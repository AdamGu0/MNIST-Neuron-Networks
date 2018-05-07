'''
This file implements a multi layer neural network for a multiclass classifier
using Python3.6
'''
import numpy as np
from load_dataset import mnist
import matplotlib.pyplot as plt


def sigmoid(Z):
    '''
    computes sigmoid activation of Z

    Inputs:
        Z is a numpy.ndarray (n, m)

    Returns:
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = 1 / (1 + np.exp(-Z))
    cache = {}
    cache["Z"] = Z
    return A, cache


def sigmoid_der(dA, cache):
    '''
    computes derivative of sigmoid activation

    Inputs:
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input
        to the activation layer during forward propagation

    Returns:
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    ### CODE HERE
    Z = cache["Z"]
    A, c = sigmoid(Z)
    dZ = dA * A * (1 - A)
    return dZ


def softmax_cross_entropy_loss(Z, Y=np.array([])):
    '''
    Computes the softmax activation of the inputs Z
    Estimates the cross entropy loss

    Inputs:
        Z - numpy.ndarray (n, m)
        Y - numpy.ndarray (1, m) of labels
            when y=[] loss is set to []

    Returns:
        A - numpy.ndarray (n, m) of softmax activations
        cache -  a dictionary to store the activations later used to estimate derivatives
        loss - cost of prediction
    '''
    ### CODE HERE
    mZ = np.max(Z, axis=0, keepdims=True)
    expZ = np.exp(Z - mZ)
    A = expZ / np.sum(expZ, axis=0, keepdims=True)

    cache = {}
    cache["A"] = A
    IA = [A[Y[0][i]][i] for i in range(Y.size)]
    IA = (np.array(IA) - 0.5) * 0.9999999999 + 0.5
    loss = -np.mean(np.log(IA))

    return A, cache, loss


def softmax_cross_entropy_loss_der(Y, cache):
    '''
    Computes the derivative of softmax activation and cross entropy loss

    Inputs:
        Y - numpy.ndarray (1, m) of labels
        cache -  a dictionary with cached activations A of size (n,m)

    Returns:
        dZ - numpy.ndarray (n, m) derivative for the previous layer
    '''
    ### CODE HERE
    dZ = cache["A"].copy()
    for i in range(Y.size):
        dZ[Y[0][i]][i] -= 1
    return dZ


def initialize_multilayer_weights(net_dims):
    '''
    Initializes the weights of the multilayer network

    Inputs:
        net_dims - tuple of network dimensions

    Returns:
        dictionary of parameters
    '''
    np.random.seed(0)
    numLayers = len(net_dims)
    parameters = {}
    for l in range(numLayers - 1):
        parameters["W" + str(l + 1)] = np.random.randn(net_dims[l + 1], net_dims[l]) * 0.01
        parameters["b" + str(l + 1)] = np.random.randn(net_dims[l + 1], 1) * 0.01
    return parameters


def linear_forward(A, W, b):
    '''
    Input A propagates through the layer
    Z = WA + b is the output of this layer.

    Inputs:
        A - numpy.ndarray (n,m) the input to the layer
        W - numpy.ndarray (n_out, n) the weights of the layer
        b - numpy.ndarray (n_out, 1) the bias of the layer

    Returns:
        Z = WA + b, where Z is the numpy.ndarray (n_out, m) dimensions
        cache - a dictionary containing the inputs A
    '''
    ### CODE HERE
    Z = np.dot(W, A) + b
    cache = {}
    cache["A"] = A
    return Z, cache


def layer_forward(A_prev, W, b):
    '''
    Input A_prev propagates through the layer and the activation

    Inputs:
        A_prev - numpy.ndarray (n,m) the input to the layer
        W - numpy.ndarray (n_out, n) the weights of the layer
        b - numpy.ndarray (n_out, 1) the bias of the layer
        activation - is the string that specifies the activation function

    Returns:
        A = g(Z), where Z = WA + b, where Z is the numpy.ndarray (n_out, m) dimensions
        g is the activation function
        cache - a dictionary containing the cache from the linear and the nonlinear propagation
        to be used for derivative
    '''
    Z, lin_cache = linear_forward(A_prev, W, b)
    A, act_cache = sigmoid(Z)

    cache = {}
    cache["lin_cache"] = lin_cache
    cache["act_cache"] = act_cache
    return A, cache


def multi_layer_forward(X, parameters):
    '''
    Forward propgation through the layers of the network

    Inputs:
        X - numpy.ndarray (n,m) with n features and m samples
        parameters - dictionary of network parameters {"W1":[..],"b1":[..],"W2":[..],"b2":[..]...}
    Returns:
        AL - numpy.ndarray (c,m)  - outputs of the last fully connected layer before softmax
            where c is number of categories and m is number of samples in the batch
        caches - a dictionary of associated caches of parameters and network inputs
    '''
    L = len(parameters) // 2
    A = X
    caches = []
    for l in range(1, L):  # since there is no W0 and b0
        A, cache = layer_forward(A, parameters["W" + str(l)], parameters["b" + str(l)])
        caches.append(cache)

    AL, cache = layer_forward(A, parameters["W" + str(L)], parameters["b" + str(L)])
    caches.append(cache)
    return AL, caches


def linear_backward(dZ, cache, W, b):
    '''
    Backward prpagation through the linear layer

    Inputs:
        dZ - numpy.ndarray (n,m) derivative dL/dz
        cache - a dictionary containing the inputs A, for the linear layer
            where Z = WA + b,
            Z is (n,m); W is (n,p); A is (p,m); b is (n,1)
        W - numpy.ndarray (n,p)
        b - numpy.ndarray (n, 1)

    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W
        db - numpy.ndarray (n, 1) the gradient of b
    '''
    A_prev = cache["A"]
    ## CODE HERE
    m = dZ.shape[1]
    dA_prev = np.dot(np.transpose(W), dZ)
    dW = np.dot(dZ, np.transpose(A_prev)) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    return dA_prev, dW, db


def layer_backward(dA, cache, W, b, KL=False):
    '''
    Backward propagation through the activation and linear layer

    Inputs:
        dA - numpy.ndarray (n,m) the derivative to the previous layer
        cache - dictionary containing the linear_cache and the activation_cache
        activation - activation of the layer
        W - numpy.ndarray (n,p)
        b - numpy.ndarray (n, 1)

    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W
        db - numpy.ndarray (n, 1) the gradient of b
    '''
    lin_cache = cache["lin_cache"]
    act_cache = cache["act_cache"]
    '''
    if activation == "sigmoid":
        dZ = sigmoid_der(dA, act_cache)
    elif activation == "tanh":
        dZ = tanh_der(dA, act_cache)
    '''
    if KL:
        b = 3
        KL = cache["KL"]
        p = cache["p"]
        dp = -np.divide(p, KL) + np.divide(1 - p, 1 - KL)
        dA += b * dp

    dZ = sigmoid_der(dA, act_cache)
    dA_prev, dW, db = linear_backward(dZ, lin_cache, W, b)
    return dA_prev, dW, db


def multi_layer_backward(dAL, caches, parameters):
    '''
    Back propgation through the layers of the network (except softmax cross entropy)
    softmax_cross_entropy can be handled separately

    Inputs:
        dAL - numpy.ndarray (n,m) derivatives from the softmax_cross_entropy layer
        caches - a dictionary of associated caches of parameters and network inputs
        parameters - dictionary of network parameters {"W1":[..],"b1":[..],"W2":[..],"b2":[..]...}

    Returns:
        gradients - dictionary of gradient of network parameters
            {"dW1":[..],"db1":[..],"dW2":[..],"db2":[..],...}
    '''
    L = len(caches)  # with one hidden layer, L = 2
    gradients = {}
    dA = dAL
    for l in reversed(range(1, L + 1)):
        dA, gradients["dW" + str(l)], gradients["db" + str(l)] = layer_backward(dA, caches[l - 1],
            parameters["W" + str(l)], parameters["b" + str(l)])
    return gradients


def classify(X, parameters):
    '''
    Network prediction for inputs X

    Inputs:
        X - numpy.ndarray (n,m) with n features and m samples
        parameters - dictionary of network parameters
            {"W1":[..],"b1":[..],"W2":[..],"b2":[..],...}
    Returns:
        YPred - numpy.ndarray (1,m) of predictions
    '''
    ### CODE HERE
    # Forward propagate X using multi_layer_forward
    YPred, caches = multi_layer_forward(X, parameters)
    return YPred


def update_parameters(parameters, gradients, epoch, learning_rate, decay_rate=0.0, weight_decay=0):
    '''
    Updates the network parameters with gradient descent

    Inputs:
        parameters - dictionary of network parameters
            {"W1":[..],"b1":[..],"W2":[..],"b2":[..],...}
        gradients - dictionary of gradient of network parameters
            {"dW1":[..],"db1":[..],"dW2":[..],"db2":[..],...}
        epoch - epoch number
        learning_rate - step size for learning
        decay_rate - rate of decay of step size - not necessary - in case you want to use
    '''
    alpha = learning_rate * (1 / (1 + decay_rate * epoch))
    L = len(parameters) // 2
    ### CODE HERE
    for l in reversed(range(1, L + 1)):
        w = parameters["W" + str(l)]
        parameters["W" + str(l)] -= alpha * (gradients["dW" + str(l)] + weight_decay * w)
        parameters["b" + str(l)] -= alpha * gradients["db" + str(l)]
    return parameters, alpha


def train_sparse_autoencoder(X, Y, net_dims, num_iterations=500, learning_rate=0.2, decay_rate=0.01, weight_decay=0, p=0.5):
    '''
    Creates the neuron network and trains the network

    Inputs:
        X - numpy.ndarray (n,m) of training data
        Y - numpy.ndarray (1,m) of training data labels
        net_dims - tuple of layer dimensions
        num_iterations - num of epochs to train
        learning_rate - step size for gradient descent

    Returns:
        costs - list of costs over training
        parameters - dictionary of trained network parameters
    '''
    parameters = initialize_multilayer_weights(net_dims)
    A0 = X
    costs = []
    for ii in range(num_iterations):
        ### CODE HERE
        # Forward Prop
        # call to layer_forward to get activations
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        A1, cache1 = layer_forward(A0, W1, b1)
        KL = np.mean(A1, axis=1, keepdims=True)
        KL = (KL - 0.5) * 0.9999999999 + 0.5
        cache1["KL"] = KL
        cache1["p"] = p
        A2, cache2 = layer_forward(A1, W2, b2)
        ## call loss function
        cost = np.mean(np.sum((Y - A2)**2, axis=0)) / 2 + weight_decay / 2 * (np.sum(W1) + np.sum(W2))
        + np.sum(p * np.log(np.divide(p, KL)) + (1 - p) * np.log(np.divide(1 - p, 1 - KL)))

        # Backward Prop
        # loss der
        m = A2.shape[1]
        dA2 = A2 - Y
        ## call to layer_backward to get gradients
        gradients = {}
        dA1, gradients["dW2"], gradients["db2"] = layer_backward(dA2, cache2, W2, b2)
        dA0, gradients["dW1"], gradients["db1"] = layer_backward(dA1, cache1, W1, b1, KL=True)
        ## call to update the parameters
        parameters, alpha = update_parameters(parameters, gradients, ii, learning_rate, decay_rate, weight_decay)

        if ii % 10 == 0:
            costs.append(cost)
        if ii % 10 == 0:
            print("Cost at iteration %i is: %.05f, learning rate: %.05f" % (ii, cost, alpha))

    return costs, parameters


def main():
    '''
    Trains a sparse autoencoder for MNIST digit data
    '''
    net_dims = [784, 200, 784]
    print("Network dimensions are:" + str(net_dims))

    # getting the subset dataset from MNIST
    train_data, train_label, test_data, test_label = \
        mnist(ntrain=1000, ntest=200, digit_range=[0, 10])
    train_label = train_data
    test_label = test_data
    # initialize learning rate and num_iterations

    num_iterations = 400
    decay_rate = 0
    weight_decay = 0.001
    ps = [0.01, 0.1, 0.5, 0.8]
    p = ps[1]
    learning_rate = 0.3

    costs, parameters = train_sparse_autoencoder(train_data, train_label, net_dims, num_iterations=num_iterations,
        learning_rate=learning_rate, decay_rate=decay_rate, weight_decay=weight_decay, p=p)

    train_Pred = classify(train_data, parameters)
    test_Pred = classify(test_data, parameters)

    # plt.imshow(np.reshape(train_label[:, 10], [28, 28]), cmap='gray')
    # plt.show()
    # plt.imshow(np.reshape(train_Pred[:, 10], [28, 28]), cmap='gray')
    # plt.show()

    # compute the accuracy for training set and testing set
    trAcc = 100 * (1 - np.mean(np.abs(train_Pred - train_label)))
    teAcc = 100 * (1 - np.mean(np.abs(test_Pred - test_label)))

    print("Accuracy for training set is {0:0.3f} %".format(trAcc))
    print("Accuracy for testing set is {0:0.3f} %".format(teAcc))

    ### CODE HERE to plot costs
    iterations = range(0, num_iterations, 10)
    plt.plot(iterations, costs)
    plt.title("Sparse Autoencoder: " + str(net_dims) + " (p = " + str(p) +
        ")\nTraining accuracy:{0:0.3f}% Testing accuracy:{1:0.3f}%".format(trAcc, teAcc))
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.show()

    W1 = parameters["W1"]
    tmp = np.reshape(W1[0:100, :], [100, 28, 28])
    for i in range(100):
        plt.subplot(10, 10, i + 1)
        plt.axis('off')
        plt.imshow(tmp[i], cmap='gray')
    plt.subplots_adjust(left=0.16, bottom=0.05, right=0.84, top=0.95, wspace=0.05, hspace=0.05)
    plt.suptitle("100 rows of W1 in 28*28 images" + " (p = " + str(p) + ")")
    plt.show()

if __name__ == "__main__":
    main()
