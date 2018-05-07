'''
This file implements a multi layer neural network for a multiclass classifier
using Python3.6
'''
import numpy as np
from load_dataset import mnist
import matplotlib.pyplot as plt
import sys
import ast
from scipy import signal


def relu(Z):
    '''
    computes relu activation of Z

    Inputs:
        Z is a numpy.ndarray (n, m)

    Returns:
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = np.maximum(0, Z)
    cache = {}
    cache["Z"] = Z
    return A, cache


def relu_der(dA, cache):
    '''
    computes derivative of relu activation

    Inputs:
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input
        to the activation layer during forward propagation

    Returns:
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    dZ = np.array(dA, copy=True)
    Z = cache["Z"]
    dZ[Z < 0] = 0
    return dZ


def linear(Z):
    '''
    computes linear activation of Z
    This function is implemented for completeness

    Inputs:
        Z is a numpy.ndarray (n, m)

    Returns:
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = Z
    cache = {}
    return A, cache


def linear_der(dA, cache):
    '''
    computes derivative of linear activation
    This function is implemented for completeness

    Inputs:
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input
        to the activation layer during forward propagation

    Returns:
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    dZ = np.array(dA, copy=True)
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

    convParameters = {}
    for i in range(5):
        convParameters["convW" + str(i + 1)] = np.random.randn(3, 3) * 0.01
        convParameters["convb" + str(i + 1)] = np.random.randn() * 0.01
    return parameters, convParameters


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


def layer_forward(A_prev, W, b, activation):
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
    if activation == "relu":
        A, act_cache = relu(Z)
    elif activation == "linear":
        A, act_cache = linear(Z)

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
        A, cache = layer_forward(A, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
        caches.append(cache)

    AL, cache = layer_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "linear")
    caches.append(cache)
    return AL, caches


def conv_forward(X, parameters):
    cache = {}
    m = X.shape[2]
    cache["A"] = X
    dim = 26 * 26
    Z = np.empty((dim * 5, m))
    for i in range(5):
        W = parameters["convW" + str(i + 1)]
        b = parameters["convb" + str(i + 1)]
        cache["convW" + str(i + 1)] = W
        cache["convb" + str(i + 1)] = b
        for j in range(m):
            z = signal.convolve2d(X[:, :, j], W, mode='valid') + b
            Z[i * dim:(i + 1) * dim, j] = z.reshape(dim)

    return Z, cache


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


def layer_backward(dA, cache, W, b, activation):
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
    if activation == "relu":
        dZ = relu_der(dA, act_cache)
    elif activation == "linear":
        dZ = linear_der(dA, act_cache)
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
    activation = "linear"
    for l in reversed(range(1, L + 1)):
        dA, gradients["dW" + str(l)], gradients["db" + str(l)] = layer_backward(dA, caches[l - 1],
            parameters["W" + str(l)], parameters["b" + str(l)], activation)
        activation = "relu"

    gradients["dA"] = dA
    return gradients


def conv_backward(dA, cache, parameters):
    m = dA.shape[1]
    A = cache["A"]
    gradients = {}
    dim = 26 * 26
    for i in range(5):
        dZ = dA[i * dim:(i + 1) * dim]
        gradients["dconvb" + str(i + 1)] = np.sum(dZ) / (m * dim)

        dW = np.zeros((3, 3))
        for j in range(dim):
            x = j // 26
            y = j % 26
            Aslice = A[x:(x + 3), y:(y + 3), :]
            dW += np.sum(Aslice * dZ[j], axis=2)
        gradients["dconvW" + str(i + 1)] = dW / (m * dim)

    return gradients


def classify(X, parameters, convParameters):
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
    A0, convCache = conv_forward(X, convParameters)
    # Forward propagate X using multi_layer_forward
    AL, caches = multi_layer_forward(A0, parameters)
    # Get predictions using softmax_cross_entropy_loss
    A, cache, cost = softmax_cross_entropy_loss(AL)
    # Estimate the class labels using predictions
    Ypred = np.argmax(A, axis=0)
    return Ypred


def update_parameters(parameters, gradients, epoch, learning_rate, decay_rate=0.0):
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
        parameters["W" + str(l)] -= alpha * gradients["dW" + str(l)]
        parameters["b" + str(l)] -= alpha * gradients["db" + str(l)]
    return parameters, alpha


def update_conv(parameters, gradients, alpha):
    for i in range(5):
        parameters["convW" + str(i + 1)] -= alpha * gradients["dconvW" + str(i + 1)]
        parameters["convb" + str(i + 1)] -= alpha * gradients["dconvb" + str(i + 1)]
    return parameters


def multi_layer_network(X, Y, net_dims, num_iterations=500, learning_rate=0.2, decay_rate=0.01):
    '''
    Creates the multilayer network and trains the network

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
    parameters, convParameters = initialize_multilayer_weights(net_dims)
    costs = []
    for ii in range(num_iterations):
        ### CODE HERE
        Y = Y.astype(int)
        # Forward Prop
        A0, convCache = conv_forward(X, convParameters)
        # call to multi_layer_forward to get activations
        AL, caches = multi_layer_forward(A0, parameters)
        ## call to softmax cross entropy loss
        A, cache, cost = softmax_cross_entropy_loss(AL, Y)

        # Backward Prop
        # call to softmax cross entropy loss der
        dZ = softmax_cross_entropy_loss_der(Y, cache)
        ## call to multi_layer_backward to get gradients
        gradients = multi_layer_backward(dZ, caches, parameters)

        convGradients = conv_backward(gradients["dA"], convCache, convParameters)
        ## call to update the parameters
        parameters, alpha = update_parameters(parameters, gradients, ii, learning_rate, decay_rate)
        convParameters = update_conv(convParameters, convGradients, alpha)

        if ii % 10 == 0:
            costs.append(cost)
        if ii % 10 == 0:
            print("Cost at iteration %i is: %.05f, learning rate: %.05f" % (ii, cost, alpha))

    return costs, parameters, convParameters


def main():
    '''
    Trains a multilayer network for MNIST digit classification (all 10 digits)
    To create a network with 1 hidden layer of dimensions 800
    Run the progam as:
        python convNeuralNetwork.py "[800]"
    The network will have the dimensions [3380,800,10]

    input size of digit images: 784 (28pix x 28pix = 784)
    after convolution with 3 x 3 filter: (28 - 3 + 1)^2 = 676
    with 5 filters: 676 x 5 = 3380
    10 is the number of digits

    To create a network with 2 hidden layers of dimensions 800 and 500
    Run the progam as:
        python deepMultiClassNetwork_starter.py "[3380,800,500]"
    The network will have the dimensions [3380,800,500,10]
    '''
    net_dims = ast.literal_eval(sys.argv[1])
    net_dims.insert(0, 3380)
    net_dims.append(10)  # Adding the digits layer with dimensionality = 10
    print("Network dimensions are:" + str(net_dims))

    # getting the subset dataset from MNIST
    train_data, train_label, test_data, test_label = \
        mnist(ntrain=800, ntest=200, digit_range=[0, 10])
    # initialize learning rate and num_iterations
    learning_rate = 0.8
    num_iterations = 300
    decay_rate = 0.003

    costs, parameters, convParameters = multi_layer_network(train_data, train_label, net_dims,
        num_iterations=num_iterations, learning_rate=learning_rate, decay_rate=decay_rate)

    # compute the accuracy for training set and testing set
    train_Pred = classify(train_data, parameters, convParameters)
    test_Pred = classify(test_data, parameters, convParameters)

    trAcc = 100 * np.mean((train_Pred == train_label).astype(int))
    teAcc = 100 * np.mean((test_Pred == test_label).astype(int))
    print("Accuracy for training set is {0:0.3f} %".format(trAcc))
    print("Accuracy for testing set is {0:0.3f} %".format(teAcc))

    ### CODE HERE to plot costs
    iterations = range(0, num_iterations, 10)
    plt.plot(iterations, costs)
    plt.title("CNN: " + str(net_dims) +
        "\nTraining accuracy:{0:0.3f}% Testing accuracy:{1:0.3f}%".format(trAcc, teAcc))
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.show()


if __name__ == "__main__":
    main()
