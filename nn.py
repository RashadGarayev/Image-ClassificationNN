
import cv2,os
import numpy as np



def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s

def propagate(w, b, X, Y):
    m = X.shape[1]
    
     # FORWARD PROPAGATION (FROM X TO COST)
    z = np.dot(w.T, X)+b # tag 1
    A = sigmoid(z) # tag 2                                    
    loss = (-np.sum(Y*np.log(A)+(1-Y)*np.log(1-A)))/m # tag 5
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = (np.dot(X,(A-Y).T))/m # tag 6
    db = np.average(A-Y) # tag 7

    cost = np.squeeze(loss)
    grads = {"dw": dw,
             "db": db}
    
    return grads, loss

def optimize(w, b, X, Y, num_iterations, learning_rate, print_loss= True):
    cost = []    
    for i in range(num_iterations):
        # Cost and gradient calculation
        grads, loss = propagate(w, b, X, Y)
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # update w and b
        w = w - learning_rate*dw
        b = b - learning_rate*db

        # Record the costs
        if i % 100 == 0:
            cost.append(loss)
            
        # Print the cost every 100 training iterations
        if print_loss and i % 100 == 0:
            print ("Loss  %i: %f" %(i, loss))

    # update w and b to dictionary
    params = {"w": w,
              "b": b}
    
    # update derivatives to dictionary
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, cost

def predict(w, b, X):    
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    
    z = np.dot(w.T, X) + b
    A = sigmoid(z)
    
    for i in range(A.shape[1]):
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if A[0,i] > 0.5:
            Y_prediction[[0],[i]] = 1
        else: 
            Y_prediction[[0],[i]] = 0
    
    return Y_prediction

def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    return w, b

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_loss = True):
    # initialize parameters with zeros
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent
    parameters, grads, cost = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_loss)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples
    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    dict = {"loss": cost,
            "Y_prediction_test": Y_prediction_test,
            "Y_prediction_train": Y_prediction_train,
            "w": w,
            "b": b,
            "learning_rate": learning_rate,
            "num_iterations:": num_iterations}
    
    return dict


