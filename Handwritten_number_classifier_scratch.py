#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 21:14:12 2022

@author: ashrith

Goal of this is to replicate Handwritten_number_classifier without using pytorch (from scratch)
"""
import numpy as np
import idx2numpy
import matplotlib.pyplot as plt


"""
Function definitions:
"""
def normalize(input_train_unnormal, input_test_unnormal):
    
    old_mean_train = np.mean(arr_train)
    old_std_train = np.std(arr_train)
    new_mean_train = 0.5
    new_std_train = 0.5

    old_mean_test = np.mean(arr_test)
    old_std_test = np.std(arr_test)
    new_mean_test = 0.5
    new_std_test = 0.5

    alpha_train = old_std_train / new_std_train
    beta_train = new_mean_train - alpha_train * old_mean_train
    alpha_test = old_std_test / new_std_test
    beta_test = new_mean_test - alpha_test * old_mean_test


    input_train = input_train_unnormal * alpha_train + beta_train    
    input_test = input_test_unnormal * alpha_test + beta_test   

    return input_train, input_test  

def ReLu(Z):
    return (Z * (Z>0))

def Sigmoid(Z):
    return (1 / (1 + np.exp(Z)))

def dReLu(Z):
    return (1.0*(Z>0))

def dSigmoid(Z):
    return (-1*np.exp(Z)/np.square(1 + np.exp(Z)))
    
    
def cross_entropy_loss(T_in , label):
    sum_classes = np.sum(np.exp(T_in), axis=0, keepdims=True)
    Log = -1 * np.log( np.exp(T_in) / sum_classes)
    loss = np.sum((Log * label))
    return loss

def matrix_constr(c, label):
    l = len(label)
    T = np.zeros((c,l))
    for i in range(len(label)):
        T[label[i],i]= 1.0
    return T
 
def create_prob(P):
    sum_prob = np.sum(np.exp(P), axis=0, keepdims = False)
    prob = np.exp(P)/ sum_prob 
    return prob     
        
        
    
def initialize_weights(n_o, n_i):
    
    W = np.random.randn(n_o , n_i) * 0.01
    B = np.zeros((n_o , 1))
    params ={"W" : W,
             "B" : B}
    return params

def forward_prop(W, B, A_i, activation_fn='Relu'):
    
    Z = np.dot(W, A_i) + B
    A_o = Sigmoid(Z) if activation_fn != 'Relu' else ReLu(Z)
    return A_o, Z
        

def back_prop(dA, A_l1, Z, W, m, activation_fn= 'Relu'):
    
    # dA = cache["dA"]
    # A_l1 = cache["A"]
    # Z = cache["Z"]
    # W = cache["W"]

    if activation_fn == 'Sigmoid':
        G = dSigmoid(Z)
    else:
        G = dReLu(Z)
    dZ = dA * G
    dW = (1/m) * np.dot(dZ ,A_l1.T)
    dB = (1/m) * np.sum(dZ, axis = 1, keepdims=True)
    dA_l1 = np.dot(W.T, dZ)
    grads = {"dW" : dW,
             "dB" : dB}
    
    return grads, dA_l1
    
def parameter_update(grad, W, B, alpha): 
    dW = grad["dW"]
    dB = grad["dB"]
    
    W = W - alpha * dW
    B = B - alpha * dB
    
    params = {"W" : W,
             "B" : B}
               
    return params








"""if activation_fn == 'Relu':
        A_o = ReLu(Z)
        
    else :
            
        A_o = Sigmoid(Z)
Download dataset
"""

file_train = '/home/ashrith/Personal/Deep learning/Pytorch_projects/Number classifier/Dataset/train-images.idx3-ubyte'
arr_train = idx2numpy.convert_from_file(file_train)
label_train = '/home/ashrith/Personal/Deep learning/Pytorch_projects/Number classifier/Dataset/train-labels.idx1-ubyte'
arr_label_train = idx2numpy.convert_from_file(label_train)
file_test = '/home/ashrith/Personal/Deep learning/Pytorch_projects/Number classifier/Dataset/t10k-images.idx3-ubyte'
arr_test = idx2numpy.convert_from_file(file_test)
label_test = '/home/ashrith/Personal/Deep learning/Pytorch_projects/Number classifier/Dataset/t10k-labels.idx1-ubyte'
arr_label_test = idx2numpy.convert_from_file(label_test)


"""
Training data contains 60000 images and testing data has 10000 images
"""
plt.imshow(arr_train[56999], cmap=plt.cm.binary)
plt.imshow(arr_test[9994], cmap=plt.cm.binary)
in_size = arr_train[0,:,:].size   #input array size

"""
reshaping inputs
"""

input_train_unnormal = arr_train.reshape(60000,in_size).T
input_test_unnormal = arr_test.reshape(10000, in_size).T

"""
Normalise data
"""
input_train, input_test = normalize(input_train_unnormal , input_test_unnormal)

"""
Building the neural network
"""
### Hyperparameters
m = 60 # minibatch size
size_layers   = (128, 64, 10)
hidden_layers = len(size_layers)
number_train  = len(arr_train)
number_test   = len(arr_test)
number_minibatch = int(number_train / m)
input_data_size = 28*28
epochs = 10
learning_rate = 0.003

# use list to cache the long list of variables and dictionary to store each layer's parameter and also pass values
# find m from array size
# intialising weights in the NN
parameters = initialize_weights(size_layers[0], input_data_size)
W = ["Null"]  # so that we can refer to W according to mathematical convension starting from 1...
B = ["Null"]

dW = ["Null"]
dB = ["Null"]

for l in range(hidden_layers):
    if l == 0:
        parameters = initialize_weights(size_layers[0], input_data_size)
    else:
        parameters = initialize_weights(size_layers[l], size_layers[l-1])
    
    W.append(np.array(parameters["W"]))
    B.append(np.array(parameters["B"]))
    print(W[l])
    
  
# params ={"W" + str(l+1): W,
#          "B" + str(l+1): B}

total_loss=0
for e in range(epochs):
    print("start epoch", e)
    for batch_no in range(number_minibatch):
        
        
        start = batch_no * m
        end = (batch_no + 1) * m
        A = [input_train[:, start:end]]
        Z = ["Null"]
        #dA = ["Null"]
        
        for l in range(1, hidden_layers+1):
            A_o, Z_o = forward_prop(W[l], B[l], A[l-1])
            A.append(A_o) # All A's stored here
            Z.append(Z_o) # All Z's stored here
        #print("here")
        dA = 1-A[l]
        
        for l in range(hidden_layers, 0, -1):
           grads, dA_l1 = back_prop(dA, A[l-1], Z[l], W[l], m,'Sigmoid' if l == hidden_layers else 'Relu')         
           dA = dA_l1
           params = parameter_update(grads, W[l], B[l], learning_rate)
           W[l] = params["W"]
           B[l] = params["B"]
        
        label= matrix_constr(len(A[hidden_layers]),arr_label_train[start:end])
        prob_softmax = create_prob(A[hidden_layers])
        loss = cross_entropy_loss(A[hidden_layers] , label)/m
        total_loss = total_loss + loss
        
    print(e,total_loss)
    total_loss =0    
         
    
    #print(e)

    




# W1=np.random.randn(size_layers[0], in_size) * 0.01

# Parameters={
#             "W1":W1,
#             "W2":W2,
#             "W3":W3,
#             "b1":b1,
#             "b2":b2,
#             "b3":b3
#             }
# for i in range(hidden_layers):
#     Parameter["W"+i] = np.random.randn((2,2)) * 0.01






        

