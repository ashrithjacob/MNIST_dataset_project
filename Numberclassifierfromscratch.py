#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 12:31:19 2022

@author: ashrith
"""
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
def normalize(input_unnormal):
    old_mean = np.mean(input_unnormal)
    old_std = np.std(input_unnormal)
    new_mean = 0.5
    new_std= 0.5
    input_normalised = input_unnormal/255.0
    input_normalised = (input_normalised - new_mean)/new_std
    

    return input_normalised

def initialize_weights(n_o, n_i):
    np.random.seed(30)
    W = np.random.randn(n_o , n_i) * (1/32)
    B = np.zeros((n_o , 1))
    params ={"W" : W,
             "B" : B}
    return params 

def ReLu(Z):
    return (Z * (Z>0))

def Sigmoid(Z):
    return (1 / (1 + np.exp(-Z)))

def dReLu(Z):
    return (1.0*(Z>0))

def dSigmoid(Z):
    return (np.exp(-Z)/np.square(1 + np.exp(-Z)))
    
    
def cross_entropy_loss(T_in , label):
    sum_classes = np.sum(np.exp(T_in), axis=0, keepdims=True)
    Log = -1 * np.log( np.exp(T_in) / sum_classes)
    loss = np.sum((Log * label))
    return loss

def matrix_constr(c, label):
    l = len(label)
    T = np.zeros((c,l))
    for i in range(len(label)):
        T[label[i],i] = 1.0
    return T
 
def create_prob(P):
    sum_prob = np.sum(np.exp(P), axis=0, keepdims = True)
    prob = np.exp(P)/ sum_prob 
    return prob     
        
def forward_prop(params, activation_fn='Relu'):
    W = params["W"]
    A = params["A"]
    B = params["B"]
   
    Z = np.dot(W, A) + B
    A_o = Sigmoid(Z) if activation_fn != 'Relu' else ReLu(Z)
    return A_o, Z
        

def back_prop(params, m, activation_fn= 'Relu'):
    
    dA = params["dA"]
    A = params["A"]
    Z = params["Z"]
    W = params["W"]

    if activation_fn == 'Sigmoid':
        G = dSigmoid(Z)
    else:
        G = dReLu(Z)
    dZ = dA * G
    dW = (1/m) * np.dot(dZ ,A.T)
    dB = (1/m) * np.sum(dZ, axis = 1, keepdims=True)
    dA_1 = np.dot(W.T, dZ)
    grads = {"dW" : dW,
             "dB" : dB}
    
    return grads, dA_1
    
def parameter_update(grad, bno, VW, VB, W, B, alpha, momentum =0.9): 
    dW = grad["dW"]
    dB = grad["dB"]
    
    if bno == 0:
        VW = dW
        VB = dB
    else:
        VW = momentum * VW + dW
        VB = momentum * VB + dB
        
        
    W = W - alpha * VW
    B = B - alpha * VB
    
    params = {"W" : W,
              "B" : B}
    
    stoch_params ={"VW" : VW,
                   "VB" : VB}
               
    return params, stoch_params








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


"""
Building the neural network
"""
### Hyperparameters
m = 64 # minibatch size
input_data_size = 28*28
size_layers   = (input_data_size, 128, 64, 10)
hidden_layers = len(size_layers)
number_train  = 60000
number_test   = 10000
number_minibatch = int(number_train / m)
epochs = 10
learning_rate = 0.003

# use list to cache the long list of variables and dictionary to store each layer's parameter and also pass values
# find m from array size
# intialising weights in the NN


W = ["Null"]  # so that we can refer to W according to mathematical convension starting from 1...
B = ["Null"]
VW = ["Null",0,0,0]
VB = ["Null",0,0,0]

for l in range(hidden_layers-1):
  
    parameters = initialize_weights(size_layers[l+1], size_layers[l])
    W.append(np.array(parameters["W"]))
    B.append(np.array(parameters["B"]))
    print("l value in intitialisation", l)

  

total_loss = 0
for e in range(epochs):
    print("start epoch", e)
    for batch_no in range(number_minibatch):
        
        start = batch_no * m
        end = (batch_no + 1) * m
        
        """Handling last minibatch"""
        if end > 60000:
            end = 60000
            m = 32
        
        """ Normalisation of each minibatch """
        input_normalised = normalize(input_train_unnormal[:, start:end])
        """ input normalised batch is stored in A[0] """
        A = [input_normalised]
        """ Z is stored as required in backpropogation step"""
        Z = ["Null"]
       
        """Forward  Propogation"""
        for l in range(1, hidden_layers):
            paramsf ={"W" : W[l], 
                      "B" : B[l],
                      "A" : A[l-1]}
            A_o, Z_o = forward_prop(paramsf, 'Sigmoid' if l == hidden_layers-1 else 'Relu')
            A.append(A_o) # All A's stored here
            Z.append(Z_o) # All Z's stored here
                
        prob_softmax = create_prob(A[l])
        dA = prob_softmax-1
        label= matrix_constr(len(A[hidden_layers-1]),arr_label_train[start:end])
        
        
        for l in range(hidden_layers-1, 0, -1):
           paramsb = {"dA" : dA,
                      "A" : A[l-1],
                      "Z" : Z[l],
                      "W" : W[l]} 
           grads, dA= back_prop(paramsb, m,'Sigmoid' if l == hidden_layers-1 else 'Relu')         
           params, sparams = parameter_update(grads, batch_no+e, VW[l], VB[l], W[l], B[l], learning_rate)
           W[l] = params["W"]
           B[l] = params["B"]
           VW[l] = sparams["VW"]
           VB[l] = sparams["VB"]
           

        loss = cross_entropy_loss(A[hidden_layers-1] , label)
        total_loss = total_loss + loss
        
    total_loss = total_loss / number_train
    print(e,total_loss)
    total_loss =0    
