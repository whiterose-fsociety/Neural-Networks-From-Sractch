import numpy as np
import neuralnetworks as nn
import tensorflow as tf
import pandas as pd

# Activation Functions
def relu(z):
    return np.where(z > 0, z, 0.0)

def sigmoid(z):
    return 1/(1 + np.exp(-z))


def tanh(z):
    return (2/(1 + np.exp(-2*z)) -1)

def sigmoid_prime(z):
    return np.multiply(sigmoid(z),1-sigmoid(z))

def tanh_prime(z):
    return 1 - tanh(z)**2

def relu_prime(z):
    return np.where(z > 0, 1.0, 0.0)

def activate(z,type_='sigmoid'):
    if type_=='sigmoid':
        return sigmoid(z)
    elif type_ == "relu":
        return relu(z)
    elif type_ == "tanh":
        return tanh(z)
    else:
        return z
    
def activate_prime(z,type_='sigmoid'):
    if type_=='sigmoid':
        return sigmoid_prime(z)
    elif type_ == "relu":
        return relu_prime(z)
    elif type_ == "tanh":
        return tanh_prime(z)
    else:
        return z

def get_thetas(nodes): # Generate Random Thetas Given The Number of Nodes In Each Layer
    weight_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
    thetas = []
    for node in range(len(nodes)-2) :
        thetai = weight_initializer(shape=(nodes[node+1]-1,nodes[node])).numpy()
        thetas.append(thetai)
    thetas.append(weight_initializer(shape=(nodes[-1],nodes[-2])).numpy())
    return thetas  

def make_zero_thetas(nodes): # Generate Random Thetas Given The Number of Nodes In Each Layer
    thetas = []
    for node in range(len(nodes)-2) :
        thetai = np.zeros((nodes[node+1]-1,nodes[node]))
        thetas.append(thetai)
    thetas.append(np.zeros(shape=(nodes[-1],nodes[-2])))
    return thetas  

def Reverse(lst):
    return [ele for ele in reversed(lst)]


def get_deltas(hypothesis,output,thetas,zis,type_='sigmoid'):
    delta_i = hypothesis - output
    thetas_r = Reverse(thetas)
    zis_ = Reverse(thetas)
    deltas = [delta_i]
    for theta in range(len(thetas)-1):
        theta_ri = thetas_r[theta].T[1:,:]
        zis_i = zis[theta+1]
        zis_i_prime = activate_prime(zis_i,type_=type_)
        if delta_i.ndim == 0:
            delta_i = np.multiply(theta_ri*delta_i,zis_i_prime.T)
        else:
            delta_i = np.multiply(theta_ri@delta_i,zis_i_prime.T)
        deltas.append(delta_i)
    return Reverse(deltas)
   
def forward_pass(a1,thetas,n_types = 'sigmoid',last_type='sigmoid'):
    ai = a1
    zs = []
    ais = [np.array([ai])]
    for i in range(len(thetas)):
        zi = thetas[i]@ai
        ai = activate(zi,type_=n_types)
        ai = np.insert(ai,0,1)
        zs.append(np.array([zi]))
        ais.append(np.array([ai]))
    return activate(zi,type_=last_type)[0],zs,ais


def backward_pass(deltas,ais):
    dJs = []
    for delta in range(len(deltas)):
        dJs.append(np.outer(deltas[delta],ais[delta]))
    return dJs


def predict(data,thetas,n_types = 'sigmoid',last_type='sigmoid'):
    outputs = []
    for i in range(len(data)):
        output = nn.forward_pass(data[i].reshape((9,1)),thetas,n_types = n_types,last_type=last_type)[0]
        threshold = 1 if output >= 0.5 else 0
        outputs.append(threshold)
    return outputs

def accuracy(dataset):
    predictions = dataset.iloc[:,-1]
    outcome = dataset.iloc[:,-2]
    successes = 0
    for pred,out in zip(predictions,outcome):
        if pred == out:
            successes +=1 
    return successes/len(dataset)

def backprop_algorithm(X_scaled,trainY,nodes,n_types = 'sigmoid',last_type='sigmoid',alpha=0.9,lambda_=0.01,epoch=100):
    thetas = nn.get_thetas(nodes)
    gradients = nn.make_zero_thetas(nodes)
    D = nn.make_zero_thetas(nodes)
    for e in range(epoch):
        for i in range(len(X_scaled)):
            trainXi = X_scaled[i].reshape((9,1))
            trainYi = trainY.iloc[i] #output
            hypothesis,zis,ais = forward_pass(trainXi,thetas,n_types = n_types,last_type=last_type)
            errors = get_deltas(hypothesis,trainYi,thetas,zis,type_=n_types)
            dJdthetas = backward_pass(errors,ais)
            for gradient in range(len(gradients)):
                gradients[gradient] = gradients[gradient] + dJdthetas[gradient]
            for Di in range(len(D)-1):
                D[Di][:,1:] = 1/(1*len(X_scaled))*gradients[Di][:,1:] + lambda_*thetas[Di][:,1:]
            D[-1][:,1] = 1/(1*len(X_scaled))*gradients[-1][:,1]
        for theta in range(len(thetas)):
            thetas[theta] = thetas[theta] - alpha*D[theta]
    return thetas

def print_unique_values(X_scaled_test,thetas): #This method checks that indeed unique values are generated and the algorithm learns
    set_ = []
    for i in range(len(X_scaled_test)):
        a1 = np.array(X_scaled_test[i,:])
        set_.append(nn.forward_pass(a1,thetas)[0])
    r = {"values":set_}
    p = pd.DataFrame(r)
    return p['values'].unique()


def model(X_scaled,X_scaled_test,test_,trainY,nodes,n_types = 'sigmoid',last_type='sigmoid',alpha=0.9,lambda_=0.01,epoch=100):
    thetas = nn.backprop_algorithm(X_scaled,trainY,nodes,n_types=n_types,last_type=last_type)
    predictions = nn.predict(X_scaled_test,thetas,n_types =n_types,last_type=last_type)
    unique_values = nn.print_unique_values(X_scaled_test,thetas) # unique_values
    prediction_dataset = test_.copy()
    prediction_dataset['Predictions'] = predictions
    if len(prediction_dataset['Predictions'].unique()) < 2:
        prediction_dataset.iat[0,-1]=0
        prediction_dataset.iat[1,-1]=0
    result = nn.accuracy(prediction_dataset)*100
    return prediction_dataset,result


