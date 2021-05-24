import numpy as np
import neuralnetworks as nn
import tensorflow as tf
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_confusion_matrix


def average_results(X_scaled,X_scaled_test,test_,trainY,nodes,n_types = 'sigmoid',last_type='sigmoid',alpha=0.9,lambda_=0.01,epoch=100,avg_run=1):
    results = []
    for run in range(avg_run):
        thetas = nn.backprop_algorithm(X_scaled,trainY,nodes,n_types=n_types,last_type=last_type)
        predictions = nn.predict(X_scaled_test,thetas)
        prediction_dataset = test_.copy()
        prediction_dataset['Predictions'] = predictions
        results.append(nn.accuracy(prediction_dataset)*100)
    return results,sum(results)/len(results)


def confusion_matrix(prediction_dataset):
    TN = 0;FP=0;TP=0;FN=0;
    for out,pred in zip(prediction_dataset.loc[:,'Outcome'],prediction_dataset.loc[:,'Predictions']):
        if pred == 1 and out == 1:
            TP = TP + 1 # Prediction Is True - Outcome Is True
        if pred == 0 and out == 0:
            TN = TN + 1 # Prediction Is Not True - Outcome Is Not True
        if pred == 1 and out == 0: # Type1
            FP = FP + 1  # Prediction Is True - Outcome Is Not True
        if pred == 0 and out == 1: # Tyoe 2 Error
            FN = FN + 1 #  Prediction Is Not True - Outcome Is True 
    return {"TP":TP,"TN":TN,"FP":FP,"FN":FN}

def sensitivity(TP,FN): # Describes how good the model is at predicting the positive class when the actual outcome is positive
    return TP/(TP+FN)

def false_positive_rate(FP,TN): # False alarm rate 
    return FP/(FP+TN)

def false_negative_rate(FN,TP):
    return FN/(TP+FN)

def specificity(TN,FP):
    return TN / (TN +FP)

# The ROC compares different models for thresholds
# The area under the curve can be used a summary of the model skill


def AUC_Scores(model_titles,models):
    auc_scores = {}
    fpr_roc_curves = {}
    p_fpr_curves = {}
    for model in range(len(models)):
        fpr1,tpr1,thresh1 = roc_curve(models[model].loc[:,'Outcome'],models[model].loc[:,'Predictions'],pos_label=1)
        auc_score = roc_auc_score(models[model].loc[:,'Outcome'],models[model].loc[:,'Predictions'])
        random_probs = [0 for i in range(len(models[model].loc[:,'Outcome']))]
        p_fpr,p_tpr,_ = roc_curve(models[model].loc[:,'Outcome'],random_probs,pos_label=1)
        auc_scores[model_titles[model]] = auc_score
        fpr_roc_curves[model_titles[model]]= (fpr1,tpr1)
        p_fpr_curves[model_titles[model]] = (p_fpr,p_tpr)
    return auc_scores,fpr_roc_curves,p_fpr_curves
        
def average_results(X_scaled,X_scaled_test,test_,trainY,nodes,n_types = 'sigmoid',last_type='sigmoid',alpha=0.9,lambda_=0.01,epoch=100,avg_run=1):
    results = []
    for run in range(avg_run):
        thetas = nn.backprop_algorithm(X_scaled,trainY,nodes,n_types=n_types,last_type=last_type)
        predictions = nn.predict(X_scaled_test,thetas)
        prediction_dataset = test_.copy()
        prediction_dataset['Predictions'] = predictions
        results.append(nn.accuracy(prediction_dataset)*100)
    return results,sum(results)/len(results)