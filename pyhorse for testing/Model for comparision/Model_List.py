#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ivan
"""

"""
---------------
Model Framework
---------------
Input
-----
DataFrame : ['RARID', 'HNAME', '...']

Prediction Output
-----------------
DataFrame : ['RARID', 'HNAME', 'Model Name']

Methods
-------
self.__init__
self.fit
self.predict
self.predict_proba
self.hyperparameter_selection
self.load_hyperparameters()

Important Attributes
--------------------
self.model_name : user defined model name
self.preprocessing : preprocessing function
self.fitted_model : fitted model
self.summary : Summary Table from estimator
"""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

=========================  Dataset Transformation =========================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Raw : No Transformation "√"
Normalise_Race : Normalise Data by Race "√"
Normalise_Profile : Normalise Data by Race Profile
Log_x : Apply Log(1+X-min(x)) to Dataset "√"
PCA : Apply Principle Component Analysis transformation to Dataset
SMOTE :
Poly2 : Create Polynomial Variables
PolyInter Create Interaction Variables

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

============================= Ensemble Models =============================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""
Custom Models
"""
Conditional Gradient Boosting Machine

"""
Conditional Logit Models (5)
"""

CL_Ridge Model : "√"
    One Stage Conditional Logsitic Regression
    Ridge Penalty on Coefficients
    Public Implied probability as an attribute

CL_Frailty Model : "√"
    One Stage Conditional Logsitic Regression
    Frailty on public implied probability

CL_Ridge_CL_Ridge Model : "√"
    Two Stage Conditional Logistic Regression
    Conditional Logistic Regression Strength Model : Ridge Penalty
    Conditional Logistic Regression Combined Model : Strength Probability + Public Odds implied Probability, Ridge Penalty

CL_Ridge_CL Model : "√"
    Two Stage Conditional Logistic Regression
    Conditional Logistic Regression Strength Model : Ridge Penalty
    Conditional Logistic Regression Combined Model : Strength Probability + Public Odds implied Probability, No Penalty

CL_Ridge_CL_Frailty Model : "√"
    Two Stage Conditional Logistic Regression
    Conditional Logistic Regression Strength Model : Ridge Penalty
    Conditional Logistic Regression Combined Model : Strength Probability + Public Odds implied Probability,
                                                     Frailty on public implied probability

"""
SVM Models (5)
"""

SVC_RBF Model : "√"
    Support Vector Classifier with RBF Kernel
    Within Race competition is not considered

SVC_Linear Model : "√"
    Support Vector Classifier with Linear kernel
    Within Race competition is not considered

SVR_RBF Model : "√"
    Support Vector Regression
    Predicting Finishing Position

SVC_CL Model : "√"
    Stage 1 : Support Vector Classifier with RBF Kernel
    Stage 2 : Conditional Logistic Regression with Ridge Penalty

SVR_CL Model : "√"
    Stage 1 : Support Vector Regression with RBF Kernal on Finishing Position
    Stage 2 : Conditional Logistic Regression with Ridge Penalty

"""
Tree Models (16)
"""

XGBoost_Class Model : "√"
    Extreme Gradient Boosting Classifier
    Within Race competition is not considered

XGBoost_Reg Model : "√"
    Extreme Gradient Boosting Regressor
    Predicting Finishing Position

XGBoost_Class_CL Model : "√"
    Stage 1 : Extreme Gradient Boosting Classifier
    Stage 2 : Conditional Logistic Regression

XGBoost_Reg_CL Model : "√"
    Stage 1 : Extreme Gradient Boosting Regressor
    Stage 2 : Conditional Logistic Regression

LightGBM_Class Model :
    Light Gradient Boosting Classifier
    Within Race competition is not considered

LightGBM_Reg Model :
    Light Gradient Boosting Regressor
    Predicting Finishing Position

LightGBM_Class_CL Model :
    Stage 1 : Light Gradient Boosting Classifier
    Stage 2 : Conditional Logistic Regression

LightGBM_Reg_CL Model :
    Stage 1 : Light Gradient Boosting Regressor
    Stage 2 : Conditional Logistic Regression

Catboost_Class Model : "√"
    Categorical Gradient Boosting Classifier
    Within Race competition is not considered

Catboost_Reg Model : "√"
    Categorical Gradient Boosting Regressor
    Predicting Finishing Position

Catboost_Class_CL Model : "√"
    Stage 1 : Categorical Gradient Boosting Classifier
    Stage 2 : Conditional Logistic Regression

Catboost_Reg_CL Model : "√"
    Stage 1 : Categorical Gradient Boosting Regressor
    Stage 2 : Conditional Logistic Regression

Extra_Trees Model :
    Extra Trees Classifier

Extra_Trees_CL Model :
    Stage 1 : Extra Trees Classifier
    Stage 2 : Conditional Logistic Regression

Random_Forest Model :
    Random Forest Classifier

Random_Forest_CL Model :
    Stage 1 : Random Forest Classifier
    Stage 2 : Conditional Logistic Regression

"""
Neural Nets Models (4)
"""
Shallow_NN Model :
    Shallow Artificial Neural Network Model
    Within Race competition is not considered

Shallow_NN_CL Model :
    Stage 1 : Shallow Artificial Neural Network Model
    Stage 2 : Conditional Logistic Regression

Deep_NN Model :
    Deep Artificial Neural Network Model
    Within Race competition is not considered

Deep_NN_CL Model :
    Stage 1 : Deep Artificial Neural Network Model
    Stage 2 : Conditional Logistic Regression

"""
Others (1)
"""
KNN Model :
    KNN Classifier

Libfm
    Factorization Machines

"""
Harville Models
"""












