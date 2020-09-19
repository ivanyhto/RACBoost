#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ivan
"""

"""
Support Vector Machines
"""

#Loading Libraries
import os
import time
import pylogit
import warnings
import numpy as np
import pandas as pd
from bayes_opt.event import Events
from collections import OrderedDict
from bayes_opt.util import load_logs
from bayes_opt import BayesianOptimization
from pyhorse.Data_Preprocessing import Raw, Normalise_Race
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split, KFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
from pyhorse.Model_Evaluation import Kelly_Profit, Prediction_accuracy, newJSONLogger

#Global Parameters
Odds_col = "OD_CR_LP"
saved_models_path = "./pyhorse/Saved_Models/"
# saved_models_path = "/content/gdrive/My Drive/pyhorse/Saved_Models/"
Betting_Fraction = 0.3

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

========================== Hyperparameter Range ==========================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#Conditional Logit Parameters
ridge_Range = (0, 101)

#SVM Parameters
C_Range = (-3, 17)
gamma_Range = (-20, 1)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

=============================== Test Cases ===============================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def _test_cases():
    from pyhorse import Dataset_Creation
    X, y = Dataset_Creation.Dataset_Extraction(Dataset_Creation.Get_RaceID(['2013','2014', '2015', '2016', '2017']))
    X, y = Dataset_Creation.Dataset_Extraction(Dataset_Creation.Get_RaceID(['2017']))

    SVC_RBF_Model = SVC_RBF(preprocessing = Normalise_Race)
    SVC_RBF_Model.fit(X, y)
    y_pred = SVC_RBF_Model.predict(X)
    SVC_RBF_Model.hyperparameter_selection(X, y, 0, 0, initial_probe = True)
    SVC_RBF_Model.load_hyperparameters()

    SVC_Linear_Model = SVC_Linear(preprocessing = Normalise_Race)
    SVC_Linear_Model.fit(X, y)
    y_pred = SVC_Linear_Model.predict(X)
    SVC_Linear_Model.hyperparameter_selection(X, y, 0, 0, initial_probe = True)
    SVC_Linear_Model.load_hyperparameters()

    SVR_RBF_Model = SVR_RBF(preprocessing = Normalise_Race)
    SVR_RBF_Model.fit(X, y)
    y_pred = SVR_RBF_Model.predict(X)
    SVR_RBF_Model.hyperparameter_selection(X, y, 0, 0, initial_probe = True)
    SVR_RBF_Model.load_hyperparameters()

    SVC_CL_Model = SVC_CL(preprocessing = Normalise_Race)
    SVC_CL_Model.fit(X, y)
    SVC_CL_Model.summary
    y_pred = SVC_CL_Model.predict(X)
    SVC_CL_Model.hyperparameter_selection(X, y, 0, 0, initial_probe = True)
    SVC_CL_Model.load_hyperparameters()

    SVR_CL_Model = SVR_CL(preprocessing = Normalise_Race)
    SVR_CL_Model.fit(X, y)
    SVR_CL_Model.summary
    y_pred = SVR_CL_Model.predict(X)
    SVR_CL_Model.hyperparameter_selection(X, y, 0, 0, initial_probe = True)
    SVR_CL_Model.load_hyperparameters()

    return None

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

========================== One Stage SVC Wrapper ==========================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class SVC_RBF_Wrapper(BaseEstimator, ClassifierMixin):

    """
    A sklearn-style wrapper for One Stage Support Vector Classifier
    """

    def __init__(self, model_class, model_name, preprocessing, kernel, C, gamma) :
        self.model_class = model_class
        self.model_name = model_name
        self.preprocessing = preprocessing

        """
        Hyperparameter
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        #Hyperparameter Dictionary - Bounds
        self.hyperparameter = {'C' : C_Range, 'gamma' : gamma_Range}
        #Track Hyperparameter Selection
        self.number_model_fitted = 0

        return None


    def fit(self, X, y, preprocess = True):

        #Making a copy of X
        X_train = X.copy()

        #Apply Preprocessing Pipeline
        if preprocess == True :
            self.preprocessing = self.preprocessing()
            X_train = self.preprocessing.fit_transform(X_train)
        else:
            #Save fitted preprocessing object passed in
            self.preprocessing = preprocess

        #Create Model Instance
        self.model = self.model_class(probability = True, kernel = self.kernel, C = np.exp(self.C), gamma = np.exp(self.gamma))

        #Slicing away [RARID, HNAME]
        X_train = X_train.loc[:,X_train.columns[2:]]
        y = y.loc[:,'RESWL']

        #Model Fitting
        self.model.fit(X_train, y)
        self.fitted_model  = self.model

        return None


    def predict(self, X, preprocess = True):

        #Making a copy of X
        X_test = X.copy()

        #Apply Preprocessing Pipeline
        if preprocess == True: X_test = self.preprocessing.transform(X_test)

        #Formatting into DataFrame
        Prediction = X_test.loc[:,['RARID','HNAME']]

        #Getting the probability of y=1
        Prediction.loc[:,self.model_name] = self.fitted_model.predict_proba(X_test[X_test.columns[2:]])[:,1]

        #Scale prediction to sum to 1
        Prediction.loc[:,self.model_name] = Prediction.groupby('RARID')[self.model_name].apply(lambda x : x / x.sum())

        return Prediction


    def predict_proba(self, X, preprocess = True):

        #Making a copy of X
        X_test = X.copy()

        #Apply Preprocessing Pipeline
        if preprocess == True: X_test = self.preprocessing.transform(X_test)

        #Formatting into DataFrame
        Prediction = X_test.loc[:,['RARID','HNAME']]

        #Getting the probability of y=1
        Prediction.loc[:,self.model_name] = self.fitted_model.predict_proba(X_test[X_test.columns[2:]])[:,1]

        #Scale prediction to sum to 1
        Prediction.loc[:,self.model_name] = Prediction.groupby('RARID')[self.model_name].apply(lambda x : x / x.sum())

        return Prediction


    def load_hyperparameters(self):

        """
        This method loads the best hyperparameters from the Hyperparameter_Selection History to the current instance
        """
        optimizer = BayesianOptimization(f=None,
                                         pbounds=self.hyperparameter,
                                         verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
                                         random_state=1)

        #Load Hyperparamter Selection History Logs
        logger_name = self.model_name
        if os.path.exists(saved_models_path + logger_name+".json"):
            load_logs(optimizer, logs=[saved_models_path + logger_name+".json"])

            #Save best hyperparameter
            best_model = optimizer.max
            self.gamma = best_model['params']['gamma']
            self.C = best_model['params']['C']

            print("The optimizer is now aware of {num_pts} points and the best result is {Result}.".format(Result = best_model['target'] ,
                                                                                                           num_pts = len(optimizer.space)))

        else :
            print('No Hyperparameters was tested.')

        return None


    def hyperparameter_selection(self, X, y, inital_pts, rounds, initial_probe = True):

        #Create Timer
        start_time = time.time()

        #Making a copy of X
        X_copy = X.copy()

        #Apply Preprocessing Pipeline
        self.preprocessing = self.preprocessing()
        X_copy = self.preprocessing.fit_transform(X_copy)

        """
        Beyesian Optimization
        """
        #Get RaceID in Dataset
        RaceID_List = X_copy.loc[:,'RARID'].unique()

        #Create Function to Optimize
        def SVC_fit_predict_score(C, gamma):
            """
            Looping Over Cross Validation Folds
            """
            Score = []
            #Cross Validation Object
            FoldCV = KFold(n_splits = 4, shuffle = True, random_state=12345).split(RaceID_List)
            for train_index, test_index in FoldCV :
                train_index, test_index = RaceID_List[train_index], RaceID_List[test_index]
                #Building Dataset
                X_train = X_copy.loc[X_copy.loc[:,'RARID'].isin(train_index), :]
                y_train = y.loc[y.loc[:,'RARID'].isin(train_index), :]
                X_test = X_copy.loc[X_copy.loc[:,'RARID'].isin(test_index), :]
                y_test = y.loc[y.loc[:,'RARID'].isin(test_index), :]

                model = self.model_class(probability = True, kernel = self.kernel, C = np.exp(C), gamma = np.exp(gamma))
                model = model.fit(X_train[X_train.columns[2:]], y_train['RESWL'])

                Prediction = X_test.loc[:,['RARID','HNAME']]
                Prediction.loc[:,'prediction'] = model.predict_proba(X_test.loc[:,X_test.columns[2:]])[:,1]
                #Scale prediction to sum to 1
                Prediction.loc[:,'prediction'] = Prediction.groupby('RARID')['prediction'].apply(lambda x : x / x.sum())

                Score.append(Kelly_Profit(Prediction, y_test, weight = Betting_Fraction))

            Score = np.mean(Score)

            return Score

        #Define BayesianOptimization instance
        optimizer = BayesianOptimization(f=SVC_fit_predict_score,
                                         pbounds=self.hyperparameter,
                                         verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
                                         random_state=1)

        #Load Hyperparamter Selection History Logs
        logger_name = self.model_name
        if os.path.exists(saved_models_path + logger_name+".json"):
            load_logs(optimizer, logs=[saved_models_path+logger_name+".json"])

        #Subscribe to Log Hyperparameter History
        logger = newJSONLogger(path=saved_models_path + logger_name+".json")
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

        #Count Initial Points
        initial_number_model_fitted = len(optimizer.space)

        #Search Default Space
        if initial_probe == True:
            optimizer.probe(params={'C' : 1, 'gamma' : -6}, lazy=False)

        #Loop over instances
        optimizer.maximize(init_points=inital_pts, n_iter=rounds)
        self.number_model_fitted = len(optimizer.res) - initial_number_model_fitted
        print("==================== %d Hyperparameters Models are fitted in %s hours ===================="
              %(self.number_model_fitted, (str(round((time.time() - start_time)/ (60*60), 2)))))

        best_model = optimizer.max
        print("The optimizer is now aware of {num_pts} points and the best result is {Result}.".format(Result = best_model['target'] ,
                                                                                                       num_pts = len(optimizer.space)))

        return None

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

======================= One Stage SVC Linear Wrapper =======================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class SVC_Linear_Wrapper(BaseEstimator, ClassifierMixin):

    """
    A sklearn-style wrapper for One Stage Linear Support Vector Classifier
    """

    def __init__(self, model_class, model_name, preprocessing, C) :
        self.model_class = model_class
        self.model_name = model_name
        self.preprocessing = preprocessing

        """
        Hyperparameter
        """
        self.kernel = 'linear'
        self.C = C
        #Hyperparameter Dictionary - Bounds
        self.hyperparameter = {'C' : C_Range}
        #Track Hyperparameter Selection
        self.number_model_fitted = 0

        return None


    def fit(self, X, y, preprocess = True):

        #Making a copy of X
        X_train = X.copy()

        #Apply Preprocessing Pipeline
        if preprocess == True :
            self.preprocessing = self.preprocessing()
            X_train = self.preprocessing.fit_transform(X_train)
        else:
            pass

        #Create Model Instance
        self.model = CalibratedClassifierCV(self.model_class(C = np.exp(self.C)), cv=5)

        #Slicing away [RARID, HNAME]
        X_train = X_train.loc[:,X_train.columns[2:]]
        y = y.loc[:,'RESWL']

        #Model Fitting
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        self.model.fit(X_train, y)
        self.fitted_model  = self.model
        warnings.filterwarnings("default", category=ConvergenceWarning)

        return None


    def predict(self, X, preprocess = True):

        #Making a copy of X
        X_test = X.copy()

        #Apply Preprocessing Pipeline
        if preprocess == True: X_test = self.preprocessing.transform(X_test)

        #Formatting into DataFrame
        Prediction = X_test.loc[:,['RARID','HNAME']]

        #Getting the probability of y=1
        Prediction.loc[:,self.model_name] = self.fitted_model.predict_proba(X_test[X_test.columns[2:]])[:,1]

        #Scale prediction to sum to 1
        Prediction.loc[:,self.model_name] = Prediction.groupby('RARID')[self.model_name].apply(lambda x : x / x.sum())

        return Prediction


    def predict_proba(self, X, preprocess = True):

        #Making a copy of X
        X_test = X.copy()

        #Apply Preprocessing Pipeline
        if preprocess == True: X_test = self.preprocessing.transform(X_test)

        #Formatting into DataFrame
        Prediction = X_test.loc[:,['RARID','HNAME']]

        #Getting the probability of y=1
        Prediction.loc[:,self.model_name] = self.fitted_model.predict_proba(X_test[X_test.columns[2:]])[:,1]

        #Scale prediction to sum to 1
        Prediction.loc[:,self.model_name] = Prediction.groupby('RARID')[self.model_name].apply(lambda x : x / x.sum())

        return Prediction


    def load_hyperparameters(self):

        """
        This method loads the best hyperparameters from the Hyperparameter_Selection History to the current instance
        """
        optimizer = BayesianOptimization(f=None,
                                         pbounds=self.hyperparameter,
                                         verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
                                         random_state=1)

        #Load Hyperparamter Selection History Logs
        logger_name = self.model_name
        if os.path.exists(saved_models_path + logger_name+".json"):
            load_logs(optimizer, logs=[saved_models_path + logger_name+".json"])

            #Save best hyperparameter
            best_model = optimizer.max
            self.C = best_model['params']['C']

            print("The optimizer is now aware of {num_pts} points and the best result is {Result}.".format(Result = best_model['target'] ,
                                                                                                           num_pts = len(optimizer.space)))

        else :
            print('No Hyperparameters was tested.')

        return None


    def hyperparameter_selection(self, X, y, inital_pts, rounds, initial_probe = True):

        #Create Timer
        start_time = time.time()

        #Making a copy of X
        X_copy = X.copy()

        #Apply Preprocessing Pipeline
        self.preprocessing = self.preprocessing()
        X_copy = self.preprocessing.fit_transform(X_copy)

        """
        Beyesian Optimization
        """
        #Get RaceID in Dataset
        RaceID_List = X_copy.loc[:,'RARID'].unique()

        #Create Function to Optimize
        def SVC_fit_predict_score(C):
            """
            Looping Over Cross Validation Folds
            """
            Score = []
            #Cross Validation Object
            FoldCV = KFold(n_splits = 4, shuffle = True, random_state=12345).split(RaceID_List)
            for train_index, test_index in FoldCV :
                train_index, test_index = RaceID_List[train_index], RaceID_List[test_index]
                #Building Dataset
                X_train = X_copy.loc[X_copy.loc[:,'RARID'].isin(train_index), :]
                y_train = y.loc[y.loc[:,'RARID'].isin(train_index), :]
                X_test = X_copy.loc[X_copy.loc[:,'RARID'].isin(test_index), :]
                y_test = y.loc[y.loc[:,'RARID'].isin(test_index), :]

                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                model = CalibratedClassifierCV(self.model_class(C = np.exp(C)), cv=5)
                model = model.fit(X_train[X_train.columns[2:]], y_train['RESWL'])

                Prediction = X_test.loc[:,['RARID','HNAME']]
                Prediction.loc[:,'prediction'] = model.predict_proba(X_test.loc[:,X_test.columns[2:]])[:,1]
                #Scale prediction to sum to 1
                Prediction.loc[:,'prediction'] = Prediction.groupby('RARID')['prediction'].apply(lambda x : x / x.sum())

                Score.append(Kelly_Profit(Prediction, y_test, weight = Betting_Fraction))

            Score = np.mean(Score)

            return Score

        #Define BayesianOptimization instance
        optimizer = BayesianOptimization(f=SVC_fit_predict_score,
                                         pbounds=self.hyperparameter,
                                         verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
                                         random_state=1)

        #Load Hyperparamter Selection History Logs
        logger_name = self.model_name
        if os.path.exists(saved_models_path +logger_name+".json"):
            load_logs(optimizer, logs=[saved_models_path + logger_name+".json"])

        #Subscribe to Log Hyperparameter History
        logger = newJSONLogger(path=saved_models_path +logger_name+".json")
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

        #Count Initial Points
        initial_number_model_fitted = len(optimizer.space)

        #Search Default Space
        if initial_probe == True:
            optimizer.probe(params={'C' : 1}, lazy=False)

        #Loop over instances
        optimizer.maximize(init_points=inital_pts, n_iter=rounds)
        self.number_model_fitted = len(optimizer.res) - initial_number_model_fitted
        print("==================== %d Hyperparameters Models are fitted in %s hours ===================="
              %(self.number_model_fitted, (str(round((time.time() - start_time)/ (60*60), 2)))))

        best_model = optimizer.max
        print("The optimizer is now aware of {num_pts} points and the best result is {Result}.".format(Result = best_model['target'] ,
                                                                                                       num_pts = len(optimizer.space)))

        return None

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

========================== One Stage SVR Wrapper ==========================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class SVR_Wrapper(BaseEstimator, ClassifierMixin):

    """
    A sklearn-style wrapper for One Stage Support Vector Regression
    """

    def __init__(self, model_class, model_name, preprocessing, kernel, C, gamma) :
        self.model_class = model_class
        self.model_name = model_name
        self.preprocessing = preprocessing

        #Hyperparameter
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        #Hyperparameter Dictionary - Bounds
        self.hyperparameter = {'C' : C_Range, 'gamma' : gamma_Range}
        #Track Hyperparameter Selection
        self.number_model_fitted = 0

        return None


    def fit(self, X, y, preprocess = True):

        #Making a copy of X
        X_train = X.copy()

        #Apply Preprocessing Pipeline
        if preprocess == True :
            self.preprocessing = self.preprocessing()
            X_train = self.preprocessing.fit_transform(X_train)
        else:
            #Save fitted preprocessing object passed in
            self.preprocessing = preprocess

        #Creating Model Instance
        self.model = self.model_class(kernel = self.kernel, C = np.exp(self.C), gamma = np.exp(self.gamma))

        #Slicing away [RARID, HNAME]
        X_train = X_train.loc[:,X_train.columns[2:]]
        y = y.loc[:,'RESFP']

        #Model Fitting
        self.model.fit(X_train, y)
        self.fitted_model  = self.model

        return None


    def predict(self, X, preprocess = True):

        #Making a copy of X
        X_test = X.copy()

        #Apply Preprocessing Pipeline
        if preprocess == True: X_test = self.preprocessing.transform(X_test)

        #Formatting into DataFrame
        Prediction = X.loc[:,['RARID','HNAME']]

        #Predicting Final Position
        Prediction.loc[:,self.model_name] = self.fitted_model.predict(X_test.loc[:,X_test.columns[2:]])

        #Scale prediction to sum to 1
        Prediction.loc[:,self.model_name] = Prediction.groupby('RARID')[self.model_name].apply(lambda x : x / x.sum())

        return Prediction


    def load_hyperparameters(self):

        """
        This method loads the best hyperparameters from the Hyperparameter_Selection History to the current instance
        """
        optimizer = BayesianOptimization(f=None,
                                         pbounds=self.hyperparameter,
                                         verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
                                         random_state=1)

        #Load Hyperparamter Selection History Logs
        logger_name = self.model_name
        if os.path.exists(saved_models_path + logger_name+".json"):
            load_logs(optimizer, logs=[saved_models_path + logger_name+".json"])

            #Save best hyperparameter
            best_model = optimizer.max
            self.gamma = best_model['params']['gamma']
            self.C = best_model['params']['C']

            print("The optimizer is now aware of {num_pts} points and the best result is {Result}.".format(Result = best_model['target'] ,
                                                                                                           num_pts = len(optimizer.space)))

        else :
            print('No Hyperparameters was tested.')

        return None


    def hyperparameter_selection(self, X, y, inital_pts, rounds, initial_probe = True):

        #Create Timer
        start_time = time.time()

        #Making a copy of X
        X_copy = X.copy()

        #Apply Preprocessing Pipeline
        self.preprocessing = self.preprocessing()
        X_copy = self.preprocessing.fit_transform(X_copy)

        """
        Beyesian Optimization
        """
        #Get RaceID in Dataset
        RaceID_List = X_copy.loc[:,'RARID'].unique()

        #Create Function to Optimize
        def SVR_fit_predict_score(C, gamma):
            """
            Looping Over Cross Validation Folds
            """
            Score = []
            #Cross Validation Object
            FoldCV = KFold(n_splits = 4, shuffle = True, random_state=12345).split(RaceID_List)
            for train_index, test_index in FoldCV :
                train_index, test_index = RaceID_List[train_index], RaceID_List[test_index]
                #Building Dataset
                X_train = X_copy.loc[X_copy.loc[:,'RARID'].isin(train_index), :]
                y_train = y.loc[y.loc[:,'RARID'].isin(train_index), :]
                X_test = X_copy.loc[X_copy.loc[:,'RARID'].isin(test_index), :]
                y_test = y.loc[y.loc[:,'RARID'].isin(test_index), :]

                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                model = CalibratedClassifierCV(self.model_class(C = np.exp(C)), cv=5)
                model = model.fit(X_train[X_train.columns[2:]], y_train.loc[:,'RESFP'])
                warnings.filterwarnings("default", category=ConvergenceWarning)

                Prediction = X_test.loc[:,['RARID','HNAME']]
                Prediction.loc[:,'prediction'] = model.predict(X_test.loc[:,X_test.columns[2:]])

                Score.append(Prediction_accuracy(Prediction, y_test))

            Score = np.mean(Score)

            return Score

        #Define BayesianOptimization instance
        optimizer = BayesianOptimization(f=SVR_fit_predict_score,
                                         pbounds=self.hyperparameter,
                                         verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
                                         random_state=1)

        #Load Hyperparamter Selection History Logs
        logger_name = self.model_name
        if os.path.exists(saved_models_path + logger_name+".json"):
            load_logs(optimizer, logs=[saved_models_path + logger_name+".json"])

        #Subscribe to Log Hyperparameter History
        logger = newJSONLogger(path=saved_models_path + logger_name+".json")
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

        #Count Initial Points
        initial_number_model_fitted = len(optimizer.space)

        #Search Default Space
        if initial_probe == True:
            optimizer.probe(params={'C' : 1, 'gamma' : 'scale'}, lazy=False)

        #Loop over instances
        optimizer.maximize(init_points=inital_pts, n_iter=rounds)
        self.number_model_fitted = len(optimizer.res) - initial_number_model_fitted
        print("==================== %d Hyperparameters Models are fitted in %s hours ===================="
              %(self.number_model_fitted, (str(round((time.time() - start_time)/ (60*60), 2)))))

        best_model = optimizer.max
        print("The optimizer is now aware of {num_pts} points and the best result is {Result}.".format(Result = best_model['target'] ,
                                                                                                       num_pts = len(optimizer.space)))

        return None


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

============================= SVC/CL Wrapper =============================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class SVC_CL_Wrapper(BaseEstimator, ClassifierMixin):

    """
    A sklearn-style wrapper for Two Stage SVM, CL Model
    """

    def __init__(self, model_class_1, model_class_2, model_name, preprocessing, kernel, C, gamma, ridge) :
        self.model_class_1 = model_class_1
        self.model_class_2 = model_class_2
        self.model_name = model_name
        self.preprocessing = preprocessing

        """
        Hyperparameter
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.ridge = ridge
        #Hyperparameter Dictionary - Bounds
        self.hyperparameter = {'C' : C_Range, 'gamma' : gamma_Range, 'ridge' : ridge_Range}
        #Track Hyperparameter Selection
        self.number_model_fitted = 0

        return None


    def fit(self, X, y, preprocess = True):

        #Making a copy of X
        X_train = X.copy()

        #Slicing Odds Columns
        self.other_col = [i for i in X_train.columns if i != Odds_col]
        X_Odds = X_train.loc[:,['RARID', 'HNAME', Odds_col]]
        X_Others = X_train.loc[:,self.other_col]

        #Apply Preprocessing Pipeline
        if preprocess == True :
            self.preprocessing = self.preprocessing()
            X_Others = self.preprocessing.fit_transform(X_Others)
        else:
            #Save fitted preprocessing object passed in
            self.preprocessing = preprocess

        #Redefine other cols
        self.other_col = list(X_Others.columns)

        #Join the two dataframes
        X_train = pd.merge(X_Odds, X_Others, on = ['RARID','HNAME'])

        #Get RaceID in Dataset
        RaceID_List = X_train.loc[:,'RARID'].unique()

        #Split into Stage 1 and Stage 2
        stage_1_index, stage_2_index = train_test_split(RaceID_List, test_size=0.5, random_state=12345)

        #Building Dataset
        X_Stage_1 = X_train.loc[X_train.loc[:,'RARID'].isin(stage_1_index), :]
        y_Stage_1 = y.loc[y.loc[:,'RARID'].isin(stage_1_index), :]
        X_Stage_2 = X_train.loc[X_train.loc[:,'RARID'].isin(stage_2_index), :]
        y_Stage_2 = y.loc[y.loc[:,'RARID'].isin(stage_2_index), :]

        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        """
        Stage 1
        """
        #Removing Odds from Stage 1
        X_Stage_1 = X_Stage_1.loc[:,self.other_col[2:]]
        y_Stage_1 = y_Stage_1.loc[:,'RESWL']

        self.model_Stage_1 = self.model_class_1(probability = True, kernel = self.kernel, C = np.exp(self.C), gamma = np.exp(self.gamma))
        self.model_Stage_1.fit(X_Stage_1, y_Stage_1)
        self.fitted_Stage_1 = self.model_Stage_1

        """
        Stage 2
        """
        #Create DataFrame for prediction
        X_predict_Stage_1 = X_Stage_2.loc[:,self.other_col]

        #Slice in Odds Columns
        X_Stage_2 = X_Stage_2.loc[:,['HNAME','RARID', Odds_col]]

        #Combining X and Y
        X_Stage_2 = X_Stage_2.merge(y_Stage_2.loc[:,['RARID','HNAME','RESWL']], on=['RARID','HNAME'])
        X_train.reset_index(inplace = True, drop = True)

        #Update Dataset with Stage 1 Prediction
        X_Stage_2.loc[:,'Fundamental_Probi'] = self.fitted_Stage_1.predict_proba(X_predict_Stage_1.loc[:,X_predict_Stage_1.columns[2:]])[:,1]

        #Scale prediction to sum to 1
        X_Stage_2.loc[:,'Fundamental_Probi'] = X_Stage_2.groupby('RARID')['Fundamental_Probi'].apply(lambda x : x / x.sum())
        X_Stage_2.loc[:,'Fundamental_Probi'] = np.log(X_Stage_2.loc[:,'Fundamental_Probi'])

        #Create specification dictionary
        model_specification_Stage_2 = OrderedDict()
        for variable in X_Stage_2.columns[2:]:
            model_specification_Stage_2[variable] = 'all_same'

        #Remove 'RESWL'
        model_specification_Stage_2.pop("RESWL")
        zeros_Stage_2 = np.zeros(len(model_specification_Stage_2))

        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        #Creating Model Instance
        self.model_Stage_2 = self.model_class_2(data = X_Stage_2,
                                                alt_id_col = 'HNAME',
                                                obs_id_col = 'RARID',
                                                choice_col = 'RESWL',
                                                specification = model_specification_Stage_2,
                                                model_type = 'MNL')

        self.model_Stage_2.fit_mle(zeros_Stage_2, print_res = False, ridge = self.ridge)

        warnings.filterwarnings("default", category=FutureWarning)
        warnings.filterwarnings("default", category=UserWarning)
        warnings.filterwarnings("default", category=RuntimeWarning)

        #Save Models
        self.fitted_Stage_2  = self.model_Stage_2

        #Feature Importance
        self.summary = self.fitted_Stage_2.get_statsmodels_summary()

        return None


    def predict(self, X, preprocess = True):

        """
        Prediction
        """
        #Making a copy of X
        X_test = X.copy()

        #Slicing Odds Columns
        if preprocess == True:
            X_Odds = X_test.loc[:,['RARID', 'HNAME', Odds_col]]
            X_Others = X_test.loc[:, [i for i in X_test.columns if i != Odds_col]]
            X_Others = self.preprocessing.transform(X_Others)
            X_test = pd.merge(X_Odds, X_Others, on = ['RARID','HNAME'])
        else:
            pass

        warnings.filterwarnings("ignore", category=FutureWarning)
        """
        Stage 1
        """
        #Removing Odds from Stage 1
        X_Stage_1 = X_test.loc[:,self.other_col[2:]]
        X_Stage_2 = X_test.loc[:,['HNAME','RARID', Odds_col]]
        X_Stage_2.loc[:,'Fundamental_Probi'] = self.fitted_Stage_1.predict_proba(X_Stage_1)[:,1]
        X_Stage_2.loc[:,'Fundamental_Probi'] = X_Stage_2.groupby('RARID')['Fundamental_Probi'].apply(lambda x : x / x.sum())
        X_Stage_2.loc[:,'Fundamental_Probi'] = np.log(X_Stage_2.loc[:,'Fundamental_Probi'])

        """
        Stage 2
        """
        Prediction =  X_test.loc[:,['RARID','HNAME']]
        Prediction.loc[:,self.model_name] = self.fitted_Stage_2.predict(X_Stage_2)
        warnings.filterwarnings("default", category=FutureWarning)

        return  Prediction


    def predict_proba(self, X, preprocess = True):

        """
        Prediction
        """
        #Making a copy of X
        X_test = X.copy()

        #Slicing Odds Columns
        if preprocess == True:
            X_Odds = X_test.loc[:,['RARID', 'HNAME', Odds_col]]
            X_Others = X_test.loc[:, [i for i in X_test.columns if i != Odds_col]]
            X_Others = self.preprocessing.transform(X_Others)
            X_test = pd.merge(X_Odds, X_Others, on = ['RARID','HNAME'])
        else:
            pass

        warnings.filterwarnings("ignore", category=FutureWarning)
        """
        Stage 1
        """
        #Removing Odds from Stage 1
        X_Stage_1 = X_test.loc[:,self.other_col[2:]]
        X_Stage_2 = X_test.loc[:,['HNAME','RARID', Odds_col]]
        X_Stage_2.loc[:,'Fundamental_Probi'] = self.fitted_Stage_1.predict_proba(X_Stage_1)[:,1]
        X_Stage_2.loc[:,'Fundamental_Probi'] = X_Stage_2.groupby('RARID')['Fundamental_Probi'].apply(lambda x : x / x.sum())
        X_Stage_2.loc[:,'Fundamental_Probi'] = np.log(X_Stage_2.loc[:,'Fundamental_Probi'])

        """
        Stage 2
        """
        Prediction =  X_test.loc[:,['RARID','HNAME']]
        Prediction.loc[:,self.model_name] = self.fitted_Stage_2.predict(X_Stage_2)
        warnings.filterwarnings("default", category=FutureWarning)

        return  Prediction


    def load_hyperparameters(self):

        """
        This method loads the best hyperparameters from the Hyperparameter_Selection History to the current instance
        """
        optimizer = BayesianOptimization(f=None,
                                         pbounds=self.hyperparameter,
                                         verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
                                         random_state=1)

        #Load Hyperparamter Selection History Logs
        logger_name = self.model_name
        if os.path.exists(saved_models_path + logger_name+".json"):
            load_logs(optimizer, logs=[saved_models_path + logger_name+".json"])

            #Save best hyperparameter
            best_model = optimizer.max
            self.C = best_model['params']['C']
            self.gamma = best_model['params']['gamma']
            self.ridge = best_model['params']['ridge']

            print("The optimizer is now aware of {num_pts} points and the best result is {Result}.".format(Result = best_model['target'] ,
                                                                                                           num_pts = len(optimizer.space)))

        else :
            print('No Hyperparameters was tested.')

        return None


    def hyperparameter_selection(self, X, y, inital_pts, rounds, initial_probe = True):

        #Create Timer
        start_time = time.time()

        #Making a copy of X
        X_copy = X.copy()

        #Slicing Odds Columns
        #Only apply mutate preprocessinging pipelines on non-Odds columns
        other_col = [i for i in X_copy.columns if i != Odds_col]
        X_Odds = X_copy.loc[:,['RARID', 'HNAME', Odds_col]]
        X_Others = X_copy.loc[:,other_col]

        #Apply Preprocessing Pipeline
        self.preprocessing = self.preprocessing()
        X_Others = self.preprocessing.fit_transform(X_Others)

        #Redefine other cols
        self.other_col = list(X_Others.columns)

        #Join the two dataframes
        X_copy = pd.merge(X_Odds, X_Others, on = ['RARID','HNAME'])

        #Get RaceID in Dataset
        RaceID_List = X_copy.loc[:,'RARID'].unique()

        """
        Beyesian Optimization
        """
        #Create FUnction to Optimize
        def SVC_CL_fit_predict_score(C, gamma, ridge):
            """
            Looping Over Cross Validation Folds
            """
            Score = []
            #Cross Validation Object
            FoldCV = KFold(n_splits = 4, shuffle = True, random_state=12345).split(RaceID_List)

            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            for train_index, test_index in FoldCV :
                train_index, test_index = RaceID_List[train_index], RaceID_List[test_index]
                #Building Dataset - Validation
                X_train = X_copy.loc[X_copy.loc[:,'RARID'].isin(train_index), :]
                y_train = y.loc[y.loc[:,'RARID'].isin(train_index), :]
                X_train = X_train.merge(y_train.loc[:,['RARID','HNAME','RESWL']], on=['RARID','HNAME'])
                X_train.reset_index(inplace=True, drop=True)

                X_test = X_copy.loc[X_copy.loc[:,'RARID'].isin(test_index), :]
                y_test = y.loc[y.loc[:,'RARID'].isin(test_index), :]

                #Building Dataset - Stages 1 and  Stage 2
                #Get RaceID in Dataset
                TrainID_List = X_train.loc[:,'RARID'].unique()
                stage_1_index, stage_2_index = train_test_split(TrainID_List, test_size=0.5, shuffle = True, random_state=12345)
                X_train_Stage_1 = X_train.loc[X_train.loc[:,'RARID'].isin(stage_1_index), self.other_col[2:]]
                y_train_Stage_1 = y.loc[y.loc[:,'RARID'].isin(stage_1_index), 'RESWL']
                X_predict_Stage_1 = X_train.loc[X_train.loc[:,'RARID'].isin(stage_2_index), self.other_col[2:]]
                X_train_Stage_2 = X_train.loc[X_train.loc[:,'RARID'].isin(stage_2_index),['HNAME', 'RARID', Odds_col, 'RESWL']]

                #Testing Dataset
                X_test_Stage_1 = X_test.loc[:,self.other_col[2:]]
                X_test_Stage_2 = X_test.loc[:,['HNAME','RARID', Odds_col]]

                """
                Training Stage 1
                """
                model_stage_1 = self.model_class_1(probability = True, kernel = self.kernel, C = np.exp(C), gamma = np.exp(gamma))
                model_stage_1 = model_stage_1.fit(X_train_Stage_1, y_train_Stage_1)

                """
                Training Stage 2
                """
                X_train_Stage_2.loc[:,'Fundamental_Probi'] = model_stage_1.predict_proba(X_predict_Stage_1)[:,1]
                X_train_Stage_2.loc[:,'Fundamental_Probi'] = X_train_Stage_2.groupby('RARID')['Fundamental_Probi'].apply(lambda x : x / x.sum())
                X_train_Stage_2.loc[:,'Fundamental_Probi'] = np.log(X_train_Stage_2.loc[:,'Fundamental_Probi'])

                #Create specification dictionary
                model_specification_Stage_2 = OrderedDict()
                for variable in X_train_Stage_2.columns[2:]:
                    model_specification_Stage_2[variable] = 'all_same'
                #Remove 'RESWL'
                model_specification_Stage_2.pop("RESWL")
                zeros_Stage_2 = np.zeros(len(model_specification_Stage_2))

                model_Stage_2 = self.model_class_2(data = X_train_Stage_2,
                                                   alt_id_col = 'HNAME',
                                                   obs_id_col = 'RARID',
                                                   choice_col = 'RESWL',
                                                   specification = model_specification_Stage_2,
                                                   model_type = 'MNL')

                model_Stage_2.fit_mle(zeros_Stage_2, print_res = False, ridge = ridge)

                """
                Prediction
                """
                #Stage 1
                X_test_Stage_2.loc[:,'Fundamental_Probi'] = model_stage_1.predict_proba(X_test_Stage_1)[:,1]
                X_test_Stage_2.loc[:,'Fundamental_Probi'] = X_test_Stage_2.groupby('RARID')['Fundamental_Probi'].apply(lambda x : x / x.sum())
                X_test_Stage_2.loc[:,'Fundamental_Probi'] = np.log(X_test_Stage_2.loc[:,'Fundamental_Probi'])

                #Stage 2
                Prediction = X_test.loc[:,['RARID','HNAME']]
                Prediction.loc[:,'prediction'] = model_Stage_2.predict(X_test_Stage_2)

                Score.append(Kelly_Profit(Prediction, y_test, weight = Betting_Fraction))

            Score = np.mean(Score)

            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            return Score

        #Define BayesianOptimization instance
        optimizer = BayesianOptimization(f=SVC_CL_fit_predict_score,
                                         pbounds=self.hyperparameter,
                                         verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
                                         random_state=1)

        #Load Hyperparamter Selection History Logs
        logger_name = self.model_name
        if os.path.exists(saved_models_path + logger_name+".json"):
            load_logs(optimizer, logs=[saved_models_path + logger_name+".json"])

        #Subscribe to Log Hyperparameter History
        logger = newJSONLogger(path=saved_models_path + logger_name+".json")
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

        #Count Initial Points
        initial_number_model_fitted = len(optimizer.space)

        #Search Default Space
        if initial_probe == True:
            optimizer.probe(params={'C' : 1, 'gamma' : -6, 'ridge' : 0}, lazy=False)

        #Loop over instances
        optimizer.maximize(init_points=inital_pts, n_iter=rounds)
        self.number_model_fitted = len(optimizer.res) - initial_number_model_fitted
        print("==================== %d Hyperparameters Models are fitted in %s hours ===================="
              %(self.number_model_fitted, (str(round((time.time() - start_time)/ (60*60), 2)))))

        best_model = optimizer.max
        print("The optimizer is now aware of {num_pts} points and the best result is {Result}.".format(Result = best_model['target'] ,
                                                                                                       num_pts = len(optimizer.space)))

        return None


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

============================= SVR/CL Wrapper =============================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class SVR_CL_Wrapper(BaseEstimator, ClassifierMixin):

    """
    A sklearn-style wrapper for Two Stage SVM, CL Model
    """

    def __init__(self, model_class_1, model_class_2, model_name, preprocessing, kernel, C, gamma, ridge) :
        self.model_class_1 = model_class_1
        self.model_class_2 = model_class_2
        self.model_name = model_name
        self.preprocessing = preprocessing

        """
        Hyperparameter
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.ridge = ridge
        #Hyperparameter Dictionary - Bounds
        self.hyperparameter = {'C' : C_Range, 'gamma' : gamma_Range, 'ridge' : ridge_Range}
        #Track Hyperparameter Selection
        self.number_model_fitted = 0

        return None


    def fit(self, X, y, preprocess = True):

        #Making a copy of X
        X_train = X.copy()

        #Slicing Odds Columns
        #Only apply mutate preprocessinging pipelines on non-Odds columns
        self.other_col = [i for i in X_train.columns if i != Odds_col]
        X_Odds = X_train.loc[:,['RARID', 'HNAME', Odds_col]]
        X_Others = X_train.loc[:,self.other_col]

        #Apply Preprocessing Pipeline
        if preprocess == True :
            self.preprocessing = self.preprocessing()
            X_Others = self.preprocessing.fit_transform(X_Others)
        else:
            #Save fitted preprocessing object passed in
            self.preprocessing = preprocess

        #Redefine other cols
        self.other_col = list(X_Others.columns)

        #Join the two dataframes
        X_train = pd.merge(X_Odds, X_Others, on = ['RARID','HNAME'])

        #Get RaceID in Dataset
        RaceID_List = X_train.loc[:,'RARID'].unique()

        #Split into Stage 1 and Stage 2
        stage_1_index, stage_2_index = train_test_split(RaceID_List, test_size=0.5, random_state=12345)

        #Building Dataset
        X_Stage_1 = X_train.loc[X.loc[:,'RARID'].isin(stage_1_index), :]
        y_Stage_1 = y.loc[y.loc[:,'RARID'].isin(stage_1_index), :]
        X_Stage_2 = X_train.loc[X.loc[:,'RARID'].isin(stage_2_index), :]
        y_Stage_2 = y.loc[y.loc[:,'RARID'].isin(stage_2_index), :]

        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        """
        Stage 1
        """
        #Removing Odds from Stage 1
        X_Stage_1 = X_Stage_1.loc[:,self.other_col[2:]]
        y_Stage_1 = y_Stage_1.loc[:,'RESFP']

        self.model_Stage_1 = self.model_class_1(kernel = self.kernel, C = np.exp(self.C), gamma = np.exp(self.gamma))
        self.model_Stage_1.fit(X_Stage_1, y_Stage_1)
        self.fitted_Stage_1 = self.model_Stage_1

        """
        Stage 2
        """
        #Create DataFrame for prediction
        X_predict_Stage_1 = X_Stage_2.loc[:,self.other_col]

        #Slice in Odds Columns
        X_Stage_2 = X_Stage_2.loc[:,['HNAME','RARID',Odds_col]]

        #Combining X and Y
        X_Stage_2 = X_Stage_2.merge(y_Stage_2.loc[:,['RARID','HNAME','RESWL']], on=['RARID','HNAME'])
        X_train.reset_index(inplace = True, drop = True)

        #Update Dataset with Stage 1 Prediction
        X_Stage_2.loc[:,'Finishing_Position'] = self.fitted_Stage_1.predict(X_predict_Stage_1.loc[:,X_predict_Stage_1.columns[2:]])

        #Scale prediction to sum to 1
        X_Stage_2.loc[:,'Finishing_Position'] = X_Stage_2.groupby('RARID')['Finishing_Position'].apply(lambda x : x / x.sum())
        X_Stage_2.loc[:,'Finishing_Position'] = np.log(X_Stage_2.loc[:,'Finishing_Position'])

        #Create specification dictionary
        model_specification_Stage_2 = OrderedDict()
        for variable in X_Stage_2.columns[2:]:
            model_specification_Stage_2[variable] = 'all_same'
        #Remove 'RESWL'
        model_specification_Stage_2.pop("RESWL")
        zeros_Stage_2 = np.zeros(len(model_specification_Stage_2))

        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        #Creating Model Instance
        self.model_Stage_2 = self.model_class_2(data = X_Stage_2,
                                                alt_id_col = 'HNAME',
                                                obs_id_col = 'RARID',
                                                choice_col = 'RESWL',
                                                specification = model_specification_Stage_2,
                                                model_type = 'MNL')

        self.model_Stage_2.fit_mle(zeros_Stage_2, print_res = False, ridge = self.ridge)

        warnings.filterwarnings("default", category=FutureWarning)
        warnings.filterwarnings("default", category=UserWarning)
        warnings.filterwarnings("default", category=RuntimeWarning)

        #Save Models
        self.fitted_Stage_2  = self.model_Stage_2

        #Feature Importance
        self.summary = self.fitted_Stage_2.get_statsmodels_summary()

        return None


    def predict(self, X, preprocess = True):

        """
        Prediction
        """
        #Making a copy of X
        X_test = X.copy()

        #Apply Preprocessing Pipeline
        if preprocess == True:
            X_Odds = X_test.loc[:,['RARID', 'HNAME', Odds_col]]
            X_Others = X_test.loc[:, [i for i in X_test.columns if i != Odds_col]]
            X_Others = self.preprocessing.transform(X_Others)
            X_test = pd.merge(X_Odds, X_Others, on = ['RARID','HNAME'])
        else:
            pass

        warnings.filterwarnings("ignore", category=FutureWarning)
        """
        Stage 1
        """
        #Removing Odds from Stage 1
        X_Stage_1 = X_test.loc[:,self.other_col[2:]]
        X_Stage_2 = X_test.loc[:,['HNAME','RARID', Odds_col]]
        X_Stage_2.loc[:,'Finishing_Position'] = self.fitted_Stage_1.predict(X_Stage_1)
        X_Stage_2.loc[:,'Finishing_Position'] = X_Stage_2.groupby('RARID')['Finishing_Position'].apply(lambda x : x / x.sum())
        X_Stage_2.loc[:,'Finishing_Position'] = np.log(X_Stage_2.loc[:,'Finishing_Position'])

        """
        Stage 2
        """
        Prediction =  X_test.loc[:,['RARID','HNAME']]
        Prediction.loc[:,self.model_name] = self.fitted_Stage_2.predict(X_Stage_2)

        warnings.filterwarnings("default", category=FutureWarning)

        return  Prediction


    def predict_proba(self, X, preprocess = True):

        """
        Prediction
        """
        #Making a copy of X
        X_test = X.copy()

        #Apply Preprocessing Pipeline
        if preprocess == True:
            X_Odds = X_test.loc[:,['RARID', 'HNAME', Odds_col]]
            X_Others = X_test.loc[:, [i for i in X_test.columns if i != Odds_col]]
            X_Others = self.preprocessing.transform(X_Others)
            X_test = pd.merge(X_Odds, X_Others, on = ['RARID','HNAME'])
        else:
            pass

        warnings.filterwarnings("ignore", category=FutureWarning)
        """
        Stage 1
        """
        #Removing Odds from Stage 1
        X_Stage_1 = X_test.loc[:,self.other_col[2:]]
        X_Stage_2 = X_test.loc[:,['HNAME','RARID', Odds_col]]
        X_Stage_2.loc[:,'Finishing_Position'] = self.fitted_Stage_1.predict(X_Stage_1)
        X_Stage_2.loc[:,'Finishing_Position'] = X_Stage_2.groupby('RARID')['Finishing_Position'].apply(lambda x : x / x.sum())
        X_Stage_2.loc[:,'Finishing_Position'] = np.log(X_Stage_2.loc[:,'Finishing_Position'])

        """
        Stage 2
        """
        Prediction =  X_test.loc[:,['RARID','HNAME']]
        Prediction.loc[:,self.model_name] = self.fitted_Stage_2.predict(X_Stage_2)

        warnings.filterwarnings("default", category=FutureWarning)

        return  Prediction


    def load_hyperparameters(self):

        """
        This method loads the best hyperparameters from the Hyperparameter_Selection History to the current instance
        """
        optimizer = BayesianOptimization(f=None,
                                         pbounds=self.hyperparameter,
                                         verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
                                         random_state=1)

        #Load Hyperparamter Selection History Logs
        logger_name = self.model_name
        if os.path.exists(saved_models_path + logger_name+".json"):
            load_logs(optimizer, logs=[saved_models_path + logger_name+".json"])

            #Save best hyperparameter
            best_model = optimizer.max
            self.C = best_model['params']['C']
            self.gamma = best_model['params']['gamma']
            self.ridge = best_model['params']['ridge']

            print("The optimizer is now aware of {num_pts} points and the best result is {Result}.".format(Result = best_model['target'] ,
                                                                                                           num_pts = len(optimizer.space)))

        else :
            print('No Hyperparameters was tested.')

        return None


    def hyperparameter_selection(self, X, y, inital_pts, rounds, initial_probe = True):

        #Create Timer
        start_time = time.time()

        #Making a copy of X
        X_copy = X.copy()

        #Slicing Odds Columns
        #Only apply mutate preprocessinging pipelines on non-Odds columns
        other_col = [i for i in X_copy.columns if i != Odds_col]
        X_Odds = X_copy.loc[:,['RARID', 'HNAME', Odds_col]]
        X_Others = X_copy.loc[:,other_col]

        #Apply Preprocessing Pipeline
        self.preprocessing = self.preprocessing()
        X_Others = self.preprocessing.fit_transform(X_Others)

        #Redefine other cols
        self.other_col = list(X_Others.columns)

        #Join the two dataframes
        X_copy = pd.merge(X_Odds, X_Others, on = ['RARID','HNAME'])

        #Get RaceID in Dataset
        RaceID_List = X_copy.loc[:,'RARID'].unique()

        """
        Beyesian Optimization
        """
        #Create Function to Optimize
        def SVR_CL_fit_predict_score(C, gamma, ridge):
            """
            Looping Over Cross Validation Folds
            """
            Score = []
            #Cross Validation Object
            FoldCV = KFold(n_splits = 4, shuffle = True, random_state=12345).split(RaceID_List)

            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            for train_index, test_index in FoldCV :
                train_index, test_index = RaceID_List[train_index], RaceID_List[test_index]

                #Building Dataset - Validation
                X_train = X_copy.loc[X_copy.loc[:,'RARID'].isin(train_index), :]
                y_train = y.loc[y.loc[:,'RARID'].isin(train_index), :]
                X_train = X_train.merge(y_train.loc[:,['RARID','HNAME','RESFP','RESWL']], on=['RARID','HNAME'])
                X_train.reset_index(inplace=True, drop=True)

                X_test = X_copy.loc[X_copy.loc[:,'RARID'].isin(test_index), :]
                y_test = y.loc[y.loc[:,'RARID'].isin(test_index), :]

                #Building Dataset - Stages 1 and  Stage 2
                #Get RaceID in Dataset
                TrainID_List = X_train.loc[:,'RARID'].unique()
                stage_1_index, stage_2_index = train_test_split(TrainID_List, test_size=0.5, shuffle = True, random_state=12345)
                X_train_Stage_1 = X_train.loc[X_train.loc[:,'RARID'].isin(stage_1_index), self.other_col[2:]]
                y_train_Stage_1 = y.loc[y.loc[:,'RARID'].isin(stage_1_index), 'RESFP']
                X_predict_Stage_1 = X_train.loc[X_train.loc[:,'RARID'].isin(stage_2_index), self.other_col[2:]]
                X_train_Stage_2 = X_train.loc[X_train.loc[:,'RARID'].isin(stage_2_index),['HNAME', 'RARID', Odds_col, 'RESWL']]

                #Testing Dataset
                X_test_Stage_1 = X_test.loc[:,self.other_col[2:]]
                X_test_Stage_2 = X_test.loc[:,['HNAME','RARID', Odds_col]]

                """
                Training Stage 1
                """
                model_stage_1 = self.model_class_1(kernel = self.kernel, C = np.exp(C), gamma = np.exp(gamma))
                model_stage_1 = model_stage_1.fit(X_train_Stage_1, y_train_Stage_1)

                """
                Training Stage 2
                """
                X_train_Stage_2.loc[:,'Finishing_Position'] = model_stage_1.predict(X_predict_Stage_1)
                X_train_Stage_2.loc[:,'Finishing_Position'] = X_train_Stage_2.groupby('RARID')['Finishing_Position'].apply(lambda x : x / x.sum())
                X_train_Stage_2.loc[:,'Finishing_Position'] = np.log(X_train_Stage_2.loc[:,'Finishing_Position'])

                #Create specification dictionary
                model_specification_Stage_2 = OrderedDict()
                for variable in X_train_Stage_2.columns[2:]:
                    model_specification_Stage_2[variable] = 'all_same'
                #Remove 'RESWL'
                model_specification_Stage_2.pop("RESWL")
                zeros_Stage_2 = np.zeros(len(model_specification_Stage_2))

                model_Stage_2 = self.model_class_2(data = X_train_Stage_2,
                                                   alt_id_col = 'HNAME',
                                                   obs_id_col = 'RARID',
                                                   choice_col = 'RESWL',
                                                   specification = model_specification_Stage_2,
                                                   model_type = 'MNL')

                model_Stage_2.fit_mle(zeros_Stage_2, print_res = False, ridge = ridge)

                """
                Prediction
                """
                #Stage 1
                X_test_Stage_2.loc[:,'Finishing_Position'] = model_stage_1.predict(X_test_Stage_1)
                X_test_Stage_2.loc[:,'Finishing_Position'] = X_test_Stage_2.groupby('RARID')['Finishing_Position'].apply(lambda x : x / x.sum())
                X_test_Stage_2.loc[:,'Finishing_Position'] = np.log(X_test_Stage_2.loc[:,'Finishing_Position'])

                #Stage 2
                Prediction = X_test.loc[:,['RARID','HNAME']]
                Prediction.loc[:,'prediction'] = model_Stage_2.predict(X_test_Stage_2)

                Score.append(Kelly_Profit(Prediction, y_test, weight = Betting_Fraction))

            Score = np.mean(Score)

            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            return Score

        #Define BayesianOptimization instance
        optimizer = BayesianOptimization(f=SVR_CL_fit_predict_score,
                                         pbounds=self.hyperparameter,
                                         verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
                                         random_state=1)

        #Load Hyperparamter Selection History Logs
        logger_name = self.model_name
        if os.path.exists(saved_models_path + logger_name+".json"):
            load_logs(optimizer, logs=[saved_models_path + logger_name+".json"])

        #Subscribe to Log Hyperparameter History
        logger = newJSONLogger(path=saved_models_path + logger_name+".json")
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

        #Count Initial Points
        initial_number_model_fitted = len(optimizer.space)

        #Search Default Space
        if initial_probe == True:
            optimizer.probe(params={'C' : 1, 'gamma' : -6, 'ridge' : 0}, lazy=False)

        #Loop over instances
        optimizer.maximize(init_points=inital_pts, n_iter=rounds)
        self.number_model_fitted = len(optimizer.res) - initial_number_model_fitted
        print("==================== %d Hyperparameters Models are fitted in %s hours ===================="
              %(self.number_model_fitted, (str(round((time.time() - start_time)/ (60*60), 2)))))

        best_model = optimizer.max
        print("The optimizer is now aware of {num_pts} points and the best result is {Result}.".format(Result = best_model['target'] ,
                                                                                                       num_pts = len(optimizer.space)))

        return None


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

============================== SVC_RBF Model ==============================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def SVC_RBF(model_name = 'SVC_RBF Model', preprocessing = Raw, C = 1, gamma=1):

    """
    Support Vector Classifier with RBF Kernel
    Within Race competition is not considered
    """

    model = SVC_RBF_Wrapper(model_class=SVC, model_name=model_name, preprocessing=preprocessing, kernel='rbf', C=C, gamma = gamma)

    return model

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

============================= SVC_Linear Model =============================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def SVC_Linear(model_name = 'SVC_Linear Model', preprocessing = Raw, C = 1):

    """
    Support Vector Classifier with Lienar Kernel
    Within Race competition is not considered
    """

    model = SVC_Linear_Wrapper(model_class=LinearSVC, model_name=model_name, preprocessing=preprocessing, C=C)

    return model

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

============================== SVR_RBF Model ==============================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def SVR_RBF(model_name = 'SVR_RBF Model', preprocessing = Raw, C = 1, gamma = 1):

    """
    Support Vector Regression
    Predicting Finishing Position
    """

    model = SVR_Wrapper(model_class=SVR, model_name=model_name, preprocessing=preprocessing, kernel='linear', C=C, gamma = gamma)

    return model

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

============================== SVC_CL Model ==============================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def SVC_CL(model_name = 'SVC_CL Model', preprocessing = Raw, C = 1, gamma = 1 , ridge=0):

    """
    Stage 1 : Support Vector Classifier with RBF Kernel
    Stage 2 : Conditional Logistic Regression with Ridge Penalty
    """

    model = SVC_CL_Wrapper(model_class_1=SVC, model_class_2=pylogit.create_choice_model, model_name=model_name,
                           preprocessing=preprocessing, kernel='rbf', C=C, gamma=gamma, ridge=ridge)

    return model

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

============================== SVR_CL Model ==============================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def SVR_CL(model_name = 'SVR_CL Model', preprocessing = Raw, C = 1, gamma = 1, ridge=0):

    """
    Stage 1 : Support Vector Regression with RBF Kernal on Finishing Position
    Stage 2 : Conditional Logistic Regression with Ridge Penalty
    """

    model = SVR_CL_Wrapper(model_class_1=SVR, model_class_2=pylogit.create_choice_model, model_name=model_name,
                           preprocessing=preprocessing, kernel='rbf', C=C, gamma=gamma, ridge=ridge)

    return model

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

============================ Model Dictionary ============================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

SVM_Dict = {'SVC_RBF': SVC_RBF,
            'SVC_Linear' : SVC_Linear,
            'SVR_RBF' : SVR_RBF,
            'SVC_CL' : SVC_CL,
            'SVR_CL' : SVR_CL}
