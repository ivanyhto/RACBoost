#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ivan
"""

"""
Tree Based Models
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
from xgboost import XGBClassifier, XGBRegressor
from sklearn.base import BaseEstimator, ClassifierMixin
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
from pyhorse.Data_Preprocessing import Raw, Normalise_Race
from sklearn.model_selection import train_test_split, KFold
from catboost import Pool, CatBoostClassifier, CatBoostRegressor
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
ridge_Range = (0, 26)

#XGBoost Parameters
XGB_max_depth = (2, 50)
XGB_gamma = (0, 9)
XGB_n_estimators = (20, 1000)
XGB_learning_rate = (0.01, 0.3)
XGB_subsample = (0.5, 1)
XGB_colsample_bytree = (0.1, 1)
XGB_colsample_bylevel = (0.1, 1)
XGB_reg_lambda = (1e-9, 1000)
XGB_reg_alpha = (1e-9, 1.0)
XGB_min_child_weight = (0,10)
XGB_scale_pos_weight = (1e-6, 500)

#LightGBM Parameters
LGB_learning_rate = (0.01, 0.5)
LGB_num_leaves = (2, 500)
LGB_max_depth = (0, 500)
LGB_min_child_samples = (0, 200)
LGB_subsample = (0.01, 1.0)
LGB_bagging_freq = (0, 100)
LGB_colsample_bytree = (0.01, 1.0)
LGB_min_child_weight = (0, 10)
LGB_subsample_for_bin = (100000, 500000)
LGB_reg_lambda = (1e-9, 1000)
LGB_reg_alpha = (1e-9, 1.0)
LGB_scale_pos_weight = (1e-6, 500)
LGB_n_estimators = (10, 10000)

#Catboost Parameters
CGB_iterations = (50,1000)
CGB_depth = (4,10)
CGB_learning_rate = (0.01, 0.5)
CGB_random_strength = (1e-9, 10)
CGB_bagging_temperature = (0.0, 1.0)
CGB_l2_leaf_reg = (2,30)
CGB_scale_pos_weight = (0.01, 1.0)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

=============================== Test Cases ===============================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# y_pred.iloc[:,-1].hist(bins=100)
# y_true = y.copy()
# print(Kelly_Profit(y_pred, y_true, weight = 0.5))
# abc = Kelly_Profit(y_pred, y_true, weight = 0.5, get_history = True)
# abc.plot.line()

def _test_cases():
    from pyhorse import Dataset_Creation
    X, y = Dataset_Creation.Dataset_Extraction(Dataset_Creation.Get_RaceID(['2013','2014', '2015', '2016', '2017']))
    X, y = Dataset_Creation.Dataset_Extraction(Dataset_Creation.Get_RaceID(['2018']))

    XGBoost_Class_Model = XGBoost_Class(preprocessing = Normalise_Race)
    XGBoost_Class_Model.fit(X, y)
    y_pred = XGBoost_Class_Model.predict(X)
    XGBoost_Class_Model.summary
    XGBoost_Class_Model.hyperparameter_selection(X,y, 1, 1, initial_probe = True)
    XGBoost_Class_Model.load_hyperparameters()

    XGBoost_Reg_Model = XGBoost_Reg(preprocessing = Normalise_Race)
    XGBoost_Reg_Model.fit(X, y)
    y_pred = XGBoost_Reg_Model.predict(X)
    XGBoost_Reg_Model.summary
    XGBoost_Reg_Model.hyperparameter_selection(X,y, 0, 0, initial_probe = True)
    XGBoost_Reg_Model.load_hyperparameters()

    XGBoost_Class_CL_Model = XGBoost_Class_CL(preprocessing = Normalise_Race)
    XGBoost_Class_CL_Model.fit(X, y)
    y_pred = XGBoost_Class_CL_Model.predict(X)
    XGBoost_Class_CL_Model.summary
    XGBoost_Class_CL_Model.hyperparameter_selection(X,y, 0, 0, initial_probe = True)
    XGBoost_Class_CL_Model.load_hyperparameters()

    XGBoost_Reg_CL_Model = XGBoost_Reg_CL(preprocessing = Normalise_Race)
    XGBoost_Reg_CL_Model.fit(X, y)
    y_pred = XGBoost_Reg_CL_Model.predict(X)
    XGBoost_Reg_CL_Model.summary
    XGBoost_Reg_CL_Model.hyperparameter_selection(X,y, 0, 0, initial_probe = True)
    XGBoost_Reg_CL_Model.load_hyperparameters()

    LightGBM_Class_Model = LightGBM_Class(preprocessing = Raw)
    LightGBM_Class_Model.fit(X, y)
    y_pred = LightGBM_Class_Model.predict(X)
    LightGBM_Class_Model.summary
    LightGBM_Class_Model.hyperparameter_selection(X,y, 1, 0, initial_probe = True)
    LightGBM_Class_Model.load_hyperparameters()

    LightGBM_Reg_Model = LightGBM_Reg(preprocessing = Raw)
    LightGBM_Reg_Model.fit(X, y)
    y_pred = LightGBM_Reg_Model.predict(X)
    LightGBM_Reg_Model.summary
    LightGBM_Reg_Model.hyperparameter_selection(X,y, 0, 0, initial_probe = True)
    LightGBM_Reg_Model.load_hyperparameters()

    LightGBM_Class_CL_Model = LightGBM_Class_CL(preprocessing = Raw)
    LightGBM_Class_CL_Model.fit(X, y)
    y_pred = LightGBM_Class_CL_Model.predict(X)
    LightGBM_Class_CL_Model.summary
    LightGBM_Class_CL_Model.hyperparameter_selection(X,y, 0, 0, initial_probe = True)
    LightGBM_Class_CL_Model.load_hyperparameters()

    LightGBM_Reg_CL_Model = LightGBM_Reg_CL(preprocessing = Raw)
    LightGBM_Reg_CL_Model.fit(X, y)
    y_pred = LightGBM_Reg_CL_Model.predict(X)
    LightGBM_Reg_CL_Model.summary
    LightGBM_Reg_CL_Model.hyperparameter_selection(X,y, 0, 0, initial_probe = True)
    LightGBM_Reg_CL_Model.load_hyperparameters()

    Catboost_Class_Model = Catboost_Class(preprocessing = Normalise_Race)
    Catboost_Class_Model.fit(X, y)
    y_pred = Catboost_Class_Model.predict(X)
    Catboost_Class_Model.summary
    Catboost_Class_Model.hyperparameter_selection(X,y, 0, 0, initial_probe = True)
    Catboost_Class_Model.load_hyperparameters()

    Catboost_Reg_Model = Catboost_Reg(preprocessing = Normalise_Race)
    Catboost_Reg_Model.fit(X, y)
    y_pred = Catboost_Reg_Model.predict(X)
    Catboost_Reg_Model.summary.sort_values()
    Catboost_Reg_Model.hyperparameter_selection(X,y, 0, 0, initial_probe = True)
    Catboost_Reg_Model.load_hyperparameters()

    Catboost_Class_CL_Model = Catboost_Class_CL(preprocessing = Normalise_Race)
    Catboost_Class_CL_Model.fit(X, y)
    y_pred = Catboost_Class_CL_Model.predict(X)
    Catboost_Class_CL_Model.summary
    Catboost_Class_CL_Model.hyperparameter_selection(X,y, 0, 0, initial_probe = True)
    Catboost_Class_CL_Model.load_hyperparameters()

    Catboost_Reg_CL_Model = Catboost_Reg_CL(preprocessing = Normalise_Race)
    Catboost_Reg_CL_Model.fit(X, y)
    y_pred = Catboost_Reg_CL_Model.predict(X)
    Catboost_Reg_CL_Model.summary
    Catboost_Reg_CL_Model.hyperparameter_selection(X,y, 0, 0, initial_probe = True)
    Catboost_Reg_CL_Model.load_hyperparameters()

    return None

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

================== One Stage XGBoost Classifier Wrapper ==================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class XGBoost_Class_Wrapper(BaseEstimator, ClassifierMixin):

    """
    A sklearn-style wrapper for One Stage Extreme Gradient Boosting Classifier
    """

    def __init__(self, model_class, model_name, preprocessing, max_depth, gamma, n_estimators, learning_rate, subsample, colsample_bytree,
                 colsample_bylevel, reg_lambda, reg_alpha, min_child_weight, scale_pos_weight):
        self.model_class = model_class
        self.model_name = model_name
        self.preprocessing = preprocessing

        """
        Hyperparameter
        """
        self.max_depth = max_depth
        self.gamma = gamma
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.min_child_weight = min_child_weight
        self.scale_pos_weight = scale_pos_weight
        #Hyperparameter Dictionary - Bounds
        self.hyperparameter = {'max_depth' : XGB_max_depth, 'gamma' : XGB_gamma,
                               'n_estimators' : XGB_n_estimators, 'learning_rate' : XGB_learning_rate,
                               'subsample' : XGB_subsample, 'colsample_bytree' : XGB_colsample_bytree,
                               'colsample_bylevel' : XGB_colsample_bylevel, 'reg_lambda' : XGB_reg_lambda,
                               'reg_alpha' : XGB_reg_alpha, 'min_child_weight' : XGB_min_child_weight, 'scale_pos_weight' : XGB_scale_pos_weight}
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
        self.model = self.model_class(verbosity = 1,
                                      objective = 'multi:softprob', num_class=2, booster = 'gbtree', max_delta_step = 1,
                                      max_depth = int(self.max_depth), gamma = self.gamma, n_estimators = int(self.n_estimators),
                                      learning_rate = self.learning_rate, subsample = self.subsample, colsample_bytree = self.colsample_bytree,
                                      colsample_bylevel = self.colsample_bylevel, reg_lambda = self.reg_lambda, reg_alpha = self.reg_alpha,
                                      min_child_weight = self.min_child_weight, scale_pos_weight = self.scale_pos_weight)

        #Slicing away [RARID, HNAME]
        X_train = X_train.loc[:,X_train.columns[2:]]
        y = y.loc[:,'RESWL']

        #Model Fitting
        self.model.fit(X_train, y)
        self.fitted_model  = self.model

        #Feature Importance
        self.summary = pd.DataFrame(self.fitted_model.feature_importances_, X_train.columns, columns = ['importance'])

        return None


    def predict(self, X, preprocess = True):

        #Making a copy of X
        X_test = X.copy()

        #Apply Preprocessing Pipeline
        if preprocess == True: X_test = self.preprocessing.transform(X_test)

        #Formatting into DataFrame
        Prediction = X_test.loc[:,['RARID', 'HNAME']]

        #Getting the probability of y=1
        Prediction[self.model_name] = self.fitted_model.predict_proba(X_test[X_test.columns[2:]])[:,1]

        #Scale prediction to sum to 1
        Prediction[self.model_name] = Prediction.groupby('RARID')[self.model_name].apply(lambda x : x / x.sum())

        return Prediction


    def predict_proba(self, X, preprocess = True):

        #Making a copy of X
        X_test = X.copy()

        #Apply Preprocessing Pipeline
        if preprocess == True: X_test = self.preprocessing.transform(X_test)

        #Formatting into DataFrame
        Prediction = X_test.loc[:,['RARID', 'HNAME']]

        #Getting the probability of y=1
        Prediction[self.model_name] = self.fitted_model.predict_proba(X_test[X_test.columns[2:]])[:,1]

        #Scale prediction to sum to 1
        Prediction[self.model_name] = Prediction.groupby('RARID')[self.model_name].apply(lambda x : x / x.sum())

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
            self.max_depth = best_model['params']['max_depth']
            self.gamma = best_model['params']['gamma']
            self.n_estimators = best_model['params']['n_estimators']
            self.learning_rate = best_model['params']['learning_rate']
            self.subsample = best_model['params']['subsample']
            self.colsample_bytree = best_model['params']['colsample_bytree']
            self.colsample_bylevel = best_model['params']['colsample_bylevel']
            self.reg_lambda = best_model['params']['reg_lambda']
            self.reg_alpha = best_model['params']['reg_alpha']
            self.min_child_weight = best_model['params']['min_child_weight']
            self.scale_pos_weight = best_model['params']['scale_pos_weight']

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
        def XGBoost_fit_predict_score(max_depth, gamma, n_estimators, learning_rate, subsample, colsample_bytree, colsample_bylevel, reg_lambda,
                                      reg_alpha, min_child_weight, scale_pos_weight):

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

                model = self.model_class(verbosity = 1,
                                         objective = 'multi:softprob', num_class=2, booster = 'gbtree', max_delta_step = 1,
                                         max_depth = int(max_depth), gamma = gamma, n_estimators = int(n_estimators),
                                         learning_rate = learning_rate, subsample = subsample, colsample_bytree = colsample_bytree,
                                         colsample_bylevel = colsample_bylevel, reg_lambda = reg_lambda, reg_alpha = reg_alpha,
                                         min_child_weight = min_child_weight, scale_pos_weight = scale_pos_weight)
                model = model.fit(X_train.loc[:,X_train.columns[2:]], y_train.loc[:,'RESWL'])

                Prediction = X_test.loc[:,['RARID','HNAME']]
                Prediction.loc[:,'prediction'] = model.predict_proba(X_test.loc[:,X_test.columns[2:]])[:,1]
                #Scale prediction to sum to 1
                Prediction.loc[:,'prediction'] = Prediction.groupby('RARID')['prediction'].apply(lambda x : x / x.sum())
                Score.append(Kelly_Profit(Prediction, y_test, weight = Betting_Fraction))

            Score = np.mean(Score)

            return Score

        #Define BayesianOptimization instance
        optimizer = BayesianOptimization(f=XGBoost_fit_predict_score,
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
            optimizer.probe(params={'max_depth' : 6,'gamma' : 0,'n_estimators' : 500,'learning_rate' : 0.3,'subsample' : 1,
                                    'colsample_bytree' : 1,'colsample_bylevel' : 1,'reg_lambda' : 1,'reg_alpha' : 0,
                                    'min_child_weight' : 1,'scale_pos_weight' : 14}, lazy=False)

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

=================== One Stage XGBoost Regresor Wrapper ===================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class XGBoost_Reg_Wrapper(BaseEstimator, ClassifierMixin):

    """
    A sklearn-style wrapper for One Stage Extreme Gradient Boosting Regressor
    """

    def __init__(self, model_class, model_name, preprocessing, max_depth, gamma, n_estimators, learning_rate, subsample, colsample_bytree,
                 colsample_bylevel, reg_lambda, reg_alpha, min_child_weight, scale_pos_weight):
        self.model_class = model_class
        self.model_name = model_name
        self.preprocessing = preprocessing

        """
        Hyperparameter
        """
        self.max_depth = max_depth
        self.gamma = gamma
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.min_child_weight = min_child_weight
        self.scale_pos_weight = scale_pos_weight

        #Hyperparameter Dictionary - Bounds
        self.hyperparameter = {'max_depth' : XGB_max_depth, 'gamma' : XGB_gamma,
                               'n_estimators' : XGB_n_estimators, 'learning_rate' : XGB_learning_rate,
                               'subsample' : XGB_subsample, 'colsample_bytree' : XGB_colsample_bytree,
                               'colsample_bylevel' : XGB_colsample_bylevel, 'reg_lambda' : XGB_reg_lambda,
                               'reg_alpha' : XGB_reg_alpha, 'min_child_weight' : XGB_min_child_weight, 'scale_pos_weight' : XGB_scale_pos_weight}
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
        self.model = self.model_class(verbosity = 1,
                                      objective = 'rank:ndcg', booster = 'gbtree', max_delta_step = 1,
                                      max_depth = int(self.max_depth), gamma = self.gamma, n_estimators = int(self.n_estimators),
                                      learning_rate = self.learning_rate, subsample = self.subsample, colsample_bytree = self.colsample_bytree,
                                      colsample_bylevel = self.colsample_bylevel, reg_lambda = self.reg_lambda, reg_alpha = self.reg_alpha,
                                      min_child_weight = self.min_child_weight, scale_pos_weight = self.scale_pos_weight)

        #Slicing away [RARID, HNAME]
        X_train = X_train.loc[:,X_train.columns[2:]]
        y = y.loc[:,'RESFP']

        #Model Fitting
        self.model.fit(X_train, y)
        self.fitted_model  = self.model

        #Feature Importance
        self.summary = pd.DataFrame(self.fitted_model.feature_importances_, X_train.columns, columns = ['importance'])

        return None


    def predict(self, X, preprocess = True):

        #Making a copy of X
        X_test = X.copy()

        #Apply Preprocessing Pipeline
        if preprocess == True: X_test = self.preprocessing.transform(X_test)

        #Formatting into DataFrame
        Prediction = X_test.loc[:,['RARID', 'HNAME']]

        #Getting the probability of y=1
        Prediction[self.model_name] = self.fitted_model.predict(X_test[X_test.columns[2:]])

        #Scale prediction to sum to 1
        Prediction[self.model_name] = Prediction.groupby('RARID')[self.model_name].apply(lambda x : x / x.sum())

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
            self.max_depth = best_model['params']['max_depth']
            self.gamma = best_model['params']['gamma']
            self.n_estimators = best_model['params']['n_estimators']
            self.learning_rate = best_model['params']['learning_rate']
            self.subsample = best_model['params']['subsample']
            self.colsample_bytree = best_model['params']['colsample_bytree']
            self.colsample_bylevel = best_model['params']['colsample_bylevel']
            self.reg_lambda = best_model['params']['reg_lambda']
            self.reg_alpha = best_model['params']['reg_alpha']
            self.min_child_weight = best_model['params']['min_child_weight']
            self.scale_pos_weight = best_model['params']['scale_pos_weight']

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
        def XGBoost_fit_predict_score(max_depth, gamma, n_estimators, learning_rate, subsample, colsample_bytree, colsample_bylevel, reg_lambda,
                                      reg_alpha, min_child_weight, scale_pos_weight):

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

                model = self.model_class(verbosity = 1,
                                         objective = 'rank:ndcg', booster = 'gbtree', max_delta_step = 1,
                                         max_depth = int(max_depth), gamma = gamma, n_estimators = int(n_estimators),
                                         learning_rate = learning_rate, subsample = subsample, colsample_bytree = colsample_bytree,
                                         colsample_bylevel = colsample_bylevel, reg_lambda = reg_lambda, reg_alpha = reg_alpha,
                                         min_child_weight = min_child_weight, scale_pos_weight = scale_pos_weight)
                model = model.fit(X_train.loc[:,X_train.columns[2:]], y_train.loc[:,'RESFP'])

                Prediction = X_test.loc[:,['RARID','HNAME']]
                Prediction.loc[:,'prediction'] = model.predict(X_test.loc[:,X_test.columns[2:]])

                Score.append(Prediction_accuracy(Prediction, y_test))

            Score = np.mean(Score)

            return Score

        #Define BayesianOptimization instance
        optimizer = BayesianOptimization(f=XGBoost_fit_predict_score,
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
            optimizer.probe(params={'max_depth' : 6,'gamma' : 0,'n_estimators' : 500,'learning_rate' : 0.3,'subsample' : 1,
                                    'colsample_bytree' : 1,'colsample_bylevel' : 1,'reg_lambda' : 1,'reg_alpha' : 0,
                                    'min_child_weight' : 1,'scale_pos_weight' : 14}, lazy=False)

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

===================== XGBoost Classifier / CL Wrapper =====================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class XGBoost_Class_CL_Wrapper(BaseEstimator, ClassifierMixin):

    """
    A sklearn-style wrapper for Two Stage XGBoost Classifier, CL Model
    """

    def __init__(self, model_class_1, model_class_2, model_name, preprocessing, max_depth, gamma, n_estimators, learning_rate,
                 subsample, colsample_bytree, colsample_bylevel, reg_lambda, reg_alpha, min_child_weight, scale_pos_weight, ridge) :
        self.model_class_1 = model_class_1
        self.model_class_2 = model_class_2
        self.model_name = model_name
        self.preprocessing = preprocessing

        """
        Hyperparameter
        """
        self.max_depth = max_depth
        self.gamma = gamma
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.min_child_weight = min_child_weight
        self.scale_pos_weight = scale_pos_weight
        self.ridge = ridge

        #Hyperparameter Dictionary - Bounds
        self.hyperparameter = {'max_depth' : XGB_max_depth, 'gamma' : XGB_gamma,
                               'n_estimators' : XGB_n_estimators, 'learning_rate' : XGB_learning_rate,
                               'subsample' : XGB_subsample, 'colsample_bytree' : XGB_colsample_bytree,
                               'colsample_bylevel' : XGB_colsample_bylevel, 'reg_lambda' : XGB_reg_lambda,
                               'reg_alpha' : XGB_reg_alpha, 'min_child_weight' : XGB_min_child_weight,
                               'scale_pos_weight' : XGB_scale_pos_weight, 'ridge' : ridge_Range}
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
        y_Stage_1 = y_Stage_1.loc[:,'RESWL']

        self.model_Stage_1 = self.model_class_1(verbosity = 1,
                                                objective = 'multi:softprob', num_class=2, booster = 'gbtree', max_delta_step = 1,
                                                max_depth = int(self.max_depth), gamma = self.gamma, n_estimators = int(self.n_estimators),
                                                learning_rate = self.learning_rate, subsample = self.subsample, colsample_bytree = self.colsample_bytree,
                                                colsample_bylevel = self.colsample_bylevel, reg_lambda = self.reg_lambda, reg_alpha = self.reg_alpha,
                                                min_child_weight = self.min_child_weight, scale_pos_weight = self.scale_pos_weight)

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
        self.summary = [pd.DataFrame(self.fitted_Stage_1.feature_importances_, X_Stage_1.columns, columns = ['importance']),
                        self.fitted_Stage_2.get_statsmodels_summary()]

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
            self.max_depth = best_model['params']['max_depth']
            self.gamma = best_model['params']['gamma']
            self.n_estimators = best_model['params']['n_estimators']
            self.learning_rate = best_model['params']['learning_rate']
            self.subsample = best_model['params']['subsample']
            self.colsample_bytree = best_model['params']['colsample_bytree']
            self.colsample_bylevel = best_model['params']['colsample_bylevel']
            self.reg_lambda = best_model['params']['reg_lambda']
            self.reg_alpha = best_model['params']['reg_alpha']
            self.min_child_weight = best_model['params']['min_child_weight']
            self.scale_pos_weight = best_model['params']['scale_pos_weight']
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
        def XGB_Class_CL_fit_predict_score(max_depth, gamma, n_estimators, learning_rate, subsample, colsample_bytree, colsample_bylevel,
                                           reg_lambda, reg_alpha, min_child_weight, scale_pos_weight, ridge):
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
                model_stage_1 = self.model_class_1(verbosity = 1,
                                                   objective = 'multi:softprob', num_class=2, booster = 'gbtree', max_delta_step = 1,
                                                   max_depth = int(max_depth), gamma = gamma, n_estimators = int(n_estimators),
                                                   learning_rate = learning_rate, subsample = subsample, colsample_bytree = colsample_bytree,
                                                   colsample_bylevel = colsample_bylevel, reg_lambda = reg_lambda, reg_alpha = reg_alpha,
                                                   min_child_weight = min_child_weight, scale_pos_weight = scale_pos_weight)
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
        optimizer = BayesianOptimization(f=XGB_Class_CL_fit_predict_score,
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
            optimizer.probe(params={'max_depth' : 6,'gamma' : 0,'n_estimators' : 500,'learning_rate' : 0.3,'subsample' : 1,
                                    'colsample_bytree' : 1,'colsample_bylevel' : 1,'reg_lambda' : 1,'reg_alpha' : 0,
                                    'min_child_weight' : 1,'scale_pos_weight' : 14, 'ridge' : 0}, lazy=False)

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

===================== XGBoost Regressor / CL Wrapper =====================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class XGBoost_Reg_CL_Wrapper(BaseEstimator, ClassifierMixin):

    """
    A sklearn-style wrapper for Two Stage XGBoost Regressor, CL Model
    """

    def __init__(self, model_class_1, model_class_2, model_name, preprocessing, max_depth, gamma, n_estimators, learning_rate,
                 subsample, colsample_bytree, colsample_bylevel, reg_lambda, reg_alpha, min_child_weight, scale_pos_weight, ridge) :
        self.model_class_1 = model_class_1
        self.model_class_2 = model_class_2
        self.model_name = model_name
        self.preprocessing = preprocessing

        """
        Hyperparameter
        """
        self.max_depth = max_depth
        self.gamma = gamma
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.min_child_weight = min_child_weight
        self.scale_pos_weight = scale_pos_weight
        self.ridge = ridge

        #Hyperparameter Dictionary - Bounds
        self.hyperparameter = {'max_depth' : XGB_max_depth, 'gamma' : XGB_gamma,
                               'n_estimators' : XGB_n_estimators, 'learning_rate' : XGB_learning_rate,
                               'subsample' : XGB_subsample, 'colsample_bytree' : XGB_colsample_bytree,
                               'colsample_bylevel' : XGB_colsample_bylevel, 'reg_lambda' : XGB_reg_lambda,
                               'reg_alpha' : XGB_reg_alpha, 'min_child_weight' : XGB_min_child_weight,
                               'scale_pos_weight' : XGB_scale_pos_weight, 'ridge' : ridge_Range}
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

        self.model_Stage_1 = self.model_class_1(verbosity = 1,
                                                objective = 'rank:ndcg', booster = 'gbtree', max_delta_step = 1,
                                                max_depth = int(self.max_depth), gamma = self.gamma, n_estimators = int(self.n_estimators),
                                                learning_rate = self.learning_rate, subsample = self.subsample, colsample_bytree = self.colsample_bytree,
                                                colsample_bylevel = self.colsample_bylevel, reg_lambda = self.reg_lambda, reg_alpha = self.reg_alpha,
                                                min_child_weight = self.min_child_weight, scale_pos_weight = self.scale_pos_weight)

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
        X_Stage_2.reset_index(inplace = True, drop = True)

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
        self.summary = [pd.DataFrame(self.fitted_Stage_1.feature_importances_, X_Stage_1.columns, columns = ['importance']),
                        self.fitted_Stage_2.get_statsmodels_summary()]

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
            self.max_depth = best_model['params']['max_depth']
            self.gamma = best_model['params']['gamma']
            self.n_estimators = best_model['params']['n_estimators']
            self.learning_rate = best_model['params']['learning_rate']
            self.subsample = best_model['params']['subsample']
            self.colsample_bytree = best_model['params']['colsample_bytree']
            self.colsample_bylevel = best_model['params']['colsample_bylevel']
            self.reg_lambda = best_model['params']['reg_lambda']
            self.reg_alpha = best_model['params']['reg_alpha']
            self.min_child_weight = best_model['params']['min_child_weight']
            self.scale_pos_weight = best_model['params']['scale_pos_weight']
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
        def XGB_Reg_CL_fit_predict_score(max_depth, gamma, n_estimators, learning_rate, subsample, colsample_bytree,
                                           colsample_bylevel, reg_lambda, reg_alpha, min_child_weight, scale_pos_weight, ridge):
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
                model_stage_1 = self.model_class_1(verbosity = 1,
                                                   objective = 'rank:ndcg', booster = 'gbtree', max_delta_step = 1,
                                                   max_depth = int(max_depth), gamma = gamma, n_estimators = int(n_estimators),
                                                   learning_rate = learning_rate, subsample = subsample, colsample_bytree = colsample_bytree,
                                                   colsample_bylevel = colsample_bylevel, reg_lambda = reg_lambda, reg_alpha = reg_alpha,
                                                   min_child_weight = min_child_weight, scale_pos_weight = scale_pos_weight)
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

                warnings.filterwarnings("ignore", category=FutureWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=RuntimeWarning)

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
        optimizer = BayesianOptimization(f=XGB_Reg_CL_fit_predict_score,
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
            optimizer.probe(params={'max_depth' : 6,'gamma' : 0,'n_estimators' : 500,'learning_rate' : 0.3,'subsample' : 1,
                                    'colsample_bytree' : 1,'colsample_bylevel' : 1,'reg_lambda' : 1,'reg_alpha' : 0,
                                    'min_child_weight' : 1,'scale_pos_weight' : 14, 'ridge' : 0}, lazy=False)

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

================== One Stage LightGBM Classifier Wrapper ==================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class LightGBM_Class_Wrapper(BaseEstimator, ClassifierMixin):

    """
    A sklearn-style wrapper for One Stage Light Gradient Boosting Classifier
    """

    def __init__(self, model_class, model_name, preprocessing, learning_rate, num_leaves, max_depth, min_child_samples, subsample_for_bin,
                 subsample, bagging_freq, colsample_bytree, min_child_weight, reg_alpha, reg_lambda, scale_pos_weight,
                 n_estimators):
        self.model_class = model_class
        self.model_name = model_name
        self.preprocessing = preprocessing

        """
        Hyperparameter
        """
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.subsample_for_bin = subsample_for_bin
        self.subsample = subsample
        self.bagging_freq = bagging_freq
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.n_estimators = n_estimators
        #Hyperparameter Dictionary - Bounds
        self.hyperparameter = {'learning_rate' : LGB_learning_rate, 'num_leaves' : LGB_num_leaves,'max_depth' : LGB_max_depth,
                               'min_child_samples' : LGB_min_child_samples,'subsample_for_bin' : LGB_subsample_for_bin, 'subsample' : LGB_subsample,
                               'bagging_freq' : LGB_bagging_freq, 'colsample_bytree' : LGB_colsample_bytree,
                               'min_child_weight' : LGB_min_child_weight, 'reg_alpha' : LGB_reg_alpha,
                               'reg_lambda' : LGB_reg_lambda, 'scale_pos_weight' : LGB_scale_pos_weight, 'n_estimators' : LGB_n_estimators}

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
        self.model = self.model_class(verbosity = 1,
                                      learning_rate = self.learning_rate, num_leaves = int(self.num_leaves), max_depth = int(self.max_depth),
                                      min_child_samples = int(self.min_child_samples), subsample_for_bin = int(self.subsample_for_bin),
                                      subsample = self.subsample, bagging_freq = int(self.bagging_freq), colsample_bytree = self.colsample_bytree,
                                      min_child_weight = self.min_child_weight, reg_alpha = self.reg_alpha,
                                      reg_lambda = self.reg_lambda, scale_pos_weight = self.scale_pos_weight, n_estimators = int(self.n_estimators))

        #Slicing away [RARID, HNAME]
        X_train = X_train.loc[:,X_train.columns[2:]]
        y = y.loc[:,'RESWL']

        #Model Fitting
        self.model.fit(X_train, y)
        self.fitted_model  = self.model

        #Feature Importance
        self.summary = pd.DataFrame(self.fitted_model.feature_importances_, X_train.columns, columns = ['importance'])

        return None


    def predict(self, X, preprocess = True):

        #Making a copy of X
        X_test = X.copy()

        #Apply Preprocessing Pipeline
        if preprocess == True: X_test = self.preprocessing.transform(X_test)

        #Formatting into DataFrame
        Prediction = X_test.loc[:,['RARID', 'HNAME']]

        #Getting the probability of y=1
        Prediction[self.model_name] = self.fitted_model.predict_proba(X_test[X_test.columns[2:]])[:,1]

        #Scale prediction to sum to 1
        Prediction[self.model_name] = Prediction.groupby('RARID')[self.model_name].apply(lambda x : x / x.sum())

        return Prediction


    def predict_proba(self, X, preprocess = True):

        #Making a copy of X
        X_test = X.copy()

        #Apply Preprocessing Pipeline
        if preprocess == True: X_test = self.preprocessing.transform(X_test)

        #Formatting into DataFrame
        Prediction = X_test.loc[:,['RARID', 'HNAME']]

        #Getting the probability of y=1
        Prediction[self.model_name] = self.fitted_model.predict_proba(X_test[X_test.columns[2:]])[:,1]

        #Scale prediction to sum to 1
        Prediction[self.model_name] = Prediction.groupby('RARID')[self.model_name].apply(lambda x : x / x.sum())

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
            self.learning_rate = best_model['params']['learning_rate']
            self.num_leaves = best_model['params']['num_leaves']
            self.max_depth = best_model['params']['max_depth']
            self.min_child_samples = best_model['params']['min_child_samples']
            self.subsample_for_bin = best_model['params']['subsample_for_bin']
            self.subsample = best_model['params']['subsample']
            self.bagging_freq = best_model['params']['bagging_freq']
            self.colsample_bytree = best_model['params']['colsample_bytree']
            self.min_child_weight = best_model['params']['min_child_weight']
            self.reg_lambda = best_model['params']['reg_lambda']
            self.reg_alpha = best_model['params']['reg_alpha']
            self.scale_pos_weight = best_model['params']['scale_pos_weight']
            self.n_estimators = best_model['params']['n_estimators']

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
        def LGBM_fit_predict_score(learning_rate, num_leaves, max_depth, min_child_samples, subsample_for_bin, subsample,
                                      bagging_freq, colsample_bytree, min_child_weight, reg_alpha, reg_lambda,
                                      scale_pos_weight, n_estimators):

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

                model = self.model_class(verbosity = 1, silent = False,
                              learning_rate = learning_rate, num_leaves = int(num_leaves), max_depth = int(max_depth),
                              min_child_samples = int(min_child_samples), subsample_for_bin = int(subsample_for_bin), subsample = subsample,
                              colsample_bytree = colsample_bytree, min_child_weight = min_child_weight,
                              reg_alpha = reg_alpha, reg_lambda = reg_lambda, bagging_freq = int(bagging_freq),
                              scale_pos_weight = scale_pos_weight, n_estimators = int(n_estimators))

                model = model.fit(X_train.loc[:,X_train.columns[2:]], y_train.loc[:,'RESWL'])

                Prediction = X_test.loc[:,['RARID','HNAME']]
                Prediction.loc[:,'prediction'] = model.predict_proba(X_test.loc[:,X_test.columns[2:]])[:,1]
                #Scale prediction to sum to 1
                Prediction.loc[:,'prediction'] = Prediction.groupby('RARID')['prediction'].apply(lambda x : x / x.sum())

                Score.append(Kelly_Profit(Prediction, y_test, weight = Betting_Fraction))

            Score = np.mean(Score)

            return Score

        #Define BayesianOptimization instance
        optimizer = BayesianOptimization(f=LGBM_fit_predict_score,
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
            optimizer.probe(params={'learning_rate' : 0.1,'num_leaves' : 31,'max_depth' : -1,'min_child_samples' : 20,'subsample' : 1,
                                    'subsample_for_bin' : 200000,'bagging_freq' : 0,
                                    'colsample_bytree' : 1,'min_child_weight' : 0.001,
                                    'reg_alpha' : 0,'reg_lambda' : 0, 'scale_pos_weight' : 1/14, 'n_estimators' : 100}, lazy=False)

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

=================== One Stage LightGBM Regresor Wrapper ===================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class LightGBM_Reg_Wrapper(BaseEstimator, ClassifierMixin):

    """
    A sklearn-style wrapper for One Stage Light Gradient Boosting Regressor
    """

    def __init__(self, model_class, model_name, preprocessing, learning_rate, num_leaves, max_depth, min_child_samples,
                 subsample, bagging_freq, colsample_bytree, min_child_weight, subsample_for_bin, reg_alpha, reg_lambda,
                 scale_pos_weight, n_estimators):
        self.model_class = model_class
        self.model_name = model_name
        self.preprocessing = preprocessing

        """
        Hyperparameter
        """
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.subsample_for_bin = subsample_for_bin
        self.subsample = subsample
        self.bagging_freq = bagging_freq
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.n_estimators = n_estimators
        #Hyperparameter Dictionary - Bounds
        self.hyperparameter = {'learning_rate' : LGB_learning_rate, 'num_leaves' : LGB_num_leaves,'max_depth' : LGB_max_depth,
                               'min_child_samples' : LGB_min_child_samples,'subsample_for_bin' : LGB_subsample_for_bin, 'subsample' : LGB_subsample,
                               'bagging_freq' : LGB_bagging_freq, 'colsample_bytree' : LGB_colsample_bytree,
                               'min_child_weight' : LGB_min_child_weight, 'reg_alpha' : LGB_reg_alpha,
                               'reg_lambda' : LGB_reg_lambda, 'scale_pos_weight' : LGB_scale_pos_weight, 'n_estimators' : LGB_n_estimators}

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
        self.model = self.model_class(verbosity = 1,
                                      learning_rate = self.learning_rate, num_leaves = int(self.num_leaves), max_depth = int(self.max_depth),
                                      min_child_samples = int(self.min_child_samples), subsample_for_bin = int(self.subsample_for_bin),
                                      subsample = self.subsample, bagging_freq = int(self.bagging_freq), colsample_bytree = self.colsample_bytree,
                                      min_child_weight = self.min_child_weight, reg_alpha = self.reg_alpha,
                                      reg_lambda = self.reg_lambda, scale_pos_weight = self.scale_pos_weight, n_estimators = int(self.n_estimators))

        #Slicing away [RARID, HNAME]
        X_train = X_train.loc[:,X_train.columns[2:]]
        y = y.loc[:,'RESFP']

        #Model Fitting
        self.model.fit(X_train, y)
        self.fitted_model  = self.model

        #Feature Importance
        self.summary = pd.DataFrame(self.fitted_model.feature_importances_, X_train.columns, columns = ['importance'])

        return None


    def predict(self, X, preprocess = True):

        #Making a copy of X
        X_test = X.copy()

        #Apply Preprocessing Pipeline
        if preprocess == True: X_test = self.preprocessing.transform(X_test)

        #Formatting into DataFrame
        Prediction = X_test.loc[:,['RARID', 'HNAME']]

        #Getting the probability of y=1
        Prediction[self.model_name] = self.fitted_model.predict(X_test[X_test.columns[2:]])

        #Scale prediction to sum to 1
        Prediction[self.model_name] = Prediction.groupby('RARID')[self.model_name].apply(lambda x : x / x.sum())

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
            self.learning_rate = best_model['params']['learning_rate']
            self.num_leaves = best_model['params']['num_leaves']
            self.max_depth = best_model['params']['max_depth']
            self.min_child_samples = best_model['params']['min_child_samples']
            self.subsample_for_bin = best_model['params']['subsample_for_bin']
            self.subsample = best_model['params']['subsample']
            self.bagging_freq = best_model['params']['bagging_freq']
            self.colsample_bytree = best_model['params']['colsample_bytree']
            self.min_child_weight = best_model['params']['min_child_weight']
            self.reg_lambda = best_model['params']['reg_lambda']
            self.reg_alpha = best_model['params']['reg_alpha']
            self.min_data_in_leaf = best_model['params']['min_data_in_leaf']
            self.scale_pos_weight = best_model['params']['scale_pos_weight']
            self.n_estimators = best_model['params']['n_estimators']

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
        def LGBM_fit_predict_score(learning_rate, num_leaves, max_depth, min_child_samples, subsample_for_bin, subsample,
                                      bagging_freq, colsample_bytree, min_child_weight, reg_alpha, reg_lambda,
                                      scale_pos_weight, n_estimators):

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

                model = self.model_class(verbosity = 1,
                              learning_rate = learning_rate, num_leaves = int(num_leaves), max_depth = int(max_depth),
                              min_child_samples = int(min_child_samples), subsample_for_bin = int(subsample_for_bin), subsample = subsample,
                              colsample_bytree = colsample_bytree, min_child_weight = min_child_weight,
                               reg_alpha = reg_alpha, reg_lambda = reg_lambda, bagging_freq = int(bagging_freq),
                              scale_pos_weight = scale_pos_weight, n_estimators = int(n_estimators))
                model = model.fit(X_train.loc[:,X_train.columns[2:]], y_train.loc[:,'RESFP'])

                Prediction = X_test.loc[:,['RARID','HNAME']]
                Prediction.loc[:,'prediction'] = model.predict(X_test.loc[:,X_test.columns[2:]])
                Score.append(Prediction_accuracy(Prediction, y_test))

            Score = np.mean(Score)

            return Score

        #Define BayesianOptimization instance
        optimizer = BayesianOptimization(f=LGBM_fit_predict_score,
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
            optimizer.probe(params={'learning_rate' : 0.1,'num_leaves' : 31,'max_depth' : -1,'min_child_samples' : 20,'subsample' : 1,
                                    'subsample_for_bin' : 200000,'bagging_freq' : 0,'colsample_bytree' : 1,'min_child_weight' : 0.001,
                                    'reg_alpha' : 0, 'reg_lambda' : 0, 'scale_pos_weight' : 1/14,
                                    'n_estimators' : 100}, lazy=False)

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

==================== LightGBM Classifier / CL Wrapper ====================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class LightGBM_Class_CL_Wrapper(BaseEstimator, ClassifierMixin):

    """
    A sklearn-style wrapper for Two Stage LightGBM Classifier, CL Model
    """

    def __init__(self, model_class_1, model_class_2, model_name, preprocessing, learning_rate, num_leaves, max_depth, min_child_samples,
                 subsample_for_bin, subsample, bagging_freq, colsample_bytree, min_child_weight, reg_alpha, reg_lambda,
                 scale_pos_weight, n_estimators, ridge):
        self.model_class_1 = model_class_1
        self.model_class_2 = model_class_2
        self.model_name = model_name
        self.preprocessing = preprocessing

        """
        Hyperparameter
        """
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.subsample_for_bin = subsample_for_bin
        self.subsample = subsample
        self.bagging_freq = bagging_freq
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.n_estimators = n_estimators
        self.ridge = ridge

        #Hyperparameter Dictionary - Bounds
        self.hyperparameter = {'learning_rate' : LGB_learning_rate, 'num_leaves' : LGB_num_leaves,'max_depth' : LGB_max_depth,
                               'min_child_samples' : LGB_min_child_samples,'subsample_for_bin' : LGB_subsample_for_bin, 'subsample' : LGB_subsample,
                               'bagging_freq' : LGB_bagging_freq, 'colsample_bytree' : LGB_colsample_bytree,
                               'min_child_weight' : LGB_min_child_weight, 'reg_alpha' : LGB_reg_alpha, 'ridge' : ridge_Range,
                               'reg_lambda' : LGB_reg_lambda, 'scale_pos_weight' : LGB_scale_pos_weight, 'n_estimators' : LGB_n_estimators}

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
        y_Stage_1 = y_Stage_1.loc[:,'RESWL']

        self.model_Stage_1 = self.model_class_1(verbosity = 1,
                                      learning_rate = self.learning_rate, num_leaves = int(self.num_leaves), max_depth = int(self.max_depth),
                                      min_child_samples = int(self.min_child_samples), subsample_for_bin = int(self.subsample_for_bin),
                                      subsample = self.subsample, bagging_freq = int(self.bagging_freq), colsample_bytree = self.colsample_bytree,
                                      min_child_weight = self.min_child_weight, reg_alpha = self.reg_alpha,
                                      reg_lambda = self.reg_lambda, scale_pos_weight = self.scale_pos_weight, n_estimators = int(self.n_estimators))

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
        X_Stage_2.reset_index(inplace = True, drop = True)

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
        self.summary = [pd.DataFrame(self.fitted_Stage_1.feature_importances_, X_Stage_1.columns, columns = ['importance']),
                        self.fitted_Stage_2.get_statsmodels_summary()]

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
            self.learning_rate = best_model['params']['learning_rate']
            self.num_leaves = best_model['params']['num_leaves']
            self.max_depth = best_model['params']['max_depth']
            self.min_child_samples = best_model['params']['min_child_samples']
            self.subsample_for_bin = best_model['params']['subsample_for_bin']
            self.subsample = best_model['params']['subsample']
            self.bagging_freq = best_model['params']['bagging_freq']
            self.colsample_bytree = best_model['params']['colsample_bytree']
            self.min_child_weight = best_model['params']['min_child_weight']
            self.reg_lambda = best_model['params']['reg_lambda']
            self.reg_alpha = best_model['params']['reg_alpha']
            self.min_data_in_leaf = best_model['params']['min_data_in_leaf']
            self.scale_pos_weight = best_model['params']['scale_pos_weight']
            self.n_estimators = best_model['params']['n_estimators']
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
        def LGBM_Class_CL_fit_predict_score(learning_rate, num_leaves, max_depth, min_child_samples, subsample_for_bin, subsample,
                                      bagging_freq, colsample_bytree, min_child_weight, reg_alpha, reg_lambda,
                                      scale_pos_weight, n_estimators, ridge):
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
                model_stage_1 = self.model_class_1(verbosity = 1,
                                                  learning_rate = learning_rate, num_leaves = int(num_leaves), max_depth = int(max_depth),
                                                  min_child_samples = int(min_child_samples), subsample_for_bin = int(subsample_for_bin),
                                                  subsample = subsample, colsample_bytree = colsample_bytree,
                                                  min_child_weight = min_child_weight, reg_alpha = reg_alpha, bagging_freq = int(bagging_freq),
                                                  reg_lambda = reg_lambda, scale_pos_weight = scale_pos_weight, n_estimators = int(n_estimators))
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
        optimizer = BayesianOptimization(f=LGBM_Class_CL_fit_predict_score,
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
            optimizer.probe(params={'learning_rate' : 0.1,'num_leaves' : 31,'max_depth' : -1,'min_child_samples' : 20,'subsample' : 1,
                                    'subsample_for_bin' : 200000,'bagging_freq' : 0,'colsample_bytree' : 1,'min_child_weight' : 0.001,
                                    'reg_alpha' : 0,'reg_lambda' : 0, 'scale_pos_weight' : 1/14,
                                    'n_estimators' : 100, 'ridge' : 0}, lazy=False)

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

===================== LightGBM Regressor / CL Wrapper =====================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class LightGBM_Reg_CL_Wrapper(BaseEstimator, ClassifierMixin):

    """
    A sklearn-style wrapper for Two Stage LightGBM Regressor, CL Model
    """

    def __init__(self, model_class_1, model_class_2, model_name, preprocessing, learning_rate, num_leaves, max_depth, min_child_samples,
                 subsample, bagging_freq, colsample_bytree, min_child_weight, subsample_for_bin, reg_alpha, reg_lambda,
                 scale_pos_weight, n_estimators, ridge):
        self.model_class_1 = model_class_1
        self.model_class_2 = model_class_2
        self.model_name = model_name
        self.preprocessing = preprocessing

        """
        Hyperparameter
        """
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.subsample_for_bin = subsample_for_bin
        self.subsample = subsample
        self.bagging_freq = bagging_freq
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.n_estimators = n_estimators
        self.ridge = ridge

        #Hyperparameter Dictionary - Bounds
        self.hyperparameter = {'learning_rate' : LGB_learning_rate, 'num_leaves' : LGB_num_leaves,'max_depth' : LGB_max_depth,
                               'min_child_samples' : LGB_min_child_samples,'subsample_for_bin' : LGB_subsample_for_bin, 'subsample' : LGB_subsample,
                               'bagging_freq' : LGB_bagging_freq, 'colsample_bytree' : LGB_colsample_bytree,
                               'min_child_weight' : LGB_min_child_weight, 'reg_alpha' : LGB_reg_alpha,'ridge' : ridge_Range,
                               'reg_lambda' : LGB_reg_lambda, 'scale_pos_weight' : LGB_scale_pos_weight, 'n_estimators' : LGB_n_estimators}

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

        self.model_Stage_1 = self.model_class_1(verbosity = 1,
                                      learning_rate = self.learning_rate, num_leaves = int(self.num_leaves), max_depth = int(self.max_depth),
                                      min_child_samples = int(self.min_child_samples), subsample_for_bin = int(self.subsample_for_bin),
                                      subsample = self.subsample, bagging_freq = int(self.bagging_freq), colsample_bytree = self.colsample_bytree,
                                      min_child_weight = self.min_child_weight, reg_alpha = self.reg_alpha,
                                      reg_lambda = self.reg_lambda, scale_pos_weight = self.scale_pos_weight, n_estimators = int(self.n_estimators))

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
        X_Stage_2.reset_index(inplace = True, drop = True)

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
        self.summary = [pd.DataFrame(self.fitted_Stage_1.feature_importances_, X_Stage_1.columns, columns = ['importance']),
                        self.fitted_Stage_2.get_statsmodels_summary()]

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
            self.learning_rate = best_model['params']['learning_rate']
            self.num_leaves = best_model['params']['num_leaves']
            self.max_depth = best_model['params']['max_depth']
            self.min_child_samples = best_model['params']['min_child_samples']
            self.subsample_for_bin = best_model['params']['subsample_for_bin']
            self.subsample = best_model['params']['subsample']
            self.bagging_freq = best_model['params']['bagging_freq']
            self.colsample_bytree = best_model['params']['colsample_bytree']
            self.min_child_weight = best_model['params']['min_child_weight']
            self.reg_lambda = best_model['params']['reg_lambda']
            self.reg_alpha = best_model['params']['reg_alpha']
            self.min_data_in_leaf = best_model['params']['min_data_in_leaf']
            self.scale_pos_weight = best_model['params']['scale_pos_weight']
            self.n_estimators = best_model['params']['n_estimators']
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
        def LGBM_Reg_CL_fit_predict_score(learning_rate, num_leaves, max_depth, min_child_samples, subsample,
                                          bagging_freq, colsample_bytree, min_child_weight, subsample_for_bin, reg_alpha, reg_lambda,
                                          scale_pos_weight, n_estimators, ridge):
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
                model_stage_1 = self.model_class_1(verbosity = 1,
                                                  learning_rate = learning_rate, num_leaves = int(num_leaves), max_depth = int(max_depth),
                                                  min_child_samples = int(min_child_samples), subsample_for_bin = int(subsample_for_bin),
                                                  subsample = subsample, colsample_bytree = colsample_bytree,
                                                  min_child_weight = min_child_weight, reg_alpha = reg_alpha, bagging_freq = int(bagging_freq),
                                                  reg_lambda = reg_lambda, scale_pos_weight = scale_pos_weight, n_estimators = int(n_estimators))
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

                warnings.filterwarnings("ignore", category=FutureWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=RuntimeWarning)

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
        optimizer = BayesianOptimization(f=LGBM_Reg_CL_fit_predict_score,
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
            optimizer.probe(params={'learning_rate' : 0.1,'num_leaves' : 31,'max_depth' : -1,'min_child_samples' : 20,'subsample' : 1,
                                    'subsample_for_bin' : 200000,'bagging_freq' : 0,'colsample_bytree' : 1,'min_child_weight' : 0.001,
                                    'reg_alpha' : 0,'reg_lambda' : 0, 'scale_pos_weight' : 1/14,
                                    'n_estimators' : 100, 'ridge' : 0}, lazy=False)

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

================== One Stage Catboost Classifier Wrapper ==================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class Catboost_Class_Wrapper(BaseEstimator, ClassifierMixin):

    """
    A sklearn-style wrapper for One Stage Categorical Boosting Classifier
    """

    def __init__(self, model_class, model_name, preprocessing, iterations, depth, learning_rate, random_strength, bagging_temperature,
                 l2_leaf_reg, scale_pos_weight):
        self.model_class = model_class
        self.model_name = model_name
        self.preprocessing = preprocessing

        """
        Hyperparameter
        """
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.random_strength = random_strength
        self.bagging_temperature = bagging_temperature
        self.l2_leaf_reg = l2_leaf_reg
        self.scale_pos_weight = scale_pos_weight

        #Hyperparameter Dictionary - Bounds
        self.hyperparameter = {'iterations' : CGB_iterations, 'depth' : CGB_depth,
                               'learning_rate' : CGB_learning_rate, 'random_strength' : CGB_random_strength,
                               'bagging_temperature' : CGB_bagging_temperature, 'l2_leaf_reg' : CGB_l2_leaf_reg,
                               'scale_pos_weight' : CGB_scale_pos_weight}
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
        self.model = self.model_class(allow_writing_files=False,
                                      iterations = int(self.iterations), depth = int(self.depth), learning_rate = self.learning_rate,
                                      random_strength = self.random_strength, bagging_temperature = self.bagging_temperature,
                                      l2_leaf_reg = self.l2_leaf_reg, scale_pos_weight = self.scale_pos_weight)

        #Slicing away [RARID, HNAME]
        X_train = X_train.loc[:,X_train.columns[2:]]
        y = y.loc[:,'RESWL']

        #Model Fitting
        self.model.fit(X_train, y)
        self.fitted_model  = self.model

        #Feature Importance
        self.summary = pd.DataFrame(self.fitted_model.get_feature_importance(Pool(X_train, label=y)), X_train.columns, columns = ['importance'])

        return None


    def predict(self, X, preprocess = True):

        #Making a copy of X
        X_test = X.copy()

        #Apply Preprocessing Pipeline
        if preprocess == True: X_test = self.preprocessing.transform(X_test)

        #Formatting into DataFrame
        Prediction = X_test.loc[:,['RARID', 'HNAME']]

        #Getting the probability of y=1
        Prediction[self.model_name] = self.fitted_model.predict_proba(X_test[X_test.columns[2:]])[:,1]

        #Scale prediction to sum to 1
        Prediction[self.model_name] = Prediction.groupby('RARID')[self.model_name].apply(lambda x : x / x.sum())

        return Prediction


    def predict_proba(self, X, preprocess = True):

        #Making a copy of X
        X_test = X.copy()

        #Apply Preprocessing Pipeline
        if preprocess == True: X_test = self.preprocessing.transform(X_test)

        #Formatting into DataFrame
        Prediction = X_test.loc[:,['RARID', 'HNAME']]

        #Getting the probability of y=1
        Prediction[self.model_name] = self.fitted_model.predict_proba(X_test[X_test.columns[2:]])[:,1]

        #Scale prediction to sum to 1
        Prediction[self.model_name] = Prediction.groupby('RARID')[self.model_name].apply(lambda x : x / x.sum())

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
            self.iterations = best_model['params']['iterations']
            self.depth = best_model['params']['depth']
            self.learning_rate = best_model['params']['learning_rate']
            self.random_strength = best_model['params']['random_strength']
            self.bagging_temperature = best_model['params']['bagging_temperature']
            self.l2_leaf_reg = best_model['params']['l2_leaf_reg']
            self.scale_pos_weight = best_model['params']['scale_pos_weight']

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
        def CatBoost_fit_predict_score(iterations, depth, learning_rate, random_strength, bagging_temperature, l2_leaf_reg,
                                       scale_pos_weight):

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

                model = self.model_class(allow_writing_files=False,
                                         iterations = int(iterations), depth = int(depth), learning_rate = learning_rate,
                                         random_strength = random_strength, bagging_temperature = bagging_temperature,
                                         l2_leaf_reg = l2_leaf_reg, scale_pos_weight = scale_pos_weight)
                model = model.fit(X_train.loc[:,X_train.columns[2:]], y_train.loc[:,'RESWL'])

                Prediction = X_test.loc[:,['RARID','HNAME']]
                Prediction.loc[:,'prediction'] = model.predict_proba(X_test.loc[:,X_test.columns[2:]])[:,1]
                #Scale prediction to sum to 1
                Prediction.loc[:,'prediction'] = Prediction.groupby('RARID')['prediction'].apply(lambda x : x / x.sum())

                Score.append(Kelly_Profit(Prediction, y_test, weight = Betting_Fraction))

            Score = np.mean(Score)

            return Score

        #Define BayesianOptimization instance
        optimizer = BayesianOptimization(f=CatBoost_fit_predict_score,
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
            optimizer.probe(params={'iterations' : 1000,'depth' : 6,'learning_rate' : 0.03,'random_strength' : 1,'bagging_temperature' : 1,
                                    'l2_leaf_reg' : 3,'scale_pos_weight' : 1}, lazy=False)

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

================== One Stage Catboost Regressor Wrapper ==================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class Catboost_Reg_Wrapper(BaseEstimator, ClassifierMixin):

    """
    A sklearn-style wrapper for One Stage Categorical Boosting Regressor
    """

    def __init__(self, model_class, model_name, preprocessing, iterations, depth, learning_rate, random_strength, bagging_temperature,
                 l2_leaf_reg, scale_pos_weight):
        self.model_class = model_class
        self.model_name = model_name
        self.preprocessing = preprocessing

        """
        Hyperparameter
        """
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.random_strength = random_strength
        self.bagging_temperature = bagging_temperature
        self.l2_leaf_reg = l2_leaf_reg
        self.scale_pos_weight = scale_pos_weight

        #Hyperparameter Dictionary - Bounds
        self.hyperparameter = {'iterations' : CGB_iterations, 'depth' : CGB_depth,
                               'learning_rate' : CGB_learning_rate, 'random_strength' : CGB_random_strength,
                               'bagging_temperature' : CGB_bagging_temperature, 'l2_leaf_reg' : CGB_l2_leaf_reg}
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
        self.model = self.model_class(allow_writing_files=False,
                                      iterations = int(self.iterations), depth = int(self.depth), learning_rate = self.learning_rate,
                                      random_strength = self.random_strength, bagging_temperature = self.bagging_temperature,
                                      l2_leaf_reg = self.l2_leaf_reg, scale_pos_weight = self.scale_pos_weight)

        #Slicing away [RARID, HNAME]
        X_train = X_train.loc[:,X_train.columns[2:]]
        y = y.loc[:,'RESFP']

        #Convert all features into floats
        X_train.loc[:,X_train.columns[2:]] = pd.DataFrame(X_train.loc[:,X_train.columns[2:]].values.astype(float))

        #Model Fitting
        self.model.fit(X_train, y)
        self.fitted_model  = self.model

        #Feature Importance
        self.summary = pd.DataFrame(self.fitted_model.get_feature_importance(Pool(X_train, label=y)), X_train.columns, columns = ['importance'])

        return None


    def predict(self, X, preprocess = True):

        #Making a copy of X
        X_test = X.copy()

        #Apply Preprocessing Pipeline
        if preprocess == True: X_test = self.preprocessing.transform(X_test)

        #Formatting into DataFrame
        Prediction = X_test.loc[:,['RARID', 'HNAME']]

        #Getting the probability of y=1
        Prediction[self.model_name] = self.fitted_model.predict(X_test[X_test.columns[2:]])

        #Scale prediction to sum to 1
        Prediction[self.model_name] = Prediction.groupby('RARID')[self.model_name].apply(lambda x : x / x.sum())

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
            self.iterations = best_model['params']['iterations']
            self.depth = best_model['params']['depth']
            self.learning_rate = best_model['params']['learning_rate']
            self.random_strength = best_model['params']['random_strength']
            self.bagging_temperature = best_model['params']['bagging_temperature']
            self.l2_leaf_reg = best_model['params']['l2_leaf_reg']
            self.scale_pos_weight = best_model['params']['scale_pos_weight']

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
        def CatBoost_fit_predict_score(iterations, depth, learning_rate, random_strength, bagging_temperature, l2_leaf_reg,
                                       scale_pos_weight):
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

                model = self.model_class(allow_writing_files=False,
                                         iterations = int(iterations), depth = int(depth), learning_rate = learning_rate,
                                         random_strength = random_strength, bagging_temperature = bagging_temperature,
                                         l2_leaf_reg = l2_leaf_reg, scale_pos_weight = scale_pos_weight)
                model = model.fit(X_train.loc[:,X_train.columns[2:]], y_train.loc[:,'RESFP'])

                Prediction = X_test.loc[:,['RARID','HNAME']]
                Prediction.loc[:,'prediction'] = model.predict(X_test.loc[:,X_test.columns[2:]])
                Score.append(Prediction_accuracy(Prediction, y_test))

            Score = np.mean(Score)

            return Score

        #Define BayesianOptimization instance
        optimizer = BayesianOptimization(f=CatBoost_fit_predict_score,
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
            optimizer.probe(params={'iterations' : 1000,'depth' : 6,'learning_rate' : 0.03,'random_strength' : 1,'bagging_temperature' : 1,
                                    'l2_leaf_reg' : 3,'scale_pos_weight' : 1}, lazy=False)

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

==================== Catboost Classifier / CL Wrapper ====================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class CatBoost_Class_CL_Wrapper(BaseEstimator, ClassifierMixin):

    """
    A sklearn-style wrapper for Two Stage Catboost Classifier, CL Model
    """

    def __init__(self, model_class_1, model_class_2, model_name, preprocessing, iterations, depth, learning_rate, random_strength,
                 bagging_temperature,l2_leaf_reg, scale_pos_weight, ridge) :
        self.model_class_1 = model_class_1
        self.model_class_2 = model_class_2
        self.model_name = model_name
        self.preprocessing = preprocessing

        """
        Hyperparameter
        """
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.random_strength = random_strength
        self.bagging_temperature = bagging_temperature
        self.l2_leaf_reg = l2_leaf_reg
        self.scale_pos_weight = scale_pos_weight
        self.ridge = ridge

        #Hyperparameter Dictionary - Bounds
        self.hyperparameter = {'iterations' : CGB_iterations, 'depth' : CGB_depth,
                               'learning_rate' : CGB_learning_rate, 'random_strength' : CGB_random_strength,
                               'bagging_temperature' : CGB_bagging_temperature, 'l2_leaf_reg' : CGB_l2_leaf_reg,
                               'scale_pos_weight' : CGB_scale_pos_weight, 'ridge' : ridge_Range}

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
        y_Stage_1 = y_Stage_1.loc[:,'RESWL']

        self.model_Stage_1 = self.model_class_1(allow_writing_files=False,
                                                iterations = int(self.iterations), depth = int(self.depth), learning_rate = self.learning_rate,
                                                random_strength = self.random_strength, bagging_temperature = self.bagging_temperature,
                                                l2_leaf_reg = self.l2_leaf_reg, scale_pos_weight = self.scale_pos_weight)

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
        X_Stage_2.reset_index(inplace = True, drop = True)

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
        self.summary = [pd.DataFrame(self.fitted_Stage_1.get_feature_importance(Pool(X_train.loc[:,X_train.columns[2:]], label=y.loc[:,'RESWL'])),
                                     X_train.loc[:,self.other_col].columns[2:], columns = ['importance']),self.fitted_Stage_2.get_statsmodels_summary()]

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
        print(saved_models_path + logger_name+".json")
        if os.path.exists(saved_models_path + logger_name+".json"):
            load_logs(optimizer, logs=[saved_models_path + logger_name+".json"])

            #Save best hyperparameter
            best_model = optimizer.max
            self.iterations = best_model['params']['iterations']
            self.depth = best_model['params']['depth']
            self.learning_rate = best_model['params']['learning_rate']
            self.random_strength = best_model['params']['random_strength']
            self.bagging_temperature = best_model['params']['bagging_temperature']
            self.l2_leaf_reg = best_model['params']['l2_leaf_reg']
            self.scale_pos_weight = best_model['params']['scale_pos_weight']
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
        def CatBoost_Class_CL_fit_predict_score(iterations, depth, learning_rate, random_strength,bagging_temperature,
                                                l2_leaf_reg, scale_pos_weight, ridge):
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
                model_stage_1 = self.model_class_1(allow_writing_files=False,
                                                   iterations = int(iterations), depth = int(depth), learning_rate = learning_rate,
                                                   random_strength = random_strength, bagging_temperature = bagging_temperature,
                                                   l2_leaf_reg = l2_leaf_reg, scale_pos_weight = scale_pos_weight)
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
        optimizer = BayesianOptimization(f=CatBoost_Class_CL_fit_predict_score,
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
            optimizer.probe(params={'iterations' : 1000,'depth' : 6,'learning_rate' : 0.03,'random_strength' : 1,'bagging_temperature' : 1,
                                    'l2_leaf_reg' : 3,'scale_pos_weight' : 1, 'ridge' : 0}, lazy=False)

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

===================== Catboost Regressor / CL Wrapper =====================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class Catboost_Reg_CL_Wrapper(BaseEstimator, ClassifierMixin):

    """
    A sklearn-style wrapper for Two Stage XGBoost Regressor, CL Model
    """

    def __init__(self, model_class_1, model_class_2, model_name, preprocessing, iterations, depth, learning_rate,
                 random_strength, bagging_temperature,l2_leaf_reg, ridge) :
        self.model_class_1 = model_class_1
        self.model_class_2 = model_class_2
        self.model_name = model_name
        self.preprocessing = preprocessing

        """
        Hyperparameter
        """
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.random_strength = random_strength
        self.bagging_temperature = bagging_temperature
        self.l2_leaf_reg = l2_leaf_reg
        self.ridge = ridge

        #Hyperparameter Dictionary - Bounds
        self.hyperparameter = {'iterations' : CGB_iterations, 'depth' : CGB_depth,
                               'learning_rate' : CGB_learning_rate, 'random_strength' : CGB_random_strength,
                               'bagging_temperature' : CGB_bagging_temperature, 'l2_leaf_reg' : CGB_l2_leaf_reg,
                               'ridge' : ridge_Range}
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

        self.model_Stage_1 = self.model_class_1(allow_writing_files=False,
                                                iterations = int(self.iterations), depth = int(self.depth), learning_rate = self.learning_rate,
                                                random_strength = self.random_strength, bagging_temperature = self.bagging_temperature,
                                                l2_leaf_reg = self.l2_leaf_reg)
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
        X_Stage_2.reset_index(inplace = True, drop = True)

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
        self.summary = [pd.DataFrame(self.fitted_Stage_1.get_feature_importance(Pool(X_train.loc[:,X_train.columns[2:]], label=y.loc[:,'RESFP'])),
                                     X_train.loc[:,self.other_col].columns[2:], columns = ['importance']),self.fitted_Stage_2.get_statsmodels_summary()]

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
            self.iterations = best_model['params']['iterations']
            self.depth = best_model['params']['depth']
            self.learning_rate = best_model['params']['learning_rate']
            self.random_strength = best_model['params']['random_strength']
            self.bagging_temperature = best_model['params']['bagging_temperature']
            self.l2_leaf_reg = best_model['params']['l2_leaf_reg']
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
        def Catboost_Reg_CL_fit_predict_score(iterations, depth, learning_rate, random_strength, bagging_temperature, l2_leaf_reg,
                                              ridge):
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
                model_stage_1 = self.model_class_1(allow_writing_files=False,
                                                    iterations = int(iterations), depth = int(depth), learning_rate = learning_rate,
                                                    random_strength = random_strength, bagging_temperature = bagging_temperature,
                                                    l2_leaf_reg = l2_leaf_reg)
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

                warnings.filterwarnings("ignore", category=FutureWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=RuntimeWarning)

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
        optimizer = BayesianOptimization(f=Catboost_Reg_CL_fit_predict_score,
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
            optimizer.probe(params={'iterations' : 1000,'depth' : 6,'learning_rate' : 0.03,'random_strength' : 1,'bagging_temperature' : 1,
                                    'l2_leaf_reg' : 3, 'ridge' : 0}, lazy=False)

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

=========================== XGBoost_Class Model ===========================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def XGBoost_Class(model_name = 'XGBoost_Class Model', preprocessing = Raw, max_depth=5, gamma=1, n_estimators=1000, learning_rate=.005,
                  subsample=0.9, colsample_bytree=0.7, colsample_bylevel=1.0, reg_lambda=1.0, reg_alpha=1.0, min_child_weight=1.0,
                  scale_pos_weight=1.0):

    """
    Extreme Gradient Boosting Classifier
    Within Race competition is not considered
    """

    model = XGBoost_Class_Wrapper(model_class=XGBClassifier, model_name=model_name,  preprocessing=preprocessing,
                                   max_depth=max_depth, gamma=gamma, n_estimators=n_estimators, learning_rate=learning_rate,
                                   subsample=subsample, colsample_bytree=colsample_bytree, colsample_bylevel=colsample_bylevel,
                                   reg_lambda=reg_lambda, reg_alpha=reg_alpha, min_child_weight=min_child_weight,
                                   scale_pos_weight=scale_pos_weight)

    return model

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

============================ XGBoost_Reg Model ============================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def XGBoost_Reg(model_name = 'XGBoost_Reg Model', preprocessing = Raw, max_depth=5, gamma=1, n_estimators=1000, learning_rate=.005,
                subsample=0.9, colsample_bytree=0.7, colsample_bylevel=1.0, reg_lambda=1.0, reg_alpha=1.0, min_child_weight=1.0,
                  scale_pos_weight=1.0):

    """
    Extreme Gradient Boosting Regressor
    Predicting Finishing Position
    """

    model = XGBoost_Reg_Wrapper(model_class=XGBRegressor, model_name=model_name,  preprocessing=preprocessing,
                                   max_depth=max_depth, gamma=gamma, n_estimators=n_estimators, learning_rate=learning_rate,
                                   subsample=subsample, colsample_bytree=colsample_bytree,colsample_bylevel=colsample_bylevel,
                                   reg_lambda=reg_lambda, reg_alpha=reg_alpha, min_child_weight=min_child_weight,
                                   scale_pos_weight=scale_pos_weight)

    return model

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

========================== XGBoost_Class_CL Model ==========================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def XGBoost_Class_CL(model_name = 'XGBoost_Class_CL Model', preprocessing = Raw, max_depth=5, gamma=1, n_estimators=1000, learning_rate=.005,
                     subsample=0.9, colsample_bytree=0.7, colsample_bylevel=1.0, reg_lambda=1.0, reg_alpha=1.0, min_child_weight=1.0,
                     scale_pos_weight=1.0, ridge=0):

    """
    Stage 1 : Extreme Gradient Boosting Classifier
    Stage 2 : Conditional Logistic Regression
    """

    model = XGBoost_Class_CL_Wrapper(model_class_1=XGBClassifier, model_class_2=pylogit.create_choice_model, model_name=model_name,
                                     preprocessing=preprocessing, max_depth=max_depth, gamma=gamma, n_estimators=n_estimators,
                                     learning_rate=learning_rate, subsample=subsample, colsample_bytree=colsample_bytree,
                                     colsample_bylevel=colsample_bylevel, reg_lambda=reg_lambda, reg_alpha=reg_alpha,
                                     min_child_weight=min_child_weight,scale_pos_weight=scale_pos_weight, ridge=ridge)

    return model

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

========================== XGBoost_Reg_CL Model ==========================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def XGBoost_Reg_CL(model_name = 'XGBoost_Reg_CL Model', preprocessing = Raw, max_depth=5, gamma=1, n_estimators=1000, learning_rate=.005,
                     subsample=0.9, colsample_bytree=0.7, colsample_bylevel=1.0, reg_lambda=1.0, reg_alpha=1.0, min_child_weight=1.0,
                     scale_pos_weight=1.0, ridge=0):

    """
    Stage 1 : Extreme Gradient Boosting Regressor
    Stage 2 : Conditional Logistic Regression
    """

    model = XGBoost_Reg_CL_Wrapper(model_class_1=XGBRegressor, model_class_2=pylogit.create_choice_model, model_name=model_name,
                                   preprocessing=preprocessing, max_depth=max_depth, gamma=gamma, n_estimators=n_estimators,
                                     learning_rate=learning_rate, subsample=subsample, colsample_bytree=colsample_bytree,
                                     colsample_bylevel=colsample_bylevel, reg_lambda=reg_lambda, reg_alpha=reg_alpha,
                                     min_child_weight=min_child_weight,scale_pos_weight=scale_pos_weight, ridge=ridge)

    return model

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

========================== LightGBM_Class Model ==========================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def LightGBM_Class(model_name = 'LightGBM_Class Model', preprocessing = Raw, learning_rate=0.1, num_leaves=31, max_depth=-1, min_child_samples=20,
                   subsample_for_bin=200000, subsample=1, subsample_freq=0, colsample_bytree=1, min_child_weight=0.001,
                   reg_alpha=0, reg_lambda=0, scale_pos_weight=1/14, n_estimators=100):

    """
    Light Gradient Boosting Classifier
    Within Race competition is not considered
    """

    model = LightGBM_Class_Wrapper(model_class=LGBMClassifier, model_name=model_name,  preprocessing=preprocessing,
                                   learning_rate=learning_rate, num_leaves=num_leaves, max_depth=max_depth, min_child_samples=min_child_samples,
                                   subsample_for_bin=subsample_for_bin, subsample=subsample, subsample_freq=subsample_freq,
                                   colsample_bytree=colsample_bytree, min_child_weight=min_child_weight,
                                   reg_alpha=reg_alpha, reg_lambda=reg_lambda, scale_pos_weight=scale_pos_weight, n_estimators=n_estimators)

    return model

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

=========================== LightGBM_Reg Model ===========================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def LightGBM_Reg(model_name = 'LightGBM_Reg Model', preprocessing = Raw, learning_rate=0.1, num_leaves=31, max_depth=-1, min_child_samples=20,
                   subsample_for_bin=200000, subsample=1, subsample_freq=0, colsample_bytree=1, min_child_weight=0.001,
                   reg_alpha=0, reg_lambda=0, scale_pos_weight=1/14, n_estimators=100):

    """
    Light Gradient Boosting Regressor
    Predicting Finishing Position
    """

    model = LightGBM_Reg_Wrapper(model_class=LGBMRegressor, model_name=model_name,  preprocessing=preprocessing,
                                   learning_rate=learning_rate, num_leaves=num_leaves, max_depth=max_depth, min_child_samples=min_child_samples,
                                   subsample_for_bin=subsample_for_bin, subsample=subsample, subsample_freq=subsample_freq,
                                   colsample_bytree=colsample_bytree, min_child_weight=min_child_weight,
                                   reg_alpha=reg_alpha, reg_lambda=reg_lambda, scale_pos_weight=scale_pos_weight, n_estimators=n_estimators)

    return model

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

========================= LightGBM_Class_CL Model =========================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def LightGBM_Class_CL(model_name = 'LightGBM_Class_CL Model', preprocessing = Raw, learning_rate=0.1, num_leaves=31, max_depth=-1, min_child_samples=20,
                      subsample_for_bin=200000, subsample=1, subsample_freq=0, colsample_bytree=1, min_child_weight=0.001,
                      reg_alpha=0, reg_lambda=0, scale_pos_weight=1/14, n_estimators=100, ridge=0):

    """
    Stage 1 : Light Gradient Boosting Classifier
    Stage 2 : Conditional Logistic Regression
    """

    model = LightGBM_Class_CL_Wrapper(model_class_1=LGBMClassifier, model_class_2=pylogit.create_choice_model, model_name=model_name,
                                     preprocessing=preprocessing, learning_rate=learning_rate, num_leaves=num_leaves, max_depth=max_depth,
                                     min_child_samples=min_child_samples, subsample_for_bin=subsample_for_bin, subsample=subsample,
                                     subsample_freq=subsample_freq, colsample_bytree=colsample_bytree, min_child_weight=min_child_weight,
                                     reg_alpha=reg_alpha, reg_lambda=reg_lambda,
                                     scale_pos_weight=scale_pos_weight, n_estimators=n_estimators, ridge=ridge)

    return model

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

========================== LightGBM_Reg_CL Model ==========================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def LightGBM_Reg_CL(model_name = 'LightGBM_Reg_CL Model', preprocessing = Raw, learning_rate=0.1, num_leaves=31, max_depth=-1, min_child_samples=20,
                    subsample_for_bin=200000, subsample=1, subsample_freq=0, colsample_bytree=1, min_child_weight=0.001,
                    reg_alpha=0, reg_lambda=0, scale_pos_weight=1/14, n_estimators=100, ridge=0):
    """
    Stage 1 : Light Gradient Boosting Regressor
    Stage 2 : Conditional Logistic Regression
    """

    model = LightGBM_Reg_CL_Wrapper(model_class_1=LGBMRegressor, model_class_2=pylogit.create_choice_model, model_name=model_name,
                                     preprocessing=preprocessing, learning_rate=learning_rate, num_leaves=num_leaves, max_depth=max_depth,
                                     min_child_samples=min_child_samples, subsample_for_bin=subsample_for_bin, subsample=subsample,
                                     subsample_freq=subsample_freq, colsample_bytree=colsample_bytree, min_child_weight=min_child_weight,
                                     reg_alpha=reg_alpha, reg_lambda=reg_lambda,
                                     scale_pos_weight=scale_pos_weight, n_estimators=n_estimators, ridge=ridge)

    return model

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

========================== Catboost_Class Model ==========================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def Catboost_Class(model_name = 'Catboost_Class Model', preprocessing = Raw, iterations=1000, depth=6, learning_rate=None,
                   random_strength=1, bagging_temperature=1, l2_leaf_reg=3, scale_pos_weight=1):

    """
    Categorical Gradient Boosting Classifier
    Within Race competition is not considered
    """

    model = Catboost_Class_Wrapper(model_class=CatBoostClassifier, model_name=model_name,  preprocessing=preprocessing,
                                  iterations=iterations, depth=depth, learning_rate=learning_rate,random_strength=random_strength,
                                  bagging_temperature=bagging_temperature, l2_leaf_reg=l2_leaf_reg, scale_pos_weight=scale_pos_weight)

    return model

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

=========================== Catboost_Reg Model ===========================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def Catboost_Reg(model_name = 'Catboost_Reg Model', preprocessing = Raw, iterations=1000, depth=6, learning_rate=None,
                   random_strength=1, bagging_temperature=1, l2_leaf_reg=3, scale_pos_weight = 1):

    """
    Categorical Gradient Boosting Regressor
    Predicting Finishing Position
    """

    model = Catboost_Reg_Wrapper(model_class=CatBoostRegressor, model_name=model_name,  preprocessing=preprocessing,
                                  iterations=iterations, depth=depth, learning_rate=learning_rate,random_strength=random_strength,
                                  bagging_temperature=bagging_temperature, l2_leaf_reg=l2_leaf_reg, scale_pos_weight=scale_pos_weight)

    return model


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

========================= Catboost_Class_CL Model =========================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def Catboost_Class_CL(model_name = 'Catboost_Class_CL Model', preprocessing = Raw, iterations=1000, depth=6, learning_rate=None,
                   random_strength=1, bagging_temperature=1, l2_leaf_reg=3, scale_pos_weight=1, ridge=0):

    """
    Stage 1 : Categorical Gradient Boosting Classifier
    Stage 2 : Conditional Logistic Regression
    """

    model = CatBoost_Class_CL_Wrapper(model_class_1=CatBoostClassifier, model_class_2=pylogit.create_choice_model, model_name=model_name,
                                     preprocessing=preprocessing, iterations=iterations, depth=depth, learning_rate=learning_rate,
                                     random_strength=random_strength, bagging_temperature=bagging_temperature, l2_leaf_reg=l2_leaf_reg,
                                     scale_pos_weight=scale_pos_weight, ridge=ridge)

    return model

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

========================== Catboost_Reg_CL Model ==========================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def Catboost_Reg_CL(model_name = 'Catboost_Reg_CL Model', preprocessing = Raw, iterations=1000, depth=6, learning_rate=None,
                   random_strength=1, bagging_temperature=1, l2_leaf_reg=3, ridge=0):

    """
    Stage 1 : Categorical Gradient Boosting Regressor
    Stage 2 : Conditional Logistic Regression
    """

    model = Catboost_Reg_CL_Wrapper(model_class_1=CatBoostClassifier, model_class_2=pylogit.create_choice_model, model_name=model_name,
                                     preprocessing=preprocessing, iterations=iterations, depth=depth, learning_rate=learning_rate,
                                     random_strength=random_strength, bagging_temperature=bagging_temperature, l2_leaf_reg=l2_leaf_reg,
                                     ridge=ridge)

    return model

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

============================ Model Dictionary ============================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Tree_Models_Dict = {'XGBoost_Class' : XGBoost_Class,
                    'XGBoost_Reg' : XGBoost_Reg,
                    'XGBoost_Class_CL' : XGBoost_Class_CL,
                    'XGBoost_Reg_CL' : XGBoost_Reg_CL,
                    'LightGBM_Class' : LightGBM_Class,
                    'LightGBM_Reg' : LightGBM_Reg,
                    'LightGBM_Class_CL' : LightGBM_Class_CL,
                    'LightGBM_Reg_CL' : LightGBM_Reg_CL,
                    'Catboost_Class' : Catboost_Class,
                    'Catboost_Reg' : Catboost_Reg,
                    'Catboost_Class_CL' : Catboost_Class_CL,
                    'Catboost_Reg_CL' : Catboost_Reg_CL}
