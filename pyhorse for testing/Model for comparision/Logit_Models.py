#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ivan
"""

"""
Conditional Logistic Regression Models
"""

#Loading Libraries
import os
import time
import pylogit
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
from bayes_opt.event import Events
from collections import OrderedDict
from bayes_opt.util import load_logs
from bayes_opt import BayesianOptimization
from sklearn.base import BaseEstimator, ClassifierMixin
from pyhorse.Data_Preprocessing import Raw, Normalise_Race
from sklearn.model_selection import train_test_split, KFold
from pyhorse.Model_Evaluation import Kelly_Profit, newJSONLogger

#Global Parameters
Odds_col = "OD_CR_LP"
saved_models_path = "./pyhorse/Saved_Models/"
# saved_models_path = "/content/gdrive/My Drive/pyhorse/Saved_Models/"
Betting_Fraction = 0.3

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

========================== Hyperparameter Range ==========================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#Conditional Logit Parameters
ridge_Range = (0, 501)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

=============================== Test Cases ===============================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def _test_cases():
    from pyhorse import Dataset_Creation
    X, y = Dataset_Creation.Dataset_Extraction(Dataset_Creation.Get_RaceID(['2013','2014', '2015', '2016', '2017']))

    CL_Ridge_Model = CL_Ridge(preprocessing = Normalise_Race)
    CL_Ridge_Model.fit(X, y)
    CL_Ridge_Model.summary
    y_pred = CL_Ridge_Model.predict(X)
    CL_Ridge_Model.hyperparameter_selection(X, y, 0, 0, initial_probe = True)
    CL_Ridge_Model.load_hyperparameters()

    CL_Frailty_Model = CL_Frailty(preprocessing = Normalise_Race)
    CL_Frailty_Model.fit(X, y)
    CL_Frailty_Model.summary
    y_pred = CL_Frailty_Model.predict(X)
    CL_Frailty_Model.hyperparameter_selection(X, y, 0, 0, initial_probe = True)
    CL_Frailty_Model.load_hyperparameters()

    CL_Ridge_CL_Ridge_Model = CL_Ridge_CL_Ridge(preprocessing = Normalise_Race)
    CL_Ridge_CL_Ridge_Model.fit(X, y)
    CL_Ridge_CL_Ridge_Model.summary
    y_pred = CL_Ridge_CL_Ridge_Model.predict(X)
    CL_Ridge_CL_Ridge_Model.hyperparameter_selection(X, y, 0, 0, initial_probe = True)
    CL_Ridge_CL_Ridge_Model.load_hyperparameters()

    CL_Ridge_CL_Model = CL_Ridge_CL(preprocessing = Normalise_Race)
    CL_Ridge_CL_Model.fit(X, y)
    CL_Ridge_CL_Model.summary
    X, y = Dataset_Creation.Dataset_Extraction(Dataset_Creation.Get_RaceID(['2018']))
    y_pred = CL_Ridge_CL_Model.predict(X)
    CL_Ridge_CL_Model.hyperparameter_selection(X, y, 0, 0, initial_probe = True)
    CL_Ridge_CL_Model.load_hyperparameters()

    CL_Ridge_CL_Frailty_Model = CL_Ridge_CL_Frailty(preprocessing = Normalise_Race)
    CL_Ridge_CL_Frailty_Model.fit(X, y)
    CL_Ridge_CL_Frailty_Model.summary
    y_pred = CL_Ridge_CL_Frailty_Model.predict(X)
    CL_Ridge_CL_Frailty_Model.hyperparameter_selection(X, y, 0, 0, initial_probe = True)
    CL_Ridge_CL_Frailty_Model.load_hyperparameters()

    return None

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

============================ One Stage Wrapper ============================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class CL_Wrapper(BaseEstimator, ClassifierMixin):

    """
    A sklearn-style wrapper for One Stage pylogit models
    """

    def __init__(self, model_class, model_name, preprocessing, ridge, frailty):
        self.model_class = model_class
        self.model_name = model_name
        self.preprocessing = preprocessing

        """
        Hyperparameter
        """
        self.ridge = ridge
        self.frailty = frailty
        #Hyperparameter Dictionary - Bounds
        self.hyperparameter = {'ridge' : ridge_Range}
        #Track Hyperparameter Selection
        self.number_model_fitted = 0

        return None


    def fit(self, X, y, preprocess = True):

        #Making a copy of X
        X_train = X.copy()

        #Apply Preprocessing Pipeline
        if self.preprocessing().__name__ == 'pca':
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
        else :
            self.preprocessing = self.preprocessing()
            X_train = self.preprocessing.fit_transform(X_train)

        #Create specification dictionary
        model_specification = OrderedDict()
        for variable in X_train.columns[2:]:
            model_specification[variable] = 'all_same'
        zeros = np.zeros(len(model_specification))

        #Combining X and Y
        X_train = X_train.merge(y.loc[:,['RARID','HNAME','RESWL']], on=['RARID','HNAME'])
        X_train.reset_index(inplace=True, drop=True)

        #frailty
        if self.frailty == True:
            const_pos = [list(model_specification.keys()).index(Odds_col)]
            zeros[const_pos] = 1.0
        else:
            const_pos = None

        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        #Creating Model Instance
        self.model = self.model_class(data = X_train,
                                      alt_id_col = 'HNAME',
                                      obs_id_col = 'RARID',
                                      choice_col = 'RESWL',
                                      specification = model_specification,
                                      model_type="MNL")
        #Model Fitting
        self.model.fit_mle(zeros, print_res = False, ridge = self.ridge,
                           constrained_pos = const_pos, just_point=False)
        warnings.filterwarnings("default", category=FutureWarning)
        warnings.filterwarnings("default", category=UserWarning)
        warnings.filterwarnings("default", category=RuntimeWarning)

        #Model Saving
        self.fitted_model  = self.model

        #Feature Importance
        self.summary = self.fitted_model.get_statsmodels_summary()

        return None


    def predict(self, X, preprocess = True):

        #Making a copy of X
        X_test = X.copy()

        #Apply Preprocessing Pipeline
        if preprocess == True :
            #Apply Preprocessing Pipeline
            if self.preprocessing().__name__ == 'pca':
                #Slicing Odds Columns
                self.other_col = [i for i in X_test.columns if i != Odds_col]
                X_Odds = X_test.loc[:,['RARID', 'HNAME', Odds_col]]
                X_Others = X_test.loc[:,self.other_col]
                self.preprocessing = self.preprocessing()
                X_Others = self.preprocessing.fit_transform(X_Others)
                #Redefine other cols
                self.other_col = list(X_Others.columns)
                #Join the two dataframes
                X_train = pd.merge(X_Odds, X_Others, on = ['RARID','HNAME'])
            else :
                self.preprocessing = self.preprocessing()
                X_train = self.preprocessing.fit_transform(X_train)
        else :
            pass

        #Formatting into DataFrame
        Prediction = X_test.loc[:,['RARID','HNAME']]
        warnings.filterwarnings("ignore", category=FutureWarning)
        Prediction.loc[:,self.model_name] = self.fitted_model.predict(X_test)
        warnings.filterwarnings("default", category=FutureWarning)

        return Prediction


    def predict_proba(self, X, preprocess = True):

        #Making a copy of X
        X_test = X.copy()

        #Apply Preprocessing Pipeline
        if preprocess == True :
            #Apply Preprocessing Pipeline
            if self.preprocessing().__name__ == 'pca':
                #Slicing Odds Columns
                self.other_col = [i for i in X_test.columns if i != Odds_col]
                X_Odds = X_test.loc[:,['RARID', 'HNAME', Odds_col]]
                X_Others = X_test.loc[:,self.other_col]
                self.preprocessing = self.preprocessing()
                X_Others = self.preprocessing.fit_transform(X_Others)
                #Redefine other cols
                self.other_col = list(X_Others.columns)
                #Join the two dataframes
                X_train = pd.merge(X_Odds, X_Others, on = ['RARID','HNAME'])
            else :
                self.preprocessing = self.preprocessing()
                X_train = self.preprocessing.fit_transform(X_train)
        else :
            pass

        #Formatting into DataFrame
        Prediction = X_test.loc[:,['RARID','HNAME']]
        warnings.filterwarnings("ignore", category=FutureWarning)
        Prediction.loc[:,self.model_name] = self.fitted_model.predict(X_test)
        warnings.filterwarnings("default", category=FutureWarning)

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
            load_logs(optimizer, logs=[saved_models_path+logger_name+".json"])

            #Save best hyperparameter
            best_model = optimizer.max
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

        #Apply Preprocessing Pipeline
        #Apply Preprocessing Pipeline
        if self.preprocessing().__name__ == 'pca':
            #Slicing Odds Columns
            self.other_col = [i for i in X_copy.columns if i != Odds_col]
            X_Odds = X_copy.loc[:,['RARID', 'HNAME', Odds_col]]
            X_Others = X_copy.loc[:,self.other_col]
            self.preprocessing = self.preprocessing()
            X_Others = self.preprocessing.fit_transform(X_Others)
            #Redefine other cols
            self.other_col = list(X_Others.columns)
            #Join the two dataframes
            X_copy = pd.merge(X_Odds, X_Others, on = ['RARID','HNAME'])
            print(X_copy.columns)
        else :
            self.preprocessing = self.preprocessing()
            X_copy = self.preprocessing.fit_transform(X_copy)

        #Create specification dictionary
        model_specification = OrderedDict()
        for variable in X_copy.columns[2:]:
            model_specification[variable] = 'all_same'
        zeros = np.zeros(len(model_specification))

        #frailty
        if self.frailty == True:
            const_pos = [list(model_specification.keys()).index(Odds_col)]
            #The Frailty term (Final Odds) must be the final column
            zeros[const_pos] = 1.0
        else:
            const_pos = None

        """
        Beyesian Optimization
        """
        #Get RaceID in Dataset
        RaceID_List = X_copy.loc[:,'RARID'].unique()

        #Create Function to Optimize
        def Logit_fit_predict_score(ridge):

            """
            Looping Over Cross Validation Folds
            """
            Score = []
            #Cross Validation Object
            FoldCV = KFold(n_splits = 4, shuffle = True, random_state=12345).split(RaceID_List)
            for train_index, test_index in FoldCV :

                try : #Prevent Singular Matrix Issue
                    train_index, test_index = RaceID_List[train_index], RaceID_List[test_index]
                    #Building Dataset
                    X_train = X_copy.loc[X_copy.loc[:,'RARID'].isin(train_index), :]
                    y_train = y.loc[y.loc[:,'RARID'].isin(train_index), :]
                    #Combining X and Y
                    X_train = X_train.merge(y_train.loc[:,['RARID','HNAME','RESWL']], on=['RARID','HNAME'])
                    X_train.reset_index(inplace=True, drop=True)
                    X_test = X_copy.loc[X_copy.loc[:,'RARID'].isin(test_index), :]
                    y_test = y.loc[y.loc[:,'RARID'].isin(test_index), :]

                    warnings.filterwarnings("ignore", category=FutureWarning)
                    warnings.filterwarnings("ignore", category=UserWarning)
                    warnings.filterwarnings("ignore", category=RuntimeWarning)

                    #Creating Model Instance
                    model = self.model_class(data = X_train,
                                             alt_id_col = 'HNAME',
                                             obs_id_col = 'RARID',
                                             choice_col = 'RESWL',
                                             specification = model_specification,
                                             model_type="MNL")
                    model.fit_mle(zeros, print_res = False, ridge = ridge, constrained_pos = const_pos)

                    Prediction = X_test.loc[:,['RARID','HNAME']]
                    Prediction.loc[:,'prediction'] = model.predict(X_test)

                    warnings.filterwarnings("default", category=FutureWarning)
                    warnings.filterwarnings("default", category=UserWarning)
                    warnings.filterwarnings("default", category=RuntimeWarning)

                    Score.append(Kelly_Profit(Prediction, y_test, weight = Betting_Fraction))
                except :
                    pass
            Score = np.mean(Score)
            if Score == np.NaN:
                Score = 0

            return Score

        #Define BayesianOptimization instance
        optimizer = BayesianOptimization(f=Logit_fit_predict_score,
                                         pbounds=self.hyperparameter,
                                         verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
                                         random_state=1)

        #Load Hyperparamter Selection History Logs
        logger_name = self.model_name
        if os.path.exists(saved_models_path+logger_name+".json"):
            load_logs(optimizer, logs=[saved_models_path+logger_name+".json"])

        #Subscribe to Log Hyperparameter History
        logger = newJSONLogger(path=saved_models_path+logger_name+".json")
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

        #Count Initial Points
        initial_number_model_fitted = len(optimizer.space)

        #Search Default Space
        if initial_probe == True:
            for ridge in range(ridge_Range[1]):
                optimizer.probe(params={'ridge' : ridge}, lazy=False)

        #Loop over instances
        optimizer.maximize(init_points=inital_pts, n_iter=rounds)
        self.number_model_fitted = len(optimizer.res) - initial_number_model_fitted
        print("==================== %d Hyperparameters Models are fitted in %s hours ===================="
              %(self.number_model_fitted, (str(round((time.time() - start_time)/ (60*60), 2)))))

        best_model = optimizer.max
        print("The {model_name} optimizer is now aware of {num_pts} points and the best result is {Result}."
              .format(model_name = self.model_name, Result = best_model['target'], num_pts = len(optimizer.space)))

        return None


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

============================ Two Stage Wrapper ============================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class CL_CL_Wrapper(BaseEstimator, ClassifierMixin):

    """
    A sklearn-style wrapper for Two Stage pylogit models
    """

    def __init__(self, model_class, model_name, preprocessing, ridge_1, ridge_2, frailty):
        self.model_class = model_class
        self.model_name = model_name
        self.preprocessing = preprocessing

        """
        Hyperparameter
        """
        # Ridge 1 is always a hyperparameter
        self.ridge_1 = ridge_1 #There must be a ridge penalty on first stage models
        self.ridge_2 = ridge_2 #Ridge penalty on second stage logit model
        self.frailty = frailty #Boolean frailty constraint on second stage model

        #Hyperparameter Dictionary - Bounds
        if self.ridge_1 != None and self.ridge_2 != None:
            self.hyperparameter = {'ridge_1' : ridge_Range, 'ridge_2' : ridge_Range}
        elif self.ridge_1 != None and self.ridge_2 == None:
            self.hyperparameter = {'ridge_1' : ridge_Range}
        elif self.ridge_1 != None and self.frailty == True:
            self.hyperparameter = {'ridge_1' : ridge_Range}
        #Track Hyperparameter Selection
        self.number_model_fitted = 0


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

        #Combining X and Y
        X_train = X_train.merge(y.loc[:,['RARID','HNAME','RESWL']], on=['RARID','HNAME'])
        X_train.reset_index(inplace = True, drop = True)

        #Get RaceID in Dataset
        RaceID_List = X_train.loc[:,'RARID'].unique()

        #Split into Stage 1 and Stage 2
        stage_1_index, stage_2_index = train_test_split(RaceID_List, test_size=0.5, random_state=12345)

        #Building Dataset
        X_Stage_1 = X_train.loc[X_train.loc[:,'RARID'].isin(stage_1_index), :]
        # y_Stage_1 = y.loc[y.loc[:,'RARID'].isin(stage_1_index), :]
        X_Stage_2 = X_train.loc[X_train.loc[:,'RARID'].isin(stage_2_index), :]
        # y_Stage_2 = y.loc[y.loc[:,'RARID'].isin(stage_2_index), :]

        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        """
        Stage 1
        """
        #Removing Odds from Stage 1
        X_Stage_1 = X_Stage_1.loc[:, self.other_col + ['RESWL']]

        #Create specification dictionary
        model_specification_Stage_1 = OrderedDict()
        for variable in X_Stage_1.columns[2:]:
            model_specification_Stage_1[variable] = 'all_same'
        #Remove 'RESWL'
        model_specification_Stage_1.pop("RESWL")
        zeros_Stage_1 = np.zeros(len(model_specification_Stage_1))

        #Creating Model Instance
        self.model_Stage_1 = self.model_class(data = X_Stage_1,
                                              alt_id_col = 'HNAME',
                                              obs_id_col = 'RARID',
                                              choice_col = 'RESWL',
                                              specification = model_specification_Stage_1,
                                              model_type = "MNL")
        #Model Fitting
        self.model_Stage_1.fit_mle(zeros_Stage_1, print_res = False, ridge = self.ridge_1)
        self.fitted_Stage_1 = self.model_Stage_1

        """
        Stage 2
        """
        #Create DataFrame for prediction
        X_predict_Stage_1 = X_Stage_2.loc[:,self.other_col]

        #Slice in Odds Columns
        X_Stage_2 = X_Stage_2.loc[:,['HNAME','RARID', Odds_col, 'RESWL']]

        #Update Dataset with Stage 1 Prediction
        X_Stage_2.loc[:,'Fundamental_Probi'] = np.log(self.fitted_Stage_1.predict(X_predict_Stage_1))

        #Create specification dictionary
        model_specification_Stage_2 = OrderedDict()
        for variable in X_Stage_2.columns[2:]:
            model_specification_Stage_2[variable] = 'all_same'
        #Remove 'RESWL'
        model_specification_Stage_2.pop("RESWL")
        zeros_Stage_2 = np.zeros(len(model_specification_Stage_2))

        #Creating Model Instance
        self.model_Stage_2 = self.model_class(data = X_Stage_2,
                                              alt_id_col = 'HNAME',
                                              obs_id_col = 'RARID',
                                              choice_col = 'RESWL',
                                              specification = model_specification_Stage_2,
                                              model_type = 'MNL')
        #Fit according to different models
        if self.ridge_1 != None and self.ridge_2 != None:
            self.model_Stage_2.fit_mle(zeros_Stage_2, print_res = False, ridge = self.ridge_2)
        elif self.ridge_1 != None and self.ridge_2 == None:
            self.model_Stage_2.fit_mle(zeros_Stage_2, print_res = False)
        elif self.ridge_1 != None and self.frailty == True:
            #frailty
            const_pos = [list(model_specification_Stage_2.keys()).index(Odds_col)]
            zeros_Stage_2[const_pos] = 1
            self.model_Stage_2.fit_mle(zeros_Stage_2, print_res = False, constrained_pos = const_pos)

        warnings.filterwarnings("default", category=FutureWarning)
        warnings.filterwarnings("default", category=UserWarning)
        warnings.filterwarnings("default", category=RuntimeWarning)

        #Save Models
        self.fitted_Stage_2  = self.model_Stage_2

        #Feature Importance
        self.summary = [self.fitted_Stage_1.get_statsmodels_summary(), self.fitted_Stage_2.get_statsmodels_summary()]

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
        X_Stage_1 = X_test.loc[:,self.other_col]
        X_Stage_2 = X_test.loc[:,['HNAME','RARID', Odds_col]]
        X_Stage_2.loc[:,'Fundamental_Probi'] = np.log(self.fitted_Stage_1.predict(X_Stage_1))

        """
        Stage 2
        """
        Prediction = X_test.loc[:,['RARID','HNAME']]
        Prediction.loc[:,self.model_name] = self.fitted_Stage_2.predict(X_Stage_2)
        warnings.filterwarnings("default", category=FutureWarning)

        return Prediction


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
        X_Stage_1 = X_test.loc[:,self.other_col]
        X_Stage_2 = X_test.loc[:,['HNAME','RARID', Odds_col]]
        X_Stage_2.loc[:,'Fundamental_Probi'] = np.log(self.fitted_Stage_1.predict(X_Stage_1))

        """
        Stage 2
        """
        Prediction = X_test.loc[:,['RARID','HNAME']]
        Prediction.loc[:,self.model_name] = self.fitted_Stage_2.predict(X_Stage_2)
        warnings.filterwarnings("default", category=FutureWarning)

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
        if os.path.exists(saved_models_path+logger_name+".json"):
            load_logs(optimizer, logs=[saved_models_path+logger_name+".json"])

            #Save best hyperparameter
            best_model = optimizer.max
            if self.ridge_2 != None:
                self.ridge_1 = best_model['params']['ridge_1']
                self.ridge_2 = best_model['params']['ridge_2']
            else :
                self.ridge_1 = best_model['params']['ridge_1']

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
        def Logit_fit_predict_score(ridge_1, ridge_2 = None):

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

                try : #Prevent Singular Matrix Issue
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
                    X_train_Stage_1 = X_train.loc[X_train.loc[:,'RARID'].isin(stage_1_index), self.other_col +['RESWL']]

                    X_predict_Stage_1 = X_train.loc[X_train.loc[:,'RARID'].isin(stage_2_index), self.other_col]
                    X_train_Stage_2 = X_train.loc[X_train.loc[:,'RARID'].isin(stage_2_index),['HNAME', 'RARID', Odds_col, 'RESWL']]

                    #Testing Dataset
                    X_test_Stage_1 = X_test.loc[:,self.other_col]
                    X_test_Stage_2 = X_test.loc[:,['HNAME','RARID', Odds_col]]

                    """
                    Training Stage 1
                    """
                    #Create specification dictionary
                    model_specification_Stage_1 = OrderedDict()
                    for variable in X_train_Stage_1.columns[2:]:
                        model_specification_Stage_1[variable] = 'all_same'
                    #Remove 'RESWL'
                    model_specification_Stage_1.pop("RESWL")
                    zeros_Stage_1 = np.zeros(len(model_specification_Stage_1))

                    model_stage_1 = self.model_class(data = X_train_Stage_1,
                                                     alt_id_col = 'HNAME',
                                                     obs_id_col = 'RARID',
                                                     choice_col = 'RESWL',
                                                     specification = model_specification_Stage_1,
                                                     model_type="MNL")
                    model_stage_1.fit_mle(zeros_Stage_1, print_res = False, ridge = ridge_1)

                    """
                    Training Stage 2
                    """
                    X_train_Stage_2.loc[:,'Fundamental_Probi'] = np.log(model_stage_1.predict(X_predict_Stage_1))
                    #Create specification dictionary
                    model_specification_Stage_2 = OrderedDict()
                    for variable in X_train_Stage_2.columns[2:]:
                        model_specification_Stage_2[variable] = 'all_same'
                    #Remove 'RESWL'
                    model_specification_Stage_2.pop("RESWL")
                    zeros_Stage_2 = np.zeros(len(model_specification_Stage_2))
                    model_Stage_2 = self.model_class(data = X_train_Stage_2,
                                                     alt_id_col = 'HNAME',
                                                     obs_id_col = 'RARID',
                                                     choice_col = 'RESWL',
                                                     specification = model_specification_Stage_2,
                                                     model_type = 'MNL')
                    #Fit according to different models
                    if self.frailty == False and ridge_2 == None:
                        model_Stage_2.fit_mle(zeros_Stage_2, print_res = False)
                    elif self.frailty == True and ridge_2 == None:
                        #frailty
                        const_pos = [list(model_specification_Stage_2.keys()).index(Odds_col)]
                        zeros_Stage_2[const_pos] = 1
                        model_Stage_2.fit_mle(zeros_Stage_2, print_res = False, constrained_pos = const_pos)
                    elif ridge_2 != None:
                        model_Stage_2.fit_mle(zeros_Stage_2, print_res = False, ridge = ridge_2)

                    """
                    Prediction
                    """
                    #Stage 1
                    X_test_Stage_2.loc[:,'Fundamental_Probi'] = np.log(model_stage_1.predict(X_test_Stage_1))

                    #Stage 2
                    Prediction = X_test.loc[:,['RARID','HNAME']]
                    Prediction.loc[:,'prediction'] = model_Stage_2.predict(X_test_Stage_2)

                    Score.append(Kelly_Profit(Prediction, y_test, weight = Betting_Fraction))
                except:
                    pass
            Score = np.mean(Score)
            if Score == np.NaN:
                Score = 0
            warnings.filterwarnings("default", category=FutureWarning)
            warnings.filterwarnings("default", category=UserWarning)
            warnings.filterwarnings("default", category=RuntimeWarning)

            return Score

        #Hyperparameter Dictionary - Bounds
        optimizer = BayesianOptimization(f=Logit_fit_predict_score,
                                         pbounds=self.hyperparameter,
                                         verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
                                         random_state=1)

        #Load Hyperparamter Selection History Logs
        logger_name = self.model_name
        if os.path.exists(saved_models_path+logger_name+".json"):
            load_logs(optimizer, logs=[saved_models_path+logger_name+".json"])

        #Subscribe to Log Hyperparameter History
        logger = newJSONLogger(path=saved_models_path+logger_name+".json")
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

        #Count Initial Points
        initial_number_model_fitted = len(optimizer.space)

        #Search Default Space
        if initial_probe == True:
            if self.ridge_1 != None and self.ridge_2 != None:
                for ridge_1 in range(ridge_Range[1]):
                    for ridge_2 in range(25):
                        optimizer.probe(params={'ridge_1' : ridge_1, 'ridge_2' : ridge_2}, lazy=False)
            elif self.ridge_1 != None and self.ridge_2 == None:
                for ridge_1 in range(ridge_Range[1]):
                    optimizer.probe(params={'ridge_1' : ridge_1}, lazy=False)
            elif self.ridge_1 != None and self.frailty == True:
                for ridge_1 in range(ridge_Range[1]):
                    optimizer.probe(params={'ridge_1' : ridge_1}, lazy=False)

        #Loop over instances
        optimizer.maximize(init_points=inital_pts, n_iter=rounds)
        self.number_model_fitted = len(optimizer.res) - initial_number_model_fitted
        print("==================== %d Hyperparameters Models are fitted in %s hours ===================="
              %(self.number_model_fitted, (str(round((time.time() - start_time)/ (60*60), 2)))))

        best_model = optimizer.max
        print("The {model_name} optimizer is now aware of {num_pts} points and the best result is {Result}."
              .format(model_name = self.model_name, Result = best_model['target'], num_pts = len(optimizer.space)))

        return None

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

============================= CL_Ridge Model =============================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def CL_Ridge(model_name = 'CL_Ridge Model', preprocessing = Raw, ridge = 0):

    """
    One Stage Conditional Logsitic Regression
    Ridge Penalty on Coefficients
    Public Implied probability as an attribute
    """

    model = CL_Wrapper(model_class=pylogit.create_choice_model, model_name=model_name, preprocessing=preprocessing, ridge=ridge, frailty = False)

    return model

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

============================= CL_Frailty Model ============================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def CL_Frailty(model_name = 'CL_Frailty Model', preprocessing = Raw, ridge = 0):

    """
    One Stage Conditional Logsitic Regression
    Frailty on public implied probability
    """

    model = CL_Wrapper(model_class=pylogit.create_choice_model,  model_name=model_name, preprocessing=preprocessing, ridge=ridge, frailty=True)

    return model

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

========================= CL_Ridge_CL_Ridge Model =========================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def CL_Ridge_CL_Ridge(model_name = 'CL_Ridge_CL_Ridge Model', preprocessing = Raw, ridge_1 = 0, ridge_2 = 0):

    """
    Two Stage Conditional Logistic Regression
    Conditional Logistic Regression Strength Model : Ridge Penalty
    Conditional Logistic Regression Combined Model : Strength Probability + Public Odds implied Probability, Ridge Penalty
    """

    model = CL_CL_Wrapper(model_class=pylogit.create_choice_model, model_name=model_name, preprocessing=preprocessing,
                          ridge_1=ridge_1, ridge_2=ridge_2, frailty=False)

    return model

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

============================ CL_Ridge_CL Model ============================


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def CL_Ridge_CL(model_name = 'CL_Ridge_CL Model', preprocessing = Raw, ridge = 0):

    """
    Two Stage Conditional Logistic Regression
    Conditional Logistic Regression Strength Model : Ridge Penalty
    Conditional Logistic Regression Combined Model : Strength Probability + Public Odds implied Probability, No Penalty
    """

    model = CL_CL_Wrapper(model_class=pylogit.create_choice_model, model_name=model_name, preprocessing=preprocessing,
                          ridge_1=ridge, ridge_2=None, frailty=False)

    return model

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

======================== CL_Ridge_CL_Frailty Model ========================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def CL_Ridge_CL_Frailty(model_name = 'CL_Ridge_CL_Frailty Model', preprocessing = Raw, ridge=0):

    """
    Two Stage Conditional Logistic Regression
    Conditional Logistic Regression Strength Model : Ridge Penalty
    Conditional Logistic Regression Combined Model : Strength Probability + Public Odds implied Probability, Frailty on public implied probability
    """

    model = CL_CL_Wrapper(model_class=pylogit.create_choice_model, model_name=model_name, preprocessing=preprocessing,
                          ridge_1=ridge, ridge_2=None, frailty=True)

    return model

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

============================ Model Dictionary ============================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Logit_Models_Dict = {'CL_Ridge' : CL_Ridge,
                     'CL_Frailty' : CL_Frailty,
                     'CL_Ridge_CL_Ridge' : CL_Ridge_CL_Ridge,
                     'CL_Ridge_CL' : CL_Ridge_CL,
                     'CL_Ridge_CL_Frailty' : CL_Ridge_CL_Frailty}
