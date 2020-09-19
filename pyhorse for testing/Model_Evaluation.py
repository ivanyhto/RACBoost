#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ivan
"""

"""
Hyperparameter Selection, Model Evaluation and Wagering Strategies
"""

#Import Libraries
from tabulate import tabulate
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from bayes_opt.logger import JSONLogger
pd.options.mode.chained_assignment = None

bet_size= 10
Final_Odds_col = "RESFO"
saved_models_path = "./pyhorse/Saved_Models/"

# saved_models_path = "./Downloads/"

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

======================== Hyperparameter Selection ========================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class newJSONLogger(JSONLogger) :

      def __init__(self, path):
        self._path=None
        super(JSONLogger, self).__init__()
        self._path = path if path[-5:] == ".json" else path + ".json"


def hyperparameter_results():

    """
    Get json files
    """
    json_files = [pos_json for pos_json in os.listdir(saved_models_path) if pos_json.endswith('.json')]

    result = []
    for logger_name in json_files:
        model_name = logger_name.split()[0]
        preprocess_name = logger_name.split()[1]
        data = []
        with open(saved_models_path + logger_name, 'r') as json_file:
            for line in json_file:
                data.append(json.loads(line))
        tested = len(data)
        space = len(data[0]['params'])
        best_result = max([i['target'] for i in data])
        avg_result = np.mean([i['target'] for i in data])
        result.append([model_name, preprocess_name, tested, space, best_result, avg_result])

    columns = ['Model', 'Pre-processing', 'Hyper Tested', 'No. Hyper', 'Best', 'Avg.']
    evaluation_results = pd.DataFrame(result, columns = columns)
    evaluation_results = evaluation_results.sort_values(['Model', 'Pre-processing'])

    """
    Get Moodel / Pre-process summary statistics
    """
    #Title
    title = 'Hyper-parameter Evaluation Results'

    #Summary
    num_model = len(evaluation_results.index)
    num_positive = sum(evaluation_results.loc[:,'Best'] > 0)
    best_result = str(evaluation_results.loc[:,'Best'].max().round(4)*100) + '%'
    num_hyper = evaluation_results.loc[:,'Hyper Tested'].sum()
    summary = pd.DataFrame([['Number of Models : ', num_model], ['Number of +ve Models : ',num_positive],
                            ['Percentage of +ve Models : ', str(round((num_positive / num_model),4)*100) + '%'], ['Best Result : ', best_result],
                           ['Number of Hyper Tested : ', num_hyper]])
    summary = tabulate(summary, showindex = False, numalign='right', tablefmt="plain")

    #Group By Model and Pre-processing
    model_best = evaluation_results.groupby('Model').max()['Best'].reset_index()
    model_best.columns = ['Model', 'Model Best']
    model_best.sort_values('Model Best', inplace = True, ascending = False)
    model_best.reset_index(drop=True, inplace = True)
    model_best.loc[:,'Model Best'] = model_best.loc[:,'Model Best'].map(lambda x : str(round(x*100, 2)) + '%')
    model_mean = evaluation_results.groupby('Model').mean()['Best'].reset_index()
    model_mean.columns = ['Model', 'Model Average']
    model_mean.loc[:,'Model Average'] = model_mean.loc[:,'Model Average'].map(lambda x : str(round(x*100, 2)) + '%')
    model = model_best.merge(model_mean, on='Model')

    preprocess_best = evaluation_results.groupby('Pre-processing').max()['Best'].reset_index()
    preprocess_best.columns = ['Pre-processing', 'Pre-processing Best']
    preprocess_best.sort_values('Pre-processing Best', inplace = True, ascending = False)
    preprocess_best.reset_index(drop=True, inplace = True)
    preprocess_best.loc[:,'Pre-processing Best'] = preprocess_best.loc[:,'Pre-processing Best'].map(lambda x : str(round(x*100, 2)) + '%')
    preprocess_mean = evaluation_results.groupby('Pre-processing').mean()['Best'].reset_index()
    preprocess_mean.columns = ['Pre-processing', 'Pre-processing Average']
    preprocess_mean.loc[:,'Pre-processing Average'] = preprocess_mean.loc[:,'Pre-processing Average'].map(lambda x : str(round(x*100, 2)) + '%')
    preprocess = preprocess_best.merge(preprocess_mean, on='Pre-processing')

    groupby_model_pre = model.copy()
    groupby_model_pre.loc[:,'Pre-processing'] = preprocess.loc[:,'Pre-processing']
    groupby_model_pre.loc[:,'Pre-processing Best'] = preprocess.loc[:,'Pre-processing Best']
    groupby_model_pre.loc[:,'Pre-processing Average'] = preprocess.loc[:,'Pre-processing Average']
    groupby_model_pre.fillna(' ', inplace = True)

    groupby_summ = tabulate(groupby_model_pre, headers = ['Model', 'Best', 'Average', 'Pre-processing', 'Best','Average'],
                            showindex = False, numalign='center', floatfmt=(".4f"))

    #Result Table
    result_table = tabulate(evaluation_results, headers = columns, showindex = False, numalign='center', floatfmt=(".4f"))
    best_table = evaluation_results.sort_values(['Best'], ascending = False).head(5)
    best_table = tabulate(best_table, headers = columns, showindex = False, numalign='center', floatfmt=(".4f"))

    table_len = len(result_table.split('\n')[0])
    seperator = '\n\n' + '='*(table_len+2) + '\n\n'
    title_center = ' ' * int((table_len-len(title))/2)

    #Construct Table
    title_table = '\n' + title_center + title + seperator \
                        + summary + seperator \
                        + groupby_summ + seperator \
                        + result_table + seperator \
                        + best_table + seperator.rstrip()

    print(evaluation_results.loc[:,'Hyper Tested'].sum())
    print(title_table)
    return None

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

============================= Exotic Betting =============================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

============================ Profit Evaluation ============================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def Prediction_accuracy(y_pred, y_true):

    """
    Normalised Discounted Cumulative Gain
    """
    # group by race -> dcg
    return ndcg_score(y_true, y_pred)


def Kelly_Profit(y_pred, y_true, get_history = False, weight = 1):

    """
    Calculate the Profit from betting accoridng to the Kelly Criterion
    Parameters
    ----------
    y_pred : DataFrame of [RARID, HNAME, prediction]
    y_true : DataFrane of [RARID, HNAME, RESFO, RESWL]
    Reaturns
    ---------
    score : float profitability
    """
    #Initial Capital Bankroll
    bankroll = 100000

    y_true = y_true.merge(y_pred, on = ['RARID','HNAME'])
    win_col = ''.join(set(y_true.columns) - set(['RARID', 'HNAME', Final_Odds_col, 'RESWL', 'RESFP', 'ODPLA']))

    profit_history = []
    #Loop over races
    for RARID, race in y_true.groupby('RARID'):
        if bankroll != 0:
            #Calculate kelly bet
            bet_history = Kelly_Criterion(race, bankroll, weight)

            #Calculate Profits
            bet_history.loc[:,'Kelly_Profit'] = race.loc[:,'RESWL'] * race.loc[:,Final_Odds_col] * bet_history.loc[:,'to_bet'] \
                                                - bet_history.loc[:,'to_bet']
            bankroll += bet_history.loc[:,'Kelly_Profit'].sum()
            profit_history.append(bankroll)
        else :
            profit_history.append(0)

    if get_history == True:
        return pd.Series(profit_history)
    else :
        return (bankroll - 100000) / 100000


def Gen_Kelly_Profit(y_pred, y_true, get_history = False, weight = 1):

    """
    Calculate the Profit from betting the Generalised Kelly Criterion
    Parameters
    ----------
    y_pred : DataFrame of [RARID, HNAME, ...]
    y_true : DataFrane of
    Reaturns
    ---------
    score : float profitability
    """
    #Initial Capital Bankroll
    bankroll = 10000

    y_true = y_true.merge(y_pred, on = ['RARID','HNAME'])
    profit_history = []
    #Loop over races
    for RARID, race in y_true.groupby('RARID'):

        if bankroll != 0:

            #Calculate kelly bet
            bet_history = Gen_Kelly_Criterion(race, bankroll, weight)

            #Calculate Profits
            bet_history.loc[:,'Kelly_Profit'] = race.loc[:,'RESWL'] * race.loc[:,Final_Odds_col] * bet_history.loc[:,'to_bet'] \
                                                - bet_history.loc[:,'to_bet']
            bankroll += bet_history.loc[:,'Kelly_Profit'].sum()
            profit_history.append(bankroll)
        else :
            profit_history.append(0)

    if get_history == True:
        return pd.Series(profit_history)
    else :
        return (bankroll - 10000) / 10000


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

===========================  Wagering Strategy ============================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def Kelly_Criterion(race, bankroll, weight=1):

    """
    Calculate Fraction of Money to Wager on According to Kelly Criterion
    Evaluated Race by Race
    Note : Odds : "RESFO : 1"
    Parameters
    ----------
    race : DataFrame of [RARID, HNAME, RESFO, Probi_WIN]
    Weight : 0-1, Percentage of Kelly Suggested Fraction to bet on
    Returns
    -------
    Fraction to Bet on for each Horse
    """
    win_col = ''.join(set(race.columns) - set(['RARID', 'HNAME', Final_Odds_col, 'RESWL', 'RESFP', 'ODPLA']))

    #Kelly Criterion
    race.loc[:,Final_Odds_col] = race.loc[:,Final_Odds_col] - 1
    race.loc[:,'Expected_Value'] = race.loc[:,win_col] * race.loc[:,Final_Odds_col]
    race.loc[:,'Kelly_Fraction'] = weight * ((race.loc[:,win_col] * (race.loc[:,Final_Odds_col] + 1) - 1) / race.loc[:,Final_Odds_col])
    race.loc[:,'Kelly_Fraction'] = np.maximum(race.loc[:,'Kelly_Fraction'], 0)
    #Round to Bet Size
    race.loc[:,'to_bet'] = (race.loc[:,'Kelly_Fraction'] * bankroll).apply(lambda x : round(x/bet_size,0)*bet_size)

    return race.loc[:,['RARID', 'HNAME', 'to_bet']]


def Gen_Kelly_Criterion(race, bankroll, weight=1):

    """
    Calculate Fraction of Money to Wager on According to Generalised Kelly Criterion
    Evaluated Race by Race
    Note : Odds : "RESFO : 1"
    Parameters
    ----------
    y_pred : DataFrame of [RARID, HNAME, RESFO, Probi_WIN]
    Weight : 0-1, Percentage of Kelly Suggested Fraction to bet on
    Returns
    -------
    Fraction to Bet on for each Horse
    """
    win_col = ''.join(set(race.columns) - set(['RARID', 'HNAME', Final_Odds_col, 'RESWL', 'RESFP', 'ODPLA']))
    race.loc[:,'Expected_Value'] = race.loc[:,win_col] * race.loc[:,Final_Odds_col]
    race.sort_values('Expected_Value', ascending = False, inplace=True)

    #Kelly Criterion
    S = pd.DataFrame()
    rS = 1
    for index, row in race.iterrows():
        if row['Expected_Value'] > rS:
            S = S.append(row)
            rS = (1 - S.loc[:,win_col].sum()) / (1 - (1/S.loc[:,Final_Odds_col]).sum())

    #Kelly Fraction
    race.loc[:,'Kelly_Fraction'] = 0
    race.loc[S.index,'Kelly_Fraction'] = race.loc[S.index,win_col] - (rS/race.loc[S.index,Final_Odds_col])

    #Round to Bet Size
    race.loc[:,'to_bet'] = (race.loc[:,'Kelly_Fraction'] * bankroll).apply(lambda x : round(x/bet_size,0)*bet_size)

    return race.loc[:,['RARID', 'HNAME', 'to_bet']]

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

==============================  Wagering Map ==============================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
wagering_map = {'Kelly_Profit' : Kelly_Criterion,
                'Gen_Kelly_Criterion' : Gen_Kelly_Criterion}