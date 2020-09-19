#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ivan
"""

"""
Feature : Racetrack Conditiion
"""

#Loading Libraries
import os
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from pyhorse.Database_Management import Extraction_Database

Aux_Reg_Path = 'pyhorse/Feature_Creation/Auxiliary_Regression/'

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

============================ Support Functions ============================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def Distance_Similarity(Distance):

    """
    Parameters
    ----------
    Distance : eg 1000, 1200, 1400, 1600, 1650, 1800, 2000, 2200, 2400

    Returns
    -------
    Dictionary of percentage difference
    """

    Distance_Dict = {1000: {1200:-0.2, 1400:-0.4, 1600:-0.6, 1650:-0.65, 1800:-0.8, 2000:-1, 2200:-1.2, 2400:-1.4},
                     1200: {1000:-1/6, 1400:-1/6, 1600:-1/3, 1650:-0.375, 1800:-0.5, 2000:-2/3, 2200:-5/6, 2400:-1},
                     1400: {1000:-2/7, 1200:-1/7, 1600:-1/7, 1650:-5/28, 1800:-2/7, 2000:-3/7, 2200:-4/7, 2400:-5/7},
                     1600: {1000:-0.375, 1200:-0.25, 1400:-0.125, 1650:-0.03125, 1800:-0.125, 2000:-0.25, 2200:-0.375, 2400:-0.5},
                     1650: {1000:-13/33, 1200:-3/11, 1400:-5/33, 1600:-1/33, 1800:-1/11, 2000:-7/33, 2200:-1/3, 2400:-5/11},
                     1800: {1000:-4/9, 1200:-1/3, 1400:-2/9, 1600:-1/9, 1650:-1/12, 2000:-1/9, 2200:-2/9, 2400:-1/3},
                     2000: {1000:-0.5, 1200:-0.4, 1400:-0.3, 1600:-0.2, 1650:-0.175, 1800:-0.1, 2200:-0.1, 2400:-0.2},
                     2200: {1000:-6/11, 1200:-5/11, 1400:-4/11, 1600:-3/11, 1650:-0.25, 1800:-2/11, 2000:-1/11, 2400:-1/11},
                     2400: {1000:-7/12, 1200:-0.5, 1400:-5/12, 1600:-1/3, 1650:-5/16, 1800:-1/4, 2000:-1/6, 2200:-1/12}}

    return Distance_Dict[Distance]


def Pref_Distance_Similarity(Distance):

    """
    Parameters
    ----------
    Distance : eg 1000, 1200, 1400, 1600, 1650, 1800, 2000, 2200, 2400

    Returns
    -------
    Dictionary of percentage difference
    """

    Distance_Dict = {1000: {1000:1, 1200:-0.2, 1400:-0.4, 1600:-0.6, 1650:-0.65, 1800:-0.8, 2000:-1, 2200:-1.2, 2400:-1.4},
                     1200: {1000:-1/6, 1200:1, 1400:-1/6, 1600:-1/3, 1650:-0.375, 1800:-0.5, 2000:-2/3, 2200:-5/6, 2400:-1},
                     1400: {1000:-2/7, 1200:-1/7, 1400:1, 1600:-1/7, 1650:-5/28, 1800:-2/7, 2000:-3/7, 2200:-4/7, 2400:-5/7},
                     1600: {1000:-0.375, 1200:-0.25, 1400:-0.125, 1600:1, 1650:-0.03125, 1800:-0.125, 2000:-0.25, 2200:-0.375, 2400:-0.5},
                     1650: {1000:-13/33, 1200:-3/11, 1400:-5/33, 1600:-1/33, 1650:1, 1800:-1/11, 2000:-7/33, 2200:-1/3, 2400:-5/11},
                     1800: {1000:-4/9, 1200:-1/3, 1400:-2/9, 1600:-1/9, 1650:-1/12, 1800:1, 2000:-1/9, 2200:-2/9, 2400:-1/3},
                     2000: {1000:-0.5, 1200:-0.4, 1400:-0.3, 1600:-0.2, 1650:-0.175, 1800:-0.1, 2000:1, 2200:-0.1, 2400:-0.2},
                     2200: {1000:-6/11, 1200:-5/11, 1400:-4/11, 1600:-3/11, 1650:-0.25, 1800:-2/11, 2000:-1/11, 2200:1, 2400:-1/11},
                     2400: {1000:-7/12, 1200:-0.5, 1400:-5/12, 1600:-1/3, 1650:-5/16, 1800:-1/4, 2000:-1/6, 2200:-1/12, 2400:1}}

    return Distance_Dict[Distance]


def Going_Similarity(Going):

    """
    Parameters
    ----------
    Going : eg 好快, 好黏, 黏地, 黏軟, 黏軟, 濕快, 泥快, 泥好, 泥好

    Returns
    -------
    Dictionary of percentage difference
    """

    Going_Dict = {'好快': {'好快':1, '好地':-1/5, '好黏':-2/11, '黏地':-3/11, '黏軟':-4/11, '軟地':-5/11},
                  '好地': {'好快':-1/6, '好地':1, '好黏':-1/12, '黏地':-1/6, '黏軟':-1/4, '軟地':-1/3},
                  '好黏': {'好快':-2/13, '好地':-1/13, '好黏':1, '黏地':-1/13, '黏軟':-2/13, '軟地':-3/13},
                  '黏地': {'好快':-3/14, '好地':-1/7, '好黏':-1/14, '黏地':1, '黏軟':-1/14, '軟地':-2/14},
                  '黏軟': {'好快':-4/15, '好地':-1/5, '好黏':-2/15, '黏地':-1/15, '黏軟':1, '軟地':-1/15},
                  '軟地': {'好快':-5/16, '好地':-1/4, '好黏':-3/16, '黏地':-2/16, '黏軟':-1/16, '軟地':1},

                  '濕快': {'濕快':1, '泥快':-1/10, '泥好':-2/10, '濕慢':-4/10},
                  '泥快': {'濕快':-1/11, '泥快':1, '泥好':-1/11, '濕慢':-3/11},
                  '泥好': {'濕快':-2/12, '泥快':-1/12, '泥好':1, '濕慢':-1/3},
                  '濕慢': {'濕快':-3/8, '泥快':-5/16, '泥好':-1/4, '濕慢':1}}

    return Going_Dict[Going]

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

========================= Preference Calculations =========================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def Fit_Residual_Model(Raceday):

    """
    This Function Fits the Residual Models for the following Preference Subjects
    HPRE_DIST, HPRE_GO, HPRE_SUR, HPRE_PFL, JPRE_DIST, JPRRE_GO, JPRE_SUR, JPRE_LOC, JPRE_PFL
    SPRE_DIST, SPRE_GO, SPRE_SUR, SPRE_LOC, SPRE_PFL
    Parameter
    ---------
    Raceday : Fit Models using Data up to this point
    """
    Preference_Dict = {'HPRE_DIST' : '_DIST','HPRE_GO' : '_GO','HPRE_SUR' : '_SUR','HPRE_PFL' : '_PFL',
                       'JPRE_DIST' : '_JDIST','JPRE_GO' : '_JGO','JPRE_SUR' : '_JSUR','JPRE_LOC' : '_JLOC',
                       'JPRE_PFL' : '_JPFL','SPRE_DIST' : '_SDIST','SPRE_GO' : '_SGO','SPRE_SUR' : '_SSUR',
                       'SPRE_LOC' : '_SLOC','SPRE_PFL' : '_SPFL'}
    """
    Extract Dataset
    """
    Extraction = Extraction_Database("""
                                     Select * from
                                     (Select * from FeatureDb where RARID <
                                     (Select min(RARID) from RaceDb where RADAT = {Raceday})) A,
                                     (Select HNAME, RARID, RESFP from RaceDb
                                     where RADAT < {Raceday}) B
                                     where A.HNAME = B.HNAME and A.RARID = B.RARID
                                     """.format(Raceday = Raceday))
    Extraction.drop(['HNAME','RARID'],  axis = 1, inplace = True)

    def _fit_Res_model(Subject, Condition, X ,y):

        Feature_Targets = [i for i in Extraction.columns if i !='RESFP' and i[-len(Condition):] != Condition]
        if Subject in ['HPRE_GO', 'HPRE_PFL']:
            Feature_Targets = [i for i in Feature_Targets if i[-6:] != '_GOPFL']
        X = X.loc[:,Feature_Targets]
        #Model Fitting
        model = CatBoostRegressor(allow_writing_files=False)
        model.fit(X, y, verbose = False)
        #Save Model
        with open(Aux_Reg_Path + Subject+'_Model.joblib', 'wb') as location:
            joblib.dump(model, location)
        return model

    for Subject, Condition in Preference_Dict.items():
        #Run Functions
        y = Extraction.loc[:,'RESFP']
        try : #Handle exception of first races, where there are no data
            _fit_Res_model(Subject, Condition, Extraction ,y)
        except :
            pass
    return None


def Preference_Residuals(Feature_DF, Result_DF):

    """
    Calculate the Preference Residuals
    For each Preference target, predict a finishing position using the respective Auxiliary Regression,
    Residual = Subtract the predicted position from the actual finishing position
    Parameters
    ----------
    Feature_DF : Feature Dataset for a raceday
    Result_DF : Post race Dataset for a raceday
    """

    Preference_Dict = {'HPRE_DIST' : '_DIST',
                       'HPRE_GO' : '_GO',
                        'HPRE_SUR' : '_SUR',
                        'HPRE_PFL' : '_PFL',
                        'JPRE_DIST' : '_JDIST',
                        'JPRE_GO' : '_JGO',
                        'JPRE_SUR' : '_JSUR',
                        'JPRE_LOC' : '_JLOC',
                        'JPRE_PFL' : '_JPFL',
                        'SPRE_DIST' : '_SDIST',
                        'SPRE_GO' : '_SGO',
                        'SPRE_SUR' : '_SSUR',
                        'SPRE_LOC' : '_SLOC',
                        'SPRE_PFL' : '_SPFL'}

    #Dataset Preperation
    Dataset = Feature_DF.merge(Result_DF.loc[:,['HNAME','RARID','RESFP']], on=['HNAME','RARID'])
    Residual_DF = Dataset.loc[:,['HNAME', 'RARID', 'RESFP']]

    def _predict_Res_model(Subject, Condition, Dataset):

        Feature_Targets = [i for i in Dataset.columns if i not in ['HNAME','RARID','RESFP'] and i[-len(Condition):] != Condition]
        if Subject in ['HPRE_GO', 'HPRE_PFL']:
            Feature_Targets = [i for i in Feature_Targets if i[-6:] != '_GOPFL']
        X = Dataset.loc[:,Feature_Targets]

        #Load Model
        if os.path.isfile(Aux_Reg_Path + Subject + '_Model.joblib'):
            with open(Aux_Reg_Path + Subject + '_Model.joblib', 'rb') as location:
                model = joblib.load(location)
            Residual_DF.loc[:,Subject + '_RES'] = model.predict(X)
        else :
            Residual_DF.loc[:,Subject + '_RES'] = Residual_DF.loc[:,'RESFP']

    for Subject, Condition in Preference_Dict.items():
        #Run Functions
        _predict_Res_model(Subject, Condition, Dataset)

    for col in Residual_DF.columns[3:]:
        Residual_DF.loc[:,col] = Residual_DF.loc[:,'RESFP'] - Residual_DF.loc[:,col]

    Residual_DF.drop(columns=['RESFP'], inplace = True)

    return Residual_DF


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

================================ Distance ================================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
RC_DIST_DAY_DIST
"""

def RC_DIST_DAY_DIST(Dataframe, HNAME_List, Raceday):

    """
    Day since Distance
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, RC_DIST_DAY_DIST]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RADIS']]
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Extraction = Extraction_Database("""
                                     Select HNAME, max(RADAT) Dist_Date from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RADIS = {Distance}
                                     Group by HNAME
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Distance = Distance))
    if len(Extraction) == 0:
        Feature_DF.loc[:,'RC_DIST_DAY_DIST'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','RC_DIST_DAY_DIST']]
        return Feature_DF

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'Dist_Date'] = Feature_DF.loc[:,'Dist_Date'].apply(float)
    Feature_DF.loc[:,'Dist_Date'].fillna(Feature_DF.loc[:,'Dist_Date'].min(), inplace = True)

    Raceday = pd.to_datetime(Raceday, format = '%Y%m%d')
    Feature_DF.loc[:,'RC_DIST_DAY_DIST'] = Raceday - pd.to_datetime(Feature_DF.loc[:, 'Dist_Date'], format = '%Y%m%d')
    Feature_DF.loc[:,'RC_DIST_DAY_DIST'] = Feature_DF.loc[:,'RC_DIST_DAY_DIST'].apply(lambda x : int(str(x).split('days')[0]))

    Feature_DF.loc[:,'RC_DIST_DAY_DIST'].fillna(Feature_DF.loc[:,'RC_DIST_DAY_DIST'].max(), inplace = True)
    Feature_DF.loc[:,'RC_DIST_DAY_DIST'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_DIST_DAY_DIST']]

    return Feature_DF

"""
RC_DIST_DAY_SIM_DIST
"""

def RC_DIST_DAY_SIM_DIST(Dataframe, HNAME_List, Raceday):

    """
    Day since similar Distance
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, RC_DIST_DAY_SIM_DIST]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RADIS']]
    Distance = Dataframe.loc[:,'RADIS'].values[0]
    Dist_Dict = Distance_Similarity(Distance)

    Extraction = Extraction_Database("""
                                     Select HNAME, max(RADAT) Dist_Date, RADIS from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RADIS != {Distance}
                                     Group by HNAME, RADIS
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Distance = Distance))

    if len(Extraction) == 0:
        Feature_DF.loc[:,'RC_DIST_DAY_SIM_DIST'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','RC_DIST_DAY_SIM_DIST']]
        return Feature_DF

    Extraction.replace({'RADIS': Dist_Dict}, inplace = True)
    Raceday = pd.to_datetime(Raceday, format = '%Y%m%d')
    Extraction.loc[:,'Days'] = Raceday - pd.to_datetime(Extraction.loc[:, 'Dist_Date'], format = '%Y%m%d')
    Extraction.loc[:,'Days'] = Extraction.loc[:,'Days'].apply(lambda x : int(str(x).split('days')[0]))

    def Normalise(group):
        group.loc[:,'Norm'] = np.exp(group.loc[:,'RADIS']) / np.exp(group.loc[:,'RADIS']).sum()
        group.loc[:,'Normed'] = group.loc[:,'Norm'] * group.loc[:,'Days']
        return group.loc[:,'Normed'].sum()

    Extraction = Extraction.groupby('HNAME').apply(Normalise).reset_index()
    Extraction.columns = ['HNAME','RC_DIST_DAY_SIM_DIST']
    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'RC_DIST_DAY_SIM_DIST'].fillna(Feature_DF.loc[:,'RC_DIST_DAY_SIM_DIST'].max(), inplace = True)
    Feature_DF.loc[:,'RC_DIST_DAY_SIM_DIST'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_DIST_DAY_SIM_DIST']]

    return Feature_DF

"""
RC_DIST_AVG_DIST
"""

def RC_DIST_AVG_DIST(Dataframe, HNAME_List, Raceday):

    """
    Average Recent Distance
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, RC_DIST_AVG_DIST]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RADIS']]

    Extraction = Extraction_Database("""
                                     Select HNAME, RADAT, RADIS from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    Extraction.loc[:, 'RADAT'] = Extraction.loc[:, 'RADAT'].astype(float)

    Dist = []
    for name, group in Extraction.groupby('HNAME'):
        Dist.append([name, group.nlargest(3, 'RADAT').loc[:,'RADIS'].mean()])
    Avg_Dist = pd.DataFrame(Dist, columns=['HNAME','Avg_Dist'])

    Feature_DF = Feature_DF.merge(Avg_Dist, how='left')
    Feature_DF.loc[:,'RC_DIST_AVG_DIST'] = ((Feature_DF.loc[:,'Avg_Dist'] - Feature_DF.loc[:,'RADIS']) / Feature_DF.loc[:,'RADIS']).abs()
    Feature_DF.loc[:,'RC_DIST_AVG_DIST'].fillna(Feature_DF.loc[:,'RC_DIST_AVG_DIST'].max(), inplace = True)
    Feature_DF.loc[:,'RC_DIST_AVG_DIST'].fillna(0, inplace=True)
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_DIST_AVG_DIST']]

    return Feature_DF

"""
RC_DIST_RANGE_DIST
"""

def RC_DIST_RANGE_DIST(Dataframe, HNAME_List, Raceday):

    """
    T3 Distribution Limit on Distance
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, RC_DIST_RANGE_DIST]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Distance = Dataframe.loc[:,'RADIS'].values[0]


    Extraction = Extraction_Database("""
                                     Select HNAME, max(RADIS) max, min(RADIS) min from RaceDb
                                     where  RADAT < {Raceday} and RESFP <= 3 and HNAME in {HNAME_List}
                                     Group by HNAME
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    Extraction.loc[:,'mid'] = (Extraction.loc[:,'max'] + Extraction.loc[:,'min']) / 2
    Feature_DF = Feature_DF.merge(Extraction, how='left')

    Feature_DF.loc[Feature_DF.loc[:,'max'] < Distance,'RC_DIST_RANGE_DIST'] = \
        Feature_DF.loc[Feature_DF.loc[:,'max'] < Distance,'max'].apply(lambda x : x - Distance )
    Feature_DF.loc[Feature_DF.loc[:,'min'] > Distance,'RC_DIST_RANGE_DIST'] = \
        Feature_DF.loc[Feature_DF.loc[:,'min'] > Distance,'min'].apply(lambda x : Distance - x)

    query1 = Feature_DF.loc[:,'min'] < Distance
    query2 = Feature_DF.loc[:,'mid'] > Distance
    Feature_DF.loc[query1 | query2, 'RC_DIST_RANGE_DIST'] = Feature_DF.loc[query1 | query2, 'min'].apply(lambda x : Distance - x)

    query1 = Feature_DF.loc[:,'max'] > Distance
    query2 = Feature_DF.loc[:,'mid'] < Distance
    Feature_DF.loc[query1 | query2, 'RC_DIST_RANGE_DIST'] = Feature_DF.loc[query1 | query2, 'max'].apply(lambda x : x - Distance)

    Feature_DF.loc[:,'RC_DIST_RANGE_DIST'] = Feature_DF.loc[:,'RC_DIST_RANGE_DIST'] / Distance
    Feature_DF.loc[:,'RC_DIST_RANGE_DIST'].fillna(Feature_DF.loc[:,'RC_DIST_RANGE_DIST'].min(), inplace = True)
    Feature_DF.loc[:,'RC_DIST_RANGE_DIST'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_DIST_RANGE_DIST']]

    return Feature_DF

"""
RC_DIST_HPRE
"""

def RC_DIST_HPRE(Dataframe, HNAME_List, Raceday):

    """
    Distance Preference of Horse
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, RC_DIST_HPRE]
    """

    Distance = Dataframe.loc[:,'RADIS'].values[0]
    Dist_Dict = Pref_Distance_Similarity(Distance)

    Feature_DF = Dataframe.loc[:,['HNAME','RADIS']]

    Extraction = Extraction_Database("""
                                     Select HNAME, RADIS, Avg(HPRE_DIST_RES) HPRE_DIST_RES from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     Group by HNAME, RADIS
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    Extraction.replace({'RADIS': Dist_Dict}, inplace = True)

    if len(Extraction) == 0:
        Feature_DF.loc[:,'RC_DIST_HPRE'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','RC_DIST_HPRE']]
        return Feature_DF

    def Normalise(group):
        group.loc[:,'Norm'] = np.exp(group.loc[:,'RADIS']) / np.exp(group.loc[:,'RADIS']).sum()
        group.loc[:,'Normed'] = group.loc[:,'Norm'] * group.loc[:,'HPRE_DIST_RES']
        return group.loc[:,'Normed'].sum()

    Extraction = Extraction.groupby('HNAME').apply(Normalise).reset_index()
    Extraction.columns = ['HNAME','RC_DIST_HPRE']
    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'RC_DIST_HPRE'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_DIST_HPRE']]

    return Feature_DF

"""
RC_DIST_JPRE
"""

def RC_DIST_JPRE(Dataframe, HNAME_List, Raceday):

    """
    Distance Preference of Jockey
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, RC_DIST_JPRE]
    """

    Distance = Dataframe.loc[:,'RADIS'].values[0]
    JNAME_List = '('+str(Dataframe.loc[:,'JNAME'].tolist())[1:-1]+')'
    Dist_Dict = Pref_Distance_Similarity(Distance)

    Feature_DF = Dataframe.loc[:,['HNAME', 'JNAME']]

    Extraction = Extraction_Database("""
                                     Select JNAME, RADIS, Avg(JPRE_DIST_RES) JPRE_DIST_RES from Race_PosteriorDb
                                     where RADAT < {Raceday} and JNAME in {JNAME_List}
                                     Group by JNAME, RADIS
                                     """.format(Raceday = Raceday, JNAME_List = JNAME_List))
    Extraction.replace({'RADIS': Dist_Dict}, inplace = True)

    if len(Extraction) == 0:
        Feature_DF.loc[:,'RC_DIST_JPRE'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','RC_DIST_JPRE']]
        return Feature_DF

    def Normalise(group):
        group.loc[:,'Norm'] = np.exp(group.loc[:,'RADIS']) / np.exp(group.loc[:,'RADIS']).sum()
        group.loc[:,'Normed'] = group.loc[:,'Norm'] * group.loc[:,'JPRE_DIST_RES']
        return group.loc[:,'Normed'].sum()

    Extraction = Extraction.groupby('JNAME').apply(Normalise).reset_index()
    Extraction.columns = ['JNAME','RC_DIST_JPRE']
    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'RC_DIST_JPRE'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_DIST_JPRE']]

    return Feature_DF

"""
RC_DIST_SPRE
"""

def RC_DIST_SPRE(Dataframe, HNAME_List, Raceday):

    """
    Distance Preference of Stable
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, RC_DIST_SPRE]
    """

    Distance = Dataframe.loc[:,'RADIS'].values[0]
    SNAME_List = '('+str(Dataframe.loc[:,'SNAME'].tolist())[1:-1]+')'
    Dist_Dict = Pref_Distance_Similarity(Distance)

    Feature_DF = Dataframe.loc[:,['HNAME', 'SNAME']]

    Extraction = Extraction_Database("""
                                     Select SNAME, RADIS, Avg(SPRE_DIST_RES) SPRE_DIST_RES from Race_PosteriorDb
                                     where RADAT < {Raceday} and SNAME in {SNAME_List}
                                     Group by SNAME, RADIS
                                     """.format(Raceday = Raceday, SNAME_List = SNAME_List))
    Extraction.replace({'RADIS': Dist_Dict}, inplace = True)

    if len(Extraction) == 0:
        Feature_DF.loc[:,'RC_DIST_SPRE'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','RC_DIST_SPRE']]
        return Feature_DF

    def Normalise(group):
        group.loc[:,'Norm'] = np.exp(group.loc[:,'RADIS']) / np.exp(group.loc[:,'RADIS']).sum()
        group.loc[:,'Normed'] = group.loc[:,'Norm'] * group.loc[:,'SPRE_DIST_RES']
        return group.loc[:,'Normed'].sum()

    Extraction = Extraction.groupby('SNAME').apply(Normalise).reset_index()
    Extraction.columns = ['SNAME','RC_DIST_SPRE']
    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'RC_DIST_SPRE'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_DIST_SPRE']]

    return Feature_DF

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

================================== Going ==================================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
RC_GO_DAY_GO
"""

def RC_GO_DAY_GO(Dataframe, HNAME_List, Raceday):

    """
    Day since Similar Going
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, RC_GO_DAY_GO]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RADIS']]
    Surface = "'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Going = Dataframe.loc[:,'RAGOG'].values[0].strip()
    Going_Dict = Going_Similarity(Going)

    Extraction = Extraction_Database("""
                                     Select HNAME, max(RADAT) Go_Date, RAGOG from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RATRA = {Surface}
                                     Group by HNAME, RAGOG
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Surface = Surface))
    if len(Extraction) == 0:
        Feature_DF.loc[:,'RC_GO_DAY_GO'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','RC_GO_DAY_GO']]
        return Feature_DF

    Extraction.loc[:, 'Go_Date'] = Extraction.loc[:, 'Go_Date'].astype(float)
    Extraction.loc[:,'RAGOG'] = Extraction.loc[:,'RAGOG'].map(lambda x : x.strip())
    Extraction.replace({'RAGOG': Going_Dict}, inplace = True)
    Raceday = pd.to_datetime(Raceday, format = '%Y%m%d')
    Extraction.loc[:,'Go_Date'] = Raceday - pd.to_datetime(Extraction.loc[:, 'Go_Date'], format = '%Y%m%d')
    Extraction.loc[:,'Go_Date'] = Extraction.loc[:,'Go_Date'].apply(lambda x : int(str(x).split('days')[0]))

    def Normalise(group):
        group.loc[:,'Norm'] = np.exp(group.loc[:,'RAGOG']) / np.exp(group.loc[:,'RAGOG']).sum()
        group.loc[:,'Normed'] = group.loc[:,'Norm'] * group.loc[:,'Go_Date']
        return group.loc[:,'Normed'].sum()

    Extraction = Extraction.groupby('HNAME').apply(Normalise).reset_index()
    Extraction.columns = ['HNAME','RC_GO_DAY_GO']
    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'RC_GO_DAY_GO'].fillna(Feature_DF.loc[:,'RC_GO_DAY_GO'].max(), inplace = True)
    Feature_DF.loc[:,'RC_GO_DAY_GO'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_GO_DAY_GO']]

    return Feature_DF

"""
RC_GO_AVG_GO
"""

def RC_GO_AVG_GO(Dataframe, HNAME_List, Raceday):

    """
    Average Recent Going
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, RC_GO_AVG_GO]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RADIS']]
    Surface = "'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Going = Dataframe.loc[:,'RAGOG'].values[0].strip()
    Going_Dict = Going_Similarity(Going)

    Extraction = Extraction_Database("""
                                     Select HNAME, RADAT, RAGOG from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RATRA = {Surface}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Surface = Surface))

    Extraction.loc[:, 'RADAT'] = Extraction.loc[:, 'RADAT'].astype(float)
    Extraction.loc[:,'RAGOG'] = Extraction.loc[:,'RAGOG'].map(lambda x : x.strip())
    Extraction.replace({'RAGOG': Going_Dict}, inplace = True)

    Dist = []
    for name, group in Extraction.groupby('HNAME'):
        Dist.append([name, group.nlargest(3, 'RADAT').loc[:,'RAGOG'].mean()])
    Avg_Going = pd.DataFrame(Dist, columns=['HNAME','RC_GO_AVG_GO'])

    Feature_DF = Feature_DF.merge(Avg_Going, how='left')
    Feature_DF.loc[:,'RC_GO_AVG_GO'].fillna(Feature_DF.loc[:,'RC_GO_AVG_GO'].min(), inplace = True)
    Feature_DF.loc[:,'RC_GO_AVG_GO'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_GO_AVG_GO']]

    return Feature_DF

"""
RC_GO_HPRE
"""

def RC_GO_HPRE(Dataframe, HNAME_List, Raceday):

    """
    Going Preference of Jockey
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, RC_GO_HPRE]
    """

    Going = Dataframe.loc[:,'RAGOG'].values[0].strip()
    Surface = "'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Going_Dict = Going_Similarity(Going)

    Feature_DF = Dataframe.loc[:,['HNAME', 'RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, RAGOG, Avg(HPRE_GO_RES) HPRE_GO_RES from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RATRA = {Surface}
                                     Group by HNAME, RAGOG
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Surface=Surface))
    Extraction.loc[:,'RAGOG'] = Extraction.loc[:,'RAGOG'].map(lambda x : x.strip())
    Extraction.replace({'RAGOG': Going_Dict}, inplace = True)

    if len(Extraction) == 0:
        Feature_DF.loc[:,'RC_GO_HPRE'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','RC_GO_HPRE']]
        return Feature_DF

    def Normalise(group):
        group.loc[:,'Norm'] = np.exp(group.loc[:,'RAGOG']) / np.exp(group.loc[:,'RAGOG']).sum()
        group.loc[:,'Normed'] = group.loc[:,'Norm'] * group.loc[:,'HPRE_GO_RES']
        return group.loc[:,'Normed'].sum()

    Extraction = Extraction.groupby('HNAME').apply(Normalise).reset_index()
    Extraction.columns = ['HNAME','RC_GO_HPRE']
    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'RC_GO_HPRE'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_GO_HPRE']]

    return Feature_DF

"""
RC_GO_JPRE
"""

def RC_GO_JPRE(Dataframe, HNAME_List, Raceday):

    """
    Going Preference of Jockey
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, RC_GO_JPRE]
    """

    Going = Dataframe.loc[:,'RAGOG'].values[0].strip()
    Surface = "'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    JNAME_List = '('+str(Dataframe.loc[:,'JNAME'].tolist())[1:-1]+')'
    Going_Dict = Going_Similarity(Going)

    Feature_DF = Dataframe.loc[:,['HNAME', 'JNAME']]

    Extraction = Extraction_Database("""
                                     Select JNAME, RAGOG, Avg(JPRE_GO_RES) JPRE_GO_RES from Race_PosteriorDb
                                     where RADAT < {Raceday} and JNAME in {JNAME_List} and RATRA = {Surface}
                                     Group by JNAME, RAGOG
                                     """.format(Raceday = Raceday, JNAME_List = JNAME_List, Surface=Surface))
    Extraction.loc[:,'RAGOG'] = Extraction.loc[:,'RAGOG'].map(lambda x : x.strip())
    Extraction.replace({'RAGOG': Going_Dict}, inplace = True)

    if len(Extraction) == 0:
        Feature_DF.loc[:,'RC_GO_JPRE'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','RC_GO_JPRE']]
        return Feature_DF

    def Normalise(group):
        group.loc[:,'Norm'] = np.exp(group.loc[:,'RAGOG']) / np.exp(group.loc[:,'RAGOG']).sum()
        group.loc[:,'Normed'] = group.loc[:,'Norm'] * group.loc[:,'JPRE_GO_RES']
        return group.loc[:,'Normed'].sum()

    Extraction = Extraction.groupby('JNAME').apply(Normalise).reset_index()
    Extraction.columns = ['JNAME','RC_GO_JPRE']
    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'RC_GO_JPRE'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_GO_JPRE']]

    return Feature_DF

"""
RC_GO_SPRE
"""

def RC_GO_SPRE(Dataframe, HNAME_List, Raceday):

    """
    Going Preference of Stable
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, RC_GO_SPRE]
    """

    Going = Dataframe.loc[:,'RAGOG'].values[0].strip()
    Surface = "'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    SNAME_List = '('+str(Dataframe.loc[:,'SNAME'].tolist())[1:-1]+')'
    Going_Dict = Going_Similarity(Going)

    Feature_DF = Dataframe.loc[:,['HNAME', 'SNAME']]

    Extraction = Extraction_Database("""
                                     Select SNAME, RAGOG, Avg(SPRE_GO_RES) SPRE_GO_RES from Race_PosteriorDb
                                     where RADAT < {Raceday} and SNAME in {SNAME_List} and RATRA = {Surface}
                                     Group by SNAME, RAGOG
                                     """.format(Raceday = Raceday, SNAME_List = SNAME_List, Surface=Surface))
    Extraction.loc[:,'RAGOG'] = Extraction.loc[:,'RAGOG'].map(lambda x : x.strip())
    Extraction.replace({'RAGOG': Going_Dict}, inplace = True)

    if len(Extraction) == 0:
        Feature_DF.loc[:,'RC_GO_SPRE'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','RC_GO_SPRE']]
        return Feature_DF

    def Normalise(group):
        group.loc[:,'Norm'] = np.exp(group.loc[:,'RAGOG']) / np.exp(group.loc[:,'RAGOG']).sum()
        group.loc[:,'Normed'] = group.loc[:,'Norm'] * group.loc[:,'SPRE_GO_RES']
        return group.loc[:,'Normed'].sum()

    Extraction = Extraction.groupby('SNAME').apply(Normalise).reset_index()
    Extraction.columns = ['SNAME','RC_GO_SPRE']
    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'RC_GO_SPRE'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_GO_SPRE']]

    return Feature_DF

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

================================== Surface ==================================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
RC_SUR_DAY_SUR
"""

def RC_SUR_DAY_SUR(Dataframe, HNAME_List, Raceday):

    """
    Day since Surface
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, RC_SUR_DAY_SUR]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RATRA']]
    Surface = "'" + Dataframe.loc[:,'RATRA'].values[0] + "'"

    Extraction = Extraction_Database("""
                                     Select HNAME, max(RADAT) Sur_Date from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RATRA = {Surface}
                                     Group by HNAME
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Surface = Surface))

    if len(Extraction) == 0:
        Feature_DF.loc[:,'RC_SUR_DAY_SUR'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','RC_SUR_DAY_SUR']]
        return Feature_DF

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'Sur_Date'] = Feature_DF.loc[:,'Sur_Date'].apply(float)
    Feature_DF.loc[:,'Sur_Date'].fillna(Feature_DF.loc[:,'Sur_Date'].min(), inplace = True)

    Feature_DF.loc[:,'RC_SUR_DAY_SUR'] = pd.to_datetime(Raceday, format = '%Y%m%d') - pd.to_datetime(Feature_DF.loc[:, 'Sur_Date'], format = '%Y%m%d')
    Feature_DF.loc[:,'RC_SUR_DAY_SUR'] = Feature_DF.loc[:,'RC_SUR_DAY_SUR'].apply(lambda x : int(str(x).split('days')[0]))

    Feature_DF.loc[:,'RC_SUR_DAY_SUR'].fillna(Feature_DF.loc[:,'RC_SUR_DAY_SUR'].max(), inplace = True)
    Feature_DF.loc[:,'RC_SUR_DAY_SUR'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_SUR_DAY_SUR']]

    return Feature_DF

"""
RC_SUR_AVG_SUR
"""

def RC_SUR_AVG_SUR(Dataframe, HNAME_List, Raceday):

    """
    Average Recent Surface
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, RC_SUR_AVG_SUR]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RADIS']]
    Surface = Dataframe.loc[:,'RATRA'].values[0]

    Extraction = Extraction_Database("""
                                     Select HNAME, RADAT, RATRA from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    Extraction.loc[:, 'RADAT'] = Extraction.loc[:, 'RADAT'].astype(float)

    Sur = []
    for name, group in Extraction.groupby('HNAME'):
        try :
            Sur.append([name, group.nlargest(3, 'RADAT').loc[:,'RATRA'].value_counts()[Surface]])
        except :
            Sur.append([name, 0])
    Avg_Sur = pd.DataFrame(Sur, columns=['HNAME','Avg_Dist'])

    Feature_DF = Feature_DF.merge(Avg_Sur, how='left')
    Feature_DF.loc[:,'RC_SUR_AVG_SUR'] = Feature_DF.loc[:,'Avg_Dist'] / 3
    Feature_DF.loc[:,'RC_SUR_AVG_SUR'].fillna(0, inplace = True)

    Feature_DF = Feature_DF.loc[:,['HNAME','RC_SUR_AVG_SUR']]

    return Feature_DF

"""
RC_SUR_HPRE
"""

def RC_SUR_HPRE(Dataframe, HNAME_List, Raceday):

    """
    Surface Preference of Horse
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, RC_SUR_HPRE]
    """

    Surface = "'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Feature_DF = Dataframe.loc[:,['HNAME', 'RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, RATRA, Avg(HPRE_SUR_RES) RC_SUR_HPRE from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RATRA = {Surface}
                                     Group by HNAME, RATRA
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Surface=Surface))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'RC_SUR_HPRE'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_SUR_HPRE']]

    return Feature_DF

"""
RC_SUR_JPRE
"""

def RC_SUR_JPRE(Dataframe, HNAME_List, Raceday):

    """
    Surface Preference of Jockey
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, RC_SUR_JPRE]
    """

    Surface = "'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    JNAME_List = '('+str(Dataframe.loc[:,'JNAME'].tolist())[1:-1]+')'
    Feature_DF = Dataframe.loc[:,['HNAME', 'JNAME']]

    Extraction = Extraction_Database("""
                                     Select JNAME, RATRA, Avg(JPRE_SUR_RES) RC_SUR_JPRE from Race_PosteriorDb
                                     where RADAT < {Raceday} and JNAME in {JNAME_List} and RATRA = {Surface}
                                     Group by JNAME, RATRA
                                     """.format(Raceday = Raceday, JNAME_List = JNAME_List, Surface=Surface))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'RC_SUR_JPRE'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_SUR_JPRE']]

    return Feature_DF

"""
RC_SUR_SPRE
"""

def RC_SUR_SPRE(Dataframe, HNAME_List, Raceday):

    """
    Surface Preference of Jockey
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, RC_SUR_SPRE]
    """

    Surface = "'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    SNAME_List = '('+str(Dataframe.loc[:,'SNAME'].tolist())[1:-1]+')'
    Feature_DF = Dataframe.loc[:,['HNAME', 'SNAME']]

    Extraction = Extraction_Database("""
                                     Select SNAME, RATRA, Avg(SPRE_SUR_RES) RC_SUR_SPRE from Race_PosteriorDb
                                     where RADAT < {Raceday} and SNAME in {SNAME_List} and RATRA = {Surface}
                                     Group by SNAME, RATRA
                                     """.format(Raceday = Raceday, SNAME_List = SNAME_List, Surface=Surface))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'RC_SUR_SPRE'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_SUR_SPRE']]

    return Feature_DF

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

================================ Location ================================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
RC_LOC_JPRE
"""

def RC_LOC_JPRE(Dataframe, HNAME_List, Raceday):

    """
    Location Preference of Jockey
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, RC_LOC_JPRE]
    """

    Location = "'" + Dataframe.loc[:,'RALOC'].values[0] + "'"
    JNAME_List = '('+str(Dataframe.loc[:,'JNAME'].tolist())[1:-1]+')'
    Feature_DF = Dataframe.loc[:,['HNAME', 'JNAME']]

    Extraction = Extraction_Database("""
                                     Select JNAME, RATRA, Avg(JPRE_LOC_RES) RC_LOC_JPRE from Race_PosteriorDb
                                     where RADAT < {Raceday} and JNAME in {JNAME_List} and RALOC = {Location}
                                     Group by JNAME, RATRA
                                     """.format(Raceday = Raceday, JNAME_List = JNAME_List, Location=Location))
    try :
        Extraction = Extraction.groupby('JNAME').mean().reset_index()
    except :
        pass
    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'RC_LOC_JPRE'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_LOC_JPRE']]

    return Feature_DF

"""
RC_LOC_SPRE
"""

def RC_LOC_SPRE(Dataframe, HNAME_List, Raceday):

    """
    Location Preference of Stable
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, RC_LOC_SPRE]
    """

    Location = "'" + Dataframe.loc[:,'RALOC'].values[0] + "'"
    SNAME_List = '('+str(Dataframe.loc[:,'SNAME'].tolist())[1:-1]+')'
    Feature_DF = Dataframe.loc[:,['HNAME', 'SNAME']]

    Extraction = Extraction_Database("""
                                     Select SNAME, RATRA, Avg(SPRE_LOC_RES) RC_LOC_SPRE from Race_PosteriorDb
                                     where RADAT < {Raceday} and SNAME in {SNAME_List} and RALOC = {Location}
                                     Group by SNAME, RATRA
                                     """.format(Raceday = Raceday, SNAME_List = SNAME_List, Location=Location))
    try :
        Extraction = Extraction.groupby('SNAME').mean().reset_index()
    except :
        pass
    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'RC_LOC_SPRE'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_LOC_SPRE']]

    return Feature_DF

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

================================= Profile =================================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
RC_PFL_HPRE
"""

def RC_PFL_HPRE(Dataframe, HNAME_List, Raceday):

    """
    Profile Preference of Horse
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, RC_PFL_HPRE]
    """

    Feature_DF = Dataframe.loc[:,['HNAME', 'RARID']]
    Location = "'" + Dataframe.loc[:,'RALOC'].values[0] + "'"
    Surface = "'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Extraction = Extraction_Database("""
                                     Select HNAME, Avg(HPRE_PFL_RES) RC_PFL_HPRE from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     and RALOC = {Location} and RATRA = {Surface} and RADIS = {Distance}
                                     Group by HNAME, RALOC, RATRA, RADIS
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List,
                                     Location = Location, Surface = Surface, Distance = Distance))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'RC_PFL_HPRE'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_PFL_HPRE']]

    return Feature_DF

"""
RC_PFL_JPRE
"""

def RC_PFL_JPRE(Dataframe, HNAME_List, Raceday):

    """
    Profile Preference of Jockey
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, RC_PFL_JPRE]
    """

    JNAME_List = '('+str(Dataframe.loc[:,'JNAME'].tolist())[1:-1]+')'
    Feature_DF = Dataframe.loc[:,['HNAME', 'JNAME']]
    Location = "'" + Dataframe.loc[:,'RALOC'].values[0] + "'"
    Surface = "'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Extraction = Extraction_Database("""
                                     Select JNAME, Avg(JPRE_PFL_RES) RC_PFL_JPRE from Race_PosteriorDb
                                     where RADAT < {Raceday} and JNAME in {JNAME_List}
                                     and RALOC = {Location} and RATRA = {Surface} and RADIS = {Distance}
                                     Group by JNAME, RALOC, RATRA, RADIS
                                     """.format(Raceday = Raceday, JNAME_List = JNAME_List,
                                     Location = Location, Surface = Surface, Distance = Distance))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'RC_PFL_JPRE'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_PFL_JPRE']]

    return Feature_DF

"""
RC_PFL_SPRE
"""

def RC_PFL_SPRE(Dataframe, HNAME_List, Raceday):

    """
    Profile Preference of Stable
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, RC_PFL_SPRE]
    """

    SNAME_List = '('+str(Dataframe.loc[:,'SNAME'].tolist())[1:-1]+')'
    Feature_DF = Dataframe.loc[:,['HNAME', 'SNAME']]
    Location = "'" + Dataframe.loc[:,'RALOC'].values[0] + "'"
    Surface = "'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Extraction = Extraction_Database("""
                                     Select SNAME, Avg(SPRE_PFL_RES) RC_PFL_SPRE from Race_PosteriorDb
                                     where RADAT < {Raceday} and SNAME in {SNAME_List}
                                     and RALOC = {Location} and RATRA = {Surface} and RADIS = {Distance}
                                     Group by SNAME, RALOC, RATRA, RADIS
                                     """.format(Raceday = Raceday, SNAME_List = SNAME_List,
                                     Location = Location, Surface = Surface, Distance = Distance))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'RC_PFL_SPRE'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_PFL_SPRE']]

    return Feature_DF

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

============================== Post Position ==============================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
RC_PP
"""

def RC_PP(Dataframe, HNAME_List, Raceday):

    """
    Post Position of Horse
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, RC_PP]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','HDRAW']]
    Feature_DF = Feature_DF.rename(columns={'HNAME': 'HNAME', 'HDRAW': 'RC_PP'})

    return Feature_DF

"""
RC_PP_W
"""

def RC_PP_W(Dataframe, HNAME_List, Raceday):

    """
    Post Position Win Percentage
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, RC_PP_W]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','HDRAW']]
    Num_Runners = len(Feature_DF.loc[:,'HDRAW'].index)

    Extraction = Extraction_Database("""
                                     Select HDRAW, sum(RESWL) Num_Win, count(RARID) Num_Races from RaceDb
                                     where RADAT < {Raceday}
                                     Group by HDRAW
                                     """.format(Raceday = Raceday))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'RC_PP_W'] = Feature_DF.loc[:,'Num_Win'] / Feature_DF.loc[:,'Num_Races']
    Feature_DF.loc[:,'RC_PP_W'].fillna(1/Num_Runners, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_PP_W']]

    return Feature_DF

"""
RC_PP_GOA
"""

def RC_PP_GOA(Dataframe, HNAME_List, Raceday):

    """
    Post Position Advantage of Going
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, RC_PP_GOA]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','HDRAW']]
    Surface ="'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Going = Dataframe.loc[:,'RAGOG'].values[0].strip()
    Going_Dict = Going_Similarity(Going)
    Num_Runners = len(Feature_DF.loc[:,'HDRAW'].index)

    Extraction = Extraction_Database("""
                                     Select HDRAW, RAGOG, sum(RESWL) Num_Win, count(RARID) Num_Races from RaceDb
                                     where RADAT < {Raceday} and RATRA = {Surface}
                                     Group by HDRAW, RAGOG
                                     """.format(Raceday = Raceday, Surface = Surface))
    Extraction.loc[:,'RAGOG'] = Extraction.loc[:,'RAGOG'].map(lambda x : x.strip())
    Extraction.replace({'RAGOG': Going_Dict}, inplace = True)

    def Normalise(group):
        group.loc[:,'Norm'] = np.exp(group.loc[:,'RAGOG']) / np.exp(group.loc[:,'RAGOG']).sum()
        group.loc[:,'Normed'] = group.loc[:,'Norm'] * (group.loc[:,'Num_Win'] / group.loc[:,'Num_Races'])
        return group.loc[:,'Normed'].sum()

    #For First Runners
    if len(Extraction) == 0:
        Feature_DF.loc[:,'RC_PP_GOA'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','RC_PP_GOA']]
        return Feature_DF

    Extraction = Extraction.groupby('HDRAW').apply(Normalise).reset_index()
    Extraction.columns = ['HDRAW','RC_PP_GOA']
    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'RC_PP_GOA'].fillna(1/Num_Runners, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_PP_GOA']]

    return Feature_DF

"""
RC_PP_PFLA
"""

def RC_PP_PFLA(Dataframe, HNAME_List, Raceday):

    """
    Post Position Advantage of Profile
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, RC_PP_PFLA]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','HDRAW']]
    Location = "'" + Dataframe.loc[:,'RALOC'].values[0] + "'"
    Surface = "'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Distance = Dataframe.loc[:,'RADIS'].values[0]
    Correction = "'" + Dataframe.loc[:,'RARAL'].values[0] + "'"
    Num_Runners = len(Feature_DF.loc[:,'HDRAW'].index)

    Extraction = Extraction_Database("""
                                     Select HDRAW, sum(RESWL) Num_Win, count(RARID) Num_Races from RaceDb
                                     where RADAT < {Raceday} and RALOC = {Location} and RATRA = {Surface}
                                     and RADIS = {Distance} and RARAL = {Correction}
                                     Group by HDRAW
                                     """.format(Raceday = Raceday, Location = Location, Surface = Surface,
                                     Correction = Correction, Distance = Distance))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'RC_PP_PFLA'] = Feature_DF.loc[:,'Num_Win'] / Feature_DF.loc[:,'Num_Races']
    Feature_DF.loc[:,'RC_PP_PFLA'].fillna(1/Num_Runners, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_PP_PFLA']]

    return Feature_DF

"""
RC_PP_GOPFLA
"""

def RC_PP_GOPFLA(Dataframe, HNAME_List, Raceday):

    """
    Post Position Advantage of Profile Going
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, RC_PP_GOPFLA]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','HDRAW']]
    Location = "'" + Dataframe.loc[:,'RALOC'].values[0] + "'"
    Surface = "'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Distance = Dataframe.loc[:,'RADIS'].values[0]
    Correction = "'" + Dataframe.loc[:,'RARAL'].values[0] + "'"
    Num_Runners = len(Feature_DF.loc[:,'HDRAW'].index)
    Going = "'" +  Dataframe.loc[:,'RAGOG'].values[0] + "'"

    Extraction = Extraction_Database("""
                                     Select HDRAW, sum(RESWL) Num_Win, count(RARID) Num_Races from RaceDb
                                     where RADAT < {Raceday} and RALOC = {Location} and RATRA = {Surface}
                                     and RADIS = {Distance} and RARAL = {Correction} and RAGOG = {Going}
                                     Group by HDRAW
                                     """.format(Raceday = Raceday, Location = Location, Surface = Surface,
                                     Correction = Correction, Distance = Distance, Going = Going))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'RC_PP_GOPFLA'] = Feature_DF.loc[:,'Num_Win'] / Feature_DF.loc[:,'Num_Races']
    Feature_DF.loc[:,'RC_PP_GOPFLA'].fillna(1/Num_Runners, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_PP_GOPFLA']]

    return Feature_DF

"""
RC_PP_PFLEP
"""

def RC_PP_PFLEP(Dataframe, HNAME_List, Raceday):

    """
    Post Position Early Pace Advantage of Profile
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, RC_PP_PFLEP]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','HDRAW']]
    Location = "'" + Dataframe.loc[:,'RALOC'].values[0] + "'"
    Surface = "'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Distance = Dataframe.loc[:,'RADIS'].values[0]
    Correction = "'" + Dataframe.loc[:,'RARAL'].values[0] + "'"

    Extraction = Extraction_Database("""
                                     Select HDRAW, EARLY_PACE RC_PP_PFLEP from Race_PosteriorDb
                                     where RADAT < {Raceday} and RALOC = {Location} and RATRA = {Surface}
                                     and RADIS = {Distance} and RARAL = {Correction}
                                     Group by HDRAW
                                     """.format(Raceday = Raceday, Location = Location, Surface = Surface,
                                     Correction = Correction, Distance = Distance))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'RC_PP_PFLEP'].fillna(Feature_DF.loc[:,'RC_PP_PFLEP'].min(), inplace = True)
    Feature_DF.loc[:,'RC_PP_PFLEP'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_PP_PFLEP']]

    return Feature_DF

"""
RC_PP_HPRE
"""

def RC_PP_HPRE(Dataframe, HNAME_List, Raceday):

    """
    Horse’s Post Position Advantage
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, RC_PP_HPRE]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','HDRAW']]

    Extraction = Extraction_Database("""
                                     Select HNAME, HDRAW, avg(EARLY_PACE) RC_PP_HPRE from Race_PosteriorDb
                                     where HNAME in {HNAME_List} and RADAT < {Raceday}
                                     Group by HNAME, HDRAW
                                     """.format(Raceday = Raceday, HNAME_List=HNAME_List))

    Feature_DF = Feature_DF.merge(Extraction, on=['HNAME','HDRAW'], how='left')
    Feature_DF.loc[:,'RC_PP_HPRE'].fillna(Feature_DF.loc[:,'RC_PP_HPRE'].min(), inplace = True)
    Feature_DF.loc[:,'RC_PP_HPRE'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_PP_HPRE']]


    return Feature_DF

"""
RC_PP_JPRE_JPFL
"""

def RC_PP_JPRE_JPFL(Dataframe, HNAME_List, Raceday):

    """
    Jockey's Post Position Advantage
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, RC_PP_JPRE_JPFL]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JNAME','HDRAW']]
    JNAME_List = '('+str(Dataframe.loc[:,'JNAME'].tolist())[1:-1]+')'
    Location = "'" + Dataframe.loc[:,'RALOC'].values[0] + "'"
    Surface = "'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Distance = Dataframe.loc[:,'RADIS'].values[0]
    Correction = "'" + Dataframe.loc[:,'RARAL'].values[0] + "'"

    Extraction = Extraction_Database("""
                                     Select JNAME, HDRAW, avg(EARLY_PACE) RC_PP_JPRE_JPFL from Race_PosteriorDb
                                     where JNAME in {JNAME_List} and RADAT < {Raceday} and RARAL = {Correction}
                                     and RADIS = {Distance} and RATRA = {Surface} and RALOC = {Location}
                                     Group by JNAME, HDRAW
                                     """.format(Raceday = Raceday, JNAME_List=JNAME_List, Correction=Correction,
                                     Location = Location, Surface = Surface, Distance = Distance))

    Feature_DF = Feature_DF.merge(Extraction, on=['JNAME','HDRAW'], how='left')
    Feature_DF.loc[:,'RC_PP_JPRE_JPFL'].fillna(Feature_DF.loc[:,'RC_PP_JPRE_JPFL'].min(), inplace = True)
    Feature_DF.loc[:,'RC_PP_JPRE_JPFL'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_PP_JPRE_JPFL']]

    return Feature_DF
