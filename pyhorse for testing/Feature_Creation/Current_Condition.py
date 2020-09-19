#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ivan
"""

"""
Feature : Current Condition of Horse
"""

#Loading Libraries
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from pyhorse.Database_Management import Extraction_Database

Aux_Reg_Path = 'pyhorse/Feature_Creation/Auxiliary_Regression/'

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

=================================== Age ===================================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
CC_AGE
"""

def CC_AGE(Dataframe, HNAME_List, Raceday):

    """
    Age of Horse
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, CC_AGE]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','HAGEI']]
    Feature_DF = Feature_DF.rename(columns={'HNAME': 'HNAME', 'HAGEI': 'CC_AGE'})

    return Feature_DF


def CC_FRB(Dataframe, HNAME_List, Raceday):

    """
    Binary indicator on whether the underlying race is the horse’s first race.
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, CC_FRB]
    """

    Feature_DF = Dataframe.loc[:, ['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, count(RARID) CC_FRB from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     Group by HNAME
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    Extraction.loc[:,'CC_FRB'] = Extraction.loc[:,'CC_FRB'].apply(lambda x : int(x<1))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'CC_FRB'].fillna(1, inplace=True)

    Feature_DF = Feature_DF.loc[:,['HNAME','CC_FRB']]

    return Feature_DF

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

================================ Recovery ================================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
CC_REC_DAYL
"""

def CC_REC_DAYL(Dataframe, HNAME_List, Raceday):

    """
    Number of Days since Last Race
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, CC_REC_DAYL]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, max(RADAT) Last_Race from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     Group by HNAME
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    #For First Runners
    if len(Extraction) == 0:
        Feature_DF.loc[:,'CC_REC_DAYL'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','CC_REC_DAYL']]
        return Feature_DF

    Raceday = pd.to_datetime(Raceday, format = '%Y%m%d')
    Extraction.loc[:,'Day_Last'] = Raceday - pd.to_datetime(Extraction.loc[:, 'Last_Race'], format = '%Y%m%d')

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[Feature_DF.loc[:,'Day_Last'].notna(),'CC_REC_DAYL'] = Feature_DF.loc[Feature_DF.loc[:,'Day_Last'].notna(),'Day_Last']\
                                                                        .apply(lambda x : int(str(x).split('days')[0]))
    Feature_DF.loc[:,'CC_REC_DAYL'].fillna(Feature_DF.loc[:,'CC_REC_DAYL'].min(), inplace = True)
    Feature_DF.loc[:,'CC_REC_DAYL'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_REC_DAYL']]

    return Feature_DF

"""
CC_REC_DAYL_DIST
"""

def CC_REC_DAYL_DIST(Dataframe, HNAME_List, Raceday):

    """
    Days since Last Race in relation with distance
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, CC_REC_DAYL_DIST]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, max(RADAT) Last_Race, RADIS Last_Dist from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     Group by HNAME
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    #For First Runners
    if len(Extraction) == 0:
        Feature_DF.loc[:,'CC_REC_DAYL_DIST'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','CC_REC_DAYL_DIST']]
        return Feature_DF
    Feature_DF = Feature_DF.merge(Extraction, how='left')

    Raceday = pd.to_datetime(Raceday, format = '%Y%m%d')
    Feature_DF.loc[:,'Day_Last'] = Raceday - pd.to_datetime(Feature_DF.loc[:, 'Last_Race'], format = '%Y%m%d')
    Feature_DF.loc[Feature_DF.loc[:,'Day_Last'].notna(),'Day_Last'] = Feature_DF.loc[Feature_DF.loc[:,'Day_Last'].notna(),'Day_Last']\
                                                                        .apply(lambda x : int(str(x).split('days')[0]))
    Feature_DF.loc[:,'CC_REC_DAYL_DIST'] = Feature_DF.loc[:,'Day_Last'] / Feature_DF.loc[:,'Last_Dist']
    Feature_DF.loc[:,'CC_REC_DAYL_DIST'].fillna(Feature_DF.loc[:,'CC_REC_DAYL_DIST'].min(), inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_REC_DAYL_DIST']]

    return Feature_DF

"""
CC_REC_DAYL_AGE
"""

def CC_REC_DAYL_AGE(Dataframe, HNAME_List, Raceday):

    """
    Days since Last Race in relation with Age
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, CC_REC_DAYL_AGE]
    """

    Feature_DF = Dataframe.loc[:,['HNAME', 'HAGEI']]

    Extraction = Extraction_Database("""
                                     Select HNAME, max(RADAT) Last_Race, HAGEI Last_Age from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     Group by HNAME
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    #For First Runners
    if len(Extraction) == 0:
        Feature_DF.loc[:,'CC_REC_DAYL_AGE'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','CC_REC_DAYL_AGE']]
        return Feature_DF

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'Last_Age'].fillna(Feature_DF.loc[:,'HAGEI'], inplace = True)

    Raceday = pd.to_datetime(Raceday, format = '%Y%m%d')
    Feature_DF.loc[:,'Day_Last'] = Raceday - pd.to_datetime(Feature_DF.loc[:, 'Last_Race'], format = '%Y%m%d')
    Feature_DF.loc[Feature_DF.loc[:,'Day_Last'].notna(),'Day_Last'] = Feature_DF.loc[Feature_DF.loc[:,'Day_Last'].notna(),'Day_Last']\
                                                                        .apply(lambda x : int(str(x).split('days')[0]))
    Feature_DF.loc[:,'CC_REC_DAYL_AGE'] = Feature_DF.loc[:,'Day_Last'] * Feature_DF.loc[:,'Last_Age']
    Feature_DF.loc[:,'CC_REC_DAYL_AGE'].fillna(Feature_DF.loc[:,'CC_REC_DAYL_AGE'].min(), inplace=True)
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_REC_DAYL_AGE']]

    return Feature_DF

"""
CC_REC_INC
"""

def CC_REC_INC(Dataframe, HNAME_List, Raceday):

    """
    Number of Days since Last Incident Date
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, CC_REC_INC]
    """

    Feature_DF = Dataframe.loc[:,['HNAME', 'RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, First_Date, Last_Incident from
                                     (Select HNAME, min(RADAT) First_Date from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     Group by HNAME) A
                                     LEFT OUTER JOIN
                                     (Select HNAME HN, max(INCIDENT_DATE) Last_Incident from Irregular_RecordDb
                                     where INCIDENT_DATE < {Raceday} and HNAME in {HNAME_List}
                                     Group by HNAME) B
                                     ON A.HNAME = B.HN
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'Last_Incident'].fillna(Feature_DF.loc[:,'First_Date'], inplace = True)
    Feature_DF.loc[:,'Last_Incident'].fillna(Raceday, inplace = True)

    Raceday = pd.to_datetime(Raceday, format = '%Y%m%d')
    Feature_DF.loc[:,'CC_REC_INC'] = Raceday - pd.to_datetime(Feature_DF.loc[:, 'Last_Incident'], format = '%Y%m%d')
    Feature_DF.loc[:,'CC_REC_INC'] = Feature_DF.loc[:,'CC_REC_INC'].apply(lambda x : int(str(x).split('days')[0]))
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_REC_INC']]

    return Feature_DF

"""
CC_REC_NUMM
"""

def CC_REC_NUMM(Dataframe, HNAME_List, Raceday):

    """
    Number of meters ran in last 3 months
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, CC_REC_NUMM]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Offset_Raceday = (pd.to_datetime(Raceday) + pd.tseries.offsets.DateOffset(months=-3)).strftime("%Y%m%d")

    Extraction = Extraction_Database("""
                                     Select HNAME, sum(RADIS) CC_REC_NUMM from RaceDb
                                     where RADAT < {Raceday} and RADAT > {Offset_Raceday} and HNAME in {HNAME_List}
                                     Group by HNAME
                                     """.format(Raceday = Raceday, Offset_Raceday = Offset_Raceday, HNAME_List = HNAME_List))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_REC_NUMM']].fillna(0)

    return Feature_DF


"""
CC_REC_DAY_LWIN
"""

def CC_REC_DAY_LWIN(Dataframe, HNAME_List, Raceday):

    """
    Number of Days since Last Win or Best Beyer Speed Figure
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, CC_REC_DAY_LWIN]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HN HNAME, Beyer_Date, Win_Date, First_Date from
                                     (Select HNAME HN, min(RADAT) First_Date from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     Group by HNAME) A
                                     LEFT OUTER JOIN
                                     (Select HNAME, Beyer_Date, Win_Date from (
                                     Select HNAME, RADAT Beyer_Date, max(BEYER_SPEED) from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     Group by HNAME) BEYER
                                     LEFT OUTER JOIN
                                     (Select HNAME HNAME_WIN, RADAT Win_Date from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RESFP = 1
                                     Group by HNAME) WIN
                                     ON BEYER.HNAME = WIN.HNAME_WIN) B
                                     ON A.HN = B.HNAME
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'Win_Date'].fillna(Feature_DF.loc[:,'Beyer_Date'], inplace = True)
    Feature_DF.loc[:,'Win_Date'].fillna(Feature_DF.loc[:,'First_Date'], inplace = True)
    Feature_DF.loc[:,'Win_Date'].fillna(Raceday, inplace = True)

    Raceday = pd.to_datetime(Raceday, format = '%Y%m%d')
    Feature_DF.loc[:,'CC_REC_DAY_LWIN'] = Raceday - pd.to_datetime(Feature_DF.loc[:, 'Win_Date'], format = '%Y%m%d')
    Feature_DF.loc[:,'CC_REC_DAY_LWIN'] = Feature_DF.loc[:,'CC_REC_DAY_LWIN'].apply(lambda x : int(str(x).split('days')[0]))
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_REC_DAY_LWIN']]

    return Feature_DF

"""
CC_REC_DAY_PT3
"""

def CC_REC_DAY_PT3(Dataframe, HNAME_List, Raceday):

    """
    Predicted days until next Top 3
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, CC_REC_DAY_PT3]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction_Performance = Extraction_Database("""
                                                 Select HNAME, First_Date, Last_T3_Date, Top_Beyer_Date from (
                                                 Select HNAME, First_Date, Top_Beyer_Date from (
                                                 Select HNAME, min(RADAT) First_Date from Race_PosteriorDb
                                                 where RADAT < {Raceday} and HNAME in {HNAME_List}
                                                 Group by HNAME) FIRST
                                                 LEFT OUTER JOIN
                                                 (Select HNAME HNAME_BEYER, RADAT Top_Beyer_Date, max(BEYER_SPEED) from Race_PosteriorDb
                                                 where RADAT < {Raceday} and HNAME in {HNAME_List}
                                                 Group by HNAME) BEYER
                                                 ON FIRST.HNAME = BEYER.HNAME_BEYER) FIRST_BEYER
                                                 LEFT OUTER JOIN
                                                 (Select HNAME HNAME_W, max(RADAT) Last_T3_Date from Race_PosteriorDb
                                                 where RADAT < {Raceday} and HNAME in {HNAME_List} and RESFP <= 3
                                                 Group by HNAME) WIN
                                                 ON WIN.HNAME_W = FIRST_BEYER.HNAME
                                                 """.format(Raceday = Raceday, HNAME_List = HNAME_List))

    Extraction_Avg = Extraction_Database("""
                                         Select HNAME, RADAT, RESFP, BEYER_SPEED from Race_PosteriorDb
                                         where RADAT < {Raceday} and HNAME in {HNAME_List}
                                         """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    Avg_Date = []
    for name, group in Extraction_Avg.groupby('HNAME'):
        Avg_T3 = group.loc[group.loc[:, 'RESFP'] <= 3, 'RADAT'].apply(lambda x: pd.to_datetime(x, format = '%Y%m%d')).diff().mean()
        Avg_Beyer = group.nlargest(3,'BEYER_SPEED').loc[:, 'RADAT'].apply(lambda x: pd.to_datetime(x, format = '%Y%m%d')).diff().mean()
        Avg_Date.append([name, Avg_T3, Avg_Beyer])
    Avg_Date = pd.DataFrame(Avg_Date, columns=['HNAME','Avg_T3','Avg_Beyer'])

    Raceday = pd.to_datetime(Raceday, format = '%Y%m%d')
    Feature_DF = Feature_DF.merge(Extraction_Performance, how='left').merge(Avg_Date, how='left')
    Feature_DF.loc[:,['First_Date','Last_T3_Date','Top_Beyer_Date']]=Feature_DF.loc[:,['First_Date','Last_T3_Date','Top_Beyer_Date']].fillna('20120101')
    Feature_DF.loc[:,['Avg_T3','Avg_Beyer']]=Feature_DF.loc[:,['Avg_T3','Avg_Beyer']].fillna(Raceday-pd.to_datetime('20120101',format='%Y%m%d'))

    Feature_DF.loc[:,'T3'] = Raceday - pd.to_datetime(Feature_DF.loc[:, 'Last_T3_Date'], format = '%Y%m%d')
    Feature_DF.loc[:,'T3'] = Feature_DF.loc[:,'T3'] - Feature_DF.loc[:,'Avg_T3'].abs()
    Feature_DF.loc[:,'Beyer'] = Raceday - pd.to_datetime(Feature_DF.loc[:, 'Top_Beyer_Date'], format = '%Y%m%d')
    Feature_DF.loc[:,'Beyer'] = Feature_DF.loc[:,'Beyer'] - Feature_DF.loc[:,'Avg_Beyer'].abs()

    Feature_DF.loc[:,'CC_REC_DAY_PT3'] = Feature_DF.loc[:,'T3'].fillna(Feature_DF.loc[:,'Beyer'])
    Feature_DF.loc[:,'CC_REC_DAY_PT3'] = Feature_DF.loc[:,'CC_REC_DAY_PT3'].apply(lambda x : int(str(x).split('days')[0]))
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_REC_DAY_PT3']]

    return Feature_DF

"""
CC_REC_NUM_LT3
"""

def CC_REC_NUM_LT3(Dataframe, HNAME_List, Raceday):

    """
    Predicted number of races until next Top 3
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, CC_REC_NUM_LT3]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction_Performance = Extraction_Database("""
                                                 Select HNAME, First_RARID, Beyer_RARID, Last_T3_RARID from (
                                                 Select HNAME, First_RARID, Beyer_RARID from (
                                                 Select HNAME, min(RARID) First_RARID from Race_PosteriorDb
                                                 where RADAT < {Raceday} and HNAME in {HNAME_List}
                                                 Group by HNAME) FIRST
                                                 LEFT OUTER JOIN
                                                 (Select HNAME HNAME_BEYER, RARID Beyer_RARID, max(BEYER_SPEED)from Race_PosteriorDb
                                                  where RADAT < {Raceday} and HNAME in {HNAME_List}
                                                  Group by HNAME) BEYER
                                                 ON FIRST.HNAME = BEYER.HNAME_BEYER) FIRST_BEYER
                                                 LEFT OUTER JOIN
                                                 (Select HNAME HNAME_T3, max(RARID) Last_T3_RARID from Race_PosteriorDb
                                                  where RADAT < {Raceday} and HNAME in {HNAME_List} and RESFP <= 3
                                                  Group by HNAME) T3
                                                 ON T3.HNAME_T3 = FIRST_BEYER.HNAME
                                                 """.format(Raceday = Raceday, HNAME_List = HNAME_List))

    Extraction_Avg = Extraction_Database("""
                                         Select HNAME, RARID, RESFP, BEYER_SPEED from Race_PosteriorDb
                                         where RADAT < {Raceday} and HNAME in {HNAME_List}
                                         """.format(Raceday = Raceday, HNAME_List = HNAME_List))

    Avg_Date = []
    for name, group in Extraction_Avg.groupby('HNAME'):
        name
        group

        group = group.sort_values('RARID').reset_index(drop = True).reset_index()
        Avg_T3 = group.loc[group.loc[:, 'RESFP'] <= 3, 'index'].diff().abs().mean()
        Avg_Beyer = group.nlargest(3,'BEYER_SPEED').loc[:, 'index'].diff().abs().mean()
        Today = group.loc[:,'index'].max()+1
        Last_Beyer = Extraction_Performance.loc[Extraction_Performance.loc[:,'HNAME']==name, 'Beyer_RARID'].values[0]
        Last_Beyer = group.loc[group.loc[:,'RARID'] == Last_Beyer,'index'].values[0]
        Diff_Beyer = Today - Last_Beyer - Avg_Beyer
        try :
            Last_T3 = Extraction_Performance.loc[Extraction_Performance.loc[:,'HNAME']==name, 'Last_T3_RARID'].values[0]
            Last_T3 = group.loc[group.loc[:,'RARID'] == Last_T3,'index'].values[0]
            Diff_T3 = Today - Last_T3 - Avg_T3
        except :
            Diff_T3 = np.NaN
        Avg_Date.append([name, Diff_T3, Diff_Beyer])
    Avg_Date = pd.DataFrame(Avg_Date, columns=['HNAME','Diff_T3','Diff_Beyer'])

    Feature_DF = Feature_DF.merge(Avg_Date, how='left')
    Feature_DF.loc[:,'CC_REC_NUM_LT3'] = Feature_DF.loc[:,'Diff_T3'].fillna(Feature_DF.loc[:,'Diff_Beyer'])
    Feature_DF.loc[:,'CC_REC_NUM_LT3'].fillna(Feature_DF.loc[:,'CC_REC_NUM_LT3'].max(), inplace = True)
    Feature_DF.loc[:,'CC_REC_NUM_LT3'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_REC_NUM_LT3']]

    return Feature_DF

"""
CC_REC_NUM_DAYB
"""

def CC_REC_NUM_DAYB(Dataframe, HNAME_List, Raceday):

    """
    Number of Days since Last Best Beyer Speed Figure
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, CC_REC_NUM_DAYB]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HN HNAME, RADAT, First_Date from
                                     (Select HNAME HN, min(RADAT) First_Date from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     Group by HNAME) A
                                     LEFT OUTER JOIN
                                     (Select HNAME, RADAT, max(BEYER_SPEED) Beyer from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     Group by HNAME) B
                                     ON A.HN = B.HNAME
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'RADAT'].fillna(Feature_DF.loc[:,'First_Date'], inplace = True)
    Feature_DF.loc[:,'RADAT'].fillna(Raceday, inplace = True)

    Raceday = pd.to_datetime(Raceday, format = '%Y%m%d')
    Feature_DF.loc[:,'CC_REC_NUM_DAYB'] = Raceday - pd.to_datetime(Feature_DF.loc[:, 'RADAT'], format = '%Y%m%d')
    Feature_DF.loc[:,'CC_REC_NUM_DAYB'] = Feature_DF.loc[:,'CC_REC_NUM_DAYB'].apply(lambda x : int(str(x).split('days')[0]))
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_REC_NUM_DAYB']]

    return Feature_DF

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

================================== Class ==================================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
CC_CLS
"""

def CC_CLS(Dataframe, HNAME_List, Raceday):

    """
    HKJC Rating of Horse
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, CC_CLS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','HJRAT']]
    Feature_DF = Feature_DF.rename(columns={'HNAME': 'HNAME', 'HJRAT': 'CC_CLS'})

    return Feature_DF

"""
CC_CLS_D
"""

def CC_CLS_D(Dataframe, HNAME_List, Raceday):

    """
    Change in HKJC Rating
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, CC_CLS_D]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, HJRAT from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))

    JRat = []
    for name, group in Extraction.groupby('HNAME'):
        Rating = (group.loc[:,'HJRAT'].diff() / group.loc[:,'HJRAT']).dropna().values
        if len(Rating) >1:
            model = SimpleExpSmoothing(Rating)
            model = model.fit()
            JRat.append([name, model.forecast()[0]])
        elif len(Rating) == 1:
            JRat.append([name,Rating[0]])
        else :
            JRat.append([name,0])
    JRat = pd.DataFrame(JRat, columns=['HNAME','CC_CLS_D'])

    Feature_DF = Feature_DF.merge(JRat, how='left')
    Feature_DF.loc[:,'CC_CLS_D'].fillna(Feature_DF.loc[:,'CC_CLS_D'].min(), inplace = True)
    Feature_DF.loc[:,'CC_CLS_D'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_CLS_D']]

    return Feature_DF

"""
CC_CLS_CC
"""

def CC_CLS_CC(Dataframe, HNAME_List, Raceday):

    """
    Horse's Compotitive Class
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, CC_CLS_CC]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','HJRAT']]
    Underlying_Class = Feature_DF.nlargest(3, 'HJRAT').mean().to_list()[0]

    Races = Extraction_Database("""
                                Select HNAME, RARID, HJRAT CC_CLS_CC from RaceDb
                                where RARID in (
                                Select RARID from RaceDb
                                where RADAT < {Raceday} and HNAME in {HNAME_List} and RESFP <= 3)
                                """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    Races_AvgHJRAT = Races.groupby('RARID')['CC_CLS_CC'].apply(lambda x: x.nlargest(3).mean())
    Races_AvgHJRAT = Races_AvgHJRAT.reset_index()
    Race_IDs = Extraction_Database("""
                                   Select HNAME, RARID from RaceDb
                                   where RADAT < {Raceday} and HNAME in {HNAME_List} and RESFP <= 3
                                   """.format(Raceday = Raceday, HNAME_List = HNAME_List))

    Merged = Race_IDs.merge(Races_AvgHJRAT)
    Feature_DF = Feature_DF.merge(Merged.groupby('HNAME').mean()['CC_CLS_CC'].reset_index(), how='left')
    Feature_DF.loc[:, 'CC_CLS_CC'].fillna(Underlying_Class, inplace = True)
    Feature_DF.loc[:, 'CC_CLS_CC'] = Underlying_Class - Feature_DF.loc[:, 'CC_CLS_CC']
    Feature_DF.loc[:, 'CC_CLS_CC'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:, ['HNAME', 'CC_CLS_CC']]

    return Feature_DF

"""
CC_CLS_CL
"""

def CC_CLS_CL(Dataframe, HNAME_List, Raceday):

    """
    Horse's Competition Level
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, CC_CLS_CL]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','HJRAT']]

    Horse_Comp = []
    #For each horse, get data for last 5 races
    for Horse in Dataframe['HNAME'].tolist():
        Extraction = Extraction_Database("""
                                         Select HNAME, RARID, HJRAT, RESFP from RaceDb
                                         where RARID in (
                                         Select RARID from RaceDb
                                         where RADAT < {Raceday} and HNAME = {Horse}
                                         ORDER BY RARID DESC
                                         LIMIT 5)
                                         """.format(Raceday = Raceday, Horse = "'" + Horse + "'"))

        for RARID, race in Extraction.groupby('RARID'):
            Horse_Rat = race.loc[race.loc[:,'HNAME']==Horse,'HJRAT'].to_list()[0]
            Horse_FP = race.loc[race.loc[:,'HNAME']==Horse,'RESFP'].to_list()[0]
            Comp_Rat = race.nlargest(3, 'HJRAT').loc[:,'HJRAT'].mean()
            Comp_Level = (Comp_Rat - Horse_Rat) / Horse_FP
            Horse_Comp.append([Horse,Comp_Level])
    Horse_Comp = pd.DataFrame(Horse_Comp, columns=['HNAME', 'Comp_Level'])

    #Recency Weighting
    Comp = []
    for name, group in Horse_Comp.groupby('HNAME'):
        Comp_Figure = group.loc[:,'Comp_Level'].dropna().values
        try :
            model = SimpleExpSmoothing(Comp_Figure)
            model = model.fit()
            Comp.append([name, model.forecast()[0]])
        except :
            Comp.append([name,Comp_Figure[0]])
    Comp = pd.DataFrame(Comp, columns=['HNAME','CC_CLS_CL'])

    Feature_DF = Feature_DF.merge(Comp, how='left')
    Feature_DF.loc[:, 'CC_CLS_CL'].fillna(Feature_DF.loc[:, 'CC_CLS_CL'].min(), inplace = True)
    Feature_DF.loc[:, 'CC_CLS_CL'].fillna(0, inplace=True)
    Feature_DF = Feature_DF.loc[:, ['HNAME', 'CC_CLS_CL']]

    return Feature_DF

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

================================ Bodyweight ================================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
CC_BWEI
"""

def CC_BWEI(Dataframe, HNAME_List, Raceday):

    """
    Current Bodyweight of Horse
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, CC_BWEI]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','HBWEI']]
    Feature_DF = Feature_DF.rename(columns={'HNAME': 'HNAME', 'HBWEI': 'CC_BWEI'})

    return Feature_DF

"""
CC_BWEI_D
"""

def CC_BWEI_D(Dataframe, HNAME_List, Raceday):

    """
    Change in Bodyweight of Horse
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, CC_BWEI_D]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, HBWEI from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))

    HBWEI = []
    for name, group in Extraction.groupby('HNAME'):
        Weight = (group.loc[:,'HBWEI'].diff() / group.loc[:,'HBWEI']).dropna().values
        if len(Weight) >1:
            model = SimpleExpSmoothing(Weight)
            model = model.fit()
            HBWEI.append([name, model.forecast()[0]])
        elif len(Weight) == 1:
            HBWEI.append([name,Weight[0]])
        else :
            HBWEI.append([name,0])
    HBWEI = pd.DataFrame(HBWEI, columns=['HNAME','CC_BWEI_D'])

    Feature_DF = Feature_DF.merge(HBWEI, how='left')
    Feature_DF.loc[:,'CC_BWEI_D'] = Feature_DF.loc[:,'CC_BWEI_D'].abs()
    Feature_DF.loc[:,'CC_BWEI_D'].fillna(Feature_DF.loc[:,'CC_BWEI_D'].max(), inplace = True)
    Feature_DF.loc[:,'CC_BWEI_D'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_BWEI_D']]

    return Feature_DF

"""
CC_BWEI_DWIN
"""

def CC_BWEI_DWIN(Dataframe, HNAME_List, Raceday):

    """
    Bodyweight difference with Winning Performance
    Abs(Current Bodyweight - Average Winning Bodyweight ) / Average Winning Bodyweight)
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, CC_BWEI_DWIN]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','HBWEI']]

    Extraction_Win = Extraction_Database("""
                                        Select HNAME, avg(HBWEI) Win_Weight from RaceDb
                                        where RADAT < {Raceday} and HNAME in {HNAME_List} and RESWL = 1
                                        Group by HNAME
                                        """.format(Raceday = Raceday, HNAME_List = HNAME_List))

    Bodyweight = Extraction_Database("""
                                     Select HNAME, RARID, HBWEI Best from RaceDb
                                     where HNAME in {HNAME_List} and RADAT < {Raceday}
                                     """.format(HNAME_List=HNAME_List, Raceday=Raceday))

    Speed_Ratings = Extraction_Database("""
                                        Select HNAME, RARID, BEYER_SPEED from Race_PosteriorDb
                                        where HNAME in {HNAME_List} and RADAT < {Raceday}
                                        """.format(HNAME_List=HNAME_List, Raceday=Raceday))

    idx = Speed_Ratings.groupby(['HNAME'])['BEYER_SPEED'].transform(max) == Speed_Ratings['BEYER_SPEED']
    Speed_Ratings_Weight = Speed_Ratings[idx].merge(Bodyweight).loc[:,['HNAME','Best']]
    Speed_Ratings_Weight = Speed_Ratings_Weight.groupby('HNAME').max().reset_index()

    Feature_DF = Feature_DF.merge(Extraction_Win, how='left').merge(Speed_Ratings_Weight, how='left')
    Feature_DF.loc[:,'Filled_Weight'] = Feature_DF.loc[:,'Win_Weight'].fillna(Feature_DF.loc[:,'Best'])

    Feature_DF.loc[:,'CC_BWEI_DWIN'] = ((Feature_DF.loc[:,'HBWEI'] - Feature_DF.loc[:,'Filled_Weight']) / Feature_DF.loc[:,'Filled_Weight']).abs()
    Feature_DF.loc[:,'CC_BWEI_DWIN'].fillna(Feature_DF.loc[:,'CC_BWEI_DWIN'].max(), inplace = True)
    Feature_DF.loc[:,'CC_BWEI_DWIN'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_BWEI_DWIN']]

    return Feature_DF

"""
CC_BWEI_DT3
"""

def CC_BWEI_DT3(Dataframe, HNAME_List, Raceday):

    """
    Absolute Difference in Bodyweight compared to average Top 3 finish of horse in percentage of Bodyweight
    Abs((Current Bodyweight - Average Top 3 Bodyweight ) / Average Top 3 Bodyweight)
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, CC_BWEI_DT3]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','HBWEI']]

    Extraction_T3 = Extraction_Database("""
                                        Select HNAME, avg(HBWEI) T3_Weight from RaceDb
                                        where RADAT < {Raceday} and HNAME in {HNAME_List} and RESFP <= 3
                                        Group by HNAME
                                        """.format(Raceday = Raceday, HNAME_List = HNAME_List))

    Bodyweight = Extraction_Database("""
                                     Select HNAME, RARID, HBWEI Best from RaceDb
                                     where HNAME in {HNAME_List} and RADAT < {Raceday}
                                     """.format(HNAME_List=HNAME_List, Raceday=Raceday))

    Speed_Ratings = Extraction_Database("""
                                        Select HNAME, RARID, BEYER_SPEED from Race_PosteriorDb
                                        where HNAME in {HNAME_List} and RADAT < {Raceday}
                                        """.format(HNAME_List=HNAME_List, Raceday=Raceday))

    idx = Speed_Ratings.groupby(['HNAME'])['BEYER_SPEED'].transform(max) == Speed_Ratings['BEYER_SPEED']
    Speed_Ratings_Weight = Speed_Ratings[idx].merge(Bodyweight).loc[:,['HNAME','Best']]
    Speed_Ratings_Weight = Speed_Ratings_Weight.groupby('HNAME').max().reset_index()

    Feature_DF = Feature_DF.merge(Extraction_T3, how='left').merge(Speed_Ratings_Weight, how='left')
    Feature_DF.loc[:,'Filled_Weight'] = Feature_DF.loc[:,'T3_Weight'].fillna(Feature_DF.loc[:,'Best'])

    Feature_DF.loc[:,'CC_BWEI_DT3'] = ((Feature_DF.loc[:,'HBWEI'] - Feature_DF.loc[:,'Filled_Weight']) / Feature_DF.loc[:,'Filled_Weight']).abs()
    Feature_DF.loc[:,'CC_BWEI_DT3'].fillna(Feature_DF.loc[:,'CC_BWEI_DT3'].max(), inplace = True)
    Feature_DF.loc[:,'CC_BWEI_DT3'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_BWEI_DT3']]

    return Feature_DF

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

============================== Weight Carried ==============================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
CC_WEI
"""

def CC_WEI(Dataframe, HNAME_List, Raceday):

    """
    Weight Carried of Horse
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, CC_WEI]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','HWEIC']]
    Feature_DF = Feature_DF.rename(columns={'HNAME': 'HNAME', 'HWEIC': 'CC_WEI'})

    return Feature_DF

"""
CC_WEI_DIST
"""

def CC_WEI_DIST(Dataframe, HNAME_List, Raceday):

    """
    Weight Carried of Horse relative to Distance of the Race
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, CC_WEI_DIST]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','HWEIC']]
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    #Normalise Weight
    Feature_DF.loc[:,'HWEIC'] = (Feature_DF.loc[:,'HWEIC'] - np.mean(Feature_DF.loc[:,'HWEIC'])) / np.std(Feature_DF.loc[:,'HWEIC'])
    Feature_DF.loc[:,'HWEIC'].fillna(Feature_DF.loc[:,'HWEIC'].max(), inplace = True)
    Feature_DF.loc[:,'HWEIC'].fillna(0, inplace = True)

    Feature_DF.loc[:,'CC_WEI_DIST'] = Feature_DF.loc[:,'HWEIC'] * Distance
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_WEI_DIST']]

    return Feature_DF

"""
CC_WEI_PER
"""

def CC_WEI_PER(Dataframe, HNAME_List, Raceday):

    """
    Weight Carried of Horse relative to Bodyweight of the Horse
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, CC_WEI_PER]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','HWEIC', 'HBWEI']]
    Feature_DF.loc[:,'CC_WEI_PER'] = Feature_DF.loc[:,'HWEIC'] / Feature_DF.loc[:,'HBWEI']
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_WEI_PER']]

    return Feature_DF

"""
CC_WEI_D
"""

def CC_WEI_D(Dataframe, HNAME_List, Raceday):

    """
    Change in Weight carried of Horse
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, CC_WEI_D]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, HWEIC from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))

    HWEIC = []
    for name, group in Extraction.groupby('HNAME'):
        Weight = (group.loc[:,'HWEIC'].diff() / group.loc[:,'HWEIC']).dropna().values
        if len(Weight) >1:
            model = SimpleExpSmoothing(Weight)
            model = model.fit()
            HWEIC.append([name, model.forecast()[0]])
        elif len(Weight) == 1:
            HWEIC.append([name,Weight[0]])
        else :
            HWEIC.append([name,0])
    HWEIC = pd.DataFrame(HWEIC, columns=['HNAME','CC_WEI_D'])

    Feature_DF = Feature_DF.merge(HWEIC, how='left')
    Feature_DF.loc[:,'CC_WEI_D'].fillna(Feature_DF.loc[:,'CC_WEI_D'].max(), inplace = True)
    Feature_DF.loc[:,'CC_WEI_D'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_WEI_D']]

    return Feature_DF

"""
CC_WEI_SP
"""

def CC_WEI_SP(Dataframe, HNAME_List, Raceday):

    """
    Weight Carried’s effect on Speed Figure
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, CC_WEI_SP]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','HWEIC']]

    Extraction_Weight = Extraction_Database("""
                                            Select HNAME, max(RARID), HWEIC Last_Weight from RaceDb
                                            where RADAT < {Raceday} and HNAME in {HNAME_List}
                                            Group by HNAME
                                            """.format(Raceday = Raceday, HNAME_List = HNAME_List))


    Extraction_Speed = Extraction_Database("""
                                           Select HNAME, BEYER_SPEED, max(RARID) from Race_PosteriorDb
                                           where RADAT < {Raceday} and HNAME in {HNAME_List}
                                           Group by HNAME
                                           """.format(Raceday = Raceday, HNAME_List = HNAME_List))

    Feature_DF = Feature_DF.merge(Extraction_Weight, how='left')
    Feature_DF.loc[:,'Weight_Change'] = Feature_DF.loc[:,'HWEIC'] - Feature_DF.loc[:,'Last_Weight']

    try :
        #Load Models
        with open(Aux_Reg_Path + 'CC_WEI_INC_Model.joblib', 'rb') as location:
            Inc_model = joblib.load(location)
        with open(Aux_Reg_Path + 'CC_WEI_DEC_Model.joblib', 'rb') as location:
            Dec_model = joblib.load(location)

        SP_Change = []
        for index, row in Feature_DF.iterrows():
            if row['Weight_Change'] < 0:
                SP_Change.append(Dec_model.predict(np.array([[row.loc['Weight_Change']]])))
            else :
                SP_Change.append(Inc_model.predict(np.array([[row.loc['Weight_Change']]])))
    except :
        pass

    Feature_DF = Feature_DF.merge(Extraction_Speed, how='left')
    try :
        Feature_DF.loc[:,'CC_WEI_SP'] = Feature_DF.loc[:,'BEYER_SPEED'] + Feature_DF.loc[:,'SP_Change']
    except :
        Feature_DF.loc[:,'CC_WEI_SP'] = Feature_DF.loc[:,'BEYER_SPEED']
    Feature_DF.loc[:,'CC_WEI_SP'].fillna(Feature_DF.loc[:,'CC_WEI_SP'].min(), inplace = True)
    Feature_DF.loc[:,'CC_WEI_SP'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_WEI_SP']]

    return Feature_DF


def Weight_Aug_Reg(Raceday):

    Extraction = Extraction_Database("""
                                     Select A.HNAME, A.RARID, BEYER_SPEED, HWEIC from
                                     (Select HNAME, RARID, BEYER_SPEED from Race_PosteriorDb
                                     where RADAT < {Raceday}) A,
                                     (Select HNAME, RARID, HWEIC from RaceDb where RADAT < {Raceday}) B
                                     where A.HNAME = B.HNAME and A.RARID = B.RARID
                                     """.format(Raceday = Raceday))
    Extraction.fillna(0, inplace=True)

    if len(Extraction) == 0:
        return None

    DF = []
    for name, group in Extraction.groupby('HNAME'):
        Speed_Figure = group.loc[:,'BEYER_SPEED'].diff().values
        Weight = group.loc[:,'HWEIC'].diff().values
        One_Horse = pd.DataFrame({'Speed': Speed_Figure, 'Weight':Weight})
        One_Horse.replace([np.inf, -np.inf], np.nan, inplace = True)
        One_Horse.dropna(inplace = True)
        DF.append(One_Horse)
    DF = pd.concat(DF)

    #Slice in increase in weight leading to decrease in Speed Figure
    Increase_Weight = DF.loc[(DF.loc[:,'Speed'] < 0) & (DF.loc[:,'Weight'] > 0), :]

    #Slice in decrease in weight leading to increase in Speed Figure
    Decrease_Weight = DF.loc[(DF.loc[:,'Speed'] > 0) & (DF.loc[:,'Weight'] < 0), :]

    #NO not fit model if there is no races
    if len(Increase_Weight) == 0 or len(Decrease_Weight) == 0:
        return None

    #Model Fitting
    model = LinearRegression()
    model.fit(Increase_Weight.loc[:,'Weight'].values.reshape(-1,1), Increase_Weight.loc[:,'Speed'])
    #Save Model
    with open(Aux_Reg_Path + 'CC_WEI_INC_Model.joblib', 'wb') as location:
        joblib.dump(model, location)

    #Model Fitting
    model = LinearRegression()
    model.fit(Decrease_Weight.loc[:,'Weight'].values.reshape(-1,1), Decrease_Weight.loc[:,'Speed'])
    #Save Model
    with open(Aux_Reg_Path + 'CC_WEI_DEC_Model.joblib', 'wb') as location:
        joblib.dump(model, location)

    return None

"""
CC_WEI_EXP
"""

def CC_WEI_EXP(Dataframe, HNAME_List, Raceday):

    """
    Weight Carrying Experience of Horse
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, CC_WEI_EXP]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','HWEIC']]
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Horse_Weight_Req = ["(HNAME = '" + row['HNAME'] + "' and HWEIC >= " + str(row['HWEIC']) + ')' for index, row in Feature_DF.iterrows()]
    Horse_Weight_Req = ' or '.join(Horse_Weight_Req)

    Races_above_today = Extraction_Database("""
                                            Select HNAME, RARID from
                                            (Select HNAME, RARID, HWEIC, RADAT, RADIS from RaceDb
                                            where {Horse_Weight_Req})
                                            where RADAT < {Raceday} and RADIS >= {Distance}
                                            """.format(Raceday = Raceday, Horse_Weight_Req = Horse_Weight_Req, Distance=Distance))
    Race_ID_List = '('+str(Races_above_today['RARID'].tolist())[1:-1]+')'
    Speed_Ratings_tdy = Extraction_Database("""
                                            Select HNAME, RARID, BEYER_SPEED from Race_PosteriorDb
                                            where RARID in {Race_ID_List} and HNAME in {HNAME_List}
                                            """.format(Race_ID_List=Race_ID_List, HNAME_List=HNAME_List))
    Best_Speed_Rating = Races_above_today.merge(Speed_Ratings_tdy, how='left').groupby('HNAME').max().reset_index().loc[:,['HNAME','BEYER_SPEED']]
    Best_Speed_Rating = Best_Speed_Rating.rename(columns={'HNAME': 'HNAME', 'BEYER_SPEED': 'Primary'})

    Race_heaviest = Extraction_Database("""
                                        Select HNAME, RARID, HWEIC from RaceDb
                                        where RADAT < {Raceday} and HNAME in {HNAME_List} and RADIS >= {Distance}
                                        """.format(Raceday = Raceday, HNAME_List = HNAME_List, Distance=Distance))
    Race_ID_List = '('+str(Race_heaviest['RARID'].tolist())[1:-1]+')'
    Speed_Ratings_heavy = Extraction_Database("""
                                              Select HNAME, RARID, BEYER_SPEED from Race_PosteriorDb
                                              where RARID in {Race_ID_List} and HNAME in {HNAME_List}
                                              """.format(Race_ID_List=Race_ID_List, HNAME_List=HNAME_List))
    Backup_Speed_Rating = Race_heaviest.merge(Speed_Ratings_heavy, on=['HNAME','RARID'], how='left')
    Backup_Speed_Rating.loc[:,'BEYER_SPEED'].fillna(Backup_Speed_Rating.loc[:,'BEYER_SPEED'].min(), inplace = True)
    if Backup_Speed_Rating.loc[:,'BEYER_SPEED'].sum() == 0:
        Backup_Speed_Rating.loc[:,'BEYER_SPEED'].fillna(0, inplace = True)

    if len(Backup_Speed_Rating) == 0:
        Feature_DF.loc[:,'CC_WEI_EXP'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','CC_WEI_EXP']]
        return Feature_DF

    idx = Backup_Speed_Rating.groupby(['HNAME'])['HWEIC'].transform(max) == Backup_Speed_Rating['HWEIC']
    Backup_Speed_Rating = Backup_Speed_Rating[idx].groupby('HNAME').mean().reset_index().loc[:,['HNAME','BEYER_SPEED']]
    Backup_Speed_Rating = Backup_Speed_Rating.rename(columns={'HNAME': 'HNAME', 'BEYER_SPEED': 'Back_Up'})

    Feature_DF = Feature_DF.merge(Best_Speed_Rating, how='left').merge(Backup_Speed_Rating, how='left')
    Feature_DF.loc[:,'CC_WEI_EXP'] = Feature_DF.loc[:,'Primary'].fillna(Feature_DF.loc[:,'Back_Up'])
    Feature_DF.loc[:,'CC_WEI_EXP'].fillna(Feature_DF.loc[:,'CC_WEI_EXP'].min(), inplace = True)
    Feature_DF.loc[:,'CC_WEI_EXP'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_WEI_EXP']]

    return Feature_DF

"""
CC_WEI_MAX
"""

def CC_WEI_MAX(Dataframe, HNAME_List, Raceday):

    """
    Weight carrying threshold
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, CC_WEI_MAX]
    """

    Feature_DF = Dataframe.loc[:,['HNAME', 'HWEIC']]

    Extraction_T3 = Extraction_Database("""
                                        Select HNAME, Avg(HWEIC) T3_Weight from RaceDb
                                        where RADAT < {Raceday} and HNAME in {HNAME_List} and RESFP <= 3
                                        Group by HNAME
                                        """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    Weight = Extraction_Database("""
                                 Select HNAME, RARID, HWEIC SP_WEI from RaceDb
                                 where HNAME in {HNAME_List} and RADAT < {Raceday}
                                 """.format(HNAME_List=HNAME_List, Raceday=Raceday))

    Speed_Ratings = Extraction_Database("""
                                        Select Distinct HNAME, RARID, BEYER_SPEED from Race_PosteriorDb
                                        where HNAME in {HNAME_List} and RADAT < {Raceday}
                                        """.format(HNAME_List=HNAME_List, Raceday=Raceday))

    idx = Speed_Ratings.groupby(['HNAME'])['BEYER_SPEED'].transform(max) == Speed_Ratings['BEYER_SPEED']
    Speed_Ratings_Weight = Speed_Ratings[idx].merge(Weight, on = ['HNAME', 'RARID']).loc[:,['HNAME','SP_WEI']]
    Speed_Ratings_Weight = Speed_Ratings_Weight.groupby('HNAME').apply(lambda x : x.sort_values('SP_WEI').max()).reset_index(drop = True)

    if len(Speed_Ratings_Weight) == 0:
        Feature_DF.loc[:,'CC_WEI_MAX'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','CC_WEI_MAX']]
        return Feature_DF

    Feature_DF = Feature_DF.merge(Extraction_T3, how='left').merge(Speed_Ratings_Weight, how='left')
    Feature_DF.loc[:,'CC_WEI_MAX'] = Feature_DF.loc[:,'T3_Weight'].fillna(Feature_DF.loc[:,'HWEIC'])
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_WEI_MAX']]

    return Feature_DF

"""
CC_WEI_BCH
"""

def CC_WEI_BCH(Dataframe, HNAME_List, Raceday):

    """
    Weight carrying over the threshold
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, CC_WEI_BCH]
    """

    Feature_DF = Dataframe.loc[:,['HNAME', 'HWEIC']]
    Location ="'" + Dataframe.loc[:,'RALOC'].values[0] + "'"
    Surface ="'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Benchmark = Extraction_Database("""
                                    Select Avg(HWEIC) from RaceDb
                                    where RADAT < {Raceday} and RESFP <= 3
                                    and RALOC = {Location} and RATRA = {Surface} and RADIS = {Distance}
                                    """.format(Raceday = Raceday, Location = Location,
                                    Surface = Surface, Distance = Distance)).values.tolist()[0][0]

    if Benchmark == None:
        Feature_DF.loc[:,'CC_WEI_BCH'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','CC_WEI_BCH']]
        return Feature_DF

    Feature_DF.loc[:,'CC_WEI_BCH'] = (Feature_DF.loc[:,'HWEIC'] - Benchmark)#.abs()
    Feature_DF.loc[:,'CC_WEI_BCH'].fillna(Feature_DF.loc[:,'CC_WEI_BCH'].max(), inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_WEI_BCH']]

    return Feature_DF

