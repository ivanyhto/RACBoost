#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ivan
"""

"""
Feature : Past Performance of Horse
"""

#Loading Libraries
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from pyhorse.Database_Management import Extraction_Database

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


def Going_Similarity(Going):

    """
    Parameters
    ----------
    Going : eg 好快, 好黏, 黏地, 黏軟, 黏軟, 濕快, 泥快, 泥好, 泥好

    Returns
    -------
    Dictionary of percentage difference
    """

    Going_Dict = {'好快': {'好快':0, '好地':-1/5, '好黏':-2/11, '黏地':-3/11, '黏軟':-4/11, '軟地':-5/11},
                  '好地': {'好快':-1/6, '好地':0, '好黏':-1/12, '黏地':-1/6, '黏軟':-1/4, '軟地':-1/3},
                  '好黏': {'好快':-2/13, '好地':-1/13, '好黏':0, '黏地':-1/13, '黏軟':-2/13, '軟地':-3/13},
                  '黏地': {'好快':-3/14, '好地':-1/7, '好黏':-1/14, '黏地':0, '黏軟':-1/14, '軟地':-2/14},
                  '黏軟': {'好快':-4/15, '好地':-1/5, '好黏':-2/15, '黏地':-1/15, '黏軟':0, '軟地':-1/15},
                  '軟地': {'好快':-5/16, '好地':-1/4, '好黏':-3/16, '黏地':-2/16, '黏軟':-1/16, '軟地':0},

                  '濕快': {'濕快':0, '泥快':-1/10, '泥好':-2/10, '濕慢':-4/10},
                  '泥快': {'濕快':-1/11, '泥快':0, '泥好':-1/11, '濕慢':-3/11},
                  '泥好': {'濕快':-2/12, '泥快':-1/12, '泥好':0, '濕慢':-1/3},
                  '濕慢': {'濕快':-3/8, '泥快':-5/16, '泥好':-1/4, '濕慢':0}}

    return Going_Dict[Going]

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

=============================== Experience ===============================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
PP_EXP_NRACE
"""

def PP_EXP_NRACE(Dataframe, HNAME_List, Raceday):

    """
    Number of Past Races Ran by Horse
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_EXP_NRACE]
    """

    Feature_DF = Dataframe.loc[:, ['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, count(RARID) PP_EXP_NRACE from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     Group by HNAME
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_EXP_NRACE'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:, ['HNAME','PP_EXP_NRACE']]

    return Feature_DF

"""
PP_EXP_NRACE_DIST
"""

def PP_EXP_NRACE_DIST(Dataframe, HNAME_List, Raceday):

    """
    Number of Past Races Ran by Horse on Underlying Distance
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_EXP_NRACE_DIST]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Extraction = Extraction_Database("""
                                     Select HNAME, count(RARID) PP_EXP_NRACE_DIST from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RADIS = {Distance}
                                     Group by HNAME
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Distance = Distance))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_EXP_NRACE_DIST'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_EXP_NRACE_DIST']]

    return Feature_DF

"""
PP_EXP_NRACE_SIM_DIST
"""

def PP_EXP_NRACE_SIM_DIST(Dataframe, HNAME_List, Raceday):

    """
    Number of Past Races Ran by Horse on Similar Distance
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_EXP_NRACE_SIM_DIST]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Distance = Dataframe.loc[:,'RADIS'].values[0]
    Dist_Dict = Distance_Similarity(Distance)


    Extraction = Extraction_Database("""
                                     Select HNAME, count(RARID) Num_Race, RADIS from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RADIS != {Distance}
                                     Group by HNAME, RADIS
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Distance = Distance))
    Extraction.replace({'RADIS': Dist_Dict}, inplace = True)

    def Normalise(group):
        group.loc[:,'Norm'] = np.exp(group.loc[:,'RADIS']) / np.exp(group.loc[:,'RADIS']).sum()
        group.loc[:,'Normed'] = group.loc[:,'Norm'] * group.loc[:,'Num_Race']
        return group.loc[:,'Normed'].sum()

    #For First Runners
    if len(Extraction) == 0:
        Feature_DF.loc[:,'PP_EXP_NRACE_SIM_DIST'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_EXP_NRACE_SIM_DIST']]
        return Feature_DF

    Extraction = Extraction.groupby('HNAME').apply(Normalise).reset_index()
    Extraction.columns = ['HNAME','PP_EXP_NRACE_SIM_DIST']
    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_EXP_NRACE_SIM_DIST'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_EXP_NRACE_SIM_DIST']]

    return Feature_DF

"""
PP_EXP_NRACE_GO
"""

def PP_EXP_NRACE_GO(Dataframe, HNAME_List, Raceday):

    """
    Number of Past Races Ran by Horse on Similar Distance
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_EXP_NRACE_GO]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Surface ="'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Going = Dataframe.loc[:,'RAGOG'].values[0].strip()
    Going_Dict = Going_Similarity(Going)

    Extraction = Extraction_Database("""
                                     Select HNAME, count(RARID) Num_Race, RAGOG from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RATRA = {Surface}
                                     Group by HNAME, RAGOG
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Surface = Surface))
    Extraction.loc[:,'RAGOG'] = Extraction.loc[:,'RAGOG'].map(lambda x : x.strip())

    def Normalise(group):
        group = pd.DataFrame(Going_Dict.keys(), columns = ['RAGOG']).merge(group.loc[:,['RAGOG','Num_Race']], how='left')
        group.fillna(0, inplace=True)
        group.replace({'RAGOG': Going_Dict}, inplace = True)
        group.loc[:,'Norm'] = np.exp(group.loc[:,'RAGOG']) / np.exp(group.loc[:,'RAGOG']).sum()
        group.loc[:,'Normed'] = group.loc[:,'Norm'] * group.loc[:,'Num_Race']
        return group.loc[:,'Normed'].sum()

    #For First Runners
    if len(Extraction) == 0:
        Feature_DF.loc[:,'PP_EXP_NRACE_GO'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_EXP_NRACE_GO']]
        return Feature_DF

    Extraction = Extraction.groupby('HNAME').apply(Normalise).reset_index()
    Extraction.columns = ['HNAME','PP_EXP_NRACE_GO']
    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_EXP_NRACE_GO'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_EXP_NRACE_GO']]

    return Feature_DF

"""
PP_EXP_NRACE_SUR
"""

def PP_EXP_NRACE_SUR(Dataframe, HNAME_List, Raceday):

    """
    Number of Past Races Ran by Horse on Surface
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_EXP_NRACE_SUR]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Surface ="'" + Dataframe.loc[:,'RATRA'].values[0] + "'"

    Extraction = Extraction_Database("""
                                     Select HNAME, count(RARID) PP_EXP_NRACE_SUR from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RATRA = {Surface}
                                     Group by HNAME
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Surface = Surface))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_EXP_NRACE_SUR'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_EXP_NRACE_SUR']]

    return Feature_DF

"""
PP_EXP_NRACE_PFL
"""

def PP_EXP_NRACE_PFL(Dataframe, HNAME_List, Raceday):

    """
    Number of Past Races Ran by Horse on Profile
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_EXP_NRACE_PFL]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Location = "'" + Dataframe.loc[:,'RALOC'].values[0] + "'"
    Surface = "'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Extraction = Extraction_Database("""
                                     Select HNAME, count(RARID) PP_EXP_NRACE_PFL from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     and RALOC = {Location} and RATRA = {Surface} and RADIS = {Distance}
                                     Group by HNAME
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List,
                                     Location = Location, Surface = Surface, Distance = Distance))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_EXP_NRACE_PFL'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_EXP_NRACE_PFL']]

    return Feature_DF

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

============================ Finishing History ============================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
PP_FH_FP_CLS
"""

def PP_FH_FP_CLS(Dataframe, HNAME_List, Raceday):

    """
    Horse's Finishing Position weighted by classs
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_FH_FP_CLS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, Avg(Class_FP) PP_FH_FP_CLS from
                                     (Select HNAME, RESFP * (1+ RACLS) Class_FP from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List})
                                     Group by HNAME
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_FH_FP_CLS'].fillna(Feature_DF.loc[:,'PP_FH_FP_CLS'].max(), inplace = True)
    Feature_DF.loc[:,'PP_FH_FP_CLS'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_FP_CLS']]

    return Feature_DF

"""
PP_FH_FP_AVG
"""

def PP_FH_FP_AVG(Dataframe, HNAME_List, Raceday):

    """
    Horse's Average Finishing Position in History
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_FH_FP_AVG]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, RESFP PP_FH_FP_AVG from RaceDb
                                     where RARID in
                                     (Select Distinct RARID from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List})
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    if len(Extraction) == 0:
        Feature_DF.loc[:,'PP_FH_FP_AVG'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_FP_AVG']]
        return Feature_DF

    races = Extraction.groupby('RARID')
    mean = races.transform(np.mean).loc[:,'PP_FH_FP_AVG']
    std = races.transform(np.std).loc[:,'PP_FH_FP_AVG']
    Extraction.loc[:,'PP_FH_FP_AVG'] = (Extraction.loc[:,'PP_FH_FP_AVG'] - mean) / std
    Extraction = Extraction.groupby('HNAME').mean().loc[:,'PP_FH_FP_AVG'].reset_index()

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_FH_FP_AVG'].fillna(Feature_DF.loc[:,'PP_FH_FP_AVG'].max(), inplace = True)
    Feature_DF.loc[:,'PP_FH_FP_AVG'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_FP_AVG']]

    return Feature_DF

"""
PP_FH_FP_AVGRW
"""

def PP_FH_FP_AVGRW(Dataframe, HNAME_List, Raceday):

    """
    Recency Weighted Avg Finishing Position of Horse
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_FH_FP_AVGRW]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, RESFP PP_FH_FP_AVGRW from RaceDb
                                     where RARID in
                                     (Select Distinct RARID from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List})
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))

    if len(Extraction) == 0:
        Feature_DF.loc[:,'PP_FH_FP_AVGRW'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_FP_AVGRW']]
        return Feature_DF

    races = Extraction.groupby('RARID')
    mean = races.transform(np.mean).loc[:,'PP_FH_FP_AVGRW']
    std = races.transform(np.std).loc[:,'PP_FH_FP_AVGRW']
    Extraction.loc[:,'PP_FH_FP_AVGRW'] = (Extraction.loc[:,'PP_FH_FP_AVGRW'] - mean) / std

    FP = []
    for name, group in Extraction.groupby('HNAME'):
        FP_History = group.loc[:,'PP_FH_FP_AVGRW'].dropna().values
        if len(FP_History) >1:
            model = SimpleExpSmoothing(FP_History)
            model = model.fit()
            FP.append([name, model.forecast()[0]])
        elif len(FP_History) == 1:
            FP.append([name,FP_History[0]])
        else :
            FP.append([name,0])
    Avg_FP = pd.DataFrame(FP, columns=['HNAME','PP_FH_FP_AVGRW'])

    Feature_DF = Feature_DF.merge(Avg_FP, how='left')
    Feature_DF.loc[:,'PP_FH_FP_AVGRW'].fillna(Feature_DF.loc[:,'PP_FH_FP_AVGRW'].min(), inplace = True)
    Feature_DF.loc[:,'PP_FH_FP_AVGRW'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_FP_AVGRW']]

    return Feature_DF

"""
PP_FH_FP_BIN
"""

def PP_FH_FP_BIN(Dataframe, HNAME_List, Raceday):

    """
    Horse's Average Binned Finishing Position in History
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_FH_FP_BIN]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, RESFP PP_FH_FP_BIN from RaceDb
                                     where RARID in
                                     (Select Distinct RARID from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List})
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))

    if len(Extraction) == 0:
        Feature_DF.loc[:,'PP_FH_FP_BIN'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_FP_BIN']]
        return Feature_DF

    Extraction.loc[:,'PP_FH_FP_BIN'] = Extraction.loc[:,'PP_FH_FP_BIN'].apply(lambda x : 4 if x > 4 else x)

    races = Extraction.groupby('RARID')
    mean = races.transform(np.mean).loc[:,'PP_FH_FP_BIN']
    std = races.transform(np.std).loc[:,'PP_FH_FP_BIN']
    Extraction.loc[:,'PP_FH_FP_BIN'] = (Extraction.loc[:,'PP_FH_FP_BIN'] - mean) / std
    Extraction = Extraction.groupby('HNAME').mean().loc[:,'PP_FH_FP_BIN'].reset_index()

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_FH_FP_BIN'].fillna(Feature_DF.loc[:,'PP_FH_FP_BIN'].max(), inplace = True)
    Feature_DF.loc[:,'PP_FH_FP_BIN'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_FP_BIN']]

    return Feature_DF

"""
PP_FH_FP_AVG
"""

def PP_FH_FP_DIST(Dataframe, HNAME_List, Raceday):

    """
    Horse's Average Finishing Position on Distance
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_FH_FP_DIST]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, RESFP PP_FH_FP_DIST from RaceDb
                                     where RARID in
                                     (Select Distinct RARID from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RADIS = {Distance})
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Distance = Distance))
    if len(Extraction) == 0:
        Feature_DF.loc[:,'PP_FH_FP_DIST'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_FP_DIST']]
        return Feature_DF

    races = Extraction.groupby('RARID')
    mean = races.transform(np.mean).loc[:,'PP_FH_FP_DIST']
    std = races.transform(np.std).loc[:,'PP_FH_FP_DIST']
    Extraction.loc[:,'PP_FH_FP_DIST'] = (Extraction.loc[:,'PP_FH_FP_DIST'] - mean) / std
    Extraction = Extraction.groupby('HNAME').mean().loc[:,'PP_FH_FP_DIST'].reset_index()

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_FH_FP_DIST'].fillna(Feature_DF.loc[:,'PP_FH_FP_DIST'].max(), inplace = True)
    Feature_DF.loc[:,'PP_FH_FP_DIST'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_FP_DIST']]

    return Feature_DF

"""
PP_FH_FP_SIM_DIST
"""

def PP_FH_FP_SIM_DIST(Dataframe, HNAME_List, Raceday):

    """
    Horse's Average Finishing Position on Similar Distance
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_FH_FP_SIM_DIST]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Distance = Dataframe.loc[:,'RADIS'].values[0]
    Dist_Dict = Distance_Similarity(Distance)

    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, RADIS, RESFP PP_FH_FP_SIM_DIST from RaceDb
                                     where RARID in
                                     (Select Distinct RARID from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RADIS != {Distance})
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Distance = Distance))
    if len(Extraction) == 0:
        Feature_DF.loc[:,'PP_FH_FP_SIM_DIST'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_FP_SIM_DIST']]
        return Feature_DF

    races = Extraction.groupby('RARID')
    mean = races.transform(np.mean).loc[:,'PP_FH_FP_SIM_DIST']
    std = races.transform(np.std).loc[:,'PP_FH_FP_SIM_DIST']
    Extraction.loc[:,'PP_FH_FP_SIM_DIST'] = (Extraction.loc[:,'PP_FH_FP_SIM_DIST'] - mean) / std
    Extraction.replace({'RADIS': Dist_Dict}, inplace = True)

    def Normalise(group):
        group.loc[:,'Norm'] = np.exp(group.loc[:,'RADIS']) / np.exp(group.loc[:,'RADIS']).sum()
        group.loc[:,'Normed'] = group.loc[:,'Norm'] * group.loc[:,'PP_FH_FP_SIM_DIST']
        return group.loc[:,'Normed'].sum()

    Extraction = Extraction.groupby('HNAME').apply(Normalise).reset_index()
    Extraction.columns = ['HNAME','PP_FH_FP_SIM_DIST']

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_FH_FP_SIM_DIST'].fillna(Feature_DF.loc[:,'PP_FH_FP_SIM_DIST'].max(), inplace = True)
    Feature_DF.loc[:,'PP_FH_FP_SIM_DIST'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_FP_SIM_DIST']]

    return Feature_DF

"""
PP_FH_FP_GO
"""

def PP_FH_FP_GO(Dataframe, HNAME_List, Raceday):

    """
    Horse's Average Finishing Position on Going
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_FH_FP_GO]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Surface ="'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Going = Dataframe.loc[:,'RAGOG'].values[0].strip()
    Going_Dict = Going_Similarity(Going)

    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, RAGOG, RESFP PP_FH_FP_GO from RaceDb
                                     where RARID in
                                     (Select Distinct RARID from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RATRA = {Surface})
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Surface = Surface))
    if len(Extraction) == 0:
        Feature_DF.loc[:,'PP_FH_FP_GO'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_FP_GO']]
        return Feature_DF

    races = Extraction.groupby('RARID')
    mean = races.transform(np.mean).loc[:,'PP_FH_FP_GO']
    std = races.transform(np.std).loc[:,'PP_FH_FP_GO']
    Extraction.loc[:,'PP_FH_FP_GO'] = (Extraction.loc[:,'PP_FH_FP_GO'] - mean) / std
    Extraction.loc[:,'RAGOG'] = Extraction.loc[:,'RAGOG'].map(lambda x : x.strip())
    Extraction.replace({'RAGOG': Going_Dict}, inplace = True)

    def Normalise(group):
        group.loc[:,'Norm'] = np.exp(group.loc[:,'RAGOG']) / np.exp(group.loc[:,'RAGOG']).sum()
        group.loc[:,'Normed'] = group.loc[:,'Norm'] * group.loc[:,'PP_FH_FP_GO']
        return group.loc[:,'Normed'].sum()

    Extraction = Extraction.groupby('HNAME').apply(Normalise).reset_index()
    Extraction.columns = ['HNAME','PP_FH_FP_GO']

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_FH_FP_GO'].fillna(Feature_DF.loc[:,'PP_FH_FP_GO'].max(), inplace = True)
    Feature_DF.loc[:,'PP_FH_FP_GO'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_FP_GO']]

    return Feature_DF

"""
PP_FH_FP_SUR
"""

def PP_FH_FP_SUR(Dataframe, HNAME_List, Raceday):

    """
    Horse's Average Finishing Position on Surface
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_FH_FP_SUR]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Surface = "'" + Dataframe.loc[:,'RATRA'].values[0] + "'"

    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, RESFP PP_FH_FP_SUR from RaceDb
                                     where RARID in
                                     (Select Distinct RARID from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RATRA = {Surface})
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Surface = Surface))
    if len(Extraction) == 0:
        Feature_DF.loc[:,'PP_FH_FP_SUR'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_FP_SUR']]
        return Feature_DF

    races = Extraction.groupby('RARID')
    mean = races.transform(np.mean).loc[:,'PP_FH_FP_SUR']
    std = races.transform(np.std).loc[:,'PP_FH_FP_SUR']
    Extraction.loc[:,'PP_FH_FP_SUR'] = (Extraction.loc[:,'PP_FH_FP_SUR'] - mean) / std
    Extraction = Extraction.groupby('HNAME').mean().loc[:,'PP_FH_FP_SUR'].reset_index()

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_FH_FP_SUR'].fillna(Feature_DF.loc[:,'PP_FH_FP_SUR'].max(), inplace = True)
    Feature_DF.loc[:,'PP_FH_FP_SUR'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_FP_SUR']]

    return Feature_DF

"""
PP_FH_FP_PFL
"""

def PP_FH_FP_PFL(Dataframe, HNAME_List, Raceday):

    """
    Horse's Average Finishing Position on Profile
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_FH_FP_PFL]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Location ="'" + Dataframe.loc[:,'RALOC'].values[0] + "'"
    Surface ="'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, RESFP PP_FH_FP_PFL from RaceDb
                                     where RARID in
                                     (Select Distinct RARID from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     and RALOC = {Location} and RATRA = {Surface} and RADIS = {Distance})
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Surface = Surface,
                                     Location = Location, Distance = Distance))
    if len(Extraction) == 0:
        Feature_DF.loc[:,'PP_FH_FP_PFL'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_FP_PFL']]
        return Feature_DF

    races = Extraction.groupby('RARID')
    mean = races.transform(np.mean).loc[:,'PP_FH_FP_PFL']
    std = races.transform(np.std).loc[:,'PP_FH_FP_PFL']
    Extraction.loc[:,'PP_FH_FP_PFL'] = (Extraction.loc[:,'PP_FH_FP_PFL'] - mean) / std
    Extraction = Extraction.groupby('HNAME').mean().loc[:,'PP_FH_FP_PFL'].reset_index()

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_FH_FP_PFL'].fillna(Feature_DF.loc[:,'PP_FH_FP_PFL'].max(), inplace = True)
    Feature_DF.loc[:,'PP_FH_FP_PFL'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_FP_PFL']]

    return Feature_DF

"""
PP_FH_FTP
"""

def PP_FH_FTP(Dataframe, HNAME_List, Raceday):

    """
    Finishing Time on Profile
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_FH_FTP]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Location ="'" + Dataframe.loc[:,'RALOC'].values[0] + "'"
    Surface ="'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, RESFT from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     and RALOC = {Location} and RATRA = {Surface} and RADIS = {Distance}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List,
                                     Location = Location, Surface = Surface, Distance = Distance))

    FT = []
    for name, group in Extraction.groupby('HNAME'):
        Time = group.loc[:,'RESFT'].dropna().values
        if len(Time) >1:
            model = SimpleExpSmoothing(Time)
            model = model.fit()
            FT.append([name, model.forecast()[0]])
        elif len(Time) == 1:
            FT.append([name,Time[0]])
        else :
            FT.append([name,0])
    FT = pd.DataFrame(FT, columns=['HNAME','PP_FH_FTP'])

    Feature_DF = Feature_DF.merge(FT, how='left')
    Feature_DF.loc[:,'PP_FH_FTP'].fillna(Feature_DF.loc[:,'PP_FH_FTP'].max(), inplace = True)
    Feature_DF.loc[:,'PP_FH_FTP'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_FTP']]

    return Feature_DF

"""
PP_FH_NUMW
"""

def PP_FH_NUMW(Dataframe, HNAME_List, Raceday):

    """
    Horse's Number of Wins in History
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_FH_NUMW]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, sum(RESWL) PP_FH_NUMW from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     Group by HNAME
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_FH_NUMW'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_NUMW']]

    return Feature_DF

"""
PP_FH_HTH
"""

def PP_FH_HTH(Dataframe, HNAME_List, Raceday):

    """
    Horse's Head to Head Wins in History
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_FH_HTH]
    """

    Feature_DF = Dataframe.loc[:,['HNAME', 'RESFO']]
    Feature_DF.loc[:, 'Probi'] = Feature_DF.loc[:, 'RESFO'].map(lambda x : (1-0.175)/x)

    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, RESFP from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))

    #For First Runners
    if len(Extraction) == 0:
        Feature_DF.loc[:,'PP_FH_HTH'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_HTH']]
        return Feature_DF

    Feature_DF.loc[:,'PP_FH_HTH'] = 0
    for Horse in Dataframe.loc[:,'HNAME'].tolist():

        Race_ID_List = Extraction.loc[Extraction.loc[:, 'HNAME'] == Horse, 'RARID'].to_list()
        #slice out Races where the horse competed
        Races = Extraction.loc[Extraction.loc[:, 'RARID'].isin(Race_ID_List), :]

        #Slice out races with multiple horses
        Races = Races[Races['RARID'].map(Races.groupby('RARID').size() > 1)]

        for name, group in Races.groupby('RARID'):

            try :
                FP_List = group.loc[:,'HNAME'].to_list()
                size = len(FP_List)
                idx_list = [idx + 1 for idx, val in enumerate(FP_List) if val == Horse]
                splitted = [FP_List[i: j] for i, j in zip([0] + idx_list, idx_list + ([size] if idx_list[-1] != size else []))]

                try :
                    lost = [Feature_DF.loc[Feature_DF.loc[:,'HNAME'] == H, 'Probi'].to_list()[0] for H in splitted[0]]
                    won = [Feature_DF.loc[Feature_DF.loc[:,'HNAME'] == H, 'Probi'].to_list()[0] for H in splitted[1]]
                    HTH_Score = sum(won) - sum(lost[:-1])
                except :
                    lost = [Feature_DF.loc[Feature_DF.loc[:,'HNAME'] == H, 'Probi'].to_list()[0] for H in splitted[0]]
                    HTH_Score = sum(lost[:-1])
            except :
                HTH_Score = 0

            Feature_DF.loc[Feature_DF.loc[:,'HNAME'] == Horse,'PP_FH_HTH'] = HTH_Score
    Feature_DF.loc[:,'PP_FH_HTH'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:, ['HNAME','PP_FH_HTH']]

    return Feature_DF

"""
PP_FH_WIN
"""

def PP_FH_WIN(Dataframe, HNAME_List, Raceday):

    """
    Whether horse has won
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_FH_WIN]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, sum(RESWL) Num_Win from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     Group by HNAME
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_FH_WIN'] = Feature_DF.loc[:,'Num_Win'].apply(lambda x : int(x>0))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_WIN']].fillna(0)

    return Feature_DF

"""
PP_FH_WINP
"""

def PP_FH_WINP(Dataframe, HNAME_List, Raceday):

    """
    Horse's Win Percentage
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_FH_WINP]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, sum(RESWL) Num_Win,  count(RARID) Num_Races from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     Group by HNAME
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:, 'PP_FH_WINP'] = Feature_DF.loc[:, 'Num_Win'] / Feature_DF.loc[:, 'Num_Races']
    Feature_DF.loc[:,'PP_FH_WINP'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_WINP']]

    return Feature_DF

"""
PP_FH_WINPY
"""

def PP_FH_WINPY(Dataframe, HNAME_List, Raceday):

    """
    Horse's Win Percentage of Horse in one year
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_FH_WINPY]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Offset_Raceday = (pd.to_datetime(Raceday) + pd.tseries.offsets.DateOffset(months=-12)).strftime("%Y%m%d")

    Extraction = Extraction_Database("""
                                     Select HNAME, sum(RESWL) Num_Win,  count(RARID) Num_Races from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RADAT > {Offset_Raceday}
                                     Group by HNAME
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Offset_Raceday = Offset_Raceday))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:, 'PP_FH_WINPY'] = Feature_DF.loc[:, 'Num_Win'] / Feature_DF.loc[:, 'Num_Races']
    Feature_DF.loc[:,'PP_FH_WINPY'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_WINPY']]

    return Feature_DF

"""
PP_FH_WINP_W
"""

def PP_FH_WINP_W(Dataframe, HNAME_List, Raceday):

    """
    Horse's Winning Percenntage in History
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_FH_WINP_W]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction_Win = Extraction_Database("""
                                         Select HNAME, RARID, RESWL from RaceDb
                                         where RADAT < {Raceday} and HNAME in {HNAME_List}
                                         """.format(Raceday = Raceday, HNAME_List = HNAME_List))

    WW_R = []
    for name, group in Extraction_Win.groupby('HNAME'):
        WW = 0
        WL = group.loc[:,'RESWL'].to_list()
        for index, item in enumerate(WL):
            try :
                if item == WL[index+1] == 1:
                    WW += 1
            except :
                 pass
        WW_R.append([name, WW])
    WW_R = pd.DataFrame(WW_R, columns=['HNAME','WW_Count'])

    Extraction_Num = Extraction_Database("""
                                         Select HNAME, sum(RESWL) Num_Win, count(RARID) Num_Races from RaceDb
                                         where RADAT < {Raceday} and HNAME in {HNAME_List}
                                         Group by HNAME
                                         """.format(Raceday = Raceday, HNAME_List = HNAME_List))

    Feature_DF = Feature_DF.merge(WW_R, how='left').merge(Extraction_Num, how='left')
    Feature_DF.loc[:, 'PP_FH_WINP_W'] = (Feature_DF.loc[:, 'WW_Count'] / Feature_DF.loc[:, 'Num_Win']).fillna(0) * \
                                        (Feature_DF.loc[:, 'Num_Win'] / Feature_DF.loc[:, 'Num_Races'])
    Feature_DF.loc[:,'PP_FH_WINP_W'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_WINP_W']]

    return Feature_DF

"""
PP_FH_WINP_DIST
"""

def PP_FH_WINP_DIST(Dataframe, HNAME_List, Raceday):

    """
    Horse's Winning Percenntage in History on Distance
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_FH_WINP_DIST]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Extraction = Extraction_Database("""
                                     Select HNAME, sum(RESWL) Num_Win,  count(RARID) Num_Races from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RADIS = {Distance}
                                     Group by HNAME
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Distance = Distance))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:, 'PP_FH_WINP_DIST'] = Feature_DF.loc[:, 'Num_Win'] / Feature_DF.loc[:, 'Num_Races']
    Feature_DF.loc[:,'PP_FH_WINP_DIST'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_WINP_DIST']]

    return Feature_DF

"""
PP_FH_WINP_SIM_DIST
"""

def PP_FH_WINP_SIM_DIST(Dataframe, HNAME_List, Raceday):

    """
    Horse's Winning Percenntage in History on Similar Distance
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_FH_WINP_SIM_DIST]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Distance = Dataframe.loc[:,'RADIS'].values[0]
    Dist_Dict = Distance_Similarity(Distance)

    Extraction = Extraction_Database("""
                                     Select HNAME, sum(RESWL) Num_Win,  count(RARID) Num_Races, RADIS from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RADIS != {Distance}
                                     Group by HNAME, RADIS
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Distance = Distance))
    #For First Runners
    if len(Extraction) == 0:
        Feature_DF.loc[:,'PP_FH_WINP_SIM_DIST'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_WINP_SIM_DIST']]
        return Feature_DF

    Extraction.replace({'RADIS': Dist_Dict}, inplace = True)
    Extraction.loc[:,'WINP'] = Extraction.loc[:,'Num_Win'] / Extraction.loc[:,'Num_Races']

    def Normalise(group):
        group.loc[:,'Norm'] = np.exp(group.loc[:,'RADIS']) / np.exp(group.loc[:,'RADIS']).sum()
        group.loc[:,'Normed'] = group.loc[:,'Norm'] * group.loc[:,'WINP']
        return group.loc[:,'Normed'].sum()

    Extraction = Extraction.groupby('HNAME').apply(Normalise).reset_index()
    Extraction.columns = ['HNAME','PP_FH_WINP_SIM_DIST']
    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_FH_WINP_SIM_DIST'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_WINP_SIM_DIST']]

    return Feature_DF

"""
PP_FH_WINP_GO
"""

def PP_FH_WINP_GO(Dataframe, HNAME_List, Raceday):

    """
    Horse's Winning Percenntage in History on Similar Going
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_FH_WINP_GO]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Surface = "'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Going = Dataframe.loc[:,'RAGOG'].values[0].strip()
    Going_Dict = Going_Similarity(Going)

    Extraction = Extraction_Database("""
                                     Select HNAME, sum(RESWL) Num_Win,  count(RARID) Num_Races, RAGOG from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RATRA = {Surface}
                                     Group by HNAME, RAGOG
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Surface = Surface))
    #For First Runners
    if len(Extraction) == 0:
        Feature_DF.loc[:,'PP_FH_WINP_GO'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_WINP_GO']]
        return Feature_DF

    Extraction.loc[:,'RAGOG'] = Extraction.loc[:,'RAGOG'].map(lambda x : x.strip())
    Extraction.replace({'RAGOG': Going_Dict}, inplace = True)
    Extraction.loc[:,'WINP'] = Extraction.loc[:,'Num_Win'] / Extraction.loc[:,'Num_Races']

    def Normalise(group):
        group.loc[:,'Norm'] = np.exp(group.loc[:,'RAGOG']) / np.exp(group.loc[:,'RAGOG']).sum()
        group.loc[:,'Normed'] = group.loc[:,'Norm'] * group.loc[:,'WINP']
        return group.loc[:,'Normed'].sum()

    Extraction = Extraction.groupby('HNAME').apply(Normalise).reset_index()
    Extraction.columns = ['HNAME','PP_FH_WINP_GO']
    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_FH_WINP_GO'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_WINP_GO']]

    return Feature_DF

"""
PP_FH_WINP_SUR
"""

def PP_FH_WINP_SUR(Dataframe, HNAME_List, Raceday):

    """
    Horse's Winning Percenntage in History on Surface
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_FH_WINP_SUR]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Surface ="'" + Dataframe.loc[:,'RATRA'].values[0] + "'"

    Extraction = Extraction_Database("""
                                     Select HNAME, sum(RESWL) Num_Win,  count(RARID) Num_Races from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RATRA = {Surface}
                                     Group by HNAME
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Surface = Surface))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_FH_WINP_SUR'] = Feature_DF.loc[:,'Num_Win'] / Feature_DF.loc[:,'Num_Races']
    Feature_DF.loc[:,'PP_FH_WINP_SUR'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_WINP_SUR']]

    return Feature_DF

"""
PP_FH_WINP_PFL
"""

def PP_FH_WINP_PFL(Dataframe, HNAME_List, Raceday):

    """
    Horse's Winning Percenntage in History on Profile
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_FH_WINP_PFL]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Location ="'" + Dataframe.loc[:,'RALOC'].values[0] + "'"
    Surface ="'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Extraction = Extraction_Database("""
                                     Select HNAME, sum(RESWL) Num_Win,  count(RARID) Num_Races from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     and RALOC = {Location} and RATRA = {Surface} and RADIS = {Distance}
                                     Group by HNAME
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List,
                                     Location = Location, Surface = Surface, Distance = Distance))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_FH_WINP_PFL'] = Feature_DF.loc[:,'Num_Win'] / Feature_DF.loc[:,'Num_Races']
    Feature_DF.loc[:,'PP_FH_WINP_PFL'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_WINP_PFL']]

    return Feature_DF

"""
PP_FH_T3P
"""

def PP_FH_T3P(Dataframe, HNAME_List, Raceday):

    """
    Horse's Top 3 Percenntage in History
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_FH_T3P]
    """
    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, Num_Races, T3 from (
                                     Select HNAME, count(RARID) Num_Races from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     Group by HNAME) NUM
                                     LEFT OUTER JOIN
                                     (Select HNAME HNAME_T3, count(RARID) T3 from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RESFP <= 3
                                     Group by HNAME) T3
                                     ON NUM.HNAME = T3.HNAME_T3
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_FH_T3P'] = Feature_DF.loc[:,'T3'] / Feature_DF.loc[:,'Num_Races']
    Feature_DF.loc[:,'PP_FH_T3P'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_T3P']]

    return Feature_DF

"""
PP_FH_T3P_T3
"""

def PP_FH_T3P_T3(Dataframe, HNAME_List, Raceday):

    """
    Horse's T3 Percentage after T3
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_FH_T3P_T3]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction_T3 = Extraction_Database("""
                                        Select HNAME, RARID, RESFP from RaceDb
                                        where RADAT < {Raceday} and HNAME in {HNAME_List}
                                        """.format(Raceday = Raceday, HNAME_List = HNAME_List))

    T3_R = []
    for name, group in Extraction_T3.groupby('HNAME'):
        T3_T3 = 0
        FP = group.loc[:,'RESFP'].to_list()
        for index, item in enumerate(FP):
            try :
                if (item <= 3) and (FP[index+1] <= 3):
                    T3_T3 += 1
            except :
                 pass
        T3_R.append([name, T3_T3])
    T3_R = pd.DataFrame(T3_R, columns=['HNAME','T3T3_Count'])

    Extraction_Num = Extraction_Database("""
                                         Select HNAME, Num_Races, T3 from (
                                         Select HNAME, count(RARID) Num_Races from RaceDb
                                         where RADAT < {Raceday} and HNAME in {HNAME_List}
                                         Group by HNAME) NUM
                                         LEFT OUTER JOIN
                                         (Select HNAME HNAME_T3, count(RARID) T3 from RaceDb
                                         where RADAT < {Raceday} and HNAME in {HNAME_List} and RESFP <= 3
                                         Group by HNAME) T3
                                         ON NUM.HNAME = T3.HNAME_T3
                                         """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    Extraction_Num.fillna(0, inplace = True)

    Feature_DF = Feature_DF.merge(T3_R, how='left').merge(Extraction_Num, how='left')
    Feature_DF.loc[:, 'PP_FH_T3P_T3'] = (Feature_DF.loc[:, 'T3T3_Count'] / Feature_DF.loc[:, 'T3']).fillna(0) * \
                                        (Feature_DF.loc[:, 'T3'] / Feature_DF.loc[:, 'Num_Races'])
    Feature_DF.loc[:,'PP_FH_T3P_T3'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_T3P_T3']]

    return Feature_DF

"""
PP_FH_T3P_DIST
"""

def PP_FH_T3P_DIST(Dataframe, HNAME_List, Raceday):

    """
    Horse's Top 3 Percenntage in History on Distance
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_FH_T3P_DIST]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Extraction = Extraction_Database("""
                                     Select HNAME, Num_Races, T3 from (
                                     Select HNAME, count(RARID) Num_Races from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}  and RADIS = {Distance}
                                     Group by HNAME) NUM
                                     LEFT OUTER JOIN
                                     (Select HNAME HNAME_T3, count(RARID) T3 from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RADIS = {Distance} and RESFP <= 3
                                     Group by HNAME) T3
                                     ON NUM.HNAME = T3.HNAME_T3
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Distance = Distance))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_FH_T3P_DIST'] = Feature_DF.loc[:,'T3'] / Feature_DF.loc[:,'Num_Races']
    Feature_DF.loc[:,'PP_FH_T3P_DIST'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_T3P_DIST']]

    return Feature_DF

"""
PP_FH_T3P_SIM_DIST
"""

def PP_FH_T3P_SIM_DIST(Dataframe, HNAME_List, Raceday):

    """
    Horse's Top 3 Percenntage in History Similar Distance
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_FH_T3P_SIM_DIST]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Distance = Dataframe.loc[:,'RADIS'].values[0]
    Dist_Dict = Distance_Similarity(Distance)

    Extraction = Extraction_Database("""
                                     Select HNAME, Num_Races, T3, RADIS from (
                                     Select HNAME, count(RARID) Num_Races, RADIS from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}  and RADIS != {Distance}
                                     Group by HNAME, RADIS) NUM
                                     LEFT OUTER JOIN
                                     (Select HNAME HNAME_T3, count(RARID) T3, RADIS RADIS_T3 from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RADIS != {Distance} and RESFP <= 3
                                     Group by HNAME, RADIS_T3) T3
                                     ON NUM.HNAME = T3.HNAME_T3 and NUM.RADIS = T3.RADIS_T3
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Distance = Distance))

    #For First Runners
    if len(Extraction) == 0:
        Feature_DF.loc[:,'PP_FH_T3P_SIM_DIST'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_T3P_SIM_DIST']]
        return Feature_DF

    Extraction.replace({'RADIS': Dist_Dict}, inplace = True)
    Extraction.loc[:,'T3P'] = Extraction.loc[:,'T3'] / Extraction.loc[:,'Num_Races']

    def Normalise(group):
        group.loc[:,'Norm'] = np.exp(group.loc[:,'RADIS']) / np.exp(group.loc[:,'RADIS']).sum()
        group.loc[:,'Normed'] = group.loc[:,'Norm'] * group.loc[:,'T3P']
        return group.loc[:,'Normed'].sum()

    Extraction = Extraction.groupby('HNAME').apply(Normalise).reset_index()
    Extraction.columns = ['HNAME','PP_FH_T3P_SIM_DIST']
    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_FH_T3P_SIM_DIST'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_T3P_SIM_DIST']]

    return Feature_DF

"""
PP_FH_T3P_GO
"""

def PP_FH_T3P_GO(Dataframe, HNAME_List, Raceday):

    """
    Horse's Top 3 Percenntage in History Similar Going
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_FH_T3P_GO]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Surface ="'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Going = Dataframe.loc[:,'RAGOG'].values[0].strip()
    Going_Dict = Going_Similarity(Going)

    Extraction = Extraction_Database("""
                                     Select HNAME, Num_Races, T3, RAGOG from (
                                     Select HNAME, count(RARID) Num_Races, RAGOG from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RATRA = {Surface}
                                     Group by HNAME, RAGOG) NUM
                                     LEFT OUTER JOIN
                                     (Select HNAME HNAME_T3, count(RARID) T3, RAGOG RAGOG_T3 from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RATRA = {Surface} and RESFP <= 3
                                     Group by HNAME, RAGOG_T3) T3
                                     ON NUM.HNAME = T3.HNAME_T3 and NUM.RAGOG = T3.RAGOG_T3
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Surface = Surface))

    #For First Runners
    if len(Extraction) == 0:
        Feature_DF.loc[:,'PP_FH_T3P_GO'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_T3P_GO']]
        return Feature_DF

    Extraction.loc[:,'RAGOG'] = Extraction.loc[:,'RAGOG'].map(lambda x : x.strip())
    Extraction.replace({'RAGOG': Going_Dict}, inplace = True)
    Extraction.loc[:,'T3P'] = Extraction.loc[:,'T3'] / Extraction.loc[:,'Num_Races']

    def Normalise(group):
        group.loc[:,'Norm'] = np.exp(group.loc[:,'RAGOG']) / np.exp(group.loc[:,'RAGOG']).sum()
        group.loc[:,'Normed'] = group.loc[:,'Norm'] * group.loc[:,'T3P']
        return group.loc[:,'Normed'].sum()

    Extraction = Extraction.groupby('HNAME').apply(Normalise).reset_index()
    Extraction.columns = ['HNAME','PP_FH_T3P_GO']
    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_FH_T3P_GO'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_T3P_GO']]

    return Feature_DF

"""
PP_FH_T3P_SUR
"""

def PP_FH_T3P_SUR(Dataframe, HNAME_List, Raceday):

    """
    Horse's Top 3 Percenntage in History on Surface
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_FH_T3P_SUR]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Surface ="'" + Dataframe.loc[:,'RATRA'].values[0] + "'"

    Extraction = Extraction_Database("""
                                     Select HNAME, Num_Races, T3 from (
                                     Select HNAME, count(RARID) Num_Races from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RATRA = {Surface}
                                     Group by HNAME) NUM
                                     LEFT OUTER JOIN
                                     (Select HNAME HNAME_T3, count(RARID) T3 from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}  and RATRA = {Surface} and RESFP <= 3
                                     Group by HNAME) T3
                                     ON NUM.HNAME = T3.HNAME_T3
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Surface = Surface))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF['PP_FH_T3P_SUR'] = Feature_DF['T3'] / Feature_DF['Num_Races']
    Feature_DF.loc[:,'PP_FH_T3P_SUR'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_T3P_SUR']]

    return Feature_DF

"""
PP_FH_T3P_PFL
"""

def PP_FH_T3P_PFL(Dataframe, HNAME_List, Raceday):

    """
    Horse's Top 3 Percenntage in History
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_FH_T3P_PFL]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Location ="'" + Dataframe.loc[:,'RALOC'].values[0] + "'"
    Surface ="'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Extraction = Extraction_Database("""
                                     Select HNAME, Num_Races, T3 from (
                                     Select HNAME, count(RARID) Num_Races from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     and RALOC = {Location} and RATRA = {Surface} and RADIS = {Distance}
                                     Group by HNAME) NUM
                                     LEFT OUTER JOIN
                                     (Select HNAME HNAME_T3, count(RARID) T3 from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RESFP <= 3
                                     and RALOC = {Location} and RATRA = {Surface} and RADIS = {Distance}
                                     Group by HNAME) T3
                                     ON NUM.HNAME = T3.HNAME_T3
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List,
                                     Location = Location, Surface = Surface, Distance = Distance))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_FH_T3P_PFL'] = Feature_DF.loc[:,'T3'] / Feature_DF.loc[:,'Num_Races']
    Feature_DF.loc[:,'PP_FH_T3P_PFL'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_T3P_PFL']]

    return Feature_DF

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

========================== Beaten Length History ==========================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
PP_BL_AVG
"""

def PP_BL_AVG(Dataframe, HNAME_List, Raceday):

    """
    Average Beaten Length
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_BL_AVG]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, RESWD from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    BL = []
    for name, group in Extraction.groupby('HNAME'):
        Beaten_Length = group.loc[:,'RESWD'].dropna().values
        if len(Beaten_Length) >1:
            model = SimpleExpSmoothing(Beaten_Length)
            model = model.fit()
            BL.append([name, model.forecast()[0]])
        elif len(Beaten_Length) == 1:
            BL.append([name,Beaten_Length[0]])
        else :
            BL.append([name,0])
    BLF = pd.DataFrame(BL, columns=['HNAME','PP_BL_AVG'])

    Feature_DF = Feature_DF.merge(BLF, how='left')
    Feature_DF.loc[:,'PP_BL_AVG'].fillna(Feature_DF.loc[:,'PP_BL_AVG'].max(), inplace = True)
    Feature_DF.loc[:,'PP_BL_AVG'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_BL_AVG']]

    return Feature_DF

"""
PP_BL_SUM
"""

def PP_BL_SUM(Dataframe, HNAME_List, Raceday):

    """
    Summing Beaten Length
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_BL_SUM]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, RESWD from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    BL = []
    for name, group in Extraction.groupby('HNAME'):
        Beaten_Length = group.loc[:,'RESWD'].dropna().values
        if len(Beaten_Length) >1:
            model = SimpleExpSmoothing(Beaten_Length)
            model = model.fit()
            BL.append([name, model.forecast()[0], Beaten_Length.sum()])
        elif len(Beaten_Length) == 1:
            BL.append([name,Beaten_Length[0], Beaten_Length.sum()])
        else :
            BL.append([name,0,0])
    BLF = pd.DataFrame(BL, columns=['HNAME', 'PP_BL_AVG', 'SumBL'])

    BLF.loc[:,'PP_BL_SUM'] = (BLF.loc[:,'SumBL'] / BLF.loc[:,'SumBL'].sum()) * BLF.loc[:,'PP_BL_AVG']

    Feature_DF = Feature_DF.merge(BLF, how='left')
    Feature_DF.loc[:,'PP_BL_SUM'].fillna(Feature_DF.loc[:,'PP_BL_SUM'].max(), inplace = True)
    Feature_DF.loc[:,'PP_BL_SUM'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_BL_SUM']]

    return Feature_DF

"""
PP_BL_AVGF
"""

def PP_BL_AVGF(Dataframe, HNAME_List, Raceday):

    """
    Average Beaten Length Figure
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_BL_AVGF]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, BEATEN_FIGURE from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    BLF = []
    for name, group in Extraction.groupby('HNAME'):
        BL_Figure = group.loc[:,'BEATEN_FIGURE'].dropna().values
        if len(BL_Figure) >1:
            model = SimpleExpSmoothing(BL_Figure)
            model = model.fit()
            BLF.append([name, model.forecast()[0]])
        elif len(BL_Figure) == 1:
            BLF.append([name,BL_Figure[0]])
        else :
            BLF.append([name,0])
    BLF = pd.DataFrame(BLF, columns=['HNAME','PP_BL_AVGF'])

    Feature_DF = Feature_DF.merge(BLF, how='left')
    Feature_DF.loc[:,'PP_BL_AVGF'].fillna(Feature_DF.loc[:,'PP_BL_AVGF'].min(), inplace = True)
    Feature_DF.loc[:,'PP_BL_AVGF'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_BL_AVGF']]

    return Feature_DF

"""
PP_BL_AVGF_SUR
"""

def PP_BL_AVGF_SUR(Dataframe, HNAME_List, Raceday):

    """
    Average Beaten Length Figure on Surface
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_BL_AVGF_SUR]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Surface = "'" + Dataframe.loc[:,'RATRA'].values[0] + "'"

    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, BEATEN_FIGURE from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RATRA = {Surface}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Surface=Surface))
    BLF = []
    for name, group in Extraction.groupby('HNAME'):
        BL_Figure = group.loc[:,'BEATEN_FIGURE'].dropna().values
        if len(BL_Figure) >1:
            model = SimpleExpSmoothing(BL_Figure)
            model = model.fit()
            BLF.append([name, model.forecast()[0]])
        elif len(BL_Figure) == 1:
            BLF.append([name,BL_Figure[0]])
        else :
            BLF.append([name,0])
    BLF = pd.DataFrame(BLF, columns=['HNAME','PP_BL_AVGF_SUR'])

    Feature_DF = Feature_DF.merge(BLF, how='left')
    Feature_DF.loc[:,'PP_BL_AVGF_SUR'].fillna(Feature_DF.loc[:,'PP_BL_AVGF_SUR'].min(), inplace = True)
    Feature_DF.loc[:,'PP_BL_AVGF_SUR'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_BL_AVGF_SUR']]

    return Feature_DF


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

=========================== Speed Figure History ===========================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
PP_SPF_L1
"""

def PP_SPF_L1(Dataframe, HNAME_List, Raceday):

    """
    Last Speed Figure
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_SPF_L1]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, BEYER_SPEED PP_SPF_L1, max(RARID) from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     Group by HNAME
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    #For First Runners
    if len(Extraction) == 0:
        Feature_DF.loc[:,'PP_SPF_L1'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_L1']]
        return Feature_DF

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_SPF_L1'].fillna(Feature_DF.loc[:,'PP_SPF_L1'].min(), inplace = True)
    Feature_DF.loc[:,'PP_SPF_L1'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_L1']]


    return Feature_DF

"""
PP_SPF_L2
"""

def PP_SPF_L2(Dataframe, HNAME_List, Raceday):

    """
    Second Last Speed Figure
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_SPF_L2]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, BEYER_SPEED PP_SPF_L2, RARID from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     Order by RARID
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    #For First Runners
    if len(Extraction) == 0:
        Feature_DF.loc[:,'PP_SPF_L2'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_L2']]
        return Feature_DF

    def Normalise(group):
        sorted_df = group.sort_values(['RARID'], ascending = False).reset_index(drop = True)
        try :
            return sorted_df.loc[1,'PP_SPF_L2']
        except :
            return np.NaN

    Extraction = Extraction.groupby('HNAME').apply(Normalise).reset_index()
    Extraction.columns = ['HNAME','PP_SPF_L2']

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_SPF_L2'].fillna(Feature_DF.loc[:,'PP_SPF_L2'].min(), inplace = True)
    Feature_DF.loc[:,'PP_SPF_L2'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_L2']]

    return Feature_DF

"""
PP_SPF_SEC
"""

def PP_SPF_SEC(Dataframe, HNAME_List, Raceday):

    """
    2/3 Speed Figure
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_SPF_SEC]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, RADAT, BEYER_SPEED from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    Beyer = []
    for name, group in Extraction.groupby('HNAME'):
        try :
            Beyer.append([name, group.nlargest(3, 'RADAT').loc[:,'BEYER_SPEED'].nlargest(2).to_list()[1]])
        except :
            #Only one race in history
            Beyer.append([name, group.nlargest(3, 'RADAT').loc[:,'BEYER_SPEED'].nlargest(2).to_list()[0]])

    Sec_Beyer = pd.DataFrame(Beyer, columns=['HNAME','PP_SPF_SEC'])

    Feature_DF = Feature_DF.merge(Sec_Beyer, how='left')
    Feature_DF.loc[:,'PP_SPF_SEC'].fillna(Feature_DF.loc[:,'PP_SPF_SEC'].min(), inplace = True)
    Feature_DF.loc[:,'PP_SPF_SEC'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_SEC']]

    return Feature_DF

"""
PP_SPF_KNN_PFL
"""

def PP_SPF_KNN_PFL(Dataframe, HNAME_List, Raceday):

    """
    KNN prediction of Speed Figure
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_SPF_KNN_PFL]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Location =  Dataframe.loc[:,'RALOC'].values[0]
    Surface = Dataframe.loc[:,'RATRA'].values[0]
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    #Creating dummy variables of categorical variables -> standardise
    def preprocess_KNN(DF):
        DF.loc[:,'Turf'] = (DF.loc[:,'RATRA'] == 'T').astype(int)
        DF.loc[:,'ST'] = (DF.loc[:,'RALOC'] == 'ST').astype(int)
        DF = DF.loc[:,['Turf','ST','RADIS']]
        return DF

    X_Test = pd.DataFrame({'RALOC': [Location], 'RATRA': [Surface], 'RADIS':[Distance]})

    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, BEYER_SPEED, RADIS, RATRA, RALOC from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))

    Beyer = []
    for name, group in Extraction.groupby('HNAME'):
        X = group.loc[:,['RADIS', 'RALOC', 'RATRA']]
        X = preprocess_KNN(X)
        X.loc[:,'RADIS'] = X.loc[:,'RADIS'] / X.loc[:,'RADIS'].sum()
        y = group.loc[:,'BEYER_SPEED']
        try :
            model = KNeighborsRegressor(n_neighbors=2)
            model.fit(X, y)
            Beyer.append([name, model.predict(preprocess_KNN(X_Test))[0]])
        except:
            Beyer.append([name, np.mean(y)])
    Beyer_Speed = pd.DataFrame(Beyer, columns=['HNAME','PP_SPF_KNN_PFL'])

    Feature_DF = Feature_DF.merge(Beyer_Speed, how='left')
    Feature_DF.loc[:,'PP_SPF_KNN_PFL'].fillna(Feature_DF.loc[:,'PP_SPF_KNN_PFL'].min(), inplace = True)
    Feature_DF.loc[:,'PP_SPF_KNN_PFL'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_KNN_PFL']]

    return Feature_DF

"""
PP_SPF_D1
"""

def PP_SPF_D1(Dataframe, HNAME_List, Raceday):

    """
    Change in Speed Figure
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_SPF_D1]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, BEYER_SPEED PP_SPF_D1, RARID from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     Order by RARID
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    #For First Runners
    if len(Extraction) == 0:
        Feature_DF.loc[:,'PP_SPF_D1'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_D1']]
        return Feature_DF

    def Normalise(group):
        sorted_df = group.sort_values(['RARID'], ascending = False).reset_index(drop = True)
        try :
            return sorted_df.loc[0,'PP_SPF_D1'] - sorted_df.loc[1,'PP_SPF_D1']
        except :
            return np.NaN

    Extraction = Extraction.groupby('HNAME').apply(Normalise).reset_index()
    Extraction.columns = ['HNAME','PP_SPF_D1']

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_SPF_D1'].fillna(Feature_DF.loc[:,'PP_SPF_D1'].min(), inplace = True)
    Feature_DF.loc[:,'PP_SPF_D1'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_D1']]

    return Feature_DF

"""
PP_SPF_D
"""

def PP_SPF_D(Dataframe, HNAME_List, Raceday):

    """
    Percentage Change in Speed Figure
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_SPF_D]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, BEYER_SPEED from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))

    Beyer = []
    for name, group in Extraction.groupby('HNAME'):
        Speed_Figure = (group.loc[:,'BEYER_SPEED'].diff() / group.loc[:,'BEYER_SPEED']).dropna().values
        if len(Speed_Figure) >1:
            model = SimpleExpSmoothing(Speed_Figure)
            model = model.fit()
            Beyer.append([name, model.forecast()[0]])
        elif len(Speed_Figure) == 1:
            Beyer.append([name,Speed_Figure[0]])
        else :
            Beyer.append([name,np.NaN])
    Beyer_Speed = pd.DataFrame(Beyer, columns=['HNAME','PP_SPF_D'])

    Feature_DF = Feature_DF.merge(Beyer_Speed, how='left')
    Feature_DF.loc[:,'PP_SPF_D'].fillna(Feature_DF.loc[:,'PP_SPF_D'].min(), inplace = True)
    Feature_DF.loc[:,'PP_SPF_D'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_D']]

    return Feature_DF

"""
PP_SPF_AVG
"""

def PP_SPF_AVG(Dataframe, HNAME_List, Raceday):

    """
    Average Speed Figure
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_SPF_AVG]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, BEYER_SPEED PP_SPF_AVG from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    if len(Extraction) == 0:
        Feature_DF.loc[:,'PP_SPF_AVG'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_AVG']]
        return Feature_DF

    Extraction = Extraction.groupby('HNAME').mean().loc[:,'PP_SPF_AVG'].reset_index()
    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_SPF_AVG'].fillna(Feature_DF.loc[:,'PP_SPF_AVG'].min(), inplace = True)
    Feature_DF.loc[:,'PP_SPF_AVG'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_AVG']]

    return Feature_DF

"""
PP_SPF_AVGRW
"""

def PP_SPF_AVGRW(Dataframe, HNAME_List, Raceday):

    """
    Recency Weighted Average Speed Figure
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_SPF_AVGRW]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, BEYER_SPEED from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    Beyer = []
    for name, group in Extraction.groupby('HNAME'):
        Speed_Figure = group.loc[:,'BEYER_SPEED'].dropna().values
        if len(Speed_Figure) >1:
            model = SimpleExpSmoothing(Speed_Figure)
            model = model.fit()
            Beyer.append([name, model.forecast()[0]])
        elif len(Speed_Figure) == 1:
            Beyer.append([name,Speed_Figure[0]])
        else :
            Beyer.append([name,0])
    Beyer_Speed = pd.DataFrame(Beyer, columns=['HNAME','PP_SPF_AVGRW'])

    Feature_DF = Feature_DF.merge(Beyer_Speed, how='left')
    Feature_DF.loc[:,'PP_SPF_AVGRW'].fillna(Feature_DF.loc[:,'PP_SPF_AVGRW'].min(), inplace = True)
    Feature_DF.loc[:,'PP_SPF_AVGRW'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_AVGRW']]

    return Feature_DF

"""
PP_SPF_AVGT
"""

def PP_SPF_AVGT(Dataframe, HNAME_List, Raceday):

    """
    Average Speed Figure in last 3 races
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_SPF_AVGT]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, BEYER_SPEED from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    Beyer = []
    for name, group in Extraction.groupby('HNAME'):
        Speed_Figure = group.loc[:,'BEYER_SPEED'].dropna().values
        if len(Speed_Figure) >1:
            model = SimpleExpSmoothing(Speed_Figure)
            model = model.fit()
            Beyer.append([name, model.forecast()[0]])
        elif len(Speed_Figure) == 1:
            Beyer.append([name,Speed_Figure[0]])
        else :
            Beyer.append([name,0])
    Beyer_Speed = pd.DataFrame(Beyer, columns=['HNAME','PP_SPF_AVGT'])

    Feature_DF = Feature_DF.merge(Beyer_Speed, how='left')
    Feature_DF.loc[:,'PP_SPF_AVGT'].fillna(Feature_DF.loc[:,'PP_SPF_AVGT'].min(), inplace = True)
    Feature_DF.loc[:,'PP_SPF_AVGT'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_AVGT']]

    return Feature_DF

"""
PP_SPF_AVG_DIST
"""

def PP_SPF_AVG_DIST(Dataframe, HNAME_List, Raceday):

    """
    Average Speed Figure on Distance
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_SPF_AVG_DIST]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, BEYER_SPEED PP_SPF_AVG_DIST from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RADIS = {Distance}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Distance = Distance))
    if len(Extraction) == 0:
        Feature_DF.loc[:,'PP_SPF_AVG_DIST'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_AVG_DIST']]
        return Feature_DF

    Extraction = Extraction.groupby('HNAME').mean().loc[:,'PP_SPF_AVG_DIST'].reset_index()
    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_SPF_AVG_DIST'].fillna(Feature_DF.loc[:,'PP_SPF_AVG_DIST'].min(), inplace = True)
    Feature_DF.loc[:,'PP_SPF_AVG_DIST'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_AVG_DIST']]

    return Feature_DF


"""
PP_SPF_AVGRW_DIST
"""

def PP_SPF_AVGRW_DIST(Dataframe, HNAME_List, Raceday):

    """
    Recency Weighted Avg Speed Figure on Distance
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_SPF_AVGRW_DIST]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, BEYER_SPEED from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RADIS = {Distance}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Distance = Distance))
    Beyer = []
    for name, group in Extraction.groupby('HNAME'):
        Speed_Figure = group.loc[:,'BEYER_SPEED'].dropna().values
        if len(Speed_Figure) >1:
            model = SimpleExpSmoothing(Speed_Figure)
            model = model.fit()
            Beyer.append([name, model.forecast()[0]])
        elif len(Speed_Figure) == 1:
            Beyer.append([name,Speed_Figure[0]])
        else :
            Beyer.append([name,0])
    Beyer_Speed = pd.DataFrame(Beyer, columns=['HNAME','PP_SPF_AVGRW_DIST'])

    Feature_DF = Feature_DF.merge(Beyer_Speed, how='left')
    Feature_DF.loc[:,'PP_SPF_AVGRW_DIST'].fillna(Feature_DF.loc[:,'PP_SPF_AVGRW_DIST'].min(), inplace = True)
    Feature_DF.loc[:,'PP_SPF_AVGRW_DIST'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_AVGRW_DIST']]

    return Feature_DF

"""
PP_SPF_AVG_SIM_DIST
"""

def PP_SPF_AVG_SIM_DIST(Dataframe, HNAME_List, Raceday):

    """
    Average Speed Figure on Similar Distance
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_SPF_AVG_SIM_DIST]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Distance = Dataframe.loc[:,'RADIS'].values[0]
    Dist_Dict = Distance_Similarity(Distance)

    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, RADIS, BEYER_SPEED Beyer from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RADIS != {Distance}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Distance = Distance))
    Extraction.replace({'RADIS': Dist_Dict}, inplace = True)
    def Normalise(group):
        group.loc[:,'Norm'] = np.exp(group.loc[:,'RADIS']) / np.exp(group.loc[:,'RADIS']).sum()
        group.loc[:,'Normed'] = group.loc[:,'Norm'] * group.loc[:,'Beyer']
        return group.loc[:,'Normed'].sum()

    if len(Extraction) == 0:
        Feature_DF.loc[:,'PP_SPF_AVG_SIM_DIST'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_AVG_SIM_DIST']]
        return Feature_DF

    Extraction = Extraction.groupby('HNAME').apply(Normalise).reset_index()
    Extraction.columns = ['HNAME','PP_SPF_AVG_SIM_DIST']

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_SPF_AVG_SIM_DIST'].fillna(Feature_DF.loc[:,'PP_SPF_AVG_SIM_DIST'].min(), inplace = True)
    Feature_DF.loc[:,'PP_SPF_AVG_SIM_DIST'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_AVG_SIM_DIST']]

    return Feature_DF

"""
PP_SPF_AVGRW_SIM_DIST
"""

def PP_SPF_AVGRW_SIM_DIST(Dataframe, HNAME_List, Raceday):

    """
    Recency Weighted Average Speed Figure on Similar Distance
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_SPF_AVGRW_SIM_DIST]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Distance = Dataframe.loc[:,'RADIS'].values[0]
    Dist_Dict = Distance_Similarity(Distance)

    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, BEYER_SPEED, RADIS from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RADIS != {Distance}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Distance = Distance))
    Beyer = []
    for name_H, group_H in Extraction.groupby('HNAME'):
        for dist, group in group_H.groupby('RADIS'):
            Speed_Figure = group.loc[:,'BEYER_SPEED'].dropna().values
            if len(Speed_Figure) >1:
                model = SimpleExpSmoothing(Speed_Figure)
                model = model.fit()
                Beyer.append([name_H, dist, model.forecast()[0]])
            elif len(Speed_Figure) == 1:
                Beyer.append([name_H,dist,Speed_Figure[0]])
            else :
                Beyer.append([name_H,dist,0])

    Beyer_Speed = pd.DataFrame(Beyer, columns=['HNAME', 'RADIS', 'Beyer'])
    Beyer_Speed.replace({'RADIS': Dist_Dict}, inplace = True)

    def Normalise(group):
        group.loc[:,'Norm'] = np.exp(group.loc[:,'RADIS']) / np.exp(group.loc[:,'RADIS']).sum()
        group.loc[:,'Normed'] = group.loc[:,'Norm'] * group.loc[:,'Beyer']
        return group.loc[:,'Normed'].sum()

    if len(Beyer_Speed) == 0:
        Feature_DF.loc[:,'PP_SPF_AVGRW_SIM_DIST'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_AVGRW_SIM_DIST']]
        return Feature_DF

    Beyer_Speed = Beyer_Speed.groupby('HNAME').apply(Normalise).reset_index()
    Beyer_Speed.columns = ['HNAME','PP_SPF_AVGRW_SIM_DIST']

    Feature_DF = Feature_DF.merge(Beyer_Speed, how='left')
    Feature_DF.loc[:,'PP_SPF_AVGRW_SIM_DIST'].fillna(Feature_DF.loc[:,'PP_SPF_AVGRW_SIM_DIST'].min(), inplace = True)
    Feature_DF.loc[:,'PP_SPF_AVGRW_SIM_DIST'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_AVGRW_SIM_DIST']]

    return Feature_DF

"""
PP_SPF_AVG_GO
"""

def PP_SPF_AVG_GO(Dataframe, HNAME_List, Raceday):

    """
    Average Speed Figure on Going
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_SPF_AVG_GO]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Surface ="'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Going = Dataframe.loc[:,'RAGOG'].values[0].strip()
    Going_Dict = Going_Similarity(Going)

    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, RAGOG, BEYER_SPEED Beyer from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RATRA = {Surface}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Surface = Surface))
    Extraction.loc[:,'RAGOG'] = Extraction.loc[:,'RAGOG'].map(lambda x : x.strip())
    Extraction.replace({'RAGOG': Going_Dict}, inplace = True)
    def Normalise(group):
        group.loc[:,'Norm'] = np.exp(group.loc[:,'RAGOG']) / np.exp(group.loc[:,'RAGOG']).sum()
        group.loc[:,'Normed'] = group.loc[:,'Norm'] * group.loc[:,'Beyer']
        return group.loc[:,'Normed'].sum()

    if len(Extraction) == 0:
        Feature_DF.loc[:,'PP_SPF_AVG_GO'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_AVG_GO']]
        return Feature_DF

    Extraction = Extraction.groupby('HNAME').apply(Normalise).reset_index()
    Extraction.columns = ['HNAME','PP_SPF_AVG_GO']

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_SPF_AVG_GO'].fillna(Feature_DF.loc[:,'PP_SPF_AVG_GO'].min(), inplace = True)
    Feature_DF.loc[:,'PP_SPF_AVG_GO'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_AVG_GO']]

    return Feature_DF

"""
PP_SPF_AVGRW_GO
"""

def PP_SPF_AVGRW_GO(Dataframe, HNAME_List, Raceday):

    """
    Average Speed Figure on Similar Going
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_SPF_AVGRW_GO]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Surface ="'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Going = Dataframe.loc[:,'RAGOG'].values[0].strip()
    Going_Dict = Going_Similarity(Going)

    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, BEYER_SPEED, RAGOG from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RATRA = {Surface}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Surface = Surface))
    Beyer = []
    for name_H, group_H in Extraction.groupby('HNAME'):
        for name, group in group_H.groupby('RAGOG'):
            Speed_Figure = group.loc[:,'BEYER_SPEED'].dropna().values
            if len(Speed_Figure) >1:
                model = SimpleExpSmoothing(Speed_Figure)
                model = model.fit()
                Beyer.append([name_H, name, model.forecast()[0]])
            elif len(Speed_Figure) == 1:
                Beyer.append([name_H,name,Speed_Figure[0]])
            else :
                Beyer.append([name_H,name,0])

    Beyer_Speed = pd.DataFrame(Beyer, columns=['HNAME', 'RAGOG', 'Beyer'])
    Beyer_Speed.loc[:,'RAGOG'] = Extraction.loc[:,'RAGOG'].map(lambda x : x.strip())
    Beyer_Speed.replace({'RAGOG': Going_Dict}, inplace = True)

    def Normalise(group):
        group.loc[:,'Norm'] = np.exp(group.loc[:,'RAGOG']) / np.exp(group.loc[:,'RAGOG']).sum()
        group.loc[:,'Normed'] = group.loc[:,'Norm'] * group.loc[:,'Beyer']
        return group.loc[:,'Normed'].sum()

    if len(Beyer_Speed) == 0:
        Feature_DF.loc[:,'PP_SPF_AVGRW_GO'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_AVGRW_GO']]
        return Feature_DF

    Beyer_Speed = Beyer_Speed.groupby('HNAME').apply(Normalise).reset_index()
    Beyer_Speed.columns = ['HNAME','PP_SPF_AVGRW_GO']

    Feature_DF = Feature_DF.merge(Beyer_Speed, how='left')
    Feature_DF.loc[:,'PP_SPF_AVGRW_GO'].fillna(Feature_DF.loc[:,'PP_SPF_AVGRW_GO'].min(), inplace = True)
    Feature_DF.loc[:,'PP_SPF_AVGRW_GO'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_AVGRW_GO']]

    return Feature_DF

"""
PP_SPF_AVG_SUR
"""

def PP_SPF_AVG_SUR(Dataframe, HNAME_List, Raceday):

    """
    Average Speed Figure on Surface
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_SPF_AVG_SUR]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Surface ="'" + Dataframe.loc[:,'RATRA'].values[0] + "'"

    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, BEYER_SPEED PP_SPF_AVG_SUR from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RATRA = {Surface}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Surface = Surface))
    if len(Extraction) == 0:
        Feature_DF.loc[:,'PP_SPF_AVG_SUR'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_AVG_SUR']]
        return Feature_DF

    Extraction = Extraction.groupby('HNAME').mean().loc[:,'PP_SPF_AVG_SUR'].reset_index()
    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_SPF_AVG_SUR'].fillna(Feature_DF.loc[:,'PP_SPF_AVG_SUR'].min(), inplace = True)
    Feature_DF.loc[:,'PP_SPF_AVG_SUR'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_AVG_SUR']]

    return Feature_DF

"""
PP_SPF_AVGRW_SUR
"""

def PP_SPF_AVGRW_SUR(Dataframe, HNAME_List, Raceday):

    """
    Recency Weighted Avg Speed Figure on Surface
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_SPF_AVGRW_SUR]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Surface ="'" + Dataframe.loc[:,'RATRA'].values[0] + "'"

    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, BEYER_SPEED from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RATRA = {Surface}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Surface = Surface))
    Beyer = []
    for name, group in Extraction.groupby('HNAME'):
        Speed_Figure = group.loc[:,'BEYER_SPEED'].dropna().values
        if len(Speed_Figure) >1:
            model = SimpleExpSmoothing(Speed_Figure)
            model = model.fit()
            Beyer.append([name, model.forecast()[0]])
        elif len(Speed_Figure) == 1:
            Beyer.append([name,Speed_Figure[0]])
        else :
            Beyer.append([name,0])
    Beyer_Speed = pd.DataFrame(Beyer, columns=['HNAME','PP_SPF_AVGRW_SUR'])

    Feature_DF = Feature_DF.merge(Beyer_Speed, how='left')
    Feature_DF.loc[:,'PP_SPF_AVGRW_SUR'].fillna(Feature_DF.loc[:,'PP_SPF_AVGRW_SUR'].min(), inplace = True)
    Feature_DF.loc[:,'PP_SPF_AVGRW_SUR'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_AVGRW_SUR']]

    return Feature_DF

"""
PP_SPF_AVG_PFL
"""

def PP_SPF_AVG_PFL(Dataframe, HNAME_List, Raceday):

    """
    Average Speed Figure on Profile
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_SPF_AVG_PFL]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Location ="'" + Dataframe.loc[:,'RALOC'].values[0] + "'"
    Surface ="'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, BEYER_SPEED PP_SPF_AVG_PFL from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     and RALOC = {Location} and RATRA = {Surface} and RADIS = {Distance}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List,
                                     Location = Location, Surface = Surface, Distance = Distance))
    if len(Extraction) == 0:
        Feature_DF.loc[:,'PP_SPF_AVG_PFL'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_AVG_PFL']]
        return Feature_DF

    Extraction = Extraction.groupby('HNAME').mean().loc[:,'PP_SPF_AVG_PFL'].reset_index()
    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_SPF_AVG_PFL'].fillna(Feature_DF.loc[:,'PP_SPF_AVG_PFL'].min(), inplace = True)
    Feature_DF.loc[:,'PP_SPF_AVG_PFL'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_AVG_PFL']]

    return Feature_DF

"""
PP_SPF_AVGRW_PFL
"""

def PP_SPF_AVGRW_PFL(Dataframe, HNAME_List, Raceday):

    """
    Recency Weighted Avg Speed Figure on Profile
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_SPF_AVGRW_PFL]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Location ="'" + Dataframe.loc[:,'RALOC'].values[0] + "'"
    Surface ="'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, BEYER_SPEED from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     and RALOC = {Location} and RATRA = {Surface} and RADIS = {Distance}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List,
                                     Location = Location, Surface = Surface, Distance = Distance))
    Beyer = []
    for name, group in Extraction.groupby('HNAME'):
        Speed_Figure = group.loc[:,'BEYER_SPEED'].dropna().values
        if len(Speed_Figure) >1:
            model = SimpleExpSmoothing(Speed_Figure)
            model = model.fit()
            Beyer.append([name, model.forecast()[0]])
        elif len(Speed_Figure) == 1:
            Beyer.append([name,Speed_Figure[0]])
        else :
            Beyer.append([name,0])
    Beyer_Speed = pd.DataFrame(Beyer, columns=['HNAME','PP_SPF_AVGRW_PFL'])

    Feature_DF = Feature_DF.merge(Beyer_Speed, how='left')
    Feature_DF.loc[:,'PP_SPF_AVGRW_PFL'].fillna(Feature_DF.loc[:,'PP_SPF_AVGRW_PFL'].min(), inplace = True)
    Feature_DF.loc[:,'PP_SPF_AVGRW_PFL'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_AVGRW_PFL']]

    return Feature_DF

"""
PP_SPF_TOP
"""

def PP_SPF_TOP(Dataframe, HNAME_List, Raceday):

    """
    Best Speed Figure
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_SPF_TOP]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, max(BEYER_SPEED) PP_SPF_TOP from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     Group by HNAME
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_SPF_TOP'].fillna(Feature_DF.loc[:,'PP_SPF_TOP'].min(), inplace = True)
    Feature_DF.loc[:,'PP_SPF_TOP'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_TOP']]

    return Feature_DF

"""
PP_SPF_TOPY
"""

def PP_SPF_TOPY(Dataframe, HNAME_List, Raceday):

    """
    Best Speed Figure in one year
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_SPF_TOPY]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Offset_Raceday = (pd.to_datetime(Raceday) + pd.tseries.offsets.DateOffset(months=-12)).strftime("%Y%m%d")

    Extraction = Extraction_Database("""
                                     Select HNAME, max(BEYER_SPEED) PP_SPF_TOPY from Race_PosteriorDb
                                     where RADAT < {Raceday} and RADAT > {Offset_Raceday} and HNAME in {HNAME_List}
                                     Group by HNAME
                                     """.format(Raceday = Raceday, Offset_Raceday = Offset_Raceday, HNAME_List = HNAME_List))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_SPF_TOPY'].fillna(0, inplace = True)
    Feature_DF.loc[:,'PP_SPF_TOPY'].fillna(Feature_DF.loc[:,'PP_SPF_TOPY'].min(), inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_TOPY']]

    return Feature_DF

"""
PP_SPF_TOP_DIST
"""

def PP_SPF_TOP_DIST(Dataframe, HNAME_List, Raceday):

    """
    Top Speed Figure on Distance
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_SPF_TOP_DIST]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Extraction = Extraction_Database("""
                                     Select HNAME, max(BEYER_SPEED) PP_SPF_TOP_DIST from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RADIS = {Distance}
                                     Group by HNAME
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Distance = Distance))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_SPF_TOP_DIST'].fillna(Feature_DF.loc[:,'PP_SPF_TOP_DIST'].min(), inplace = True)
    Feature_DF.loc[:,'PP_SPF_TOP_DIST'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_TOP_DIST']]

    return Feature_DF

"""
PP_SPF_TOP_SIM_DIST
"""

def PP_SPF_TOP_SIM_DIST(Dataframe, HNAME_List, Raceday):

    """
    Top Speed Figure on Similar Distance
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_SPF_TOP_SIM_DIST]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Distance = Dataframe.loc[:,'RADIS'].values[0]
    Dist_Dict = Distance_Similarity(Distance)

    Extraction = Extraction_Database("""
                                     Select HNAME, max(BEYER_SPEED) Beyer, RADIS from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RADIS != {Distance}
                                     Group by HNAME, RADIS
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Distance = Distance))

    Extraction.replace({'RADIS': Dist_Dict}, inplace = True)

    def Normalise(group):
        group.loc[:,'Norm'] = np.exp(group.loc[:,'RADIS']) / np.exp(group.loc[:,'RADIS']).sum()
        group.loc[:,'Normed'] = group.loc[:,'Norm'] * group.loc[:,'Beyer']
        return group.loc[:,'Normed'].sum()

    if len(Extraction) == 0:
        Feature_DF.loc[:,'PP_SPF_TOP_SIM_DIST'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_TOP_SIM_DIST']]
        return Feature_DF

    Extraction = Extraction.groupby('HNAME').apply(Normalise).reset_index()
    Extraction.columns = ['HNAME','PP_SPF_TOP_SIM_DIST']
    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_SPF_TOP_SIM_DIST'].fillna(Feature_DF.loc[:,'PP_SPF_TOP_SIM_DIST'].min(), inplace = True)
    Feature_DF.loc[:,'PP_SPF_TOP_SIM_DIST'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_TOP_SIM_DIST']]

    return Feature_DF

"""
PP_SPF_TOP_GO
"""

def PP_SPF_TOP_GO(Dataframe, HNAME_List, Raceday):

    """
    Top Speed Figure on Similar Going
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_SPF_TOP_GO]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Surface ="'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Going = Dataframe.loc[:,'RAGOG'].values[0].strip()
    Going_Dict = Going_Similarity(Going)

    Extraction = Extraction_Database("""
                                     Select HNAME, max(BEYER_SPEED) Beyer, RAGOG from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RATRA = {Surface}
                                     Group by HNAME, RAGOG
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Surface = Surface))

    Extraction.loc[:,'RAGOG'] = Extraction.loc[:,'RAGOG'].map(lambda x : x.strip())
    Extraction.replace({'RAGOG': Going_Dict}, inplace = True)

    def Normalise(group):
        group.loc[:,'Norm'] = np.exp(group.loc[:,'RAGOG']) / np.exp(group.loc[:,'RAGOG']).sum()
        group.loc[:,'Normed'] = group.loc[:,'Norm'] * group.loc[:,'Beyer']
        return group.loc[:,'Normed'].sum()

    if len(Extraction) == 0:
        Feature_DF.loc[:,'PP_SPF_TOP_GO'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_TOP_GO']]
        return Feature_DF

    Extraction = Extraction.groupby('HNAME').apply(Normalise).reset_index()
    Extraction.columns = ['HNAME','PP_SPF_TOP_GO']
    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_SPF_TOP_GO'].fillna(Feature_DF.loc[:,'PP_SPF_TOP_GO'].min(), inplace = True)
    Feature_DF.loc[:,'PP_SPF_TOP_GO'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_TOP_GO']]

    return Feature_DF

"""
PP_SPF_TOP_SUR
"""

def PP_SPF_TOP_SUR(Dataframe, HNAME_List, Raceday):

    """
    Top Speed Figure on Surface
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_SPF_TOP_SUR]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Surface ="'" + Dataframe.loc[:,'RATRA'].values[0] + "'"

    Extraction = Extraction_Database("""
                                     Select HNAME, max(BEYER_SPEED) PP_SPF_TOP_SUR from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RATRA = {Surface}
                                     Group by HNAME
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Surface = Surface))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_SPF_TOP_SUR'].fillna(Feature_DF.loc[:,'PP_SPF_TOP_SUR'].min(), inplace = True)
    Feature_DF.loc[:,'PP_SPF_TOP_SUR'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_TOP_SUR']]

    return Feature_DF

"""
PP_SPF_TOP_PFL
"""

def PP_SPF_TOP_PFL(Dataframe, HNAME_List, Raceday):

    """
    Top Speed Figure on Profile
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_SPF_TOP_PFL]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Location ="'" + Dataframe.loc[:,'RALOC'].values[0] + "'"
    Surface ="'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Extraction = Extraction_Database("""
                                     Select HNAME, max(BEYER_SPEED) PP_SPF_TOP_PFL from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     and RALOC = {Location} and RATRA = {Surface} and RADIS = {Distance}
                                     Group by HNAME
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List,
                                     Location = Location, Surface = Surface, Distance = Distance))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_SPF_TOP_PFL'].fillna(Feature_DF.loc[:,'PP_SPF_TOP_PFL'].min(), inplace = True)
    Feature_DF.loc[:,'PP_SPF_TOP_PFL'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_TOP_PFL']]

    return Feature_DF

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

=========================== Pace Figure History ===========================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
PP_PAF_SPT
"""

def PP_PAF_SPT(Dataframe, HNAME_List, Raceday):

    """
    Speed Point
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_PAF_SPT]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    SP_List = []
    if Distance in [1000, 1200, 1400, 1600, 1650]:
        Dist_List = '(1000, 1200, 1400, 1600, 1650, 1800)'
    elif Distance in [1800, 2000, 2200, 2400]:
        Dist_List = '(1800, 2000, 2200, 2400)'

    for Horse in Feature_DF.loc[:,'HNAME']:
        Speed_Point = 0
        Extraction = Extraction_Database("""
                                         Select HNAME, RARID, RESP1, RESS1 from RaceDb
                                         where RARID in (
                                         Select Distinct RARID from RaceDb
                                         where RADAT < {Raceday} and HNAME = {Horse} and RADIS in {Dist_List}
                                         Order by RARID DESC
                                         LIMIT 3)
                                         """.format(Raceday = Raceday, Horse = "'"+Horse+"'", Dist_List=Dist_List))
        for RARID, race in Extraction.groupby('RARID'):
            Leader_time = race.loc[:,'RESS1'].min()

            if race.loc[race.loc[:,'HNAME']==Horse, 'RESS1'].values - Leader_time < 0.2:
                Speed_Point += 1
            if race.loc[race.loc[:,'HNAME']==Horse, 'RESP1'].values <= 3:
                Speed_Point += 1
        SP_List.append([Horse, Speed_Point])

    SP_List = pd.DataFrame(SP_List, columns=['HNAME','PP_PAF_SPT'])
    Feature_DF = Feature_DF.merge(SP_List, how='left')

    Feature_DF.loc[:,'PP_PAF_SPT'].fillna(Feature_DF.loc[:,'PP_PAF_SPT'].min(), inplace = True)
    Feature_DF.loc[:,'PP_PAF_SPT'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_SPT']]

    return Feature_DF

"""
PP_PAF_SP_DIST
"""

def PP_PAF_SPT_DIST(Dataframe, HNAME_List, Raceday):

    """
    Speed Point relative to Distance
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_PAF_SPT_DIST]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    SP_List = []
    if Distance in [1000, 1200, 1400, 1600, 1650]:
        Dist_List = '(1000, 1200, 1400, 1600, 1650, 1800)'
    elif Distance in [1800, 2000, 2200, 2400]:
        Dist_List = '(1800, 2000, 2200, 2400)'

    for Horse in Feature_DF.loc[:,'HNAME']:
        Speed_Point = 0
        Extraction = Extraction_Database("""
                                         Select HNAME, RARID, RESP1, RESS1 from RaceDb
                                         where RARID in (
                                         Select Distinct RARID from RaceDb
                                         where RADAT < {Raceday} and HNAME = {Horse} and RADIS in {Dist_List}
                                         Order by RARID DESC
                                         LIMIT 3)
                                         """.format(Raceday = Raceday, Horse = "'"+Horse+"'", Dist_List=Dist_List))
        for RARID, race in Extraction.groupby('RARID'):
            Leader_time = race.loc[:,'RESS1'].min()

            if race.loc[race.loc[:,'HNAME']==Horse, 'RESS1'].values - Leader_time < 0.2:
                Speed_Point += 1
            if race.loc[race.loc[:,'HNAME']==Horse, 'RESP1'].values <= 3:
                Speed_Point += 1
        SP_List.append([Horse, Speed_Point])

    SP_List = pd.DataFrame(SP_List, columns=['HNAME','SP'])
    Feature_DF = Feature_DF.merge(SP_List, how='left')
    Feature_DF.loc[:,'PP_PAF_SPT_DIST'] = Feature_DF.loc[:,'SP'] / Distance

    Feature_DF.loc[:,'PP_PAF_SPT_DIST'].fillna(Feature_DF.loc[:,'PP_PAF_SPT_DIST'].min(), inplace = True)
    Feature_DF.loc[:,'PP_PAF_SPT_DIST'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_SPT_DIST']]

    return Feature_DF

"""
PP_PAF_EP_AVG
"""

def PP_PAF_EP_AVG(Dataframe, HNAME_List, Raceday):

    """
    Average Speed Figure
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_PAF_EP_AVG]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, EARLY_PACE PP_PAF_EP_AVG from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    if len(Extraction) == 0:
        Feature_DF.loc[:,'PP_PAF_EP_AVG'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_EP_AVG']]
        return Feature_DF

    Extraction = Extraction.groupby('HNAME').mean().loc[:,'PP_PAF_EP_AVG'].reset_index()
    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_PAF_EP_AVG'].fillna(Feature_DF.loc[:,'PP_PAF_EP_AVG'].min(), inplace = True)
    Feature_DF.loc[:,'PP_PAF_EP_AVG'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_EP_AVG']]

    return Feature_DF

"""
PP_PAF_EP_AVGRW
"""

def PP_PAF_EP_AVGRW(Dataframe, HNAME_List, Raceday):

    """
    Recency Weighted Avg Early Pace
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_PAF_EP_AVGRW]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, EARLY_PACE from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    EarlyPace = []
    for name, group in Extraction.groupby('HNAME'):
        EP_Figure = group.loc[:,'EARLY_PACE'].dropna().values
        if len(EP_Figure) >1:
            model = SimpleExpSmoothing(EP_Figure)
            model = model.fit()
            EarlyPace.append([name, model.forecast()[0]])
        elif len(EP_Figure) == 1:
            EarlyPace.append([name,EP_Figure[0]])
        else :
            EarlyPace.append([name,0])
    EarlyPace = pd.DataFrame(EarlyPace, columns=['HNAME','PP_PAF_EP_AVGRW'])

    Feature_DF = Feature_DF.merge(EarlyPace, how='left')
    Feature_DF.loc[:,'PP_PAF_EP_AVGRW'].fillna(Feature_DF.loc[:,'PP_PAF_EP_AVGRW'].min(), inplace = True)
    Feature_DF.loc[:,'PP_PAF_EP_AVGRW'].fillna(0, inplace=True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_EP_AVGRW']]

    return Feature_DF

"""
PP_PAF_EP_ADV_GOPFL
"""

def PP_PAF_EP_ADV_GOPFL(Dataframe, HNAME_List, Raceday):

    """
    Early Pace Advantage on Going and Profile
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_PAF_EP_ADV_GOPFL]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Location ="'" + Dataframe.loc[:,'RALOC'].values[0] + "'"
    Surface ="'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Distance = Dataframe.loc[:,'RADIS'].values[0]
    Going = "'" + Dataframe.loc[:,'RAGOG'].values[0] + "'"

    Extraction_History = Extraction_Database("""
                                             Select HNAME, RARID, EARLY_PACE, RESFP PP_PAF_EP_ADV_GOPFL from Race_PosteriorDb
                                             where RADAT < {Raceday} and RADIS = {Distance} and RATRA = {Surface}
                                             and RALOC = {Location} and RAGOG = {Going}
                                             """.format(Raceday = Raceday, HNAME_List = HNAME_List, Distance=Distance,
                                             Going=Going, Surface=Surface, Location=Location))
    if len(Extraction_History) == 0:
        Feature_DF.loc[:,'PP_PAF_EP_ADV_GOPFL'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_EP_ADV_GOPFL']]
        return Feature_DF

    def EP_Rank(race):
        race.loc[:,'rank'] = race.loc[:,'EARLY_PACE'].rank(method = 'max', ascending=False)
        return race
    Extraction_History = Extraction_History.groupby('RARID').apply(EP_Rank)
    History_tomap = Extraction_History.groupby('rank').mean()['PP_PAF_EP_ADV_GOPFL'].reset_index(drop=False)

    #Average Early Pace
    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, EARLY_PACE from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    EarlyPace = []
    for name, group in Extraction.groupby('HNAME'):
        EP_Figure = group.loc[:,'EARLY_PACE'].dropna().values
        if len(EP_Figure) >1:
            model = SimpleExpSmoothing(EP_Figure)
            model = model.fit()
            EarlyPace.append([name, model.forecast()[0]])
        elif len(EP_Figure) == 1:
            EarlyPace.append([name,EP_Figure[0]])
        else :
            EarlyPace.append([name,0])
    EarlyPace = pd.DataFrame(EarlyPace, columns=['HNAME','EP'])

    # EarlyPace = Extraction_Database("""
    #                                  Select HNAME, RARID, EARLY_PACE EP from Race_PosteriorDb
    #                                  where RADAT < {Raceday} and HNAME in {HNAME_List}
    #                                  """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    # if len(EarlyPace) == 0:
    #     Feature_DF.loc[:,'PP_PAF_EP_ADV_GOPFL'] = 0
    #     Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_EP_ADV_GOPFL']]
    #     return Feature_DF
    # EarlyPace = EarlyPace.groupby('HNAME').mean().loc[:,'EP'].reset_index()

    Feature_DF = Feature_DF.merge(EarlyPace, how='left')

    Feature_DF.loc[:,'rank'] = Feature_DF.loc[:,'EP'].rank(method = 'max', ascending=False)

    Feature_DF = Feature_DF.merge(History_tomap, how='left')
    Feature_DF.loc[:,'PP_PAF_EP_ADV_GOPFL'].fillna(Feature_DF.loc[:,'PP_PAF_EP_ADV_GOPFL'].min(), inplace = True)
    Feature_DF.loc[:,'PP_PAF_EP_ADV_GOPFL'].fillna(0, inplace=True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_EP_ADV_GOPFL']]

    return Feature_DF

"""
PP_PAF_EP_ADV_PFL
"""

def PP_PAF_EP_ADV_PFL(Dataframe, HNAME_List, Raceday):

    """
    Early Pace Advantage on Profile
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_PAF_EP_ADV_PFL]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Location ="'" + Dataframe.loc[:,'RALOC'].values[0] + "'"
    Surface ="'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Extraction_History = Extraction_Database("""
                                             Select HNAME, RARID, EARLY_PACE, RESFP PP_PAF_EP_ADV_PFL from Race_PosteriorDb
                                             where RADAT < {Raceday} and RADIS = {Distance} and RATRA = {Surface}
                                             and RALOC = {Location}
                                             """.format(Raceday = Raceday, HNAME_List = HNAME_List, Distance=Distance,
                                             Surface=Surface, Location=Location))
    if len(Extraction_History) == 0:
        Feature_DF.loc[:,'PP_PAF_EP_ADV_PFL'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_EP_ADV_PFL']]
        return Feature_DF

    def EP_Rank(race):
        race.loc[:,'rank'] = race.loc[:,'EARLY_PACE'].rank(method = 'max', ascending=False)
        return race
    Extraction_History = Extraction_History.groupby('RARID').apply(EP_Rank)
    History_tomap = Extraction_History.groupby('rank').mean()['PP_PAF_EP_ADV_PFL'].reset_index(drop=False)

    #Average Early Pace
    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, EARLY_PACE from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    EarlyPace = []
    for name, group in Extraction.groupby('HNAME'):
        EP_Figure = group.loc[:,'EARLY_PACE'].dropna().values
        if len(EP_Figure) >1:
            model = SimpleExpSmoothing(EP_Figure)
            model = model.fit()
            EarlyPace.append([name, model.forecast()[0]])
        elif len(EP_Figure) == 1:
            EarlyPace.append([name,EP_Figure[0]])
        else :
            EarlyPace.append([name,0])
    EarlyPace = pd.DataFrame(EarlyPace, columns=['HNAME','EP'])

    # EarlyPace = Extraction_Database("""
    #                                  Select HNAME, RARID, EARLY_PACE EP from Race_PosteriorDb
    #                                  where RADAT < {Raceday} and HNAME in {HNAME_List}
    #                                  """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    # if len(EarlyPace) == 0:
    #     Feature_DF.loc[:,'PP_PAF_EP_ADV_GOPFL'] = 0
    #     Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_EP_ADV_GOPFL']]
    #     return Feature_DF
    # EarlyPace = EarlyPace.groupby('HNAME').mean().loc[:,'EP'].reset_index()
    Feature_DF = Feature_DF.merge(EarlyPace, how='left')

    Feature_DF.loc[:,'rank'] = Feature_DF.loc[:,'EP'].rank(method = 'max', ascending=False)

    Feature_DF = Feature_DF.merge(History_tomap, how='left')
    Feature_DF.loc[:,'PP_PAF_EP_ADV_PFL'].fillna(Feature_DF.loc[:,'PP_PAF_EP_ADV_PFL'].min(), inplace = True)
    Feature_DF.loc[:,'PP_PAF_EP_ADV_PFL'].fillna(0, inplace=True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_EP_ADV_PFL']]

    return Feature_DF

"""
PP_PAF_EP_WIN_PFL
"""

def PP_PAF_EP_WIN_PFL(Dataframe, HNAME_List, Raceday):

    """
    Distance from winning Early Pace
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_PAF_EP_WIN_PFL]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Location ="'" + Dataframe.loc[:,'RALOC'].values[0] + "'"
    Surface ="'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Win_EP = Extraction_Database("""
                                 Select avg(EARLY_PACE) from Race_PosteriorDb
                                 where RADAT < {Raceday} and RADIS = {Distance} and RATRA = {Surface}
                                 and RALOC = {Location} and RESFP = 1
                                 """.format(Raceday = Raceday, HNAME_List = HNAME_List, Distance=Distance,
                                 Surface=Surface, Location=Location)).values.tolist()[0][0]
    #No Winning EP
    if Win_EP == None:
        Win_EP = 1

    #Average Early Pace
    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, EARLY_PACE from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    EarlyPace = []
    for name, group in Extraction.groupby('HNAME'):
        EP_Figure = group.loc[:,'EARLY_PACE'].dropna().values
        if len(EP_Figure) >1:
            model = SimpleExpSmoothing(EP_Figure)
            model = model.fit()
            EarlyPace.append([name, model.forecast()[0]])
        elif len(EP_Figure) == 1:
            EarlyPace.append([name,EP_Figure[0]])
        else :
            EarlyPace.append([name,0])
    EarlyPace = pd.DataFrame(EarlyPace, columns=['HNAME','EP'])

    # EarlyPace = Extraction_Database("""
    #                                  Select HNAME, RARID, EARLY_PACE EP from Race_PosteriorDb
    #                                  where RADAT < {Raceday} and HNAME in {HNAME_List}
    #                                  """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    # if len(EarlyPace) == 0:
    #     Feature_DF.loc[:,'PP_PAF_EP_ADV_GOPFL'] = 0
    #     Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_EP_ADV_GOPFL']]
    #     return Feature_DF
    # EarlyPace = EarlyPace.groupby('HNAME').mean().loc[:,'EP'].reset_index()

    Feature_DF = Feature_DF.merge(EarlyPace, how='left')
    Feature_DF.loc[:,'PP_PAF_EP_WIN_PFL'] = ((Feature_DF.loc[:,'EP'] - Win_EP)/Win_EP).abs()

    Feature_DF.loc[:,'PP_PAF_EP_WIN_PFL'].fillna(Feature_DF.loc[:,'PP_PAF_EP_WIN_PFL'].min(), inplace = True)
    Feature_DF.loc[:,'PP_PAF_EP_WIN_PFL'].fillna(0, inplace=True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_EP_WIN_PFL']]

    return Feature_DF

"""
PP_PAF_EP_DIST
"""

def PP_PAF_EP_DIST(Dataframe, HNAME_List, Raceday):

    """
    Early Pace relative to Distance
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_PAF_EP_DIST]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    #Average Early Pace
    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, EARLY_PACE from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    EarlyPace = []
    for name, group in Extraction.groupby('HNAME'):
        EP_Figure = group.loc[:,'EARLY_PACE'].dropna().values
        if len(EP_Figure) >1:
            model = SimpleExpSmoothing(EP_Figure)
            model = model.fit()
            EarlyPace.append([name, model.forecast()[0]])
        elif len(EP_Figure) == 1:
            EarlyPace.append([name,EP_Figure[0]])
        else :
            EarlyPace.append([name,0])
    EarlyPace = pd.DataFrame(EarlyPace, columns=['HNAME','EP'])

    # EarlyPace = Extraction_Database("""
    #                                  Select HNAME, RARID, EARLY_PACE EP from Race_PosteriorDb
    #                                  where RADAT < {Raceday} and HNAME in {HNAME_List}
    #                                  """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    # if len(EarlyPace) == 0:
    #     Feature_DF.loc[:,'PP_PAF_EP_ADV_GOPFL'] = 0
    #     Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_EP_ADV_GOPFL']]
    #     return Feature_DF
    # EarlyPace = EarlyPace.groupby('HNAME').mean().loc[:,'EP'].reset_index()
    Feature_DF = Feature_DF.merge(EarlyPace, how='left')

    Feature_DF.loc[:,'PP_PAF_EP_DIST'] = Feature_DF.loc[:,'EP'] / Distance

    Feature_DF.loc[:,'PP_PAF_EP_DIST'].fillna(Feature_DF.loc[:,'PP_PAF_EP_DIST'].min(), inplace = True)
    Feature_DF.loc[:,'PP_PAF_EP_DIST'].fillna(0, inplace=True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_EP_DIST']]

    return Feature_DF

"""
PP_PAF_SP_AVG
"""

def PP_PAF_SP_AVG(Dataframe, HNAME_List, Raceday):

    """
    Average Sustained Pace
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_PAF_SP_AVG]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, SUSTAINED_PACE PP_PAF_SP_AVG from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    if len(Extraction) == 0:
        Feature_DF.loc[:,'PP_PAF_SP_AVG'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_SP_AVG']]
        return Feature_DF

    Extraction = Extraction.groupby('HNAME').mean().loc[:,'PP_PAF_SP_AVG'].reset_index()
    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_PAF_SP_AVG'].fillna(Feature_DF.loc[:,'PP_PAF_SP_AVG'].min(), inplace = True)
    Feature_DF.loc[:,'PP_PAF_SP_AVG'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_SP_AVG']]

    return Feature_DF

"""
PP_PAF_SP_AVGRW
"""

def PP_PAF_SP_AVGRW(Dataframe, HNAME_List, Raceday):

    """
    Average Sustained Pace
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_PAF_SP_AVGRW]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, SUSTAINED_PACE from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    SustainedPace = []
    for name, group in Extraction.groupby('HNAME'):
        SP_Figure = group.loc[:,'SUSTAINED_PACE'].dropna().values
        if len(SP_Figure) >1:
            model = SimpleExpSmoothing(SP_Figure)
            model = model.fit()
            SustainedPace.append([name, model.forecast()[0]])
        elif len(SP_Figure) == 1:
            SustainedPace.append([name,SP_Figure[0]])
        else :
            SustainedPace.append([name,0])

    SustainedPace = pd.DataFrame(SustainedPace, columns=['HNAME','PP_PAF_SP_AVGRW'])

    Feature_DF = Feature_DF.merge(SustainedPace, how='left')
    Feature_DF.loc[:,'PP_PAF_SP_AVGRW'].fillna(Feature_DF.loc[:,'PP_PAF_SP_AVGRW'].min(), inplace = True)
    Feature_DF.loc[:,'PP_PAF_SP_AVGRW'].fillna(0, inplace=True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_SP_AVGRW']]

    return Feature_DF

"""
PP_PAF_SP_ADV_GOPFL
"""

def PP_PAF_SP_ADV_GOPFL(Dataframe, HNAME_List, Raceday):

    """
    Sustained Pace Advantage on Going and Profile
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_PAF_SP_ADV_GOPFL]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Location ="'" + Dataframe.loc[:,'RALOC'].values[0] + "'"
    Surface ="'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Distance = Dataframe.loc[:,'RADIS'].values[0]
    Going = "'" + Dataframe.loc[:,'RAGOG'].values[0] + "'"

    Extraction_History = Extraction_Database("""
                                             Select HNAME, RARID, SUSTAINED_PACE, RESFP PP_PAF_SP_ADV_GOPFL from Race_PosteriorDb
                                             where RADAT < {Raceday} and RADIS = {Distance} and RATRA = {Surface}
                                             and RALOC = {Location} and RAGOG = {Going}
                                             """.format(Raceday = Raceday, HNAME_List = HNAME_List, Distance=Distance,
                                             Going=Going, Surface=Surface, Location=Location))
    if len(Extraction_History) == 0:
        Feature_DF.loc[:,'PP_PAF_SP_ADV_GOPFL'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_SP_ADV_GOPFL']]
        return Feature_DF

    def EP_Rank(race):
        race.loc[:,'rank'] = race.loc[:,'SUSTAINED_PACE'].rank(method = 'max', ascending=False)
        return race
    Extraction_History = Extraction_History.groupby('RARID').apply(EP_Rank)
    History_tomap = Extraction_History.groupby('rank').mean()['PP_PAF_SP_ADV_GOPFL'].reset_index(drop=False)

    #Average Sustained Pace
    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, SUSTAINED_PACE from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    SustainedPace = []
    for name, group in Extraction.groupby('HNAME'):
        SP_Figure = group.loc[:,'SUSTAINED_PACE'].dropna().values
        if len(SP_Figure) >1:
            model = SimpleExpSmoothing(SP_Figure)
            model = model.fit()
            SustainedPace.append([name, model.forecast()[0]])
        elif len(SP_Figure) == 1:
            SustainedPace.append([name,SP_Figure[0]])
        else :
            SustainedPace.append([name,0])
    SustainedPace = pd.DataFrame(SustainedPace, columns=['HNAME','SP'])
    # SustainedPace = Extraction_Database("""
    #                                     Select HNAME, RARID, SUSTAINED_PACE SP from Race_PosteriorDb
    #                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
    #                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    # if len(SustainedPace) == 0:
    #     Feature_DF.loc[:,'PP_PAF_SP_ADV_GOPFL'] = 0
    #     Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_SP_ADV_GOPFL']]
    #     return Feature_DF
    # SustainedPace = SustainedPace.groupby('HNAME').mean().loc[:,'SP'].reset_index()

    Feature_DF = Feature_DF.merge(SustainedPace, how='left')

    Feature_DF.loc[:,'rank'] = Feature_DF.loc[:,'SP'].rank(method = 'max', ascending=False)

    Feature_DF = Feature_DF.merge(History_tomap, how='left')
    Feature_DF.loc[:,'PP_PAF_SP_ADV_GOPFL'].fillna(Feature_DF.loc[:,'PP_PAF_SP_ADV_GOPFL'].min(), inplace = True)
    Feature_DF.loc[:,'PP_PAF_SP_ADV_GOPFL'].fillna(0, inplace=True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_SP_ADV_GOPFL']]

    return Feature_DF

"""
PP_PAF_SP_ADV_PFL
"""

def PP_PAF_SP_ADV_PFL(Dataframe, HNAME_List, Raceday):

    """
    Sustained Pace Advantage on Profile
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_PAF_SP_ADV_PFL]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Location ="'" + Dataframe.loc[:,'RALOC'].values[0] + "'"
    Surface ="'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Extraction_History = Extraction_Database("""
                                             Select HNAME, RARID, SUSTAINED_PACE, RESFP PP_PAF_SP_ADV_PFL from Race_PosteriorDb
                                             where RADAT < {Raceday} and RADIS = {Distance} and RATRA = {Surface}
                                             and RALOC = {Location}
                                             """.format(Raceday = Raceday, HNAME_List = HNAME_List, Distance=Distance,
                                             Surface=Surface, Location=Location))

    if len(Extraction_History) == 0:
        Feature_DF.loc[:,'PP_PAF_SP_ADV_PFL'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_SP_ADV_PFL']]
        return Feature_DF

    def EP_Rank(race):
        race.loc[:,'rank'] = race.loc[:,'SUSTAINED_PACE'].rank(method = 'max', ascending=False)
        return race
    Extraction_History = Extraction_History.groupby('RARID').apply(EP_Rank)
    History_tomap = Extraction_History.groupby('rank').mean()['PP_PAF_SP_ADV_PFL'].reset_index(drop=False)

    #Average Sustained Pace
    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, SUSTAINED_PACE from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    SustainedPace = []
    for name, group in Extraction.groupby('HNAME'):
        SP_Figure = group.loc[:,'SUSTAINED_PACE'].dropna().values
        if len(SP_Figure) >1:
            model = SimpleExpSmoothing(SP_Figure)
            model = model.fit()
            SustainedPace.append([name, model.forecast()[0]])
        elif len(SP_Figure) == 1:
            SustainedPace.append([name,SP_Figure[0]])
        else :
            SustainedPace.append([name,0])
    SustainedPace = pd.DataFrame(SustainedPace, columns=['HNAME','SP'])
    # SustainedPace = Extraction_Database("""
    #                                     Select HNAME, RARID, SUSTAINED_PACE SP from Race_PosteriorDb
    #                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
    #                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    # if len(SustainedPace) == 0:
    #     Feature_DF.loc[:,'PP_PAF_SP_ADV_GOPFL'] = 0
    #     Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_SP_ADV_GOPFL']]
    #     return Feature_DF
    # SustainedPace = SustainedPace.groupby('HNAME').mean().loc[:,'SP'].reset_index()
    Feature_DF = Feature_DF.merge(SustainedPace, how='left')
    Feature_DF.loc[:,'rank'] = Feature_DF.loc[:,'SP'].rank(method = 'max', ascending=False)

    Feature_DF = Feature_DF.merge(History_tomap, how='left')
    Feature_DF.loc[:,'PP_PAF_SP_ADV_PFL'].fillna(Feature_DF.loc[:,'PP_PAF_SP_ADV_PFL'].min(), inplace = True)
    Feature_DF.loc[:,'PP_PAF_SP_ADV_PFL'].fillna(0, inplace=True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_SP_ADV_PFL']]

    return Feature_DF

"""
PP_PAF_SP_WIN_PFL
"""

def PP_PAF_SP_WIN_PFL(Dataframe, HNAME_List, Raceday):

    """
    Distance from winning Sustained Pace
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_PAF_SP_WIN_PFL]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Location ="'" + Dataframe.loc[:,'RALOC'].values[0] + "'"
    Surface ="'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Win_SP = Extraction_Database("""
                                 Select avg(SUSTAINED_PACE) from Race_PosteriorDb
                                 where RADAT < {Raceday} and RADIS = {Distance} and RATRA = {Surface}
                                 and RALOC = {Location} and RESFP = 1
                                 """.format(Raceday = Raceday, HNAME_List = HNAME_List, Distance=Distance,
                                 Surface=Surface, Location=Location)).values.tolist()[0][0]
    #No Winning SP
    if Win_SP == None:
        Win_SP = 1

    #Average Sustained Pace
    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, SUSTAINED_PACE from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    SustainedPace = []
    for name, group in Extraction.groupby('HNAME'):
        SP_Figure = group.loc[:,'SUSTAINED_PACE'].dropna().values
        if len(SP_Figure) >1:
            model = SimpleExpSmoothing(SP_Figure)
            model = model.fit()
            SustainedPace.append([name, model.forecast()[0]])
        elif len(SP_Figure) == 1:
            SustainedPace.append([name,SP_Figure[0]])
        else :
            SustainedPace.append([name,0])
    SustainedPace = pd.DataFrame(SustainedPace, columns=['HNAME','SP'])
    # SustainedPace = Extraction_Database("""
    #                                     Select HNAME, RARID, SUSTAINED_PACE SP from Race_PosteriorDb
    #                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
    #                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    # if len(SustainedPace) == 0:
    #     Feature_DF.loc[:,'PP_PAF_SP_ADV_GOPFL'] = 0
    #     Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_SP_ADV_GOPFL']]
    #     return Feature_DF
    # SustainedPace = SustainedPace.groupby('HNAME').mean().loc[:,'SP'].reset_index()
    Feature_DF = Feature_DF.merge(SustainedPace, how='left')

    Feature_DF.loc[:,'PP_PAF_SP_WIN_PFL'] = ((Feature_DF.loc[:,'SP'] - Win_SP)/Win_SP).abs()

    Feature_DF.loc[:,'PP_PAF_SP_WIN_PFL'].fillna(Feature_DF.loc[:,'PP_PAF_SP_WIN_PFL'].min(), inplace = True)
    Feature_DF.loc[:,'PP_PAF_SP_WIN_PFL'].fillna(0, inplace=True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_SP_WIN_PFL']]

    return Feature_DF

"""
PP_PAF_SP_DIST
"""

def PP_PAF_SP_DIST(Dataframe, HNAME_List, Raceday):

    """
    Sustained Pace relative to Distance
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_PAF_SP_DIST]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    #Average Sustained Pace
    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, SUSTAINED_PACE from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    SustainedPace = []
    for name, group in Extraction.groupby('HNAME'):
        SP_Figure = group.loc[:,'SUSTAINED_PACE'].dropna().values
        if len(SP_Figure) >1:
            model = SimpleExpSmoothing(SP_Figure)
            model = model.fit()
            SustainedPace.append([name, model.forecast()[0]])
        elif len(SP_Figure) == 1:
            SustainedPace.append([name,SP_Figure[0]])
        else :
            SustainedPace.append([name,0])
    SustainedPace = pd.DataFrame(SustainedPace, columns=['HNAME','SP'])
    # SustainedPace = Extraction_Database("""
    #                                     Select HNAME, RARID, SUSTAINED_PACE SP from Race_PosteriorDb
    #                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
    #                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    # if len(SustainedPace) == 0:
    #     Feature_DF.loc[:,'PP_PAF_SP_ADV_GOPFL'] = 0
    #     Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_SP_ADV_GOPFL']]
    #     return Feature_DF
    # SustainedPace = SustainedPace.groupby('HNAME').mean().loc[:,'SP'].reset_index()
    Feature_DF = Feature_DF.merge(SustainedPace, how='left')

    Feature_DF.loc[:,'PP_PAF_SP_DIST'] = Feature_DF.loc[:,'SP'] * Distance

    Feature_DF.loc[:,'PP_PAF_SP_DIST'].fillna(Feature_DF.loc[:,'PP_PAF_SP_DIST'].min(), inplace = True)
    Feature_DF.loc[:,'PP_PAF_SP_DIST'].fillna(0, inplace=True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_SP_DIST']]

    return Feature_DF

"""
PP_PAF_AP_AVG
"""

def PP_PAF_AP_AVG(Dataframe, HNAME_List, Raceday):

    """
    Average Average Pace
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_PAF_AP_AVG]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, AVERAGE_PAGE PP_PAF_AP_AVG from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    if len(Extraction) == 0:
        Feature_DF.loc[:,'PP_PAF_AP_AVG'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_AP_AVG']]
        return Feature_DF

    Extraction = Extraction.groupby('HNAME').mean().loc[:,'PP_PAF_AP_AVG'].reset_index()
    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_PAF_AP_AVG'].fillna(Feature_DF.loc[:,'PP_PAF_AP_AVG'].min(), inplace = True)
    Feature_DF.loc[:,'PP_PAF_AP_AVG'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_AP_AVG']]

    return Feature_DF

"""
PP_PAF_AP_AVGRW
"""

def PP_PAF_AP_AVGRW(Dataframe, HNAME_List, Raceday):

    """
    Recency Weighted Avg Average Pace
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_PAF_AP_AVGRW]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, AVERAGE_PAGE from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    AvgPace = []
    for name, group in Extraction.groupby('HNAME'):
        AP_Figure = group.loc[:,'AVERAGE_PAGE'].dropna().values
        if len(AP_Figure) >1:
            model = SimpleExpSmoothing(AP_Figure)
            model = model.fit()
            AvgPace.append([name, model.forecast()[0]])
        elif len(AP_Figure) == 1:
            AvgPace.append([name,AP_Figure[0]])
        else :
            AvgPace.append([name,0])

    AvgPace = pd.DataFrame(AvgPace, columns=['HNAME','PP_PAF_AP_AVGRW'])

    Feature_DF = Feature_DF.merge(AvgPace, how='left')
    Feature_DF.loc[:,'PP_PAF_AP_AVGRW'].fillna(Feature_DF.loc[:,'PP_PAF_AP_AVGRW'].min(), inplace = True)
    Feature_DF.loc[:,'PP_PAF_AP_AVGRW'].fillna(0, inplace=True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_AP_AVGRW']]

    return Feature_DF

"""
PP_PAF_AP_ADV_GOPFL
"""

def PP_PAF_AP_ADV_GOPFL(Dataframe, HNAME_List, Raceday):

    """
    Average Pace Advantage on Going and Profile
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_PAF_AP_ADV_GOPFL]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Location ="'" + Dataframe.loc[:,'RALOC'].values[0] + "'"
    Surface ="'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Distance = Dataframe.loc[:,'RADIS'].values[0]
    Going = "'" + Dataframe.loc[:,'RAGOG'].values[0] + "'"

    Extraction_History = Extraction_Database("""
                                             Select HNAME, RARID, AVERAGE_PAGE, RESFP PP_PAF_AP_ADV_GOPFL from Race_PosteriorDb
                                             where RADAT < {Raceday} and RADIS = {Distance} and RATRA = {Surface}
                                             and RALOC = {Location} and RAGOG = {Going}
                                             """.format(Raceday = Raceday, HNAME_List = HNAME_List, Distance=Distance,
                                             Going=Going, Surface=Surface, Location=Location))

    if len(Extraction_History) == 0:
        Feature_DF.loc[:,'PP_PAF_AP_ADV_GOPFL'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_AP_ADV_GOPFL']]
        return Feature_DF

    def EP_Rank(race):
        race.loc[:,'rank'] = race.loc[:,'AVERAGE_PAGE'].rank(method = 'max', ascending=False)
        return race
    Extraction_History = Extraction_History.groupby('RARID').apply(EP_Rank)
    History_tomap = Extraction_History.groupby('rank').mean()['PP_PAF_AP_ADV_GOPFL'].reset_index(drop=False)

    #Average Average Pace
    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, AVERAGE_PAGE from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    AvgPace = []
    for name, group in Extraction.groupby('HNAME'):
        AP_Figure = group.loc[:,'AVERAGE_PAGE'].dropna().values
        if len(AP_Figure) >1:
            model = SimpleExpSmoothing(AP_Figure)
            model = model.fit()
            AvgPace.append([name, model.forecast()[0]])
        elif len(AP_Figure) == 1:
            AvgPace.append([name,AP_Figure[0]])
        else :
            AvgPace.append([name,0])

    AvgPace = pd.DataFrame(AvgPace, columns=['HNAME','AP'])
    # AvgPace = Extraction_Database("""
    #                               Select HNAME, RARID, AVERAGE_PAGE AP from Race_PosteriorDb
    #                               where RADAT < {Raceday} and HNAME in {HNAME_List}
    #                               """.format(Raceday = Raceday, HNAME_List = HNAME_List))

    # if len(AvgPace) == 0:
    #     Feature_DF.loc[:,'PP_PAF_AP_ADV_GOPFL'] = 0
    #     Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_AP_ADV_GOPFL']]
    #     return Feature_DF

    # AvgPace = AvgPace.groupby('HNAME').mean().loc[:,'AP'].reset_index()
    Feature_DF = Feature_DF.merge(AvgPace, how='left')

    Feature_DF.loc[:,'rank'] = Feature_DF.loc[:,'AP'].rank(method = 'max', ascending=False)

    Feature_DF = Feature_DF.merge(History_tomap, how='left')
    Feature_DF.loc[:,'PP_PAF_AP_ADV_GOPFL'].fillna(Feature_DF.loc[:,'PP_PAF_AP_ADV_GOPFL'].min(), inplace = True)
    Feature_DF.loc[:,'PP_PAF_AP_ADV_GOPFL'].fillna(0, inplace=True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_AP_ADV_GOPFL']]

    return Feature_DF

"""
PP_PAF_AP_ADV_PFL
"""

def PP_PAF_AP_ADV_PFL(Dataframe, HNAME_List, Raceday):

    """
    Early Pace Advantage on Profile
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_PAF_AP_ADV_PFL]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Location ="'" + Dataframe.loc[:,'RALOC'].values[0] + "'"
    Surface ="'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Extraction_History = Extraction_Database("""
                                             Select HNAME, RARID, AVERAGE_PAGE, RESFP PP_PAF_AP_ADV_PFL from Race_PosteriorDb
                                             where RADAT < {Raceday} and RADIS = {Distance} and RATRA = {Surface}
                                             and RALOC = {Location}
                                             """.format(Raceday = Raceday, HNAME_List = HNAME_List, Distance=Distance,
                                             Surface=Surface, Location=Location))

    if len(Extraction_History) == 0:
        Feature_DF.loc[:,'PP_PAF_AP_ADV_PFL'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_AP_ADV_PFL']]
        return Feature_DF

    def EP_Rank(race):
        race.loc[:,'rank'] = race.loc[:,'AVERAGE_PAGE'].rank(method = 'max', ascending=False)
        return race
    Extraction_History = Extraction_History.groupby('RARID').apply(EP_Rank)
    History_tomap = Extraction_History.groupby('rank').mean()['PP_PAF_AP_ADV_PFL'].reset_index(drop=False)

    #Average Average Pace
    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, AVERAGE_PAGE from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    AvgPace = []
    for name, group in Extraction.groupby('HNAME'):
        AP_Figure = group.loc[:,'AVERAGE_PAGE'].dropna().values
        if len(AP_Figure) >1:
            model = SimpleExpSmoothing(AP_Figure)
            model = model.fit()
            AvgPace.append([name, model.forecast()[0]])
        elif len(AP_Figure) == 1:
            AvgPace.append([name,AP_Figure[0]])
        else :
            AvgPace.append([name,0])

    AvgPace = pd.DataFrame(AvgPace, columns=['HNAME','AP'])
    # AvgPace = Extraction_Database("""
    #                               Select HNAME, RARID, AVERAGE_PAGE AP from Race_PosteriorDb
    #                               where RADAT < {Raceday} and HNAME in {HNAME_List}
    #                               """.format(Raceday = Raceday, HNAME_List = HNAME_List))

    # if len(AvgPace) == 0:
    #     Feature_DF.loc[:,'PP_PAF_AP_ADV_GOPFL'] = 0
    #     Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_AP_ADV_GOPFL']]
    #     return Feature_DF

    # AvgPace = AvgPace.groupby('HNAME').mean().loc[:,'AP'].reset_index()
    Feature_DF = Feature_DF.merge(AvgPace, how='left')

    Feature_DF.loc[:,'rank'] = Feature_DF.loc[:,'AP'].rank(method = 'max', ascending=False)

    Feature_DF = Feature_DF.merge(History_tomap, how='left')
    Feature_DF.loc[:,'PP_PAF_AP_ADV_PFL'].fillna(Feature_DF.loc[:,'PP_PAF_AP_ADV_PFL'].min(), inplace = True)
    Feature_DF.loc[:,'PP_PAF_AP_ADV_PFL'].fillna(0, inplace=True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_AP_ADV_PFL']]

    return Feature_DF

"""
PP_PAF_AP_WIN_PFL
"""

def PP_PAF_AP_WIN_PFL(Dataframe, HNAME_List, Raceday):

    """
    Distance from winning Average Pace
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_PAF_AP_WIN_PFL]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Location ="'" + Dataframe.loc[:,'RALOC'].values[0] + "'"
    Surface ="'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Win_AP = Extraction_Database("""
                                 Select avg(AVERAGE_PAGE) from Race_PosteriorDb
                                 where RADAT < {Raceday} and RADIS = {Distance} and RATRA = {Surface}
                                 and RALOC = {Location} and RESFP = 1
                                 """.format(Raceday = Raceday, HNAME_List = HNAME_List, Distance=Distance,
                                 Surface=Surface, Location=Location)).values.tolist()[0][0]
    #No Winning AP
    if Win_AP == None:
        Win_AP = 1

    #Average Average Pace
    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, AVERAGE_PAGE from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    AvgPace = []
    for name, group in Extraction.groupby('HNAME'):
        AP_Figure = group.loc[:,'AVERAGE_PAGE'].dropna().values
        if len(AP_Figure) >1:
            model = SimpleExpSmoothing(AP_Figure)
            model = model.fit()
            AvgPace.append([name, model.forecast()[0]])
        elif len(AP_Figure) == 1:
            AvgPace.append([name,AP_Figure[0]])
        else :
            AvgPace.append([name,0])

    AvgPace = pd.DataFrame(AvgPace, columns=['HNAME','AP'])
    # AvgPace = Extraction_Database("""
    #                               Select HNAME, RARID, AVERAGE_PAGE AP from Race_PosteriorDb
    #                               where RADAT < {Raceday} and HNAME in {HNAME_List}
    #                               """.format(Raceday = Raceday, HNAME_List = HNAME_List))

    # if len(AvgPace) == 0:
    #     Feature_DF.loc[:,'PP_PAF_AP_ADV_GOPFL'] = 0
    #     Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_AP_ADV_GOPFL']]
    #     return Feature_DF

    # AvgPace = AvgPace.groupby('HNAME').mean().loc[:,'AP'].reset_index()
    Feature_DF = Feature_DF.merge(AvgPace, how='left')

    Feature_DF.loc[:,'PP_PAF_AP_WIN_PFL'] = ((Feature_DF.loc[:,'AP'] - Win_AP)/Win_AP).abs()

    Feature_DF.loc[:,'PP_PAF_AP_WIN_PFL'].fillna(Feature_DF.loc[:,'PP_PAF_AP_WIN_PFL'].min(), inplace = True)
    Feature_DF.loc[:,'PP_PAF_AP_WIN_PFL'].fillna(0, inplace=True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_AP_WIN_PFL']]

    return Feature_DF

"""
PP_PAF_FP_AVG
"""

def PP_PAF_FP_AVG(Dataframe, HNAME_List, Raceday):

    """
    Average Final Fraction Pace
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_PAF_FP_AVG]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, FINAL_FRACTION_PACE PP_PAF_FP_AVG from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))

    if len(Extraction) == 0:
        Feature_DF.loc[:,'PP_PAF_FP_AVG'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_FP_AVG']]
        return Feature_DF

    Extraction = Extraction.groupby('HNAME').mean().loc[:,'PP_PAF_FP_AVG'].reset_index()
    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_PAF_FP_AVG'].fillna(Feature_DF.loc[:,'PP_PAF_FP_AVG'].min(), inplace = True)
    Feature_DF.loc[:,'PP_PAF_FP_AVG'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_FP_AVG']]

    return Feature_DF

"""
PP_PAF_FP_AVGRW
"""

def PP_PAF_FP_AVGRW(Dataframe, HNAME_List, Raceday):

    """
    Recency Weighted Avg Final Fraction Pace
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_PAF_FP_AVGRW]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, FINAL_FRACTION_PACE from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    FFPace = []
    for name, group in Extraction.groupby('HNAME'):
        FFP_Figure = group.loc[:,'FINAL_FRACTION_PACE'].dropna().values
        if len(FFP_Figure) >1:
            model = SimpleExpSmoothing(FFP_Figure)
            model = model.fit()
            FFPace.append([name, model.forecast()[0]])
        elif len(FFP_Figure) == 1:
            FFPace.append([name,FFP_Figure[0]])
        else :
            FFPace.append([name,0])

    FFPace = pd.DataFrame(FFPace, columns=['HNAME','PP_PAF_FP_AVGRW'])

    Feature_DF = Feature_DF.merge(FFPace, how='left')
    Feature_DF.loc[:,'PP_PAF_FP_AVGRW'].fillna(Feature_DF.loc[:,'PP_PAF_FP_AVGRW'].min(), inplace = True)
    Feature_DF.loc[:,'PP_PAF_FP_AVGRW'].fillna(0, inplace=True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_FP_AVGRW']]

    return Feature_DF

"""
PP_PAF_FP_ADV_GOPFL
"""

def PP_PAF_FP_ADV_GOPFL(Dataframe, HNAME_List, Raceday):

    """
    Final Fraction Pace Advantage on Going and Profile
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_PAF_FP_ADV_GOPFL]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Location ="'" + Dataframe.loc[:,'RALOC'].values[0] + "'"
    Surface ="'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Distance = Dataframe.loc[:,'RADIS'].values[0]
    Going = "'" + Dataframe.loc[:,'RAGOG'].values[0] + "'"

    Extraction_History = Extraction_Database("""
                                             Select HNAME, RARID, FINAL_FRACTION_PACE, RESFP PP_PAF_FP_ADV_GOPFL from Race_PosteriorDb
                                             where RADAT < {Raceday} and RADIS = {Distance} and RATRA = {Surface}
                                             and RALOC = {Location} and RAGOG = {Going}
                                             """.format(Raceday = Raceday, HNAME_List = HNAME_List, Distance=Distance,
                                             Going=Going, Surface=Surface, Location=Location))

    if len(Extraction_History) == 0:
        Feature_DF.loc[:,'PP_PAF_FP_ADV_GOPFL'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_FP_ADV_GOPFL']]
        return Feature_DF

    def EP_Rank(race):
        race.loc[:,'rank'] = race.loc[:,'FINAL_FRACTION_PACE'].rank(method = 'max', ascending=False)
        return race
    Extraction_History = Extraction_History.groupby('RARID').apply(EP_Rank)
    History_tomap = Extraction_History.groupby('rank').mean()['PP_PAF_FP_ADV_GOPFL'].reset_index(drop=False)

    #Average Final Fraction Pace
    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, FINAL_FRACTION_PACE from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    FFPace = []
    for name, group in Extraction.groupby('HNAME'):
        FFP_Figure = group.loc[:,'FINAL_FRACTION_PACE'].dropna().values
        if len(FFP_Figure) >1:
            model = SimpleExpSmoothing(FFP_Figure)
            model = model.fit()
            FFPace.append([name, model.forecast()[0]])
        elif len(FFP_Figure) == 1:
            FFPace.append([name,FFP_Figure[0]])
        else :
            FFPace.append([name,0])

    FFPace = pd.DataFrame(FFPace, columns=['HNAME','FFP'])
    Feature_DF = Feature_DF.merge(FFPace, how='left')
    Feature_DF.loc[:,'rank'] = Feature_DF.loc[:,'FFP'].rank(method = 'max', ascending=False)

    Feature_DF = Feature_DF.merge(History_tomap, how='left')
    Feature_DF.loc[:,'PP_PAF_FP_ADV_GOPFL'].fillna(Feature_DF.loc[:,'PP_PAF_FP_ADV_GOPFL'].min(), inplace = True)
    Feature_DF.loc[:,'PP_PAF_FP_ADV_GOPFL'].fillna(0, inplace=True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_FP_ADV_GOPFL']]

    return Feature_DF

"""
PP_PAF_FP_ADV_PFL
"""

def PP_PAF_FP_ADV_PFL(Dataframe, HNAME_List, Raceday):

    """
    Final Fraction Pace Advantage on Profile
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_PAF_FP_ADV_PFL]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Location ="'" + Dataframe.loc[:,'RALOC'].values[0] + "'"
    Surface ="'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Extraction_History = Extraction_Database("""
                                             Select HNAME, RARID, FINAL_FRACTION_PACE, RESFP PP_PAF_FP_ADV_PFL from Race_PosteriorDb
                                             where RADAT < {Raceday} and RADIS = {Distance} and RATRA = {Surface}
                                             and RALOC = {Location}
                                             """.format(Raceday = Raceday, HNAME_List = HNAME_List, Distance=Distance,
                                             Surface=Surface, Location=Location))

    if len(Extraction_History) == 0:
        Feature_DF.loc[:,'PP_PAF_FP_ADV_PFL'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_FP_ADV_PFL']]
        return Feature_DF

    def EP_Rank(race):
        race.loc[:,'rank'] = race.loc[:,'FINAL_FRACTION_PACE'].rank(method = 'max', ascending=False)
        return race
    Extraction_History = Extraction_History.groupby('RARID').apply(EP_Rank)
    History_tomap = Extraction_History.groupby('rank').mean()['PP_PAF_FP_ADV_PFL'].reset_index(drop=False)

    #Average Final Fraction Pace
    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, FINAL_FRACTION_PACE from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    FFPace = []
    for name, group in Extraction.groupby('HNAME'):
        FFP_Figure = group.loc[:,'FINAL_FRACTION_PACE'].dropna().values
        if len(FFP_Figure) >1:
            model = SimpleExpSmoothing(FFP_Figure)
            model = model.fit()
            FFPace.append([name, model.forecast()[0]])
        elif len(FFP_Figure) == 1:
            FFPace.append([name,FFP_Figure[0]])
        else :
            FFPace.append([name,0])

    FFPace = pd.DataFrame(FFPace, columns=['HNAME','FFP'])
    # FFPace = Extraction_Database("""
    #                               Select HNAME, RARID, FINAL_FRACTION_PACE FFP from Race_PosteriorDb
    #                               where RADAT < {Raceday} and HNAME in {HNAME_List}
    #                               """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    # if len(FFPace) == 0:
    #     Feature_DF.loc[:,'PP_PAF_FP_ADV_PFL'] = 0
    #     Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_FP_ADV_PFL']]
    #     return Feature_DF
    # FFPace = FFPace.groupby('HNAME').mean().loc[:,'FFP'].reset_index()

    Feature_DF = Feature_DF.merge(FFPace, how='left')
    Feature_DF.loc[:,'rank'] = Feature_DF.loc[:,'FFP'].rank(method = 'max', ascending=False)

    Feature_DF = Feature_DF.merge(History_tomap, how='left')
    Feature_DF.loc[:,'PP_PAF_FP_ADV_PFL'].fillna(Feature_DF.loc[:,'PP_PAF_FP_ADV_PFL'].min(), inplace = True)
    Feature_DF.loc[:,'PP_PAF_FP_ADV_PFL'].fillna(0, inplace=True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_FP_ADV_PFL']]

    return Feature_DF

"""
PP_PAF_FP_WIN_PFL
"""

def PP_PAF_FP_WIN_PFL(Dataframe, HNAME_List, Raceday):

    """
    Distance from winning Final Fraction Pace
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_PAF_FP_WIN_PFL]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Location ="'" + Dataframe.loc[:,'RALOC'].values[0] + "'"
    Surface ="'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Win_FFP = Extraction_Database("""
                                 Select avg(FINAL_FRACTION_PACE) from Race_PosteriorDb
                                 where RADAT < {Raceday} and RADIS = {Distance} and RATRA = {Surface}
                                 and RALOC = {Location} and RESFP = 1
                                 """.format(Raceday = Raceday, HNAME_List = HNAME_List, Distance=Distance,
                                 Surface=Surface, Location=Location)).values.tolist()[0][0]
    #No Winning FFP
    if Win_FFP == None:
        Win_FFP = 1

    #Average Final Fraction Pace
    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, FINAL_FRACTION_PACE from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    FFPace = []
    for name, group in Extraction.groupby('HNAME'):
        FFP_Figure = group.loc[:,'FINAL_FRACTION_PACE'].dropna().values
        if len(FFP_Figure) >1:
            model = SimpleExpSmoothing(FFP_Figure)
            model = model.fit()
            FFPace.append([name, model.forecast()[0]])
        elif len(FFP_Figure) == 1:
            FFPace.append([name,FFP_Figure[0]])
        else :
            FFPace.append([name,0])

    FFPace = pd.DataFrame(FFPace, columns=['HNAME','FFP'])
    # FFPace = Extraction_Database("""
    #                               Select HNAME, RARID, FINAL_FRACTION_PACE FFP from Race_PosteriorDb
    #                               where RADAT < {Raceday} and HNAME in {HNAME_List}
    #                               """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    # if len(FFPace) == 0:
    #     Feature_DF.loc[:,'PP_PAF_FP_ADV_PFL'] = 0
    #     Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_FP_ADV_PFL']]
    #     return Feature_DF
    # FFPace = FFPace.groupby('HNAME').mean().loc[:,'FFP'].reset_index()

    Feature_DF = Feature_DF.merge(FFPace, how='left')
    Feature_DF.loc[:,'PP_PAF_FP_WIN_PFL'] = ((Feature_DF.loc[:,'FFP'] - Win_FFP) / Win_FFP).abs()

    Feature_DF.loc[:,'PP_PAF_FP_WIN_PFL'].fillna(Feature_DF.loc[:,'PP_PAF_FP_WIN_PFL'].min(), inplace = True)
    Feature_DF.loc[:,'PP_PAF_FP_WIN_PFL'].fillna(0, inplace=True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_FP_WIN_PFL']]

    return Feature_DF

"""
PP_PAF_EDW_DIST
"""

def PP_PAF_EDW_DIST(Dataframe, HNAME_List, Raceday):

    """
    Winning Energy Distribution on Distance
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_PAF_EDW_DIST]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Win_EEP = Extraction_Database("""
                                  Select avg(EARLY_ENERGY) from Race_PosteriorDb
                                  where RADAT < {Raceday} and RADIS = {Distance} and RESFP = 1
                                  """.format(Raceday = Raceday, HNAME_List = HNAME_List, Distance=Distance
                                  )).values.tolist()[0][0]
    #No Winning EEP
    if Win_EEP == None:
        Win_EEP = 0.5
    #Average Eearly Energy Profile
    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, EARLY_ENERGY from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    EEP = []
    for name, group in Extraction.groupby('HNAME'):
        EEP_Figure = group.loc[:,'EARLY_ENERGY'].dropna().values
        if len(EEP_Figure) >1:
            model = SimpleExpSmoothing(EEP_Figure)
            model = model.fit()
            EEP.append([name, model.forecast()[0]])
        elif len(EEP_Figure) == 1:
            EEP.append([name,EEP_Figure[0]])
        else :
            EEP.append([name,0])

    EEP = pd.DataFrame(EEP, columns=['HNAME','EEP'])

    Feature_DF = Feature_DF.merge(EEP, how='left')
    Feature_DF.loc[:,'PP_PAF_EDW_DIST'] = ((Feature_DF.loc[:,'EEP'] - Win_EEP) / Win_EEP).abs()
    Feature_DF.loc[:,'PP_PAF_EDW_DIST'].fillna(Feature_DF.loc[:,'PP_PAF_EDW_DIST'].min(), inplace = True)
    Feature_DF.loc[:,'PP_PAF_EDW_DIST'].fillna(0, inplace=True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_EDW_DIST']]

    return Feature_DF

"""
PP_PAF_EDW_PFL
"""

def PP_PAF_EDW_PFL(Dataframe, HNAME_List, Raceday):

    """
    Winning Energy Distribution on Profile
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_PAF_EDW_PFL]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Location ="'" + Dataframe.loc[:,'RALOC'].values[0] + "'"
    Surface ="'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Win_EEP = Extraction_Database("""
                                  Select avg(EARLY_ENERGY) from Race_PosteriorDb
                                  where RADAT < {Raceday} and RADIS = {Distance} and RATRA = {Surface}
                                  and RALOC = {Location} and RESFP = 1
                                  """.format(Raceday = Raceday, HNAME_List = HNAME_List, Distance=Distance,
                                  Surface=Surface, Location=Location)).values.tolist()[0][0]
    #No Winning EEP
    if Win_EEP == None:
        Win_EEP = 0.5
    #Average Eearly Energy Profile
    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, EARLY_ENERGY from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    EEP = []
    for name, group in Extraction.groupby('HNAME'):
        EEP_Figure = group.loc[:,'EARLY_ENERGY'].dropna().values
        if len(EEP_Figure) >1:
            model = SimpleExpSmoothing(EEP_Figure)
            model = model.fit()
            EEP.append([name, model.forecast()[0]])
        elif len(EEP_Figure) == 1:
            EEP.append([name,EEP_Figure[0]])
        else :
            EEP.append([name,0])

    EEP = pd.DataFrame(EEP, columns=['HNAME','EEP'])

    Feature_DF = Feature_DF.merge(EEP, how='left')
    Feature_DF.loc[:,'PP_PAF_EDW_PFL'] = ((Feature_DF.loc[:,'EEP'] - Win_EEP) / Win_EEP).abs()
    Feature_DF.loc[:,'PP_PAF_EDW_PFL'].fillna(Feature_DF.loc[:,'PP_PAF_EDW_PFL'].min(), inplace = True)
    Feature_DF.loc[:,'PP_PAF_EDW_PFL'].fillna(0, inplace=True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_EDW_PFL']]

    return Feature_DF

"""
PP_PAF_EDL_DIST
"""

def PP_PAF_EDL_DIST(Dataframe, HNAME_List, Raceday):

    """
    Winning Energy Distribution Limit on Distance
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_PAF_EDL_DIST]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Win_EEP = Extraction_Database("""
                                  Select EARLY_ENERGY from Race_PosteriorDb
                                  where RADAT < {Raceday} and RADIS = {Distance} and RESFP = 1
                                  """.format(Raceday = Raceday, Distance=Distance))

    if len(Win_EEP) == 0:
        Feature_DF.loc[:,'PP_PAF_EDL_DIST'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_EDL_DIST']]
        return Feature_DF

    #Quantile Calculations
    minEEP = Win_EEP.quantile(0.1)['EARLY_ENERGY']
    maxEEP = Win_EEP.quantile(0.9)['EARLY_ENERGY']
    avgEEP = np.mean([minEEP, maxEEP])

    #Average Eearly Energy Profile
    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, EARLY_ENERGY from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    EEP = []
    for name, group in Extraction.groupby('HNAME'):
        EEP_Figure = group.loc[:,'EARLY_ENERGY'].dropna().values
        if len(EEP_Figure) >1:
            model = SimpleExpSmoothing(EEP_Figure)
            model = model.fit()
            EEP.append([name, model.forecast()[0]])
        elif len(EEP_Figure) == 1:
            EEP.append([name,EEP_Figure[0]])
        else :
            EEP.append([name,0])

    EEP = pd.DataFrame(EEP, columns=['HNAME','EEP'])

    Feature_DF = Feature_DF.merge(EEP, how='left')
    Feature_DF.loc[:,'PP_PAF_EDL_DIST'] = ((Feature_DF.loc[:,'EEP'] - Win_EEP) / Win_EEP).abs()

    #Quantile Difference
    Feature_DF.loc[Feature_DF.loc[:,'EEP'] > maxEEP,'PP_PAF_EDL_DIST'] = \
    Feature_DF.loc[Feature_DF.loc[:,'EEP'] > maxEEP,'EEP'].apply(lambda x : x-maxEEP)
    Feature_DF.loc[Feature_DF.loc[:,'EEP'] < minEEP,'PP_PAF_EDL_DIST'] = \
    Feature_DF.loc[Feature_DF.loc[:,'EEP'] < minEEP,'EEP'].apply(lambda x : minEEP-x)
    Feature_DF.loc[(Feature_DF.loc[:,'EEP'] > minEEP) & (Feature_DF.loc[:,'EEP'] < avgEEP),'PP_PAF_EDL_PFL'] = \
    Feature_DF.loc[(Feature_DF.loc[:,'EEP'] > minEEP) & (Feature_DF.loc[:,'EEP'] < avgEEP),'EEP'].apply(lambda x : minEEP-x)
    Feature_DF.loc[(Feature_DF.loc[:,'EEP'] < maxEEP) & (Feature_DF.loc[:,'EEP'] > avgEEP),'PP_PAF_EDL_PFL'] = \
    Feature_DF.loc[(Feature_DF.loc[:,'EEP'] < maxEEP) & (Feature_DF.loc[:,'EEP'] > avgEEP),'EEP'].apply(lambda x : x-maxEEP)

    Feature_DF.loc[:,'PP_PAF_EDL_DIST'].fillna(Feature_DF.loc[:,'PP_PAF_EDL_DIST'].max(), inplace = True)
    Feature_DF.loc[:,'PP_PAF_EDL_DIST'].fillna(0, inplace=True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_EDL_DIST']]

    return Feature_DF

"""
PP_PAF_EDL_PFL
"""

def PP_PAF_EDL_PFL(Dataframe, HNAME_List, Raceday):

    """
    Winning Energy Distribution Limit on Profile
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_PAF_EDL_PFL]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Location ="'" + Dataframe.loc[:,'RALOC'].values[0] + "'"
    Surface ="'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Win_EEP = Extraction_Database("""
                                  Select EARLY_ENERGY from Race_PosteriorDb
                                  where RADAT < {Raceday} and RADIS = {Distance} and RATRA = {Surface}
                                  and RALOC = {Location} and RESFP = 1
                                  """.format(Raceday = Raceday, Distance=Distance,
                                  Surface=Surface, Location=Location))

    if len(Win_EEP) == 0:
        Feature_DF.loc[:,'PP_PAF_EDL_PFL'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_EDL_PFL']]
        return Feature_DF

    #Quantile Calculations
    minEEP = Win_EEP.quantile(0.1)['EARLY_ENERGY']
    maxEEP = Win_EEP.quantile(0.9)['EARLY_ENERGY']
    avgEEP = np.mean([minEEP, maxEEP])

    #Average Eearly Energy Profile
    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, EARLY_ENERGY from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))
    EEP = []
    for name, group in Extraction.groupby('HNAME'):
        EEP_Figure = group.loc[:,'EARLY_ENERGY'].dropna().values
        if len(EEP_Figure) >1:
            model = SimpleExpSmoothing(EEP_Figure)
            model = model.fit()
            EEP.append([name, model.forecast()[0]])
        elif len(EEP_Figure) == 1:
            EEP.append([name,EEP_Figure[0]])
        else :
            EEP.append([name,0])

    EEP = pd.DataFrame(EEP, columns=['HNAME','EEP'])

    Feature_DF = Feature_DF.merge(EEP, how='left')
    Feature_DF.loc[:,'PP_PAF_EDL_PFL'] = ((Feature_DF.loc[:,'EEP'] - Win_EEP) / Win_EEP).abs()

    #Quantile Difference
    Feature_DF.loc[Feature_DF.loc[:,'EEP'] > maxEEP,'PP_PAF_EDL_PFL'] = \
    Feature_DF.loc[Feature_DF.loc[:,'EEP'] > maxEEP,'EEP'].apply(lambda x : x-maxEEP)
    Feature_DF.loc[Feature_DF.loc[:,'EEP'] < minEEP,'PP_PAF_EDL_PFL'] = \
    Feature_DF.loc[Feature_DF.loc[:,'EEP'] < minEEP,'EEP'].apply(lambda x : minEEP-x)
    Feature_DF.loc[(Feature_DF.loc[:,'EEP'] > minEEP) & (Feature_DF.loc[:,'EEP'] < avgEEP),'PP_PAF_EDL_PFL'] = \
    Feature_DF.loc[(Feature_DF.loc[:,'EEP'] > minEEP) & (Feature_DF.loc[:,'EEP'] < avgEEP),'EEP'].apply(lambda x : minEEP-x)
    Feature_DF.loc[(Feature_DF.loc[:,'EEP'] < maxEEP) & (Feature_DF.loc[:,'EEP'] > avgEEP),'PP_PAF_EDL_PFL'] = \
    Feature_DF.loc[(Feature_DF.loc[:,'EEP'] < maxEEP) & (Feature_DF.loc[:,'EEP'] > avgEEP),'EEP'].apply(lambda x : x-maxEEP)

    Feature_DF.loc[:,'PP_PAF_EDL_PFL'].fillna(Feature_DF.loc[:,'PP_PAF_EDL_PFL'].max(), inplace = True)
    Feature_DF.loc[:,'PP_PAF_EDL_PFL'].fillna(0, inplace=True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_EDL_PFL']]

    return Feature_DF

"""
PP_PAF_STL_AVG_PFL
"""

def PP_PAF_STL_AVG_PFL(Dataframe, HNAME_List, Raceday):

    """
    Average Final Straight Line Speed on Distance
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_PAF_STL_AVG_PFL]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Extraction = Extraction_Database("""
                                     Select HNAME, avg(PACE_S3) S3, avg(PACE_S4) S4,
                                     avg(PACE_S5) S5, avg(PACE_S6) S6 from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RADIS = {Distance}
                                     Group by HNAME
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Distance=Distance))

    if Distance in [1000,1200]:
        Extraction = Extraction.loc[:,['HNAME','S3']]
    elif Distance in [1400,1600,1650]:
        Extraction = Extraction.loc[:,['HNAME','S4']]
    elif Distance in [1800,2000]:
        Extraction = Extraction.loc[:,['HNAME','S5']]
    elif Distance in [2200,2400]:
        Extraction = Extraction.loc[:,['HNAME','S6']]
    Extraction.columns = ['HNAME','PP_PAF_STL_AVG_PFL']

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_PAF_STL_AVG_PFL'].fillna(Feature_DF.loc[:,'PP_PAF_STL_AVG_PFL'].min(), inplace = True)
    Feature_DF.loc[:,'PP_PAF_STL_AVG_PFL'].fillna(0, inplace=True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_STL_AVG_PFL']]

    return Feature_DF

"""
PP_PAF_STL_B_PFL
"""

def PP_PAF_STL_B_PFL(Dataframe, HNAME_List, Raceday):

    """
    Best Final Straight Line Speed on Distance
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_PAF_STL_B_PFL]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Extraction = Extraction_Database("""
                                     Select HNAME, max(PACE_S3) S3, max(PACE_S4) S4,
                                     max(PACE_S5) S5, max(PACE_S6) S6 from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RADIS = {Distance}
                                     Group by HNAME
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Distance=Distance))

    if Distance in [1000,1200]:
        Extraction = Extraction.loc[:,['HNAME','S3']]
    elif Distance in [1400,1600,1650]:
        Extraction = Extraction.loc[:,['HNAME','S4']]
    elif Distance in [1800,2000]:
        Extraction = Extraction.loc[:,['HNAME','S5']]
    elif Distance in [2200,2400]:
        Extraction = Extraction.loc[:,['HNAME','S6']]
    Extraction.columns = ['HNAME','PP_PAF_STL_B_PFL']

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_PAF_STL_B_PFL'].fillna(Feature_DF.loc[:,'PP_PAF_STL_B_PFL'].min(), inplace = True)
    Feature_DF.loc[:,'PP_PAF_STL_B_PFL'].fillna(0, inplace=True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_STL_B_PFL']]

    return Feature_DF

"""
PP_PAF_BEST
"""

def PP_PAF_BEST(Dataframe, HNAME_List, Raceday):

    """
    Sum of fastest section time
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_PAF_BEST]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Extraction = Extraction_Database("""
                                     Select HNAME, max(PACE_S1) S1, max(PACE_S2) S2, max(PACE_S3) S3,
                                     max(PACE_S4) S4, max(PACE_S5) S5, max(PACE_S6) S6 from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RADIS = {Distance}
                                     Group by HNAME
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Distance=Distance))

    if Distance in [1000,1200]:
        Extraction.loc[:,'PP_PAF_BEST'] = Extraction.loc[:,['S1','S2','S3']].sum(axis=1)
    elif Distance in [1400,1600,1650]:
        Extraction.loc[:,'PP_PAF_BEST'] = Extraction.loc[:,['S1','S2','S3','S4']].sum(axis=1)
    elif Distance in [1800,2000]:
        Extraction.loc[:,'PP_PAF_BEST'] = Extraction.loc[:,['S1','S2','S3','S4','S5']].sum(axis=1)
    elif Distance in [2200,2400]:
        Extraction.loc[:,'PP_PAF_BEST'] = Extraction.loc[:,['S1','S2','S3','S4','S5','S6']].sum(axis=1)

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_PAF_BEST'].fillna(Feature_DF.loc[:,'PP_PAF_BEST'].min(), inplace = True)
    Feature_DF.loc[:,'PP_PAF_BEST'].fillna(0, inplace=True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_BEST']]

    return Feature_DF


"""
PP_PAF_BEST_GOPFL
"""

def PP_PAF_BEST_GOPFL(Dataframe, HNAME_List, Raceday):

    """
    Best Final Straight Line Speed on Distance
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_PAF_BEST_GOPFL]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, max(PACE_S2, PACE_S3, PACE_S4, PACE_S5, PACE_S6) PP_PAF_BEST_GOPFL
                                     from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     Group by HNAME
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'PP_PAF_BEST_GOPFL'].fillna(Feature_DF.loc[:,'PP_PAF_BEST_GOPFL'].min(), inplace = True)
    Feature_DF.loc[:,'PP_PAF_BEST_GOPFL'].fillna(0, inplace=True)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_BEST_GOPFL']]

    return Feature_DF

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

========================= Earnings / Price Money =========================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
PP_EPM
"""

def PP_EPM(Dataframe, HNAME_List, Raceday):

    """
    Cumulative Price Money earnings in History
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_EPM]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, sum(RESWB * 10 * RESWL) + sum(RESWB * 10) PP_EPM from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RESFP <= 3
                                     Group by HNAME
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_EPM']].fillna(0)

    return Feature_DF

"""
PP_EPM_AVG
"""

def PP_EPM_AVG(Dataframe, HNAME_List, Raceday):

    """
    Average Price Money
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_EPM_AVG]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, PP_EPM, Num_Races from (
                                     Select HNAME, count(RARID) Num_Races from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     Group by HNAME) RACE
                                     LEFT OUTER JOIN
                                     (Select HNAME HNAME_EPM, sum(RESWB * 10 * RESWL) + sum(RESWB * 10) PP_EPM from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RESFP <= 3
                                     Group by HNAME) EPM
                                     ON RACE.HNAME = EPM.HNAME_EPM
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:, 'PP_EPM_AVG'] = Feature_DF.loc[:,'PP_EPM'] / Feature_DF.loc[:,'Num_Races']
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_EPM_AVG']].fillna(0)

    return Feature_DF

"""
PP_EMP_AVG_WIN
"""

def PP_EMP_AVG_WIN(Dataframe, HNAME_List, Raceday):

    """
    Average Winning Price Money
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_EMP_AVG_WIN]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, Win_EPM, Num_Win from(
                                     Select HNAME HNAME_EPM, sum(RESWB * 10 * RESWL)  Win_EPM from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     Group by HNAME) EPM
                                     LEFT OUTER JOIN
                                     (Select HNAME, sum(RESWL) Num_Win from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     Group by HNAME) NUM
                                     ON NUM.HNAME = EPM.HNAME_EPM
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:, 'PP_EMP_AVG_WIN'] = Feature_DF.loc[:,'Win_EPM'] / Feature_DF.loc[:,'Num_Win']

    Feature_DF = Feature_DF.loc[:,['HNAME','PP_EMP_AVG_WIN']].fillna(0)

    return Feature_DF

"""
PP_EMP_AVG_PLA
"""

def PP_EMP_AVG_PLA(Dataframe, HNAME_List, Raceday):

    """
    Average Winning Place Money
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_EMP_AVG_PLA]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, Place_EPM, T3 from (
                                     Select HNAME HNAME_EPM,  sum(RESWB * 10)  Place_EPM from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RESFP <= 3
                                     Group by HNAME) EPM
                                     LEFT OUTER JOIN
                                     (Select HNAME, count(RARID) T3 from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List} and RESFP <= 3
                                     Group by HNAME) T
                                     ON EPM.HNAME_EPM = T.HNAME
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:, 'PP_EMP_AVG_PLA'] = Feature_DF.loc[:,'Place_EPM'] / Feature_DF.loc[:,'T3']
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_EMP_AVG_PLA']].fillna(0)

    return Feature_DF

"""
PP_EMP_YR
"""

def PP_EMP_YR(Dataframe, HNAME_List, Raceday):

    """
    Cumulative Price Money earnings in one year
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, PP_EMP_YR]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]
    Offset_Raceday = (pd.to_datetime(Raceday) + pd.tseries.offsets.DateOffset(months=-12)).strftime("%Y%m%d")

    Extraction = Extraction_Database("""
                                     Select HNAME, sum(RESWB * 10 * RESWL) + sum(RESWB * 10) PP_EMP_YR from RaceDb
                                     where RADAT < {Raceday} and RADAT > {Offset_Raceday} and HNAME in {HNAME_List} and RESFP <= 3
                                     Group by HNAME
                                     """.format(Raceday = Raceday, Offset_Raceday = Offset_Raceday, HNAME_List = HNAME_List))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_EMP_YR']].fillna(0)

    return Feature_DF
