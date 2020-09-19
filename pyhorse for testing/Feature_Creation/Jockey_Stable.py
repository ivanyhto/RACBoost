#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ivan
"""

"""
Feature : Jocky and Stable
"""

#Loading Libraries
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from pyhorse.Database_Management import Extraction_Database

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

============================ Support Functions ============================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

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

================================= Jockey =================================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
JS_J_FP
"""

def JS_J_FP(Dataframe, HNAME_List, Raceday):

    """
    Jockey's Average Finishing Position in History
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, JS_J_FP]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JNAME']]
    JNAME_List = '('+str(Dataframe.loc[:,'JNAME'].tolist())[1:-1]+')'

    Extraction = Extraction_Database("""
                                     Select JNAME, RARID, RESFP JS_J_FP from RaceDb
                                     where RARID in
                                     (Select Distinct RARID from RaceDb
                                     where RADAT < {Raceday} and JNAME in {JNAME_List})
                                     """.format(Raceday = Raceday, JNAME_List = JNAME_List))

    if len(Extraction) == 0:
        Feature_DF.loc[:,'JS_J_FP'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_FP']]
        return Feature_DF

    races = Extraction.groupby('RARID')
    mean = races.transform(np.mean).loc[:,'JS_J_FP']
    std = races.transform(np.std).loc[:,'JS_J_FP']
    Extraction.loc[:,'JS_J_FP'] = (Extraction.loc[:,'JS_J_FP'] - mean) / std
    Extraction = Extraction.groupby('JNAME').mean().loc[:,'JS_J_FP'].reset_index()

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'JS_J_FP'].fillna(Feature_DF.loc[:,'JS_J_FP'].max(), inplace = True)
    Feature_DF.loc[:,'JS_J_FP'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_FP']]

    return Feature_DF

"""
JS_J_FPRW
"""

def JS_J_FPRW(Dataframe, HNAME_List, Raceday):

    """
    Recency Weighted Avg Finishing Position of Jockey
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, JS_J_FPRW]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JNAME']]
    JNAME_List = '('+str(Dataframe.loc[:,'JNAME'].tolist())[1:-1]+')'

    Extraction = Extraction_Database("""
                                     Select JNAME, RARID, RESFP JS_J_FPRW from RaceDb
                                     where RARID in
                                     (Select Distinct RARID from RaceDb
                                     where RADAT < {Raceday} and JNAME in {JNAME_List})
                                     """.format(Raceday = Raceday, JNAME_List = JNAME_List))
    if len(Extraction) == 0:
        Feature_DF.loc[:,'JS_J_FPRW'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_FPRW']]
        return Feature_DF

    races = Extraction.groupby('RARID')
    mean = races.transform(np.mean).loc[:,'JS_J_FPRW']
    std = races.transform(np.std).loc[:,'JS_J_FPRW']
    Extraction.loc[:,'JS_J_FPRW'] = (Extraction.loc[:,'JS_J_FPRW'] - mean) / std

    FP = []
    for name, group in Extraction.groupby('JNAME'):
        FP_History = group.loc[:,'JS_J_FPRW'].dropna().values
        if len(FP_History) >1:
            model = SimpleExpSmoothing(FP_History)
            model = model.fit()
            FP.append([name, model.forecast()[0]])
        elif len(FP_History) == 1:
            FP.append([name,FP_History[0]])
        else :
            FP.append([name,0])
    Avg_FP = pd.DataFrame(FP, columns=['JNAME','JS_J_FPRW'])

    Feature_DF = Feature_DF.merge(Avg_FP, how='left')
    Feature_DF.loc[:,'JS_J_FPRW'].fillna(Feature_DF.loc[:,'JS_J_FPRW'].min(), inplace = True)
    Feature_DF.loc[:,'JS_J_FPRW'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_FPRW']]

    return Feature_DF

"""
JS_J_WINP
"""

def JS_J_WINP(Dataframe, HNAME_List, Raceday):

    """
    Jockey's Winning Percenntage in History
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, JS_J_WINP]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JNAME']]
    JNAME_List = '('+str(Dataframe.loc[:,'JNAME'].tolist())[1:-1]+')'

    Extraction = Extraction_Database("""
                                     Select JNAME, sum(RESWL) Num_Win,  count(RARID) Num_Races from RaceDb
                                     where RADAT < {Raceday} and JNAME in {JNAME_List}
                                     Group by JNAME
                                     """.format(Raceday = Raceday, JNAME_List = JNAME_List))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'JS_J_WINP'] = Feature_DF.loc[:,'Num_Win'] / Feature_DF.loc[:,'Num_Races']
    Feature_DF.loc[:,'JS_J_WINP'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_WINP']]

    return Feature_DF

"""
JS_J_WINP_JDIST
"""

def JS_J_WINP_JDIST(Dataframe, HNAME_List, Raceday):

    """
    Jockey's Winning Percenntage in History on Distance
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, JS_J_WINP_JDIST]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JNAME']]
    JNAME_List = '('+str(Dataframe.loc[:,'JNAME'].tolist())[1:-1]+')'
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Extraction = Extraction_Database("""
                                     Select JNAME, sum(RESWL) Num_Win,  count(RARID) Num_Races from RaceDb
                                     where RADAT < {Raceday} and JNAME in {JNAME_List} and RADIS = {Distance}
                                     Group by JNAME
                                     """.format(Raceday = Raceday, JNAME_List = JNAME_List, Distance = Distance))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'JS_J_WINP_JDIST'] = Feature_DF.loc[:,'Num_Win'] / Feature_DF.loc[:,'Num_Races']
    Feature_DF.loc[:,'JS_J_WINP_JDIST'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_WINP_JDIST']]

    return Feature_DF

"""
JS_J_WINP_JGO
"""

def JS_J_WINP_JGO(Dataframe, HNAME_List, Raceday):

    """
    Jockey's Winning Percenntage in History on Similar Going
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, JS_J_WINP_JGO]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JNAME']]
    JNAME_List = '('+str(Dataframe.loc[:,'JNAME'].tolist())[1:-1]+')'
    Surface = "'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Going = Dataframe.loc[:,'RAGOG'].values[0].strip()
    Going_Dict = Going_Similarity(Going)

    Extraction = Extraction_Database("""
                                     Select JNAME, sum(RESWL) Num_Win,  count(RARID) Num_Races, RAGOG from RaceDb
                                     where RADAT < {Raceday} and JNAME in {JNAME_List} and RATRA = {Surface}
                                     Group by JNAME, RAGOG
                                     """.format(Raceday = Raceday, JNAME_List = JNAME_List, Surface = Surface))

    if len(Extraction) == 0:
        Feature_DF.loc[:,'JS_J_WINP_JGO'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_WINP_JGO']]
        return Feature_DF

    Extraction.loc[:,'RAGOG'] = Extraction.loc[:,'RAGOG'].map(lambda x : x.strip())
    Extraction.replace({'RAGOG': Going_Dict}, inplace = True)
    Extraction.loc[:,'WINP'] = Extraction.loc[:,'Num_Win'] / Extraction.loc[:,'Num_Races']

    def Normalise(group):
        group.loc[:,'Norm'] = np.exp(group.loc[:,'RAGOG']) / np.exp(group.loc[:,'RAGOG']).sum()
        group.loc[:,'Normed'] = group.loc[:,'Norm'] * group.loc[:,'WINP']
        return group.loc[:,'Normed'].sum()

    Extraction = Extraction.groupby('JNAME').apply(Normalise).reset_index()
    Extraction.columns = ['JNAME','JS_J_WINP_JGO']
    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'JS_J_WINP_JGO'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_WINP_JGO']]

    return Feature_DF

"""
JS_J_WINP_JSUR
"""

def JS_J_WINP_JSUR(Dataframe, HNAME_List, Raceday):

    """
    Jockey's Winning Percenntage in History on Surface
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, JS_J_WINP_JSUR]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JNAME']]
    JNAME_List = '('+str(Dataframe.loc[:,'JNAME'].tolist())[1:-1]+')'
    Surface = "'" + Dataframe.loc[:,'RATRA'].values[0] + "'"

    Extraction = Extraction_Database("""
                                     Select JNAME, sum(RESWL) Num_Win,  count(RARID) Num_Races from RaceDb
                                     where RADAT < {Raceday} and JNAME in {JNAME_List} and RATRA = {Surface}
                                     Group by JNAME
                                     """.format(Raceday = Raceday, JNAME_List = JNAME_List, Surface = Surface))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'JS_J_WINP_JSUR'] = Feature_DF.loc[:,'Num_Win'] / Feature_DF.loc[:,'Num_Races']
    Feature_DF.loc[:,'JS_J_WINP_JSUR'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_WINP_JSUR']]

    return Feature_DF

"""
JS_J_WINP_JLOC
"""

def JS_J_WINP_JLOC(Dataframe, HNAME_List, Raceday):

    """
    Jockey's Winning Percenntage in History on Location
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, JS_J_WINP_JLOC]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JNAME']]
    JNAME_List = '('+str(Dataframe.loc[:,'JNAME'].tolist())[1:-1]+')'
    Location = "'" + Dataframe.loc[:,'RALOC'].values[0] + "'"

    Extraction = Extraction_Database("""
                                     Select JNAME, sum(RESWL) Num_Win,  count(RARID) Num_Races from RaceDb
                                     where RADAT < {Raceday} and JNAME in {JNAME_List} and RALOC = {Location}
                                     Group by JNAME
                                     """.format(Raceday = Raceday, JNAME_List = JNAME_List, Location = Location))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'JS_J_WINP_JLOC'] = Feature_DF.loc[:,'Num_Win'] / Feature_DF.loc[:,'Num_Races']
    Feature_DF.loc[:,'JS_J_WINP_JLOC'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_WINP_JLOC']]

    return Feature_DF

"""
JS_J_WINP_JPFL
"""

def JS_J_WINP_JPFL(Dataframe, HNAME_List, Raceday):

    """
    Jockey's Winning Percenntage in History on Profile
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, JS_J_WINP_JPFL]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JNAME']]
    JNAME_List = '('+str(Dataframe.loc[:,'JNAME'].tolist())[1:-1]+')'
    Location = "'" + Dataframe.loc[:,'RALOC'].values[0] + "'"
    Surface = "'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Extraction = Extraction_Database("""
                                     Select JNAME, sum(RESWL) Num_Win,  count(RARID) Num_Races from RaceDb
                                     where RADAT < {Raceday} and JNAME in {JNAME_List}
                                     and RALOC = {Location} and RATRA = {Surface} and RADIS = {Distance}
                                     Group by JNAME
                                     """.format(Raceday = Raceday, JNAME_List = JNAME_List,
                                     Location = Location, Surface = Surface, Distance = Distance))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'JS_J_WINP_JPFL'] = Feature_DF.loc[:,'Num_Win'] / Feature_DF.loc[:,'Num_Races']
    Feature_DF.loc[:,'JS_J_WINP_JPFL'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_WINP_JPFL']]

    return Feature_DF

"""
JS_J_T3P
"""

def JS_J_T3P(Dataframe, HNAME_List, Raceday):

    """
    Jockey's Top 3 Percenntage in History
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, JS_J_T3P]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JNAME']]
    JNAME_List = '('+str(Dataframe.loc[:,'JNAME'].tolist())[1:-1]+')'

    Extraction = Extraction_Database("""
                                     Select JNAME, Num_Races, T3 from (
                                     Select JNAME, count(RARID) Num_Races from RaceDb
                                     where RADAT < {Raceday} and JNAME in {JNAME_List}
                                     Group by JNAME) NUM
                                     LEFT OUTER JOIN
                                     (Select JNAME JNAME_T, count(RARID) T3 from RaceDb
                                     where RADAT < {Raceday} and JNAME in {JNAME_List} and RESFP <= 3
                                     Group by JNAME) T
                                     ON T.JNAME_T = NUM.JNAME
                                     """.format(Raceday = Raceday, JNAME_List = JNAME_List))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'JS_J_T3P'] = Feature_DF.loc[:,'T3'] / Feature_DF.loc[:,'Num_Races']
    Feature_DF.loc[:,'JS_J_T3P'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_T3P']]

    return Feature_DF

"""
JS_J_T3P_JDIST
"""

def JS_J_T3P_JDIST(Dataframe, HNAME_List, Raceday):

    """
    Jockey's Top 3 Percenntage in History on Distance
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, JS_J_T3P_JDIST]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JNAME']]
    JNAME_List = '('+str(Dataframe.loc[:,'JNAME'].tolist())[1:-1]+')'
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Extraction = Extraction_Database("""
                                     Select JNAME, Num_Races, T3 from (
                                     Select JNAME, count(RARID) Num_Races from RaceDb
                                     where RADAT < {Raceday} and JNAME in {JNAME_List} and RADIS = {Distance}
                                     Group by JNAME) NUM
                                     LEFT OUTER JOIN
                                     (Select JNAME JNAME_T, count(RARID) T3 from RaceDb
                                     where RADAT < {Raceday} and JNAME in {JNAME_List} and RADIS = {Distance} and RESFP <= 3
                                     Group by JNAME) T
                                     ON T.JNAME_T = NUM.JNAME
                                     """.format(Raceday = Raceday, JNAME_List = JNAME_List, Distance = Distance))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'JS_J_T3P_JDIST'] = Feature_DF.loc[:,'T3'] / Feature_DF.loc[:,'Num_Races']
    Feature_DF.loc[:,'JS_J_T3P_JDIST'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_T3P_JDIST']]

    return Feature_DF

"""
JS_J_T3P_JGO
"""

def JS_J_T3P_JGO(Dataframe, HNAME_List, Raceday):

    """
    Jockey's Top 3 Percenntage in History on Going
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, JS_J_T3P_JGO]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JNAME']]
    JNAME_List = '('+str(Dataframe.loc[:,'JNAME'].tolist())[1:-1]+')'
    Surface = "'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Going = Dataframe.loc[:,'RAGOG'].values[0].strip()
    Going_Dict = Going_Similarity(Going)

    Extraction = Extraction_Database("""
                                     Select JNAME, Num_Races, T3, RAGOG from (
                                     Select JNAME, count(RARID) Num_Races, RAGOG from RaceDb
                                     where RADAT < {Raceday} and JNAME in {JNAME_List} and RATRA = {Surface}
                                     Group by JNAME, RAGOG) NUM
                                     LEFT OUTER JOIN
                                     (Select JNAME JNAME_T, count(RARID) T3, RAGOG RAGOG_T3 from RaceDb
                                     where RADAT < {Raceday} and JNAME in {JNAME_List} and RATRA = {Surface} and RESFP <= 3
                                     Group by JNAME, RAGOG_T3) T3
                                     ON NUM.JNAME = T3.JNAME_T and NUM.RAGOG = T3.RAGOG_T3
                                     """.format(Raceday = Raceday, JNAME_List = JNAME_List, Surface = Surface))
    if len(Extraction) == 0:
        Feature_DF.loc[:,'JS_J_T3P_JGO'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_T3P_JGO']]
        return Feature_DF

    Extraction.loc[:,'RAGOG'] = Extraction.loc[:,'RAGOG'].map(lambda x : x.strip())
    Extraction.replace({'RAGOG': Going_Dict}, inplace = True)
    Extraction.loc[:,'T3P'] = Extraction.loc[:,'T3'] / Extraction.loc[:,'Num_Races']

    def Normalise(group):
        group.loc[:,'Norm'] = np.exp(group.loc[:,'RAGOG']) / np.exp(group.loc[:,'RAGOG']).sum()
        group.loc[:,'Normed'] = group.loc[:,'Norm'] * group.loc[:,'T3P']
        return group.loc[:,'Normed'].sum()

    Extraction = Extraction.groupby('JNAME').apply(Normalise).reset_index()
    Extraction.columns = ['JNAME','JS_J_T3P_JGO']
    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'JS_J_T3P_JGO'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_T3P_JGO']]

    return Feature_DF

"""
JS_J_T3P_JSUR
"""

def JS_J_T3P_JSUR(Dataframe, HNAME_List, Raceday):

    """
    Jockey's Top 3 Percenntage in History on Surface
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, JS_J_T3P_JSUR]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JNAME']]
    JNAME_List = '('+str(Dataframe.loc[:,'JNAME'].tolist())[1:-1]+')'
    Surface = "'" + Dataframe.loc[:,'RATRA'].values[0] + "'"

    Extraction = Extraction_Database("""
                                     Select JNAME, Num_Races, T3 from (
                                     Select JNAME, count(RARID) Num_Races from RaceDb
                                     where RADAT < {Raceday} and JNAME in {JNAME_List} and RATRA = {Surface}
                                     Group by JNAME) NUM
                                     LEFT OUTER JOIN
                                     (Select JNAME JNAME_T, count(RARID) T3 from RaceDb
                                     where RADAT < {Raceday} and JNAME in {JNAME_List} and RATRA = {Surface} and RESFP <= 3
                                     Group by JNAME) T
                                     ON T.JNAME_T = NUM.JNAME
                                     """.format(Raceday = Raceday, JNAME_List = JNAME_List, Surface = Surface))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'JS_J_T3P_JSUR'] = Feature_DF.loc[:,'T3'] / Feature_DF.loc[:,'Num_Races']
    Feature_DF.loc[:,'JS_J_T3P_JSUR'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_T3P_JSUR']]

    return Feature_DF

"""
JS_J_T3P_JLOC
"""

def JS_J_T3P_JLOC(Dataframe, HNAME_List, Raceday):

    """
    Jockey's Top 3 Percenntage in History on Location
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, JS_J_T3P_JLOC]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JNAME']]
    JNAME_List = '('+str(Dataframe.loc[:,'JNAME'].tolist())[1:-1]+')'
    Location = "'" + Dataframe.loc[:,'RALOC'].values[0] + "'"

    Extraction = Extraction_Database("""
                                     Select JNAME, Num_Races, T3 from (
                                     Select JNAME, count(RARID) Num_Races from RaceDb
                                     where RADAT < {Raceday} and JNAME in {JNAME_List} and RALOC = {Location}
                                     Group by JNAME) NUM
                                     LEFT OUTER JOIN
                                     (Select JNAME JNAME_T, count(RARID) T3 from RaceDb
                                     where RADAT < {Raceday} and JNAME in {JNAME_List} and RALOC = {Location} and RESFP <= 3
                                     Group by JNAME) T
                                     ON T.JNAME_T = NUM.JNAME
                                     """.format(Raceday = Raceday, JNAME_List = JNAME_List, Location = Location))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'JS_J_T3P_JLOC'] = Feature_DF.loc[:,'T3'] / Feature_DF.loc[:,'Num_Races']
    Feature_DF.loc[:,'JS_J_T3P_JLOC'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_T3P_JLOC']]

    return Feature_DF

"""
JS_J_T3P_JPFL
"""

def JS_J_T3P_JPFL(Dataframe, HNAME_List, Raceday):

    """
    Jockey's Top 3 Percenntage in History on Profile
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, JS_J_T3P_JPFL]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JNAME']]
    JNAME_List = '('+str(Dataframe.loc[:,'JNAME'].tolist())[1:-1]+')'
    Location = "'" + Dataframe.loc[:,'RALOC'].values[0] + "'"
    Surface = "'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Extraction = Extraction_Database("""
                                     Select JNAME, Num_Races, T3 from (
                                     Select JNAME, count(RARID) Num_Races from RaceDb
                                     where RADAT < {Raceday} and JNAME in {JNAME_List}
                                     and RALOC = {Location} and RATRA = {Surface} and RADIS = {Distance}
                                     Group by JNAME) NUM
                                     LEFT OUTER JOIN
                                     (Select JNAME JNAME_T, count(RARID) T3 from RaceDb
                                     where RADAT < {Raceday} and JNAME in {JNAME_List} and RESFP <= 3
                                     and RALOC = {Location} and RATRA = {Surface} and RADIS = {Distance}
                                     Group by JNAME) T
                                     ON T.JNAME_T = NUM.JNAME
                                     """.format(Raceday = Raceday, JNAME_List = JNAME_List,
                                     Location = Location, Surface = Surface, Distance = Distance))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'JS_J_T3P_JPFL'] = Feature_DF.loc[:,'T3'] / Feature_DF.loc[:,'Num_Races']
    Feature_DF.loc[:,'JS_J_T3P_JPFL'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_T3P_JPFL']]

    return Feature_DF

"""
JS_J_NUMR
"""

def JS_J_NUMR(Dataframe, HNAME_List, Raceday):

    """
    Number of Past Races Ran by Jockey
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, JS_J_NUMR]
    """
    Feature_DF = Dataframe.loc[:,['HNAME','JNAME']]

    JNAME_List = '('+str(Dataframe.loc[:,'JNAME'].tolist())[1:-1]+')'

    Extraction = Extraction_Database("""
                                     Select JNAME, count(RARID) JS_J_NUMR from RaceDb
                                     where RADAT < {Raceday} and JNAME in {JNAME_List}
                                     Group by JNAME
                                     """.format(Raceday = Raceday, JNAME_List = JNAME_List))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'JS_J_NUMR'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_NUMR']]

    return Feature_DF

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

=========================== Jockey Combinations ===========================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
JS_J_HJ_NUM
"""

def JS_J_HJ_NUM(Dataframe, HNAME_List, Raceday):

    """
    Jockey Horse Number of Runs
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, JS_J_HJ_NUM]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JNAME']]
    Horse_Jockey_Combi = ["(HNAME = '" + row['HNAME'] + "' and JNAME = '" + str(row['JNAME']) + "')" for index, row in Feature_DF.iterrows()]
    Horse_Jockey_Combi = ' or '.join(Horse_Jockey_Combi)
    Horse_Jockey_Combi  = '(' + Horse_Jockey_Combi + ')'

    Extraction = Extraction_Database("""
                                     Select HNAME, count(RARID) JS_J_HJ_NUM from RaceDb
                                     where RADAT < {Raceday} and {Horse_Jockey_Combi}
                                     Group by HNAME
                                     """.format(Raceday = Raceday, Horse_Jockey_Combi = Horse_Jockey_Combi))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'JS_J_HJ_NUM'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_HJ_NUM']]

    return Feature_DF

"""
JS_J_HJ_NWIN
"""

def JS_J_HJ_NWIN(Dataframe, HNAME_List, Raceday):

    """
    Jockey Horse Number of Wins
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, JS_J_HJ_NWIN]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JNAME']]
    Horse_Jockey_Combi = ["(HNAME = '" + row['HNAME'] + "' and JNAME = '" + str(row['JNAME']) + "')" for index, row in Feature_DF.iterrows()]
    Horse_Jockey_Combi = ' or '.join(Horse_Jockey_Combi)
    Horse_Jockey_Combi  = '(' + Horse_Jockey_Combi + ')'

    Extraction = Extraction_Database("""
                                     Select HNAME, sum(RESWL) JS_J_HJ_NWIN from RaceDb
                                     where RADAT < {Raceday} and {Horse_Jockey_Combi}
                                     Group by HNAME
                                     """.format(Raceday = Raceday, Horse_Jockey_Combi = Horse_Jockey_Combi))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'JS_J_HJ_NWIN'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_HJ_NWIN']]

    return Feature_DF

"""
JS_J_HJ_NT3
"""

def JS_J_HJ_NT3(Dataframe, HNAME_List, Raceday):

    """
    Jockey Horse Number of T3
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, JS_J_HJ_NT3]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JNAME']]
    Horse_Jockey_Combi = ["(HNAME = '" + row['HNAME'] + "' and JNAME = '" + str(row['JNAME']) + "')" for index, row in Feature_DF.iterrows()]
    Horse_Jockey_Combi = ' or '.join(Horse_Jockey_Combi)
    Horse_Jockey_Combi  = '(' + Horse_Jockey_Combi + ')'

    Extraction = Extraction_Database("""
                                     Select HNAME, count(RARID) JS_J_HJ_NT3 from RaceDb
                                     where RADAT < {Raceday} and {Horse_Jockey_Combi} and RESFP <= 3
                                     Group by HNAME
                                     """.format(Raceday = Raceday, Horse_Jockey_Combi = Horse_Jockey_Combi))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'JS_J_HJ_NT3'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_HJ_NT3']]

    return Feature_DF

"""
JS_J_HJ_SPAVG
"""

def JS_J_HJ_SPAVG(Dataframe, HNAME_List, Raceday):

    """
    Jockey Horse Speed Rating
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, JS_J_HJ_SPAVG]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JNAME']]
    Horse_Jockey_Combi = ["(HNAME = '" + row['HNAME'] + "' and JNAME = '" + str(row['JNAME']) + "')" for index, row in Feature_DF.iterrows()]
    Horse_Jockey_Combi = ' or '.join(Horse_Jockey_Combi)
    Horse_Jockey_Combi  = '(' + Horse_Jockey_Combi + ')'

    Extraction = Extraction_Database("""
                                     Select HNAME, avg(BEYER_SPEED) JS_J_HJ_SPAVG from Race_PosteriorDb
                                     where RADAT < {Raceday} and {Horse_Jockey_Combi}
                                     Group by HNAME
                                     """.format(Raceday = Raceday, Horse_Jockey_Combi = Horse_Jockey_Combi))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'JS_J_HJ_SPAVG'].fillna(Feature_DF.loc[:,'JS_J_HJ_SPAVG'].min(), inplace = True)
    Feature_DF.loc[:,'JS_J_HJ_SPAVG'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_HJ_SPAVG']]

    return Feature_DF

"""
JS_J_HJ_CON
"""

def JS_J_HJ_CON(Dataframe, HNAME_List, Raceday):

    """
    Jockey’s contribution to the horse’s past performances.
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, JS_J_HJ_CON]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JNAME']]
    Horse_Jockey_Combi = ["(HNAME = '" + row['HNAME'] + "' and JNAME = '" + str(row['JNAME']) + "')" for index, row in Feature_DF.iterrows()]
    Horse_Jockey_Combi = ' or '.join(Horse_Jockey_Combi)
    Horse_Jockey_Combi  = '(' + Horse_Jockey_Combi + ')'

    Extraction = Extraction_Database("""
                                     Select HNAME, Combi, Horse from (
                                     Select HNAME , avg(BEYER_SPEED) Horse from Race_PosteriorDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     Group by HNAME) BEYER
                                     LEFT OUTER JOIN
                                     (Select HNAME HNAME_C, avg(BEYER_SPEED) Combi from Race_PosteriorDb
                                     where RADAT < {Raceday} and {Horse_Jockey_Combi}
                                     Group by HNAME) COMB
                                     ON BEYER.HNAME = COMB.HNAME_C
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List, Horse_Jockey_Combi = Horse_Jockey_Combi))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'Combi'].fillna(Feature_DF.loc[:,'Horse'], inplace = True)
    Feature_DF.loc[:,'JS_J_HJ_CON'] = Feature_DF.loc[:,'Combi'] - Feature_DF.loc[:,'Horse']
    Feature_DF.loc[:,'JS_J_HJ_CON'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_HJ_CON']]

    return Feature_DF

"""
JS_J_SJ_WIN
"""

def JS_J_SJ_WIN(Dataframe, HNAME_List, Raceday):

    """
    Jockey Stable Win Percentage
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, JS_J_SJ_WIN]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','SNAME', 'JNAME']]
    Stable_Jockey_Combi = ["(SNAME = '" + row['SNAME'] + "' and JNAME = '" + str(row['JNAME']) + "')" for index, row in Feature_DF.iterrows()]
    Stable_Jockey_Combi = ' or '.join(Stable_Jockey_Combi)
    Stable_Jockey_Combi  = '(' + Stable_Jockey_Combi + ')'

    Extraction = Extraction_Database("""
                                     Select JNAME, sum(RESWL) Num_Win,  count(RARID) Num_Races from RaceDb
                                     where RADAT < {Raceday} and {Stable_Jockey_Combi}
                                     Group by JNAME
                                     """.format(Raceday = Raceday, Stable_Jockey_Combi = Stable_Jockey_Combi))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'JS_J_SJ_WIN'] = Feature_DF['Num_Win'] / Feature_DF['Num_Races']
    Feature_DF.loc[:,'JS_J_SJ_WIN'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_SJ_WIN']]

    return Feature_DF

"""
JS_J_SJ_T3
"""

def JS_J_SJ_T3(Dataframe, HNAME_List, Raceday):

    """
    Jockey Stable Top 3 Percentage
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, JS_J_SJ_T3]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','SNAME', 'JNAME']]
    Stable_Jockey_Combi = ["(SNAME = '" + row['SNAME'] + "' and JNAME = '" + str(row['JNAME']) + "')" for index, row in Feature_DF.iterrows()]
    Stable_Jockey_Combi = ' or '.join(Stable_Jockey_Combi)
    Stable_Jockey_Combi  = '(' + Stable_Jockey_Combi + ')'

    Extraction = Extraction_Database("""
                                     Select JNAME, Num_Races, T3 from (
                                     Select JNAME, count(RARID) Num_Races from RaceDb
                                     where RADAT < {Raceday} and {Stable_Jockey_Combi}
                                     Group by JNAME) NUM
                                     LEFT OUTER JOIN
                                     (Select JNAME JNAME_T, count(RARID) T3 from RaceDb
                                     where RADAT < {Raceday} and {Stable_Jockey_Combi} and RESFP <= 3
                                     Group by JNAME) T
                                     ON T.JNAME_T = NUM.JNAME
                                     """.format(Raceday = Raceday, Stable_Jockey_Combi = Stable_Jockey_Combi))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'JS_J_SJ_T3'] = Feature_DF.loc[:,'T3'] / Feature_DF.loc[:,'Num_Races']
    Feature_DF.loc[:,'JS_J_SJ_T3'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_SJ_T3']]

    return Feature_DF

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

================================= Stable =================================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
JS_S_FP
"""

def JS_S_FP(Dataframe, HNAME_List, Raceday):

    """
    Stable's Average Finishing Position in History
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, JS_S_FP]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','SNAME']]
    SNAME_List = '('+str(Dataframe.loc[:,'SNAME'].tolist())[1:-1]+')'

    Extraction = Extraction_Database("""
                                     Select SNAME, RARID, RESFP JS_S_FP from RaceDb
                                     where RARID in
                                     (Select Distinct RARID from RaceDb
                                     where RADAT < {Raceday} and SNAME in {SNAME_List})
                                     """.format(Raceday = Raceday, SNAME_List = SNAME_List))

    if len(Extraction) == 0:
        Feature_DF.loc[:,'JS_S_FP'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','JS_S_FP']]
        return Feature_DF

    races = Extraction.groupby('RARID')
    mean = races.transform(np.mean).loc[:,'JS_S_FP']
    std = races.transform(np.std).loc[:,'JS_S_FP']
    Extraction.loc[:,'JS_S_FP'] = (Extraction.loc[:,'JS_S_FP'] - mean) / std
    Extraction = Extraction.groupby('SNAME').mean().loc[:,'JS_S_FP'].reset_index()

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'JS_S_FP'].fillna(Feature_DF.loc[:,'JS_S_FP'].max(), inplace = True)
    Feature_DF.loc[:,'JS_S_FP'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_S_FP']]

    return Feature_DF

"""
JS_S_FPRW
"""

def JS_S_FPRW(Dataframe, HNAME_List, Raceday):

    """
    Recency Weighted Avg Finishing Position of Stable
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, JS_S_FPRW]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','SNAME']]
    SNAME_List = '('+str(Dataframe.loc[:,'SNAME'].tolist())[1:-1]+')'

    Extraction = Extraction_Database("""
                                     Select SNAME, RARID, RESFP JS_S_FPRW from RaceDb
                                     where RARID in
                                     (Select Distinct RARID from RaceDb
                                     where RADAT < {Raceday} and SNAME in {SNAME_List})
                                     """.format(Raceday = Raceday, SNAME_List = SNAME_List))
    if len(Extraction) == 0:
        Feature_DF.loc[:,'JS_S_FPRW'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','JS_S_FPRW']]
        return Feature_DF

    races = Extraction.groupby('RARID')
    mean = races.transform(np.mean).loc[:,'JS_S_FPRW']
    std = races.transform(np.std).loc[:,'JS_S_FPRW']
    Extraction.loc[:,'JS_S_FPRW'] = (Extraction.loc[:,'JS_S_FPRW'] - mean) / std

    FP = []
    for name, group in Extraction.groupby('SNAME'):
        FP_History = group.loc[:,'JS_S_FPRW'].dropna().values
        if len(FP_History) >1:
            model = SimpleExpSmoothing(FP_History)
            model = model.fit()
            FP.append([name, model.forecast()[0]])
        elif len(FP_History) == 1:
            FP.append([name,FP_History[0]])
        else :
            FP.append([name,0])
    Avg_FP = pd.DataFrame(FP, columns=['SNAME','JS_S_FPRW'])

    Feature_DF = Feature_DF.merge(Avg_FP, how='left')
    Feature_DF.loc[:,'JS_S_FPRW'].fillna(Feature_DF.loc[:,'JS_S_FPRW'].min(), inplace = True)
    Feature_DF.loc[:,'JS_S_FPRW'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_S_FPRW']]

    return Feature_DF

"""
JS_S_WINP
"""

def JS_S_WINP(Dataframe, HNAME_List, Raceday):

    """
    Stable's Winning Percenntage in History
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, JS_S_WINP]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','SNAME']]
    SNAME_List = '('+str(Dataframe.loc[:,'SNAME'].tolist())[1:-1]+')'

    Extraction = Extraction_Database("""
                                     Select SNAME, sum(RESWL) Num_Win,  count(RARID) Num_Races from RaceDb
                                     where RADAT < {Raceday} and SNAME in {SNAME}
                                     Group by SNAME
                                     """.format(Raceday = Raceday, SNAME = SNAME_List))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'JS_S_WINP'] = Feature_DF.loc[:,'Num_Win'] / Feature_DF.loc[:,'Num_Races']
    Feature_DF.loc[:,'JS_S_WINP'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_S_WINP']]

    return Feature_DF

"""
JS_S_WINP_SDIST
"""

def JS_S_WINP_SDIST(Dataframe, HNAME_List, Raceday):

    """
    Stable's Winning Percenntage in History on Distance
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, JS_S_WINP_SDIST]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','SNAME']]
    SNAME_List = '('+str(Dataframe.loc[:,'SNAME'].tolist())[1:-1]+')'
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Extraction = Extraction_Database("""
                                     Select SNAME, sum(RESWL) Num_Win,  count(RARID) Num_Races from RaceDb
                                     where RADAT < {Raceday} and SNAME in {SNAME} and RADIS = {Distance}
                                     Group by SNAME
                                     """.format(Raceday = Raceday, SNAME = SNAME_List, Distance = Distance))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'JS_S_WINP_SDIST'] = Feature_DF.loc[:,'Num_Win'] / Feature_DF.loc[:,'Num_Races']
    Feature_DF.loc[:,'JS_S_WINP_SDIST'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_S_WINP_SDIST']]

    return Feature_DF

"""
JS_S_WINP_SGO
"""

def JS_S_WINP_SGO(Dataframe, HNAME_List, Raceday):

    """
    Stable's Top 3 Percenntage in History on Going
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, JS_S_WINP_SGO]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','SNAME']]
    SNAME_List = '('+str(Dataframe.loc[:,'SNAME'].tolist())[1:-1]+')'
    Surface = "'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Going = Dataframe.loc[:,'RAGOG'].values[0].strip()
    Going_Dict = Going_Similarity(Going)

    Extraction = Extraction_Database("""
                                     Select SNAME, Num_Races, T3, RAGOG from (
                                     Select SNAME, count(RARID) Num_Races, RAGOG from RaceDb
                                     where RADAT < {Raceday} and SNAME in {SNAME_List} and RATRA = {Surface}
                                     Group by SNAME, RAGOG) NUM
                                     LEFT OUTER JOIN
                                     (Select SNAME SNAME_T, count(RARID) T3, RAGOG RAGOG_T3 from RaceDb
                                     where RADAT < {Raceday} and SNAME in {SNAME_List} and RATRA = {Surface} and RESFP <= 3
                                     Group by SNAME, RAGOG_T3) T3
                                     ON NUM.SNAME = T3.SNAME_T and NUM.RAGOG = T3.RAGOG_T3
                                     """.format(Raceday = Raceday, SNAME_List = SNAME_List, Surface = Surface))
    if len(Extraction) == 0:
        Feature_DF.loc[:,'JS_S_WINP_SGO'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','JS_S_WINP_SGO']]
        return Feature_DF

    Extraction.loc[:,'RAGOG'] = Extraction.loc[:,'RAGOG'].map(lambda x : x.strip())
    Extraction.replace({'RAGOG': Going_Dict}, inplace = True)
    Extraction.loc[:,'T3P'] = Extraction.loc[:,'T3'] / Extraction.loc[:,'Num_Races']

    def Normalise(group):
        group.loc[:,'Norm'] = np.exp(group.loc[:,'RAGOG']) / np.exp(group.loc[:,'RAGOG']).sum()
        group.loc[:,'Normed'] = group.loc[:,'Norm'] * group.loc[:,'T3P']
        return group.loc[:,'Normed'].sum()

    Extraction = Extraction.groupby('SNAME').apply(Normalise).reset_index()
    Extraction.columns = ['SNAME','JS_S_WINP_SGO']
    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'JS_S_WINP_SGO'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_S_WINP_SGO']]

    return Feature_DF

"""
JS_S_WINP_SSUR
"""

def JS_S_WINP_SSUR(Dataframe, HNAME_List, Raceday):

    """
    Stable's Winning Percenntage in History on Surface
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, JS_S_WINP_SSUR]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','SNAME']]
    SNAME_List = '('+str(Dataframe.loc[:,'SNAME'].tolist())[1:-1]+')'
    Surface = "'" + Dataframe.loc[:,'RATRA'].values[0] + "'"

    Extraction = Extraction_Database("""
                                     Select SNAME, sum(RESWL) Num_Win,  count(RARID) Num_Races from RaceDb
                                     where RADAT < {Raceday} and SNAME in {SNAME} and RATRA = {Surface}
                                     Group by SNAME
                                     """.format(Raceday = Raceday, SNAME = SNAME_List, Surface = Surface))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'JS_S_WINP_SSUR'] = Feature_DF.loc[:,'Num_Win'] / Feature_DF.loc[:,'Num_Races']
    Feature_DF.loc[:,'JS_S_WINP_SSUR'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_S_WINP_SSUR']]

    return Feature_DF

"""
JS_S_WINP_SLOC
"""

def JS_S_WINP_SLOC(Dataframe, HNAME_List, Raceday):

    """
    Stable's Winning Percenntage in History on Location
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, JS_S_WINP_SLOC]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','SNAME']]
    SNAME_List = '('+str(Dataframe.loc[:,'SNAME'].tolist())[1:-1]+')'
    Location = "'" + Dataframe.loc[:,'RALOC'].values[0] + "'"

    Extraction = Extraction_Database("""
                                     Select SNAME, sum(RESWL) Num_Win,  count(RARID) Num_Races from RaceDb
                                     where RADAT < {Raceday} and SNAME in {SNAME} and RALOC = {Location}
                                     Group by SNAME
                                     """.format(Raceday = Raceday, SNAME = SNAME_List, Location = Location))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'JS_S_WINP_SLOC'] = Feature_DF.loc[:,'Num_Win'] / Feature_DF.loc[:,'Num_Races']
    Feature_DF.loc[:,'JS_S_WINP_SLOC'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_S_WINP_SLOC']]

    return Feature_DF

"""
JS_S_WINP_SPFL
"""

def JS_S_WINP_SPFL(Dataframe, HNAME_List, Raceday):

    """
    Stable's Winning Percenntage in History on Profile
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, JS_S_WINP_SPFL]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','SNAME']]
    SNAME_List = '('+str(Dataframe.loc[:,'SNAME'].tolist())[1:-1]+')'
    Location = "'" + Dataframe.loc[:,'RALOC'].values[0] + "'"
    Surface = "'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Extraction = Extraction_Database("""
                                     Select SNAME, sum(RESWL) Num_Win,  count(RARID) Num_Races from RaceDb
                                     where RADAT < {Raceday} and SNAME in {SNAME}
                                     and RALOC = {Location} and RATRA = {Surface} and RADIS = {Distance}
                                     Group by SNAME
                                     """.format(Raceday = Raceday, SNAME = SNAME_List,
                                     Location = Location, Surface = Surface, Distance = Distance))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'JS_S_WINP_SPFL'] = Feature_DF.loc[:,'Num_Win'] / Feature_DF.loc[:,'Num_Races']
    Feature_DF.loc[:,'JS_S_WINP_SPFL'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_S_WINP_SPFL']]

    return Feature_DF

"""
JS_S_T3P
"""

def JS_S_T3P(Dataframe, HNAME_List, Raceday):

    """
    Stable's Top 3 Percenntage in History
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, JS_J_T3P]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','SNAME']]
    SNAME_List = '('+str(Dataframe.loc[:,'SNAME'].tolist())[1:-1]+')'

    Extraction = Extraction_Database("""
                                     Select SNAME, Num_Races, T3 from (
                                     Select SNAME, count(RARID) Num_Races from RaceDb
                                     where RADAT < {Raceday} and SNAME in {SNAME_List}
                                     Group by SNAME) NUM
                                     LEFT OUTER JOIN
                                     (Select SNAME SNAME_T, count(RARID) T3 from RaceDb
                                     where RADAT < {Raceday} and SNAME in {SNAME_List}  and RESFP <= 3
                                     Group by SNAME) T
                                     ON T.SNAME_T = NUM.SNAME
                                     """.format(Raceday = Raceday, SNAME_List = SNAME_List))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'JS_S_T3P'] = Feature_DF.loc[:,'T3'] / Feature_DF.loc[:,'Num_Races']
    Feature_DF.loc[:,'JS_S_T3P'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_S_T3P']]

    return Feature_DF

"""
JS_S_T3P_SDIST
"""

def JS_S_T3P_SDIST(Dataframe, HNAME_List, Raceday):

    """
    Stable's Top 3 Percenntage in History on Distance
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, JS_S_T3P_SDIST]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','SNAME']]
    SNAME_List = '('+str(Dataframe.loc[:,'SNAME'].tolist())[1:-1]+')'
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Extraction = Extraction_Database("""
                                     Select SNAME, Num_Races, T3 from (
                                     Select SNAME, count(RARID) Num_Races from RaceDb
                                     where RADAT < {Raceday} and SNAME in {SNAME_List} and RADIS = {Distance}
                                     Group by SNAME) NUM
                                     LEFT OUTER JOIN
                                     (Select SNAME SNAME_T, count(RARID) T3 from RaceDb
                                     where RADAT < {Raceday} and SNAME in {SNAME_List} and RADIS = {Distance} and RESFP <= 3
                                     Group by SNAME) T
                                     ON T.SNAME_T = NUM.SNAME
                                     """.format(Raceday = Raceday, SNAME_List = SNAME_List, Distance = Distance))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'JS_S_T3P_SDIST'] = Feature_DF.loc[:,'T3'] / Feature_DF.loc[:,'Num_Races']
    Feature_DF.loc[:,'JS_S_T3P_SDIST'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_S_T3P_SDIST']]

    return Feature_DF

"""
JS_S_T3P_SGO
"""

def JS_S_T3P_SGO(Dataframe, HNAME_List, Raceday):

    """
    Stable's Winning Percenntage in History on Similar Going
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, JS_S_T3P_SGO]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','SNAME']]
    SNAME_List = '('+str(Dataframe.loc[:,'SNAME'].tolist())[1:-1]+')'
    Surface = "'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Going = Dataframe.loc[:,'RAGOG'].values[0].strip()
    Going_Dict = Going_Similarity(Going)

    Extraction = Extraction_Database("""
                                     Select SNAME, sum(RESWL) Num_Win,  count(RARID) Num_Races, RAGOG from RaceDb
                                     where RADAT < {Raceday} and SNAME in {SNAME_List} and RATRA = {Surface}
                                     Group by SNAME, RAGOG
                                     """.format(Raceday = Raceday, SNAME_List = SNAME_List, Surface = Surface))
    if len(Extraction) == 0:
        Feature_DF.loc[:,'JS_S_T3P_SGO'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','JS_S_T3P_SGO']]
        return Feature_DF

    Extraction.loc[:,'RAGOG'] = Extraction.loc[:,'RAGOG'].map(lambda x : x.strip())
    Extraction.replace({'RAGOG': Going_Dict}, inplace = True)
    Extraction.loc[:,'WINP'] = Extraction.loc[:,'Num_Win'] / Extraction.loc[:,'Num_Races']

    def Normalise(group):
        group.loc[:,'Norm'] = np.exp(group.loc[:,'RAGOG']) / np.exp(group.loc[:,'RAGOG']).sum()
        group.loc[:,'Normed'] = group.loc[:,'Norm'] * group.loc[:,'WINP']
        return group.loc[:,'Normed'].sum()

    Extraction = Extraction.groupby('SNAME').apply(Normalise).reset_index()
    Extraction.columns = ['SNAME','JS_S_T3P_SGO']
    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'JS_S_T3P_SGO'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_S_T3P_SGO']]

    return Feature_DF

"""
JS_S_T3P_SSUR
"""

def JS_S_T3P_SSUR(Dataframe, HNAME_List, Raceday):

    """
    Stable's Top 3 Percenntage in History on Surface
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, JS_S_T3P_SSUR]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','SNAME']]
    SNAME_List = '('+str(Dataframe.loc[:,'SNAME'].tolist())[1:-1]+')'
    Surface = "'" + Dataframe.loc[:,'RATRA'].values[0] + "'"

    Extraction = Extraction_Database("""
                                     Select SNAME, Num_Races, T3 from (
                                     Select SNAME, count(RARID) Num_Races from RaceDb
                                     where RADAT < {Raceday} and SNAME in {SNAME_List} and RATRA = {Surface}
                                     Group by SNAME) NUM
                                     LEFT OUTER JOIN
                                     (Select SNAME SNAME_T, count(RARID) T3 from RaceDb
                                     where RADAT < {Raceday} and SNAME in {SNAME_List} and RATRA = {Surface} and RESFP <= 3
                                     Group by SNAME) T
                                     ON T.SNAME_T = NUM.SNAME
                                     """.format(Raceday = Raceday, SNAME_List = SNAME_List, Surface = Surface))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'JS_S_T3P_SSUR'] = Feature_DF.loc[:,'T3'] / Feature_DF.loc[:,'Num_Races']
    Feature_DF.loc[:,'JS_S_T3P_SSUR'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_S_T3P_SSUR']]

    return Feature_DF

"""
JS_S_T3P_SLOC
"""

def JS_S_T3P_SLOC(Dataframe, HNAME_List, Raceday):

    """
    Stable's Top 3 Percenntage in History on Location
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, JS_S_T3P_SLOC]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','SNAME']]
    SNAME_List = '('+str(Dataframe.loc[:,'SNAME'].tolist())[1:-1]+')'
    Location = "'" + Dataframe.loc[:,'RALOC'].values[0] + "'"

    Extraction = Extraction_Database("""
                                     Select SNAME, Num_Races, T3 from (
                                     Select SNAME, count(RARID) Num_Races from RaceDb
                                     where RADAT < {Raceday} and SNAME in {SNAME_List} and RALOC = {Location}
                                     Group by SNAME) NUM
                                     LEFT OUTER JOIN
                                     (Select SNAME SNAME_T, count(RARID) T3 from RaceDb
                                     where RADAT < {Raceday} and SNAME in {SNAME_List} and RALOC = {Location} and RESFP <= 3
                                     Group by SNAME) T
                                     ON T.SNAME_T = NUM.SNAME
                                     """.format(Raceday = Raceday, SNAME_List = SNAME_List, Location = Location))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'JS_S_T3P_SLOC'] = Feature_DF.loc[:,'T3'] / Feature_DF.loc[:,'Num_Races']
    Feature_DF.loc[:,'JS_S_T3P_SLOC'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_S_T3P_SLOC']]

    return Feature_DF

"""
JS_S_T3P_SPFL
"""

def JS_S_T3P_SPFL(Dataframe, HNAME_List, Raceday):

    """
    Stable's Top 3 Percenntage in History on Profile
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, JS_S_T3P_SPFL]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','SNAME']]
    SNAME_List = '('+str(Dataframe.loc[:,'SNAME'].tolist())[1:-1]+')'
    Location = "'" + Dataframe.loc[:,'RALOC'].values[0] + "'"
    Surface = "'" + Dataframe.loc[:,'RATRA'].values[0] + "'"
    Distance = Dataframe.loc[:,'RADIS'].values[0]

    Extraction = Extraction_Database("""
                                     Select SNAME, Num_Races, T3 from (
                                     Select SNAME, count(RARID) Num_Races from RaceDb
                                     where RADAT < {Raceday} and SNAME in {SNAME_List}
                                     and RALOC = {Location} and RATRA = {Surface} and RADIS = {Distance}
                                     Group by SNAME) NUM
                                     LEFT OUTER JOIN
                                     (Select SNAME SNAME_T, count(RARID) T3 from RaceDb
                                     where RADAT < {Raceday} and SNAME in {SNAME_List} and RESFP <= 3
                                     and RALOC = {Location} and RATRA = {Surface} and RADIS = {Distance}
                                     Group by SNAME) T
                                     ON T.SNAME_T = NUM.SNAME
                                     """.format(Raceday = Raceday, SNAME_List = SNAME_List,
                                     Location = Location, Surface = Surface, Distance = Distance))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'JS_S_T3P_SPFL'] = Feature_DF.loc[:,'T3'] / Feature_DF.loc[:,'Num_Races']
    Feature_DF.loc[:,'JS_S_T3P_SPFL'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_S_T3P_SPFL']]

    return Feature_DF

"""
JS_S_PRE
"""

def JS_S_PRE(Dataframe, HNAME_List, Raceday):

    """
    Stable Preference
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, JS_S_PRE]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','SNAME','HJRAT']]

    Grouped_Stable = Feature_DF.groupby('SNAME').count()['HNAME'].reset_index()
    Grouped_Stable = Grouped_Stable.rename(columns={'SNAME': 'SNAME', 'HNAME': 'Num_Horse'})
    Feature_DF = Feature_DF.merge(Grouped_Stable, how='left')

    Multiple_Horse = []
    for name, group in Feature_DF.groupby('SNAME'):
        if group.loc[:,'Num_Horse'].to_list()[0] > 1:
            group = group.sort_values(by=['HJRAT'], ascending=False)
            group.loc[:,'JS_S_PRE'] = [-i for i in range(len(group))]
            group = group.replace(0, 1)
            Multiple_Horse.append(group)

    if len(Multiple_Horse) == 0:
        Feature_DF.loc[:,'JS_S_PRE'] = 0
        Feature_DF = Feature_DF.loc[:,['HNAME','JS_S_PRE']]
        return Feature_DF

    Feature_DF = Feature_DF.merge(pd.concat(Multiple_Horse), how='left')
    Feature_DF.loc[:,'JS_S_PRE'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_S_PRE']]

    return Feature_DF
