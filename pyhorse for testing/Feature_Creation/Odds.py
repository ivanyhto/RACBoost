#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ivan
"""

"""
Feature : Odds
"""

#Loading Libraries
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from pyhorse.Database_Management import Extraction_Database

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

============================== Current Race ==============================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
OD_CR_LP
"""

def OD_CR_LP(Dataframe, HNAME_List, Raceday):

    """
    Logged odds implied probability of underlying race
    P = (1 - track take) / odds, track take = 17.5%
    Normalised across race
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, OD_CR_LP]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RESFO']]

    Feature_DF.loc[:,'OD_CR_LP'] = Feature_DF.loc[:,'RESFO'].map(lambda x : (1-0.175)/x)

    #Scale Probability to sum to 1
    Feature_DF.loc[:,'OD_CR_LP'] = np.log(Feature_DF.loc[:,'OD_CR_LP']/ Feature_DF.loc[:,'OD_CR_LP'].sum())
    Feature_DF = Feature_DF.loc[:,['HNAME','OD_CR_LP']]

    return Feature_DF

"""
OD_CR_FAVT
"""

def OD_CR_FAVT(Dataframe, HNAME_List, Raceday):

    """
    The Favourite
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, OD_CR_FAVT]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RESFO']]

    #Difference between favourite and second favourite
    Favourite = Feature_DF.loc[:, 'RESFO'].min() - sorted(Feature_DF.loc[:,'RESFO'])[1]
    Feature_DF.loc[:,'OD_CR_FAVT'] = 0
    Feature_DF.loc[Feature_DF.loc[:,'RESFO'].min() == Feature_DF.loc[:,'RESFO'], 'OD_CR_FAVT'] = Favourite

    Feature_DF = Feature_DF.loc[:,['HNAME','OD_CR_FAVT']]

    return Feature_DF

"""
OD_CR_FAVO
"""

def OD_CR_FAVO(Dataframe, HNAME_List, Raceday):

    """
    Not Favourite
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, OD_CR_FAVO]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RESFO']]

    #Difference between favourite and other odds
    Feature_DF.loc[:, 'OD_CR_FAVO'] = Feature_DF.loc[:, 'RESFO'] - Feature_DF.loc[:, 'RESFO'].min()
    Feature_DF = Feature_DF.loc[:,['HNAME','OD_CR_FAVO']]

    return Feature_DF

"""
OD_CR_FAVW
"""

def OD_CR_FAVW(Dataframe, HNAME_List, Raceday):

    """
    Favourite of current race assigned value of 1 if favouriate won last race
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, OD_CR_FAVW]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RESFO']]
    Race_ID = Dataframe.loc[:,'RARID'].values[0]

    Extraction = Extraction_Database("""
                                     Select RESFO, RESWL from RaceDb
                                     where RARID = (Select RARID from RaceDb
                                                     where RARID < {Race_ID}
                                                     Group by RARID
                                                     Order by RARID DESC
                                                     limit 1)
                                     """.format(Race_ID = Race_ID))

    OD_FAVW = Extraction.loc[Extraction.loc[:,'RESFO'] == Extraction.loc[:,'RESFO'].min(),'RESWL'].to_string(index = False, header = False).strip()
    try :
        OD_FAVW = int(OD_FAVW)
    except :
        OD_FAVW = 0
    Feature_DF.loc[:,'OD_CR_FAVW'] = 0
    Feature_DF.loc[Feature_DF.loc[:,'RESFO'] == Feature_DF.loc[:,'RESFO'].min(),'OD_CR_FAVW'] = OD_FAVW
    Feature_DF = Feature_DF.loc[:,['HNAME','OD_CR_FAVW']]

    return Feature_DF

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

============================== Previous Race ==============================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
OD_PR_LPAVG
"""

def OD_PR_LPAVG(Dataframe, HNAME_List, Raceday):

    """
    Average Log Odds implied Probability
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, OD_PR_LPAVG]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, RARID, RESFO from RaceDb
                                     where RADAT < {Raceday} and HNAME in {HNAME_List}
                                     """.format(Raceday = Raceday, HNAME_List = HNAME_List))

    Odds = []
    for name, group in Extraction.groupby('HNAME'):
        Probi = group.loc[:,'RESFO'].map(lambda x : np.log((1-0.175)/x)).dropna().values
        if len(Probi) >1:
            model = SimpleExpSmoothing(Probi)
            model = model.fit()
            Odds.append([name, model.forecast()[0]])
        elif len(Probi) == 1:
            Odds.append([name,Probi[0]])
        else :
            Odds.append([name,0])
    Odds = pd.DataFrame(Odds, columns=['HNAME','OD_PR_LPAVG'])

    Feature_DF = Feature_DF.merge(Odds, how='left')
    Feature_DF.loc[:,'OD_PR_LPAVG'].fillna(Feature_DF.loc[:,'OD_PR_LPAVG'].min(), inplace = True)
    Feature_DF.loc[:,'OD_PR_LPAVG'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','OD_PR_LPAVG']]

    return Feature_DF

"""
OD_PR_LPW
"""

def OD_PR_LPW(Dataframe, HNAME_List, Raceday):

    """
    Average Winning Odds
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, OD_PR_LPW]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RESFO']]

    Extraction_Win = Extraction_Database("""
                                        Select HNAME, avg(RESFO) Win_Odds from RaceDb
                                        where RADAT < {Raceday} and HNAME in {HNAME_List} and RESWL = 1
                                        Group by HNAME
                                        """.format(Raceday = Raceday, HNAME_List = HNAME_List))

    Odds = Extraction_Database("""
                               Select HNAME, RARID, RESFO Odds from RaceDb
                               where HNAME in {HNAME_List} and RADAT < {Raceday}
                               """.format(HNAME_List=HNAME_List, Raceday=Raceday))

    Speed_Ratings = Extraction_Database("""
                                        Select HNAME, RARID, BEYER_SPEED from Race_PosteriorDb
                                        where HNAME in {HNAME_List} and RADAT < {Raceday}
                                        """.format(HNAME_List=HNAME_List, Raceday=Raceday))

    idx = Speed_Ratings.groupby(['HNAME'])['BEYER_SPEED'].transform(max) == Speed_Ratings['BEYER_SPEED']
    Speed_Ratings_Odds = Speed_Ratings[idx].merge(Odds).loc[:,['HNAME','Odds']]
    try :
        #Exception for first season
        Speed_Ratings_Odds = Speed_Ratings_Odds.groupby('HNAME').mean().reset_index()
    except :
        pass
    Feature_DF = Feature_DF.merge(Extraction_Win, how='left').merge(Speed_Ratings_Odds, how='left')
    Feature_DF.loc[:,'Filled_Odds'] = Feature_DF.loc[:,'Win_Odds'].fillna(Feature_DF.loc[:,'Odds'])
    Feature_DF.loc[:,'Filled_Odds'] = Feature_DF.loc[:,'Filled_Odds'].map(lambda x : np.log((1-0.175)/x))
    Feature_DF.loc[:,'RESFO'] = Feature_DF.loc[:,'RESFO'].map(lambda x : np.log((1-0.175)/x))

    Feature_DF.loc[:,'OD_PR_LPW'] = ((Feature_DF.loc[:,'RESFO'] - Feature_DF.loc[:,'Filled_Odds']) / Feature_DF.loc[:,'Filled_Odds']).abs()
    Feature_DF.loc[:,'OD_PR_LPW'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','OD_PR_LPW']]

    return Feature_DF

"""
OD_PR_FAVB
"""

def OD_PR_FAVB(Dataframe, HNAME_List, Raceday):

    """
    Number of favourites that ran behind the underlying horse in the last 5 races.
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, OD_PR_FAVB]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RESFO']]

    Fav_Beaten_List = []
    #For each horse, get data for last 5 races
    for Horse in Dataframe['HNAME'].tolist():
        Extraction = Extraction_Database("""
                                         Select HNAME, RARID, RESFO, RESFP from RaceDb
                                         where RARID in (
                                         Select RARID from RaceDb
                                         where RADAT < {Raceday} and HNAME = {Horse}
                                         ORDER BY RARID DESC
                                         LIMIT 5)
                                         """.format(Raceday = Raceday, Horse = "'" + Horse + "'"))

        Won_Fav_tot = 0
        for RARID, race in Extraction.groupby('RARID'):
            fav_con = race.loc[:,'RESFO'] == race.loc[:,'RESFO'].min()
            horse_con = race.loc[:,'HNAME'] == Horse
            Only_Fav_Horse = race.loc[fav_con | horse_con,['HNAME','RESFP']].sort_values('RESFP').reset_index(drop = True)
            if len(Only_Fav_Horse) != 1:
                Won_Fav = float(not bool(Only_Fav_Horse.loc[Only_Fav_Horse.loc[:,'HNAME'] == Horse,:].index.values))
            else :
                Won_Fav = 0
            Won_Fav_tot += Won_Fav
        Fav_Beaten_List.append([Horse, Won_Fav_tot])
    Fav_Beaten = pd.DataFrame(Fav_Beaten_List, columns=['HNAME','OD_PR_FAVB'])

    Feature_DF = Feature_DF.merge(Fav_Beaten, how='left')
    Feature_DF.loc[:,'OD_PR_FAVB'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','OD_PR_FAVB']]

    return Feature_DF

"""
OD_PR_BFAV
"""

def OD_PR_BFAV(Dataframe, HNAME_List, Raceday):

    """
    Number of races the underlying horse is a beaten favourite is the last 5 races.
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, OD_PR_BFAV]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RESFO']]

    Fav_Beaten_List = []
    #For each horse, get data for last 5 races
    for Horse in Dataframe.loc[:,'HNAME'].tolist():
        Extraction = Extraction_Database("""
                                         Select HNAME, RARID, RESFO, RESFP from RaceDb
                                         where RARID in (
                                         Select RARID from RaceDb
                                         where RADAT < {Raceday} and HNAME = {Horse}
                                         ORDER BY RARID DESC
                                         LIMIT 5)
                                         """.format(Raceday = Raceday, Horse = "'" + Horse + "'"))

        Lost_Fav_tot = 0
        for RARID, race in Extraction.groupby('RARID'):
            fav_con = race.loc[:,'RESFO'] == race.loc[:,'RESFO'].min()
            horse_con = race.loc[:,'HNAME'] == Horse
            Only_Fav_Horse = race.loc[fav_con & horse_con,['HNAME','RESFP']]
            if len(Only_Fav_Horse) == 1:
                Lost_Fav = float(Only_Fav_Horse.loc[:,'RESFP'] == 1)
            else :
                Lost_Fav = 0
            Lost_Fav_tot += Lost_Fav
        Fav_Beaten_List.append([Horse, Lost_Fav_tot])
    Fav_Beaten = pd.DataFrame(Fav_Beaten_List, columns=['HNAME','OD_PR_BFAV'])

    Feature_DF = Feature_DF.merge(Fav_Beaten, how='left')
    Feature_DF.loc[:,'OD_PR_BFAV'].fillna(0, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME','OD_PR_BFAV']]

    return Feature_DF

