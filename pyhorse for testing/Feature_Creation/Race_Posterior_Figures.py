#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ivan
"""

"""
Update All Race_PosteriorDb Features after raceday
"""

#Loading Libraries
import sqlite3
import numpy as np
import pandas as pd
from functools import reduce
from pyhorse.Feature_Creation.Racetrack_Condition import Preference_Residuals
from pyhorse.Database_Management import Extraction_Database, Load_Dataset_toDatabase

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

================================= Figures =================================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Result_DF = Extraction_Database("""Select * from RaceDb where RARID = 2013050808 """)

def Update_Race_PosteriroDb(index):
    """
    Update All Race_PosteriorDb Features after raceday
    Parameter
    ---------
    Feature_DF : Feature Dataset for a raceday
    Result_DF : Post race Dataset for a raceday
    """

    Feature_DF, Result_DF = index

    Result_DF.loc[:,'RADIS'] = Result_DF.loc[:,'RADIS'].map(str)
    Result_DF.loc[:,'Profile'] = Result_DF.loc[:,['RALOC', 'RADIS', 'RATRA']].agg('_'.join, axis=1)
    Raceday = int(Result_DF.loc[:,'RADAT'].to_list()[0])
    Figures = Result_DF.loc[:, ['RARID', 'HNAME', 'JNAME', 'SNAME','RADAT', 'Profile', 'RARAL','RALOC', 'RAGOG', 'RADIS','HDRAW', 'RATRA','RACLS','RESFP']]

    """
    Recalculate Parallel Charts
    """
    # parallel_speed_charts(Raceday)
    # parallel_pace_charts(Raceday)

    """
    Beyer Speed Figure, Beaten Length Figure, Sartin Pace Figures, Preference Residual
    """
    #Beyer Speed Figure
    Beyer_DF = Beyer_Speed(Result_DF)

    #Pace Figures
    Pace_DF = Pace_Figure(Result_DF)

    #Calculating Daily Track Variant
    Beyer_DF, Pace_DF = Track_Variant(Beyer_DF, Pace_DF, Result_DF)

    #Sartin Pace Figures
    Sartin_DF = Sartin_Pace(pd.merge(Pace_DF, Result_DF, on = ['HNAME','RARID']))

    #Beaten Length Figure
    BL_DF = Beaten_Length(Result_DF)

    #Preference Residual
    Preference_Res_DF = Preference_Residuals(Feature_DF, Result_DF)

    #Combining Everything
    To_combine = [Figures ,Beyer_DF, BL_DF, Pace_DF, Sartin_DF, Preference_Res_DF]

    Figures_DF = reduce(lambda x, y: pd.merge(x, y, on = ['HNAME','RARID']), To_combine)

    """
    Load to Race_PosteriorDb
    """
    Figures_DF = Figures_DF.loc[:,['HNAME','JNAME','SNAME','RARID','RADAT','RESFP','RARAL','Profile','RALOC','HDRAW','RADIS','RATRA','RAGOG',
                             'RACLS','RESFP','BEYER_SPEED','EARLY_PACE','FINAL_FRACTION_PACE','AVERAGE_PAGE','SUSTAINED_PACE',
                             'EARLY_ENERGY','BEATEN_FIGURE','PACE_S1','PACE_S2','PACE_S3','PACE_S4','PACE_S5','PACE_S6',
                             'HPRE_DIST_RES','HPRE_GO_RES','HPRE_SUR_RES','HPRE_PFL_RES','JPRE_DIST_RES','JPRE_GO_RES',
                             'JPRE_SUR_RES','JPRE_LOC_RES','JPRE_PFL_RES','SPRE_DIST_RES','SPRE_GO_RES','SPRE_SUR_RES',
                             'SPRE_LOC_RES','SPRE_PFL_RES']]

    Load_Dataset_toDatabase('Race_PosteriorDb', Figures_DF)


    return None


def Track_Variant(Beyer_DF, Pace_DF, Result_DF):

    #Merge Dataset
    Beyer_DF = Result_DF.merge(Beyer_DF, on = ['HNAME','RARID'])
    Pace_DF = Result_DF.merge(Pace_DF, on = ['HNAME','RARID'])

    #Slice in the winning Horses
    Beyer_DF_Won = Beyer_DF.loc[Beyer_DF.loc[:,'RESWL']==1, ['RARID', 'HNAME', 'BEYER_SPEED', 'Profile', 'RACLS']]
    Pace_DF_Won = Pace_DF.loc[Pace_DF.loc[:,'RESWL']==1, ['RARID', 'RADIS','HNAME', 'PACE_S1', 'PACE_S2', 'PACE_S3',
                                                      'PACE_S4', 'PACE_S5', 'PACE_S6','Profile', 'RACLS']]
    Pace_DF.loc[:,'RADIS'] = Pace_DF.loc[:,'RADIS'].map(int)
    #Expected_Figures
    Beyer_Exp = Extraction_Database("""
                                    Select Profile, RACLS, avg(BEYER_SPEED) Win_Beyer from Race_PosteriorDb
                                    where RESFP = 1
                                    Group by Profile, RACLS
                                    """)
    Beyer_Exp.loc[:,'RACLS'] = Beyer_Exp.loc[:,'RACLS'].map(int)
    Pace_Exp = Extraction_Database("""
                                   Select Profile, RACLS, avg(PACE_S1), avg(PACE_S2), avg(PACE_S3),
                                   avg(PACE_S4), avg(PACE_S5), avg(PACE_S6)
                                   from Race_PosteriorDb
                                   where RESFP = 1
                                   Group by Profile, RACLS
                                   """)
    Pace_Exp.loc[:,'RACLS'] = Pace_Exp.loc[:,'RACLS'].map(int)

    #Merge Expectation
    Beyer_DF_Won = pd.merge(Beyer_DF_Won, Beyer_Exp, how = 'left',on=['Profile','RACLS'])
    Beyer_DF_Won.loc[:,'Win_Beyer'].fillna(Beyer_DF_Won.loc[:,'BEYER_SPEED'], inplace = True)
    Beyer_DF_Won.loc[:,'Variant'] = Beyer_DF_Won.loc[:,'BEYER_SPEED'] - Beyer_DF_Won.loc[:,'Win_Beyer']

    Pace_DF_Won = pd.merge(Pace_DF_Won, Pace_Exp, how = 'left',on=['Profile','RACLS'])
    Pace_DF_Won.loc[:,'avg(PACE_S1)'].fillna(Pace_DF_Won.loc[:,'PACE_S1'], inplace = True)
    Pace_DF_Won.loc[:,'avg(PACE_S2)'].fillna(Pace_DF_Won.loc[:,'PACE_S2'], inplace = True)
    Pace_DF_Won.loc[:,'avg(PACE_S3)'].fillna(Pace_DF_Won.loc[:,'PACE_S3'], inplace = True)
    Pace_DF_Won.loc[:,'avg(PACE_S4)'].fillna(Pace_DF_Won.loc[:,'PACE_S4'], inplace = True)
    Pace_DF_Won.loc[:,'avg(PACE_S5)'].fillna(Pace_DF_Won.loc[:,'PACE_S5'], inplace = True)
    Pace_DF_Won.loc[:,'avg(PACE_S6)'].fillna(Pace_DF_Won.loc[:,'PACE_S6'], inplace = True)
    Pace_DF_Won.loc[:,'S1_Variant'] = Pace_DF_Won.loc[:,'PACE_S1'] - Pace_DF_Won.loc[:,'avg(PACE_S1)']
    Pace_DF_Won.loc[:,'S2_Variant'] = Pace_DF_Won.loc[:,'PACE_S2'] - Pace_DF_Won.loc[:,'avg(PACE_S2)']
    Pace_DF_Won.loc[:,'S3_Variant'] = Pace_DF_Won.loc[:,'PACE_S3'] - Pace_DF_Won.loc[:,'avg(PACE_S3)']
    Pace_DF_Won.loc[:,'S4_Variant'] = Pace_DF_Won.loc[:,'PACE_S4'] - Pace_DF_Won.loc[:,'avg(PACE_S4)']
    Pace_DF_Won.loc[:,'S5_Variant'] = Pace_DF_Won.loc[:,'PACE_S5'] - Pace_DF_Won.loc[:,'avg(PACE_S5)']
    Pace_DF_Won.loc[:,'S6_Variant'] = Pace_DF_Won.loc[:,'PACE_S6'] - Pace_DF_Won.loc[:,'avg(PACE_S6)']

    Variant = []
    for Profile in Result_DF.loc[:,'Profile'].unique():
        Distance = int(Profile[3:7])
        speed_variant = Beyer_DF_Won.loc[Beyer_DF_Won.loc[:,'Profile']==Profile, 'Variant'].values.tolist()
        if Distance in [1000, 1200]:
            pace_variant = Pace_DF_Won.loc[Pace_DF_Won.loc[:,'Profile']==Profile, ['S1_Variant','S2_Variant',
                                                                            'S3_Variant']].values.tolist()
        elif Distance in [1400, 1600, 1650]:
            pace_variant = Pace_DF_Won.loc[Pace_DF_Won.loc[:,'Profile']==Profile, ['S1_Variant','S2_Variant',
                                                                            'S3_Variant','S4_Variant']].values.tolist()
        elif Distance in [1800, 2000]:
            pace_variant = Pace_DF_Won.loc[Pace_DF_Won.loc[:,'Profile']==Profile, ['S1_Variant','S2_Variant',
                                                                            'S3_Variant','S4_Variant',
                                                                            'S5_Variant']].values.tolist()
        elif Distance in [2200, 2400]:
            pace_variant = Pace_DF_Won.loc[Pace_DF_Won.loc[:,'Profile']==Profile, ['S1_Variant','S2_Variant',
                                                                            'S3_Variant','S4_Variant',
                                                                            'S5_Variant','S6_Variant']].values.tolist()
        speed_variant = np.mean(speed_variant)
        pace_variant = np.mean(pace_variant)

        Variant.append(pd.DataFrame([Profile, speed_variant, pace_variant]).transpose())
    Variant = pd.concat(Variant)
    Variant.fillna(0, inplace=True)
    Variant.columns = ['Profile', 'Speed_Variant', 'Pace_Variant']

    #Correcting Beyer Speed and Pace Figures by Daily Track Variant
    Beyer_DF = Beyer_DF.merge(Variant, on =['Profile'])
    Beyer_DF.loc[:,'BEYER_SPEED'] = Beyer_DF.loc[:,'BEYER_SPEED'] - Beyer_DF.loc[:,'Speed_Variant']

    Pace_DF = Pace_DF.merge(Variant, on =['Profile'])
    Pace_DF.loc[:,'PACE_S1'] = Pace_DF.loc[:,'PACE_S1'] - Pace_DF.loc[:,'Pace_Variant']
    Pace_DF.loc[:,'PACE_S2'] = Pace_DF.loc[:,'PACE_S2'] - Pace_DF.loc[:,'Pace_Variant']
    Pace_DF.loc[:,'PACE_S3'] = Pace_DF.loc[:,'PACE_S3'] - Pace_DF.loc[:,'Pace_Variant']
    Pace_DF.loc[Pace_DF.loc[:,'RADIS'].isin([1400,1600,1650,1800,2000,2200,2400]),'PACE_S4']\
        = Pace_DF.loc[Pace_DF.loc[:,'RADIS'].isin([1400,1600,1650,1800,2000,2200,2400]),'PACE_S4']\
            - Pace_DF.loc[:,'Pace_Variant']
    Pace_DF.loc[Pace_DF.loc[:,'RADIS'].isin([1800,2000,2200,2400]),'PACE_S5'] \
        = Pace_DF.loc[Pace_DF.loc[:,'RADIS'].isin([1800,2000,2200,2400]),'PACE_S5'] - Pace_DF.loc[:,'Pace_Variant']
    Pace_DF.loc[Pace_DF.loc[:,'RADIS'].isin([2200,2400]),'PACE_S6'] \
        = Pace_DF.loc[Pace_DF.loc[:,'RADIS'].isin([2200,2400]),'PACE_S6'] - Pace_DF.loc[:,'Pace_Variant']

    Beyer_DF = Beyer_DF.loc[:,['RARID', 'HNAME', 'BEYER_SPEED']]
    Pace_DF = Pace_DF.loc[:,['RARID','HNAME','PACE_S1','PACE_S2','PACE_S3','PACE_S4','PACE_S5','PACE_S6']]

    return Beyer_DF, Pace_DF


def Beaten_Length(Dataset):

    """
    A numeric indicator of the strength of the horse at the finish of a race.
    For races with 10 or less horses, the Power Point equals the midpoint of the second and third horse.
    For races with 11 or more horses, the Power Point equals the third horse.
    The Figure measure how many lengths is the underlying horse behind or in front of the Power Point.
    The log transformation is used to correct for over and under estimates.
    """
    BL_DF = Dataset.loc[:, ['RARID', 'HNAME', 'RARUN','RESWD']]
    BL_Figure = []
    for name, race in BL_DF.groupby('RARID'):
        Num_Runners = race.loc[:, 'RARUN'].to_list()[0]
        Beaten_Length = race['RESWD'].to_list()
        Beaten_Length[0] = 0
        if Num_Runners < 11:
            Power_Point = (Beaten_Length[1] + Beaten_Length[2])/2
        else :
            Power_Point = Beaten_Length[2]
        [BL_Figure.append(round(Power_Point - i, 2)) for i in Beaten_Length]
    BL_DF.loc[:,'BEATEN_FIGURE'] = BL_Figure

    BL_DF = BL_DF.loc[:, ['RARID', 'HNAME', 'BEATEN_FIGURE']]

    return BL_DF


def Beyer_Speed(Dataset):

    """
    Parameter
    ---------
    Dataset : Daily Datasets
    Step 1 : Calculate Actual Speed Rating from the parallel speed charts
    Step 2 : Calculate Expected Speed Figures / Par Figures of each winner of the day
    Expected Figures is defined as the Average Winning Speed Rating of races that are of the underlying class ran on underlying profile
    """
    #Get Parallel Charts
    Parallel_SpeedDb = Extraction_Database(""" Select * from Parallel_SpeedDb""")

    Beyer = Dataset.loc[:, ['RARID', 'HNAME', 'Profile', 'RESFT']]
    Beyer_Parallel = []
    for name, race in Beyer.groupby('RARID'):
        profile = race['Profile'].values[0]
        [Beyer_Parallel.append(Parallel_SpeedDb.loc[Parallel_SpeedDb['time_vector']==round(race.loc[i, 'RESFT'],2), profile].values[0]) for i in race.index]
    Beyer.loc[:,'BEYER_SPEED'] = Beyer_Parallel

    Beyer = Beyer.loc[:, ['RARID','HNAME','BEYER_SPEED']]

    return Beyer


def Pace_Figure(Dataset):

    """
    Parameter
    ---------
    Dataset : Daily Datasets
    Calculate Pace Rating from the parallel speed charts
    """

    #Get Parallel Charts
    Parallel_PaceDb = Extraction_Database(""" Select * from Parallel_PaceDb""")
    Pace = Dataset.loc[:, ['RARID', 'HNAME', 'Profile', 'RESS1', 'RESS2', 'RESS3', 'RESS4', 'RESS5', 'RESS6']]
    Pace_Parallel = []

    for rarid, race in Pace.groupby('RARID'):
        profile = race['Profile'].values[0]
        One_Race = pd.DataFrame()
        One_Race.loc[:,'HNAME'] = race.loc[:,'HNAME']
        One_Race.loc[:,'RARID'] = rarid
        One_Race.loc[:,'PACE_S1'] = [Parallel_PaceDb.loc[Parallel_PaceDb.loc[:,'time_vector'] == round(race.loc[i, 'RESS1'],3), profile+'_S1'].values[0] for i in race.index]
        One_Race.loc[:,'PACE_S2'] = [Parallel_PaceDb.loc[Parallel_PaceDb.loc[:,'time_vector'] == round(race.loc[i, 'RESS2'],3), profile+'_S2'].values[0] for i in race.index]
        One_Race.loc[:,'PACE_S3'] = [Parallel_PaceDb.loc[Parallel_PaceDb.loc[:,'time_vector'] == round(race.loc[i, 'RESS3'],3), profile+'_S3'].values[0] for i in race.index]

        try :
            One_Race.loc[:,'PACE_S4'] = [Parallel_PaceDb.loc[Parallel_PaceDb.loc[:,'time_vector'] == round(race.loc[i, 'RESS4'],3), profile+'_S4'].values[0] for i in race.index]
        except :
            One_Race.loc[:,'PACE_S4'] = 0.0
        try :
            One_Race.loc[:,'PACE_S5'] = [Parallel_PaceDb.loc[Parallel_PaceDb.loc[:,'time_vector'] == round(race.loc[i, 'RESS5'],3), profile+'_S5'].values[0] for i in race.index]
        except :
            One_Race.loc[:,'PACE_S5'] = 0.0
        try :
            One_Race.loc[:,'PACE_S6'] = [Parallel_PaceDb.loc[Parallel_PaceDb.loc[:,'time_vector'] == round(race.loc[i, 'RESS6'],3), profile+'_S6'].values[0] for i in race.index]
        except :
            One_Race.loc[:,'PACE_S6'] = 0.0
        Pace_Parallel.append(One_Race)

    Pace = pd.concat(Pace_Parallel)

    return Pace


def Sartin_Pace(Dataset):
    """
    Parameter
    ---------
    Dataset : Pace Figure Dataset

    """
    Sartin = Dataset.loc[:,['HNAME','RARID','RADIS', 'PACE_S1','PACE_S2','PACE_S3','PACE_S4','PACE_S5','PACE_S6']]
    Sartin_Results = []
    for rarid, race in Sartin.groupby('RARID'):
        Distance = int(race['RADIS'].values[0])
        One_Race = race.loc[:,['HNAME','RARID']]
        if Distance in [1000, 1200]:
            One_Race.loc[:,'EARLY_PACE'] = race.loc[:,['PACE_S1','PACE_S2']].mean(axis=1)
            One_Race.loc[:,'FINAL_FRACTION_PACE'] = race.loc[:,'PACE_S3']
            One_Race.loc[:,'AVERAGE_PAGE'] = race.loc[:,['PACE_S1','PACE_S2','PACE_S3']].mean(axis=1)
            One_Race.loc[:,'SUSTAINED_PACE'] = (One_Race.loc[:,'EARLY_PACE'] + 1.5 * race.loc[:,'PACE_S3']) / 2.5
            One_Race.loc[:,'EARLY_ENERGY'] = One_Race.loc[:,'EARLY_PACE'] / (One_Race.loc[:,'EARLY_PACE'] + One_Race.loc[:,'FINAL_FRACTION_PACE'])

        elif Distance in [1400,1600,1650]:
            One_Race.loc[:,'EARLY_PACE'] = race.loc[:,['PACE_S1','PACE_S2']].mean(axis=1)
            One_Race.loc[:,'FINAL_FRACTION_PACE'] = race.loc[:,'PACE_S4']
            One_Race.loc[:,'AVERAGE_PAGE'] = race.loc[:,['PACE_S1','PACE_S2','PACE_S3','PACE_S4']].mean(axis=1)
            One_Race.loc[:,'SUSTAINED_PACE'] = (One_Race.loc[:,'EARLY_PACE'] + 1.5 * race.loc[:,'PACE_S4']) / 2.5
            One_Race.loc[:,'EARLY_ENERGY'] = One_Race.loc[:,'EARLY_PACE'] / (One_Race.loc[:,'EARLY_PACE'] + One_Race.loc[:,'FINAL_FRACTION_PACE'])

        elif Distance in [1800,2000]:
            One_Race.loc[:,'EARLY_PACE'] = race.loc[:,['PACE_S1','PACE_S2','PACE_S3']].mean(axis=1)
            One_Race.loc[:,'FINAL_FRACTION_PACE'] = race.loc[:,'PACE_S5']
            One_Race.loc[:,'AVERAGE_PAGE'] = race.loc[:,['PACE_S1','PACE_S2','PACE_S3','PACE_S4','PACE_S5']].mean(axis=1)
            One_Race.loc[:,'SUSTAINED_PACE'] = (One_Race.loc[:,'EARLY_PACE'] + 1.5 * race.loc[:,'PACE_S5']) / 2.5
            One_Race.loc[:,'EARLY_ENERGY'] = One_Race.loc[:,'EARLY_PACE'] / (One_Race.loc[:,'EARLY_PACE'] + One_Race.loc[:,'FINAL_FRACTION_PACE'])

        elif Distance in [2200,2400]:
            One_Race.loc[:,'EARLY_PACE'] = race.loc[:,['PACE_S1','PACE_S2','PACE_S3']].mean(axis=1)
            One_Race.loc[:,'FINAL_FRACTION_PACE'] = race.loc[:,'PACE_S6']
            One_Race.loc[:,'AVERAGE_PAGE'] = race.loc[:,['PACE_S1','PACE_S2','PACE_S3','PACE_S4','PACE_S5','PACE_S6']].mean(axis=1)
            One_Race.loc[:,'SUSTAINED_PACE'] = (One_Race.loc[:,'EARLY_PACE'] + 1.5 * race.loc[:,'PACE_S6']) / 2.5
            One_Race.loc[:,'EARLY_ENERGY'] = One_Race.loc[:,'EARLY_PACE'] / (One_Race.loc[:,'EARLY_PACE'] + One_Race.loc[:,'FINAL_FRACTION_PACE'])

        Sartin_Results.append(One_Race)

    Sartin = pd.concat(Sartin_Results)

    return Sartin



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

============================= Parallel Charts =============================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Sectional_Dict = {1000:[200,400,400],
                  1200:[400,400,400],
                  1400:[200,400,400,400],
                  1600:[400,400,400,400],
                  1650:[250,400,400,400],
                  1800:[200,400,400,400,400],
                  2000:[400,400,400,400,400],
                  2200:[200,400,400,400,400,400],
                  2400:[400,400,400,400,400,400]}


def parallel_speed_charts(Raceday):

    #For the 2012 season, give the model something to start with
    Raceday = max(Raceday, 20130710)

    """
    Time Candidates
    """
    Parallel_Chart = pd.DataFrame()
    Parallel_Chart['time_vector'] = [round(i,2) for i in np.arange(45, 175, 0.01)]

    """
    Equivalent Time
    """
    equivalent_time = Extraction_Database("""Select round(avg(RESFT),2) eq_time, RADIS, RATRA, RALOC from RaceDb
                                          where RADAT < {Raceday}
                                          group by RADIS, RATRA, RALOC""".format(Raceday = Raceday))
    equivalent_time.loc[:,'RADIS'] = equivalent_time['RADIS'].map(str)
    equivalent_time.loc[:,'profile'] = equivalent_time.loc[:,['RALOC', 'RADIS', 'RATRA']].agg('_'.join, axis=1)
    equivalent_time.loc[:,'RADIS'] = equivalent_time.loc[:,'RADIS'].map(int)
    equivalent_time.loc[:,'1s'] = round(equivalent_time.loc[:,'RADIS'] / (equivalent_time.loc[:,'eq_time'] * 1), 3)

    """
    Parallel Charts
    """
    for index, row in equivalent_time.iterrows():

        Dist_DF = []
        eq_time = row['eq_time']
        increment = row['1s']

        #Top Half
        for i in np.arange(eq_time, 175, 0.01):
            Dist_DF.append([round(i,2), round(80 - (round(i,2) - eq_time) * increment, 3)])

        #Botton Half
        for i in np.arange(45, eq_time, 0.01):
            Dist_DF.append([round(i,2), round(80 + (eq_time - round(i,2)) * increment, 3)])

        #Merging to Parallel Chart
        Parallel_Chart = Parallel_Chart.merge(pd.DataFrame(Dist_DF, columns=['time', row['profile']]), how='left',left_on='time_vector',right_on='time')
        Parallel_Chart.drop(columns=['time'], inplace = True)

    """
    Uploading to Database
    """
    #Create Connection
    with sqlite3.connect('Data.db') as db:
        sql = db.cursor()
        db.commit()

    sql.execute("drop table if exists Parallel_SpeedDb")
    db.commit()
    sql.execute("""
                Create Table Parallel_SpeedDb(
                time_vector real, HV_1000_T real, ST_1000_T real, ST_1200_AW real, HV_1200_T real, ST_1200_T real, ST_1400_T real, ST_1600_T real,
                ST_1650_AW real, HV_1650_T real, ST_1800_AW real, HV_1800_T real, ST_1800_T real, ST_2000_T real, HV_2200_T real, ST_2200_T real, ST_2400_T real)
                """)
    db.commit()

    #Closing Connection
    sql.close()
    db.close()

    Load_Dataset_toDatabase('Parallel_SpeedDb', Parallel_Chart)

    return None #print("---- Parallel_SpeedDb is Created ----")


def parallel_pace_charts(Raceday):

    #For the 2012 season, give the model something to start with
    Raceday = max(Raceday, 20130710)

    """
    Time Candidates
    """
    Parallel_Chart = pd.DataFrame()
    min_time = max(0, Extraction_Database("""
                                   Select * from
                                   (Select min(RESS1) from RaceDb where RESS1 > 0),
                                   (Select min(RESS2) from RaceDb where RESS2 > 0),
                                   (Select min(RESS3) from RaceDb where RESS3 > 0),
                                   (Select min(RESS4) from RaceDb where RESS4 > 0),
                                   (Select min(RESS5) from RaceDb where RESS5 > 0),
                                   (Select min(RESS6) from RaceDb where RESS6 > 0)
                                   """.format(Raceday = Raceday)).values.min())
    max_time = Extraction_Database("""
                                   Select * from
                                   (Select max(RESS1) from RaceDb where RESS1 > 0),
                                   (Select max(RESS2) from RaceDb where RESS2 > 0),
                                   (Select max(RESS3) from RaceDb where RESS3 > 0),
                                   (Select max(RESS4) from RaceDb where RESS4 > 0),
                                   (Select max(RESS5) from RaceDb where RESS5 > 0),
                                   (Select max(RESS6) from RaceDb where RESS6 > 0)
                                   """.format(Raceday = Raceday)).values.max()
    Parallel_Chart.loc[:,'time_vector'] = [round(i,2) for i in np.arange(min_time-10, max_time+3, 0.01)]

    """
    Equivalent Time
    """
    equivalent_time = Extraction_Database("""Select RADIS, RATRA, RALOC, round(avg(RESS1),2) S1_eq_time, round(avg(RESS2),2) S2_eq_time,
                                          round(avg(RESS3),2) S3_eq_time, round(avg(RESS4),2) S4_eq_time, round(avg(RESS5),2) S5_eq_time,
                                          round(avg(RESS6),2) S6_eq_time from RaceDb
                                          where RADAT < {Raceday}
                                          group by RADIS, RATRA, RALOC""".format(Raceday = Raceday))
    equivalent_time.loc[:,'RADIS'] = equivalent_time.loc[:,'RADIS'].map(str)
    equivalent_time.loc[:,'profile'] = equivalent_time.loc[:,['RALOC', 'RADIS', 'RATRA']].agg('_'.join, axis=1)
    equivalent_time.loc[:,'RADIS'] = equivalent_time.loc[:,'RADIS'].map(int)

    Distance_List = equivalent_time.loc[:,'RADIS'].unique()
    for distance in Distance_List:
        condition = equivalent_time.loc[:,'RADIS'] == distance
        equivalent_time.loc[condition,'S1_1s'] = round(Sectional_Dict[distance][0] / (equivalent_time.loc[condition,'S1_eq_time'] * 1), 3)
        equivalent_time.loc[condition,'S2_1s'] = round(Sectional_Dict[distance][1] / (equivalent_time.loc[condition,'S2_eq_time'] * 1), 3)
        equivalent_time.loc[condition,'S3_1s'] = round(Sectional_Dict[distance][2] / (equivalent_time.loc[condition,'S3_eq_time'] * 1), 3)
        try :
            equivalent_time.loc[condition,'S4_1s'] = round(Sectional_Dict[distance][3] / (equivalent_time.loc[condition,'S4_eq_time'] * 1), 3)
            equivalent_time.loc[condition,'S5_1s'] = round(Sectional_Dict[distance][4] / (equivalent_time.loc[condition,'S5_eq_time'] * 1), 3)
            equivalent_time.loc[condition,'S6_1s'] = round(Sectional_Dict[distance][5] / (equivalent_time.loc[condition,'S6_eq_time'] * 1), 3)
        except :
            pass

    """
    Parallel Charts
    """
    for index, row in equivalent_time.iterrows():
        for Section in ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']:
            Time_DF = []
            eq_time = row[Section+'_eq_time']
            increment = row[Section+'_1s']
            if eq_time != 0:
                #Top Half
                for i in np.arange(eq_time, max_time, 0.01):
                    Time_DF.append([round(i,2), round(80 - (round(i,2) - eq_time) * increment, 3)])
                #Botton Half
                for i in np.arange(min_time, eq_time, 0.01):
                    Time_DF.append([round(i,2), round(80 + (eq_time - round(i,2)) * increment, 3)])

                #Merging to Parallel Chart
                Parallel_Chart = Parallel_Chart.merge(pd.DataFrame(Time_DF, columns=['time', row['profile']+'_'+Section]), how='left',left_on='time_vector',right_on='time')
                Parallel_Chart.drop(columns=['time'], inplace = True)

    """
    Uploading to Database
    """
    #Create Connection
    with sqlite3.connect('Data.db') as db:
        sql = db.cursor()
        db.commit()

    sql.execute("drop table if exists Parallel_PaceDb")
    db.commit()
    sql.execute("""
                Create Table Parallel_PaceDb(
                time_vector real, HV_1000_T_S1 real, HV_1000_T_S2 real, HV_1000_T_S3 real, ST_1000_T_S1 real, ST_1000_T_S2 real, ST_1000_T_S3 real,
                ST_1200_AW_S1 real, ST_1200_AW_S2 real, ST_1200_AW_S3 real, HV_1200_T_S1 real, HV_1200_T_S2 real, HV_1200_T_S3 real, ST_1200_T_S1 real,
                ST_1200_T_S2 real, ST_1200_T_S3 real, ST_1400_T_S1 real, ST_1400_T_S2 real, ST_1400_T_S3 real, ST_1400_T_S4 real, ST_1600_T_S1 real,
                ST_1600_T_S2 real, ST_1600_T_S3 real, ST_1600_T_S4 real, ST_1650_AW_S1 real, ST_1650_AW_S2 real, ST_1650_AW_S3 real, ST_1650_AW_S4 real,
                HV_1650_T_S1 real, HV_1650_T_S2 real, HV_1650_T_S3 real, HV_1650_T_S4 real, ST_1800_AW_S1 real, ST_1800_AW_S2 real, ST_1800_AW_S3 real,
                ST_1800_AW_S4 real, ST_1800_AW_S5 real, HV_1800_T_S1 real, HV_1800_T_S2 real, HV_1800_T_S3 real, HV_1800_T_S4 real, HV_1800_T_S5 real,
                ST_1800_T_S1 real, ST_1800_T_S2 real, ST_1800_T_S3 real, ST_1800_T_S4 real, ST_1800_T_S5 real,ST_2000_T_S1 real, ST_2000_T_S2 real,
                ST_2000_T_S3 real, ST_2000_T_S4 real, ST_2000_T_S5 real, HV_2200_T_S1 real, HV_2200_T_S2 real, HV_2200_T_S3 real, HV_2200_T_S4 real,
                HV_2200_T_S5 real, HV_2200_T_S6 real, ST_2200_T_S1 real, ST_2200_T_S2 real, ST_2200_T_S3 real, ST_2200_T_S4 real, ST_2200_T_S5 real,
                ST_2200_T_S6 real, ST_2400_T_S1 real, ST_2400_T_S2 real, ST_2400_T_S3 real, ST_2400_T_S4 real, ST_2400_T_S5 real, ST_2400_T_S6 real)
                """)
    db.commit()

    #Closing Connection
    sql.close()
    db.close()

    Load_Dataset_toDatabase('Parallel_PaceDb', Parallel_Chart)

    return None


