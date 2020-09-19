#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ivan
"""

"""
Dataset Creation
"""

#Loading Libraries
import time
import warnings
import concurrent
import pandas as pd
from functools import reduce
from pyhorse.Feature_Creation import Feature_Storage
from pyhorse.Database_Management import Extraction_Database, Load_Dataset_toDatabase
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# MatchDay_DF = MatchDay_Dataset(Extraction_Database("""
#                                                    Select Distinct RARID from RaceDb where RADAT = {Date}
#                                                    """.format(Date='20130123')))

# One_Race_Feature(MatchDay_DF)

def One_Race_Feature(Dataframe):

    """
    Feature Creation for one Race
    Parameter
    ---------
    Matchday_Dataset from Racecard
    Return
    ------
    Feature DataFrame
    """
    """
    Get Feature Names
    """
    #Get Feature_List from FeatureDb
    Feature_List = Feature_Storage.Feature_List#list(Extraction_Database("""PRAGMA table_info('FeatureDb')""")['name'])
    # Feature_List.remove('RARID')    Feature_List.remove('HNAME')
    Features_Dataframe = Dataframe.loc[:,['RARID', 'HNAME']]

    """
    Create Features in Parallel
    """
    #Prepare Matchday Dataset
    HNAME_List = '('+str(Dataframe['HNAME'].tolist())[1:-1]+')'
    Raceday = Dataframe.loc[:,'RADAT'].values[0]

    # results = []
    # for Feature in Feature_List:
    #     results.append(Create_Features([Feature, Dataframe, HNAME_List, Raceday]))

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for Feature in Feature_List:
            #Run Functions
            """
            All Feature Functions accepts a Matchday Dataframe
            then return a dataframe of a race,
            containing the following columns in the order of :
            [HNAME, Feature Name]
            """
            results.append(executor.submit(Create_Features, [Feature, Dataframe, HNAME_List, Raceday]))
    results = [i.result() for i in results]

    warnings.filterwarnings("default", category=RuntimeWarning)
    warnings.filterwarnings("default", category=ConvergenceWarning)

    #Combine all features into one dataframe
    Features_DF = reduce(lambda x, y: pd.merge(x, y, on = 'HNAME'), results)

    #Combine all features into one dataframe
    # Features_Dataframe = pd.merge(Features_Dataframe, Features_DF, on = 'HNAME', how='left')

    """
    Feature Transformation
    """
    Transformation_List = Feature_Storage.Transformation_List

    # results = []
    # for Feature in Transformation_List:
    #     results.append(Transform_Features([Feature, Features_DF]))

    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for Feature in Transformation_List:
            #Run Functions
            """
            All Feature Functions accepts a Base Features_Dataframe
            then return a dataframe of a race,
            containing the following columns in the order of :
            [HNAME, Feature Name]
            """
            results.append(executor.submit(Transform_Features, [Feature, Features_DF]))
    results = [i.result() for i in results]

    #Combine all features into one dataframe
    Transformation_DF = reduce(lambda x, y: pd.merge(x, y, on = 'HNAME'), results)

    #Combine all features into one dataframe
    Features_Dataframe = reduce(lambda x, y: pd.merge(x, y, on = 'HNAME'), [Features_Dataframe, Features_DF, Transformation_DF])

    if not len(Features_Dataframe.index)==len(Dataframe.index):
        print(Features_Dataframe)
        # print(Dataframe.loc[:,'RARID'].tolist()[0])

    #Inserting Features_Dataframe to Database
    Load_Dataset_toDatabase('FeatureDb', Features_Dataframe)

    if sum(Features_Dataframe.isna().sum()) != 0 :
        print(Dataframe.loc[:,'RARID'].to_list()[0])

    Features_Dataframe.loc[:,Features_Dataframe.columns[2:]] = \
        pd.DataFrame(Features_Dataframe.loc[:,Features_Dataframe.columns[2:]].values.astype(float))

    return Features_Dataframe


def Create_Features(index):

    Feature, Dataframe, HNAME_List, Raceday = index

    Feature_Function = getattr(Feature_Storage, Feature)
    Dataframe_Feature = Feature_Function(Dataframe, HNAME_List, Raceday)

    return Dataframe_Feature


def Transform_Features(index):

    Feature, Dataframe = index

    Feature_Function = getattr(Feature_Storage, Feature)
    Dataframe_Feature = Feature_Function(Dataframe)

    return Dataframe_Feature


def Post_Raceday_Update(Raceday, Feature_DF, Result_DF):

    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     executor.submit(Feature_Storage.Update_Running_Stat, Result_DF)
    #     executor.submit(Feature_Storage.Fit_Residual_Model, Raceday)
    #     executor.submit(Feature_Storage.Weight_Aug_Reg, Raceday)
    #     executor.submit(Feature_Storage.Update_Race_PosteriroDb, [Feature_DF, Result_DF])

    """
    Post-Raceday - ELO Figures
    """
    #Update Running Statistics Table with Results after feature is created
    Feature_Storage.Update_Running_Stat(Result_DF)

    """
    Post-Raceday - Auxiliary Regressions
    """
    Feature_Storage.Fit_Residual_Model(Raceday)
    Feature_Storage.Weight_Aug_Reg(Raceday)

    """
    Post-Raceday - Race_PosteriroDb - Speed and Pace Figures, Preference Residuals
    """
    #Updating Speed and Pace Figures
    Feature_Storage.Update_Race_PosteriroDb([Feature_DF, Result_DF])

    return None


def Feature_Creation(Dataframe):

    """
    Creates Feature from Feature List and Insert into FeatureDb
    Parameter
    ---------
    Dataframe : MatchDay Data Format
    Return
    ------
    Dataframe
    """
    #Start Timer
    start_time = time.time()

    for RADAT, Race_Day in Dataframe.groupby('RADAT'):
        print(RADAT)

        """
        Day by Day
        """
        for RARID, Race in Race_Day.groupby('RARID'):
            """
            Race to Race
            """
            # print(RARID)
            One_Race_Feature(Race)
        """
        Post-Day
        """
        Result_DF = Extraction_Database(""" Select * from RaceDb where RADAT = ? """,[RADAT])
        Features_Dataframe = Extraction_Database(""" Select * from FeatureDb where RARID BETWEEN ? and ? """,[int(str(RADAT)+'00'), int(str(RADAT)+'99')])
        Post_Raceday_Update(RADAT, Features_Dataframe, Result_DF)

    print("---- %s Races are Created to FeatureDb in %s seconds ----" \
                 %(Dataframe['RARID'].nunique(), (str(round((time.time() - start_time),4)))))

    return None


def MatchDay_Dataset(Race_ID):

    """
    Extracting MatchDay Data from RaceDb
    Parameter
    ---------
    Race_ID : Dataframe of RaceID
    Return
    ------
    Dataframe
    """
    #Start Timer
    start_time = time.time()

    Dataset = pd.DataFrame()
    if len(Race_ID)>1:
        Race_ID_List = [i for i in Race_ID['RARID'].tolist()]
        Dataset = Extraction_Database("""
                                      Select Distinct RARID, HNAME, HAGEI, HBWEI, HDRAW, HJRAT, HWEIC, JNAME, RESFO, RACLS, RADAT, RARAL,
                                      RADIS, RAGOG, RALOC, RARUN, RATRA, SNAME
                                      from RaceDb where RARID in {RARID}
                                      Order By RARID, HNAME
                                      """.format(RARID = '('+str(Race_ID_List)[1:-1]+')'))
    else:
        Dataset = Extraction_Database("""
                                      Select Distinct RARID, HNAME, HAGEI, HBWEI, HDRAW, HJRAT, HWEIC, JNAME, RESFO, RACLS, RADAT, RARAL,
                                      RADIS, RAGOG, RALOC, RARUN, RATRA, SNAME
                                      from RaceDb where RARID = ?
                                      Order By HNAME
                                      """, [int(list(Race_ID.values)[0])])
    #Print Time Taken to Load
    print("---- %s Races are Extracted from RaceDb in %s seconds / %s minutes----" \
    %(len(Race_ID), (str(round((time.time() - start_time),4))),(str(round(((time.time() - start_time))/60,4)))))

    return Dataset


def Dataset_Extraction(Race_ID_List):

    """
    Function for Extracting Datasets from FeatureDb
    Feature Set : Data used for Feature Engineering
    Modelling Set : Data used for trining base models and Hyperparameter Selection
    Ensemble Set : Data used for training Ensemble Model
    Testing Set : Testing Final Model
    X Dataset for all Sets should be the same
    They should go through the same Preprocessing Pipeline
    Parameter
    --------
    Dataset_Type : Feature, Modelling, Ensemble, Harville, Testing
    Race_ID : pd.Dataframe of RaceID
    Return
    ------
    Output ndArray  of Panda Series

    """
    #Start Timer
    start_time = time.time()

    #Select Race ID where
    Race_ID_List = Extraction_Database("""
                                       Select Distinct RARID from FeatureDb
                                       where RARID in {RARID} and CC_FRB = 0
                                       """.format(RARID = '('+str(Race_ID_List['RARID'].tolist())[1:-1]+')'))
    """
    Constructing X_Dataset
    """
    #Get Feature for one race
    X_Dataset = Extraction_Database("""
                                     Select * from FeatureDb where RARID in {RARID}
                                     Order By RARID, HNAME
                                     """.format(RARID = '('+str(Race_ID_List['RARID'].tolist())[1:-1]+')'))

    #Convert all features into floats
    col_list = X_Dataset.columns[2:]
    for col in col_list:
        X_Dataset[col] = X_Dataset[col].astype(float)

    #Get RADIS, RALOC, RATRA
    X_Condition = Extraction_Database("""
                                      Select RARID, HNAME, RADIS, RALOC, RATRA from RaceDb where RARID in {RARID}
                                      Order by RARID, HNAME
                                      """.format(RARID = '('+str(Race_ID_List['RARID'].tolist())[1:-1]+')'))
    #Merging Dataset
    X_Dataset = X_Condition.merge(X_Dataset, on = ['HNAME', 'RARID'])

    """
    Constructing Y_Dataset
    """
    #Ensemble Model
    Y_Dataset = Extraction_Database("""
                                    Select RARID, HNAME, RESFO, RESWL, RESFP, ODPLA
                                     from RaceDb where RARID in {RARID}
                                     Order By RARID, HNAME
                                    """.format(RARID = '('+str(Race_ID_List['RARID'].tolist())[1:-1]+')'))
    #Convert all features into floats
    col_list = Y_Dataset.columns[2:]
    for col in col_list:
        Y_Dataset[col] = Y_Dataset[col].astype(float)

    #Print Time Taken to Load
    print("---- %s Races are Extracted from FeatureDb in %s seconds ----"%(len(Race_ID_List), (str(round((time.time() - start_time),4)))))

    return X_Dataset, Y_Dataset


def Get_RaceID(Season_List):

    """
    Function for Extracting RaceID from RaceDb
    Parameter
    --------
    Season : Season of Data to extract, in format of a List ['2017', '2018']
    Return
    ------
    Race_ID : Dataframe of RaceID
    """
    #Extracting Race ID from RaceDb
    Season_List = '('+str([int(i) for i in Season_List])[1:-1]+')'

    Extraction = Extraction_Database("""
                                     Select Distinct RARID from RaceDb where RASEA in {Season_List}
                                     """.format(Season_List=Season_List))

    return Extraction

