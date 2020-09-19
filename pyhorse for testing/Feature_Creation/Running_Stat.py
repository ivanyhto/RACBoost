#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ivan
"""

"""
Feature : Running Statistics - ELO
Returning Updated Rating before current race and probability for current race
Update_Running_Stat_ELO should be ran after the completion of every raceday
"""

#Loading Libraries
import numpy as np
import pandas as pd
from functools import reduce
from pyhorse.Database_Management import Extraction_Database, General_Query_Database, Load_Dataset_toDatabase

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

============================ Update Functions ============================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Going_map = {'好快': 'GF',
              '好地': 'G',
              '好黏': 'GL',
              '黏地': 'L',
              '黏軟': 'LS',
              '軟地': 'S',
              '濕快': 'WF',
              '泥快': 'MF',
              '泥好': 'MG',
              '濕慢': 'WS'}

def Update_Running_Stat(Dataset):

    """
    Update All Running Statistics Table after raceday
    Parameter
    ---------
    Dataset : Dataframe of Post-Match Data of one Raceday
    """
    for RARID, Race in Dataset.groupby('RARID'):
        Horse_ELO = []
        HNAME_List = '('+str(Race['HNAME'].tolist())[1:-1]+')'
        JNAME_List = '('+str(Race['JNAME'].tolist())[1:-1]+')'
        SNAME_List = '('+str(Race['SNAME'].tolist())[1:-1]+')'
        Dist_ELO = 'HELO_'+str(Race.loc[:,'RADIS'].values[0])
        Sur_ELO = 'HELO_' + Race.loc[:,'RATRA'].apply(lambda x : 'TURF' if x == 'T' else 'AW').values[0]
        GOG_ELO = 'HELO_' + Going_map[Race.loc[:,'RAGOG'].values[0].strip()]
        PFL_ELO = str(Race.loc[:,'RALOC'].values[0])+'_'+str(Race.loc[:,'RADIS'].values[0])+'_'\
            +str(Race.loc[:,'RATRA'].values[0])
        for Target in ['HELO',Dist_ELO,Sur_ELO,GOG_ELO,PFL_ELO]:
            Horse_ELO.append(Calculate_HELO(Target, Race, K = 128))
        HELO_DF = reduce(lambda x, y: pd.merge(x, y, on = 'HNAME'), Horse_ELO)
        JELO_DF = Calculate_JELO(Race, K = 128)
        SELO_DF = Calculate_SELO(Race, K = 128)

        #Update HELO Score to Database
        HPrior_DF = Extraction_Database("""
                                       Select * from RS_HORSE_ELO where HNAME in {HNAME_List}
                                       """.format(HNAME_List = HNAME_List))

        HPrior_DF = HPrior_DF.loc[:,[i for i in HPrior_DF.columns if i not in HELO_DF.columns[1:]]]
        HELO_DF = HELO_DF.merge(HPrior_DF, how='left')

        General_Query_Database("""
                               DELETE FROM RS_HORSE_ELO where HNAME in {HNAME_List}
                               """.format(HNAME_List = HNAME_List))
        Load_Dataset_toDatabase('RS_HORSE_ELO',HELO_DF)

        #Update JELO Score to Database
        JPrior_DF = Extraction_Database("""
                                       Select * from RS_JOCKEY_ELO where JNAME in {JNAME_List}
                                       """.format(JNAME_List = JNAME_List))

        JPrior_DF = JPrior_DF.loc[:,[i for i in JPrior_DF.columns if i not in JELO_DF.columns[1:]]]
        JELO_DF = JELO_DF.merge(JPrior_DF, how='left')

        General_Query_Database("""
                               DELETE FROM RS_JOCKEY_ELO where JNAME in {JNAME_List}
                               """.format(JNAME_List = JNAME_List))
        Load_Dataset_toDatabase('RS_JOCKEY_ELO',JELO_DF)

        #Update SELO Score to Database
        SPrior_DF = Extraction_Database("""
                                       Select * from RS_STABLE_ELO where SNAME in {SNAME_List}
                                       """.format(SNAME_List = SNAME_List))

        SPrior_DF = SPrior_DF.loc[:,[i for i in SPrior_DF.columns if i not in SELO_DF.columns[1:]]]
        SELO_DF = SELO_DF.merge(SPrior_DF, how='left')

        General_Query_Database("""
                               DELETE FROM RS_STABLE_ELO where SNAME in {SNAME_List}
                               """.format(SNAME_List = SNAME_List))
        Load_Dataset_toDatabase('RS_STABLE_ELO',SELO_DF)

    return None

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

=============================== ELO Rating ===============================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
RS_ELO_H
"""

def RS_ELO_H(Dataframe, HNAME_List, Raceday):

    """
    ELO Score of Horse
    Parameter
    ---------
    Matchday : Matchday Dataframe
    HNAME_List : String of List of Horse Names
    Raceday : Date of Race
    Return
    ------
    Dataframe [HNAME, RS_ELO_H]
    """

    Feature_DF = Dataframe.loc[:,['HNAME', 'RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, HELO RS_ELO_H from RS_HORSE_ELO
                                     where HNAME in {HNAME_List}
                                     """.format(HNAME_List = HNAME_List))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'RS_ELO_H'].fillna(1500, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME', 'RS_ELO_H']]

    return Feature_DF

"""
RS_ELO_HP
"""

def RS_ELO_HP(Dataframe, HNAME_List, Raceday):

    """
    Horse's ELO Score Implied Probability
    Parameter
    ---------
    Matchday : Matchday Dataframe
    Return
    ------
    Dataframe [HNAME, RS_ELO_HP]
    """

    Feature_DF = Dataframe.loc[:,['HNAME', 'RARID']]

    Extraction = Extraction_Database("""
                                     Select HNAME, HELO RS_ELO_H from RS_HORSE_ELO
                                     where HNAME in {HNAME_List}
                                     """.format(HNAME_List = HNAME_List))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'RS_ELO_H'].fillna(1500, inplace = True)

    #Implied Probability Function
    Feature_DF = Feature_DF.merge(ELO_Expected(Feature_DF, 'HNAME').loc[:,'Expected_Score'], on = 'HNAME')
    Feature_DF.rename(columns={'Expected_Score':'RS_ELO_HP'},inplace=True)
    Feature_DF = Feature_DF.loc[:,['HNAME', 'RS_ELO_HP']]

    return Feature_DF

"""
RS_ELO_H_DIST
"""

def RS_ELO_H_DIST(Dataframe, HNAME_List, Raceday):

    """
    Horse’s ELO Score on Distance
    Parameter
    ---------
    Matchday : Matchday Dataframe
    Return
    ------
    Dataframe [HNAME, RS_ELO_H_DIST]
    """

    Feature_DF = Dataframe.loc[:,['HNAME', 'RARID']]
    Distance = Dataframe.loc[:,'RADIS'].values[0]
    Dist_Dict = Distance_Similarity(Distance)

    Extraction = Extraction_Database("""
                                     Select HNAME, HELO_1000, HELO_1200, HELO_1400, HELO_1600,
                                     HELO_1650, HELO_1800, HELO_2000, HELO_2200, HELO_2400
                                     from RS_HORSE_ELO
                                     where HNAME in {HNAME_List}
                                     """.format(HNAME_List = HNAME_List))
    Feature_DF = Feature_DF.merge(Extraction, how='left')

    ELO_DIST = []
    for index, row in Feature_DF.iterrows():
        Horse = row['HNAME']
        try :
            df = row.reset_index().loc[2:,:].dropna()
            df.columns = ['RADIS', 'ELO']
            df.loc[:,'RADIS'] = df.loc[:,'RADIS'].apply(lambda x : int(x[5:]))
            df.replace({'RADIS': Dist_Dict}, inplace = True)
            df.loc[:,'RADIS'] = np.exp(df.loc[:,'RADIS']) / np.exp(df.loc[:,'RADIS']).sum()
            ELO = df.loc[:,'RADIS'].dot(df.loc[:,'ELO'])
            ELO_DIST.append([Horse, ELO])
        except :
            ELO_DIST.append([Horse, 1500])
    Feature_DF = pd.DataFrame(ELO_DIST, columns = ['HNAME','RS_ELO_H_DIST'])
    Feature_DF.loc[:,'RS_ELO_H_DIST'].fillna(1500, inplace=True)
    Feature_DF = Feature_DF.loc[:,['HNAME', 'RS_ELO_H_DIST']]

    return Feature_DF

"""
RS_ELO_HP_DIST
"""

def RS_ELO_HP_DIST(Dataframe, HNAME_List, Raceday):

    """
    Horse’s ELO Score Implied Probability on Distance
    Parameter
    ---------
    Matchday : Matchday Dataframe
    Return
    ------
    Dataframe [HNAME, RS_ELO_HP_DIST]
    """

    Feature_DF = Dataframe.loc[:,['HNAME', 'RARID']]
    Distance = Dataframe.loc[:,'RADIS'].values[0]
    Dist_Dict = Distance_Similarity(Distance)

    Extraction = Extraction_Database("""
                                     Select HNAME, HELO_1000, HELO_1200, HELO_1400, HELO_1600,
                                     HELO_1650, HELO_1800, HELO_2000, HELO_2200, HELO_2400
                                     from RS_HORSE_ELO
                                     where HNAME in {HNAME_List}
                                     """.format(HNAME_List = HNAME_List))
    Feature_DF = Feature_DF.merge(Extraction, how='left')

    ELO_DIST = []
    for index, row in Feature_DF.iterrows():
        Horse = row['HNAME']
        try :
            df = row.reset_index().loc[2:,:].dropna()
            df.columns = ['RADIS', 'ELO']
            df.loc[:,'RADIS'] = df.loc[:,'RADIS'].apply(lambda x : int(x[5:]))
            df.replace({'RADIS': Dist_Dict}, inplace = True)
            df.loc[:,'RADIS'] = np.exp(df.loc[:,'RADIS']) / np.exp(df.loc[:,'RADIS']).sum()
            ELO = df.loc[:,'RADIS'].dot(df.loc[:,'ELO'])
            ELO_DIST.append([Horse, ELO])
        except :
            ELO_DIST.append([Horse, 1500])
    Feature_DF = Dataframe.loc[:,['HNAME', 'RARID']]
    Feature_DF = Feature_DF.merge(pd.DataFrame(ELO_DIST, columns = ['HNAME','RS_ELO_H_DIST']), how='left')
    Feature_DF.loc[:,'RS_ELO_H_DIST'].fillna(1500, inplace=True)

    #Implied Probability Function
    Feature_DF = Feature_DF.merge(ELO_Expected(Feature_DF, 'HNAME').loc[:,'Expected_Score'], on = 'HNAME')
    Feature_DF.rename(columns={'Expected_Score':'RS_ELO_HP_DIST'},inplace=True)
    Feature_DF = Feature_DF.loc[:,['HNAME', 'RS_ELO_HP_DIST']]

    return Feature_DF

"""
RS_ELO_H_GO
"""

def RS_ELO_H_GO(Dataframe, HNAME_List, Raceday):

    """
    Horse’s ELO Score on Going
    Parameter
    ---------
    Matchday : Matchday Dataframe
    Return
    ------
    Dataframe [HNAME, RS_ELO_H_GO]
    """

    Feature_DF = Dataframe.loc[:,['HNAME', 'RARID']]
    Going = Dataframe.loc[:,'RAGOG'].values[0].strip()
    Going_Dict = Going_Similarity(Going)
    Surface = Dataframe.loc[:,'RATRA'].values[0]
    if Surface == 'T' :
        Target_Col = 'HELO_GF, HELO_G, HELO_GL, HELO_L, HELO_LS, HELO_S'
    elif Surface == 'AW':
        Target_Col = 'HELO_WF, HELO_MF, HELO_MG, HELO_WS'

    Extraction = Extraction_Database("""
                                     Select HNAME, {Target_Col}
                                     from RS_HORSE_ELO
                                     where HNAME in {HNAME_List}
                                     """.format(HNAME_List = HNAME_List, Target_Col=Target_Col))
    Feature_DF = Feature_DF.merge(Extraction, how='left')

    ELO_GOG = []
    for index, row in Feature_DF.iterrows():
        row
        Horse = row['HNAME']
        try :
            df = row.reset_index().loc[2:,:].dropna()
            df.columns = ['RAGOG', 'ELO']
            df.loc[:,'RAGOG'] = df.loc[:,'RAGOG'].apply(lambda x : str(x[5:]))
            df.replace({'RAGOG': Going_Dict}, inplace = True)
            df.loc[:,'RAGOG'] = np.exp(df.loc[:,'RAGOG']) / np.exp(df.loc[:,'RAGOG']).sum()
            ELO = df.loc[:,'RAGOG'].dot(df.loc[:,'ELO'])
            ELO_GOG.append([Horse, ELO])
        except :
            ELO_GOG.append([Horse, 1500])
    Feature_DF = pd.DataFrame(ELO_GOG, columns = ['HNAME','RS_ELO_H_GO'])
    Feature_DF.loc[:,'RS_ELO_H_GO'].fillna(1500, inplace=True)
    Feature_DF = Feature_DF.loc[:,['HNAME', 'RS_ELO_H_GO']]

    return Feature_DF

"""
RS_ELO_HP_GO
"""

def RS_ELO_HP_GO(Dataframe, HNAME_List, Raceday):

    """
    Horse’s ELO Score Implied Probability on Going
    Parameter
    ---------
    Matchday : Matchday Dataframe
    Return
    ------
    Dataframe [HNAME, RS_ELO_HP_GO]
    """

    Feature_DF = Dataframe.loc[:,['HNAME', 'RARID']]
    Going = Dataframe.loc[:,'RAGOG'].values[0].strip()
    Going_Dict = Going_Similarity(Going)
    Surface = Dataframe.loc[:,'RATRA'].values[0]
    if Surface == 'T' :
        Target_Col = 'HELO_GF, HELO_G, HELO_GL, HELO_L, HELO_LS, HELO_S'
    elif Surface == 'AW':
        Target_Col = 'HELO_WF, HELO_MF, HELO_MG, HELO_WS'

    Extraction = Extraction_Database("""
                                     Select HNAME, {Target_Col}
                                     from RS_HORSE_ELO
                                     where HNAME in {HNAME_List}
                                     """.format(HNAME_List = HNAME_List, Target_Col=Target_Col))
    Feature_DF = Feature_DF.merge(Extraction, how='left')

    ELO_GOG = []
    for index, row in Feature_DF.iterrows():
        row
        Horse = row['HNAME']
        try :
            df = row.reset_index().loc[2:,:].dropna()
            df.columns = ['RAGOG', 'ELO']
            df.loc[:,'RAGOG'] = df.loc[:,'RAGOG'].apply(lambda x : str(x[5:]))
            df.replace({'RAGOG': Going_Dict}, inplace = True)
            df.loc[:,'RAGOG'] = np.exp(df.loc[:,'RAGOG']) / np.exp(df.loc[:,'RAGOG']).sum()
            ELO = df.loc[:,'RAGOG'].dot(df.loc[:,'ELO'])
            ELO_GOG.append([Horse, ELO])
        except :
            ELO_GOG.append([Horse, 1500])
    Feature_DF = Dataframe.loc[:,['HNAME', 'RARID']]
    Feature_DF = Feature_DF.merge(pd.DataFrame(ELO_GOG, columns = ['HNAME','RS_ELO_H_GO']), how='left')
    Feature_DF.loc[:,'RS_ELO_H_GO'].fillna(1500, inplace=True)

    #Implied Probability Function
    Feature_DF = Feature_DF.merge(ELO_Expected(Feature_DF, 'HNAME').loc[:,'Expected_Score'], on = 'HNAME')
    Feature_DF.rename(columns={'Expected_Score':'RS_ELO_HP_GO'},inplace=True)
    Feature_DF = Feature_DF.loc[:,['HNAME', 'RS_ELO_HP_GO']]

    return Feature_DF

"""
RS_ELO_H_SUR
"""

def RS_ELO_H_SUR(Dataframe, HNAME_List, Raceday):

    """
    Horse’s ELO Score on Surface
    Parameter
    ---------
    Matchday : Matchday Dataframe
    Return
    ------
    Dataframe [HNAME, RS_ELO_H_SUR]
    """

    Feature_DF = Dataframe.loc[:,['HNAME', 'RARID']]
    Surface = Dataframe.loc[:,'RATRA'].values[0]
    if Surface == 'T' :
        Target = 'HELO_TURF'
    elif Surface == 'AW':
        Target = 'HELO_AW'

    Extraction = Extraction_Database("""
                                     Select HNAME, {Target} RS_ELO_H_SUR from RS_HORSE_ELO
                                     where HNAME in {HNAME_List}
                                     """.format(HNAME_List = HNAME_List, Target=Target))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'RS_ELO_H_SUR'].fillna(1500, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME', 'RS_ELO_H_SUR']]

    return Feature_DF

"""
RS_ELO_HP_SUR
"""

def RS_ELO_HP_SUR(Dataframe, HNAME_List, Raceday):

    """
    Horse’s ELO Score Implied Probability on Surface
    Parameter
    ---------
    Matchday : Matchday Dataframe
    Return
    ------
    Dataframe [HNAME, RS_ELO_HP_SUR]
    """

    Feature_DF = Dataframe.loc[:,['HNAME', 'RARID']]
    Surface = Dataframe.loc[:,'RATRA'].values[0]
    if Surface == 'T' :
        Target = 'HELO_TURF'
    elif Surface == 'AW':
        Target = 'HELO_AW'

    Extraction = Extraction_Database("""
                                     Select HNAME, {Target} RS_ELO_H_SUR from RS_HORSE_ELO
                                     where HNAME in {HNAME_List}
                                     """.format(HNAME_List = HNAME_List, Target=Target))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'RS_ELO_H_SUR'].fillna(1500, inplace = True)

    #Implied Probability Function
    Feature_DF = Feature_DF.merge(ELO_Expected(Feature_DF, 'HNAME').loc[:,'Expected_Score'], on = 'HNAME')
    Feature_DF.rename(columns={'Expected_Score':'RS_ELO_HP_SUR'},inplace=True)
    Feature_DF = Feature_DF.loc[:,['HNAME', 'RS_ELO_HP_SUR']]

    return Feature_DF

"""
RS_ELO_H_PFL
"""

def RS_ELO_H_PFL(Dataframe, HNAME_List, Raceday):

    """
    Horse’s ELO Score on Profile
    Parameter
    ---------
    Matchday : Matchday Dataframe
    Return
    ------
    Dataframe [HNAME, RS_ELO_H_PFL]
    """

    Feature_DF = Dataframe.loc[:,['HNAME', 'RARID']]
    Target = Dataframe.loc[:,'RALOC'].values[0] + "_" + str(Dataframe.loc[:,'RADIS'].values[0]) + "_" +Dataframe.loc[:,'RATRA'].values[0]

    Extraction = Extraction_Database("""
                                     Select HNAME, {Target} RS_ELO_H_PFL from RS_HORSE_ELO
                                     where HNAME in {HNAME_List}
                                     """.format(HNAME_List = HNAME_List, Target=Target))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'RS_ELO_H_PFL'].fillna(1500, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME', 'RS_ELO_H_PFL']]

    return Feature_DF

"""
RS_ELO_HP_PFL
"""

def RS_ELO_HP_PFL(Dataframe, HNAME_List, Raceday):

    """
    Horse’s ELO Score Implied Probability on Profile
    Parameter
    ---------
    Matchday : Matchday Dataframe
    Return
    ------
    Dataframe [HNAME, RS_ELO_HP_PFL]
    """

    Feature_DF = Dataframe.loc[:,['HNAME', 'RARID']]
    Target = Dataframe.loc[:,'RALOC'].values[0] + "_" + str(Dataframe.loc[:,'RADIS'].values[0]) + "_" +Dataframe.loc[:,'RATRA'].values[0]

    Extraction = Extraction_Database("""
                                     Select HNAME, {Target} RS_ELO_H_PFL from RS_HORSE_ELO
                                     where HNAME in {HNAME_List}
                                     """.format(HNAME_List = HNAME_List, Target=Target))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'RS_ELO_H_PFL'].fillna(1500, inplace = True)

    #Implied Probability Function
    Feature_DF = Feature_DF.merge(ELO_Expected(Feature_DF, 'HNAME').loc[:,'Expected_Score'], on = 'HNAME')
    Feature_DF.rename(columns={'Expected_Score':'RS_ELO_HP_PFL'},inplace=True)
    Feature_DF = Feature_DF.loc[:,['HNAME', 'RS_ELO_HP_PFL']]

    return Feature_DF

"""
RS_ELO_J
"""

def RS_ELO_J(Dataframe, HNAME_List, Raceday):

    """
    Jockey’s ELO Score
    Parameter
    ---------
    Matchday : Matchday Dataframe
    Return
    ------
    Dataframe [HNAME, RS_ELO_J]
    """

    Feature_DF = Dataframe.loc[:,['HNAME', 'JNAME']]
    JNAME_List = '('+str(Dataframe['JNAME'].tolist())[1:-1]+')'

    Extraction = Extraction_Database("""
                                     Select JNAME, JELO RS_ELO_J from RS_JOCKEY_ELO
                                     where JNAME in {JNAME_List}
                                     """.format(JNAME_List = JNAME_List))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'RS_ELO_J'].fillna(1500, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME', 'RS_ELO_J']]

    return Feature_DF

"""
RS_ELO_JP
"""

def RS_ELO_JP(Dataframe, HNAME_List, Raceday):

    """
    Jockey’s ELO Score Implied Probability
    Parameter
    ---------
    Matchday : Matchday Dataframe
    Return
    ------
    Dataframe [HNAME, RS_ELO_JP]
    """

    Feature_DF = Dataframe.loc[:,['HNAME', 'JNAME']]
    JNAME_List = '('+str(Dataframe['JNAME'].tolist())[1:-1]+')'

    Extraction = Extraction_Database("""
                                     Select JNAME, JELO RS_ELO_J from RS_JOCKEY_ELO
                                     where JNAME in {JNAME_List}
                                     """.format(JNAME_List = JNAME_List))

    Feature_DF = Feature_DF.merge(Extraction, how='left')
    Feature_DF.loc[:,'RS_ELO_J'].fillna(1500, inplace = True)

    #Implied Probability Function
    Feature_DF = Feature_DF.merge(ELO_Expected(Feature_DF, 'JNAME').loc[:,'Expected_Score'], on = 'JNAME')
    Feature_DF.rename(columns={'Expected_Score':'RS_ELO_JP'},inplace=True)
    Feature_DF = Feature_DF.loc[:,['HNAME', 'RS_ELO_JP']]

    return Feature_DF

"""
RS_ELO_S
"""

def RS_ELO_S(Dataframe, HNAME_List, Raceday):

    """
    Stable’s ELO Score
    Parameter
    ---------
    Matchday : Matchday Dataframe
    Return
    ------
    Dataframe [HNAME, RS_ELO_S]
    """

    Feature_DF = Dataframe.loc[:,['HNAME', 'SNAME']]
    SNAME_List = '('+str(Dataframe['SNAME'].tolist())[1:-1]+')'

    Extraction = Extraction_Database("""
                                     Select SNAME, SELO RS_ELO_S from RS_STABLE_ELO
                                     where SNAME in {SNAME_List}
                                     """.format(SNAME_List = SNAME_List))

    Feature_DF = Feature_DF.merge(Extraction, on='SNAME', how='left')
    Feature_DF.loc[:,'RS_ELO_S'].fillna(1500, inplace = True)
    Feature_DF = Feature_DF.loc[:,['HNAME', 'RS_ELO_S']]

    return Feature_DF

"""
RS_ELO_SP
"""

def RS_ELO_SP(Dataframe, HNAME_List, Raceday):

    """
    Stable’s ELO Score Implied Probability
    Parameter
    ---------
    Matchday : Matchday Dataframe
    Return
    ------
    Dataframe [HNAME, RS_ELO_SP]
    """

    Feature_DF = Dataframe.loc[:,['HNAME', 'SNAME']]
    SNAME_List = '('+str(Dataframe['SNAME'].tolist())[1:-1]+')'

    Extraction = Extraction_Database("""
                                     Select SNAME, SELO RS_ELO_S from RS_STABLE_ELO
                                     where SNAME in {SNAME_List}
                                     """.format(SNAME_List = SNAME_List))

    Feature_DF = Feature_DF.merge(Extraction, on='SNAME', how='left')
    Feature_DF.loc[:,'RS_ELO_S'].fillna(1500, inplace = True)

    #Implied Probability Function
    Feature_DF = Feature_DF.merge(ELO_Expected(Feature_DF, 'SNAME').loc[:,'Expected_Score'].reset_index(), on = 'SNAME')
    Feature_DF.rename(columns={'Expected_Score':'RS_ELO_SP'},inplace=True)
    Feature_DF = Feature_DF.groupby('HNAME').mean().reset_index()
    Feature_DF = Feature_DF.loc[:,['HNAME', 'RS_ELO_SP']]

    return Feature_DF

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

    Going_Dict = {'好快': {'GF':1, 'G':-1/5, 'GL':-2/11, 'L':-3/11, 'LS':-4/11, 'S':-5/11},
                  '好地': {'GF':-1/6, 'G':1, 'GL':-1/12, 'L':-1/6, 'LS':-1/4, 'S':-1/3},
                  '好黏': {'GF':-2/13, 'G':-1/13, 'GL':1, 'L':-1/13, 'LS':-2/13, 'S':-3/13},
                  '黏地': {'GF':-3/14, 'G':-1/7, 'GL':-1/14, 'L':1, 'LS':-1/14, 'S':-2/14},
                  '黏軟': {'GF':-4/15, 'G':-1/5, 'GL':-2/15, 'L':-1/15, 'LS':1, 'S':-1/15},
                  '軟地': {'GF':-5/16, 'G':-1/4, 'GL':-3/16, 'L':-2/16, 'LS':-1/16, 'S':1},

                  '濕快': {'WF':1, 'MF':-1/10, 'MG':-2/10, 'WS':-4/10},
                  '泥快': {'WF':-1/11, 'MF':1, 'MG':-1/11, 'WS':-3/11},
                  '泥好': {'WF':-2/12, 'MF':-1/12, 'MG':1, 'WS':-1/3},
                  '濕慢': {'WF':-3/8, 'MF':-5/16, 'MG':-1/4, 'WS':1}}

    return Going_Dict[Going]


def ELO_Expected(DataFrame, Target = 'HNAME'):

    """
    Parameter
    ---------
    Dataframe : DataFrame containing a vector of Previous ELO Score
    Returns
    -------
    Dataframe : DataFrame with extra column of Expected Score
    """
    #Copy Dataframe to prevent cloning
    df = DataFrame.copy()
    df.reset_index(inplace = True, drop = True)

    #Number of Horse / Jockey / Stable
    Num = len(df)

    #Number of Games
    Num_Game = Num * (Num - 1) / 2

    #Expected score
    i = 0
    for cur in df.itertuples():
        Current_ELO = float(cur[3])
        #Calculate Estimated Score
        Expected = 0
        for opp in df.itertuples():
            Opponent_ELO = float(opp[3])
            if opp[1] != cur[1]: #opp[1] != cur[1] :
                Expected +=  1 / ( 1 + 10 ** ((Opponent_ELO - Current_ELO)/400) )
            #M = 400, control how much of a degree of luck is involved in the game

        #Update Expected Score in DataFrame
        df.loc[i,'Expected_Score'] = Expected / Num_Game
        i += 1

    #Normalilse Expected Score
    df.loc[:,'Expected_Score'] = df.loc[:,'Expected_Score'].map(lambda x : x/ df['Expected_Score'].sum())
    #Set Index to HNAME
    df.set_index(Target, inplace = True)

    return df


def ELO_Actual(DataFrame, Target = 'HNAME'):

    """
    Parameter
    ---------
    Dataframe : DataFrame containing a vector of Race Result
    Returns
    -------
    Dataframe : DataFrame with extra column of Actual Score
    """
    #Copy Dataframe to prevent cloning
    df = DataFrame.copy()

    #Number of Games
    Num = len(df)

    #Number of Games
    Num_Game = Num * (Num - 1) / 2

    df['Actual_Score'] = ( Num - df['RESFP'] ) / Num_Game

    #Set Index to HNAME
    df.set_index(Target, inplace = True)

    return df


def Calculate_HELO(Target, Dataframe, K = 128):

    """
    Parameter
    Dataset : DataFrame of Daily Post Match Result
    """

    ELO_DF = Dataframe.loc[:,['HNAME','RESFP']]
    HNAME_List = '('+str(Dataframe['HNAME'].tolist())[1:-1]+')'

    Extraction = Extraction_Database("""
                                     Select HNAME, {Target} Prior, {Mom} Momentum, {Sto} Storage
                                     from RS_HORSE_ELO where HNAME in {HNAME_List}
                                     """.format(Target=Target, Mom=Target+'_MOM',Sto=Target+'_STO',
                                     HNAME_List = HNAME_List))

    #Filling ELO Prior - First Racers with 1500 + Random
    ELO_DF = ELO_DF.merge(Extraction, on=['HNAME'], how='left')

    # print(ELO_DF.loc[:, 'Prior'])

    # ran = pd.DataFrame(1500 + np.random.randn(len(ELO_DF.index),1), columns=ELO_DF.Prior, index=ELO_DF.index)
    # ELO_DF.loc[:, 'Prior'].update(ran)

    for row in ELO_DF.loc[ELO_DF['Prior'].isnull()].index.values:
        ELO_DF.loc[row, 'Prior'] = 1500 + np.random.randint(-4,4)

    #Filling Momentum - First Racers with WWWW
    ELO_DF.loc[:, 'Momentum'].fillna('WWWW', inplace = True)

    #Filling StorageFirst Racers with 0
    ELO_DF.loc[:, 'Storage'].fillna(0, inplace = True)

    """
    Posterior ELO Score with Rx = Rx' + k(Ax - Ex)
    #K is used to control the variability of ELO
    """
    Adjustment = (ELO_Actual(ELO_DF,'HNAME').loc[:,'Actual_Score']
                  - ELO_Expected(ELO_DF, 'HNAME').loc[:,'Expected_Score']).reset_index()
    Adjustment.columns = ['HNAME','Adj']
    ELO_DF = ELO_DF.merge(Adjustment, on=['HNAME'])
    ELO_DF.loc[:,'ELO'] = ELO_DF.loc[:,'Prior'] + K * ELO_DF.loc[:,'Adj']

    """
    Momentum
    """
    #Make a Copy before iterating
    df = ELO_DF.copy()
    #Update Momentum according to Last Race's Result
    for index, row in df.iterrows():
        if row['ELO'] > row['Prior']:
            #Won
            ELO_DF.loc[index,'Momentum'] =  ELO_DF.loc[index,'Momentum'][1:]+'W'
        else :
            #Lost
            ELO_DF.loc[index,'Momentum'] =  ELO_DF.loc[index,'Momentum'][1:]+'L'

    #Copy updated Momentum
    df = ELO_DF.copy()
    #Update ELO Score with Momentum
    for index, row in df.iterrows():
        if row['Momentum'] in ['WLLW', 'LLLW', 'WWWL','LWWL']:
            #DO not update score when violating a streak
            ELO_DF.loc[index, 'ELO'] = ELO_DF.loc[index, 'Prior']

        if row['Momentum'] in ['WWLL', 'LLWW']:
            #Accounting back old scores
            ELO_DF.loc[index, 'ELO'] = ELO_DF.loc[index, 'ELO'] + ELO_DF.loc[index, 'Storage']

    #Update Storage
    for index, row in df.iterrows():
        if row['Momentum'] in ['WLLW', 'LLLW', 'WWWL','LWWL']:
            #Store Win as +ve , Lost as -ve
            ELO_DF.loc[index,'Storage'] = row['ELO'] - row['Prior']
        else :
            ELO_DF.loc[index,'Storage'] = 0

    #Rename Columns
    ELO_DF = ELO_DF.loc[:,['HNAME', 'ELO', 'Momentum', 'Storage']]
    ELO_DF.rename(columns={'ELO':Target, 'Momentum':Target+'_MOM', 'Storage':Target+'_STO'},inplace=True)


    return ELO_DF


def Calculate_JELO(Dataframe, K = 128):

    """
    Parameter
    Dataset : DataFrame of Daily Post Match Result
    """

    ELO_DF = Dataframe.loc[:,['JNAME','RESFP']]
    JNAME_List = '('+str(Dataframe['JNAME'].tolist())[1:-1]+')'

    Extraction_J = Extraction_Database("""
                                       Select JNAME, JELO Prior, JELO_MOM Momentum, JELO_STO Storage
                                       from RS_JOCKEY_ELO where JNAME in {JNAME_List}
                                       """.format(JNAME_List = JNAME_List))

    #Filling ELO Prior - First Racers with 1500 + Random
    JELO_DF = ELO_DF.merge(Extraction_J, on=['JNAME'], how='left')

    # ran = pd.DataFrame(1500 + np.random.randn(len(ELO_DF.index),1),
    # columns=ELO_DF.Prior, index=ELO_DF.index)
    # ELO_DF.loc[:, 'Prior'].update(ran)

    for row in JELO_DF[JELO_DF['Prior'].isnull()].index.values:
        JELO_DF.loc[row, 'Prior'] = 1500 + np.random.randint(-4,4)

    #Filling Momentum - First Racers with WWWW
    JELO_DF.loc[:, 'Momentum'].fillna('WWWW', inplace = True)

    #Filling StorageFirst Racers with 0
    JELO_DF.loc[:, 'Storage'].fillna(0, inplace = True)

    """
    Posterior ELO Score with Rx = Rx' + k(Ax - Ex)
    #K is used to control the variability of ELO
    """
    Adjustment = (ELO_Actual(JELO_DF,'JNAME').loc[:,'Actual_Score']
                  - ELO_Expected(JELO_DF, 'JNAME').loc[:,'Expected_Score']).reset_index()
    Adjustment.columns = ['JNAME','Adj']
    JELO_DF = JELO_DF.merge(Adjustment, on=['JNAME'])
    JELO_DF.loc[:,'ELO'] = JELO_DF.loc[:,'Prior'] + K * JELO_DF.loc[:,'Adj']

    """
    Momentum
    """
    #Make a Copy before iterating
    df = JELO_DF.copy()
    #Update Momentum according to Last Race's Result
    for index, row in df.iterrows():
        if row['ELO'] > row['Prior']:
            #Won
            JELO_DF.loc[index,'Momentum'] =  JELO_DF.loc[index,'Momentum'][1:]+'W'
        else :
            #Lost
            JELO_DF.loc[index,'Momentum'] =  JELO_DF.loc[index,'Momentum'][1:]+'L'

    #Copy updated Momentum
    df = JELO_DF.copy()
    #Update ELO Score with Momentum
    for index, row in df.iterrows():
        if row['Momentum'] in ['WLLW', 'LLLW', 'WWWL','LWWL']:
            #DO not update score when violating a streak
            JELO_DF.loc[index, 'ELO'] = JELO_DF.loc[index, 'Prior']

        if row['Momentum'] in ['WWLL', 'LLWW']:
            #Accounting back old scores
            JELO_DF.loc[index, 'ELO'] = JELO_DF.loc[index, 'ELO'] + JELO_DF.loc[index, 'Storage']

    #Update Storage
    for index, row in df.iterrows():
        if row['Momentum'] in ['WLLW', 'LLLW', 'WWWL','LWWL']:
            #Store Win as +ve , Lost as -ve
            JELO_DF.loc[index,'Storage'] = row['ELO'] - row['Prior']
        else :
            JELO_DF.loc[index,'Storage'] = 0

    #Rename Columns
    JELO_DF = JELO_DF.loc[:,['JNAME', 'ELO', 'Momentum', 'Storage']]
    JELO_DF.rename(columns={'ELO':'JELO', 'Momentum':'JELO_MOM', 'Storage':'JELO_STO'},inplace=True)

    return JELO_DF


def Calculate_SELO(Dataframe, K = 128):

    """
    Parameter
    Dataset : DataFrame of Daily Post Match Result
    """

    ELO_DF = Dataframe.loc[:,['SNAME','RESFP']]
    SNAME_List = '('+str(Dataframe['SNAME'].tolist())[1:-1]+')'

    Extraction_S = Extraction_Database("""
                                       Select SNAME, SELO Prior, SELO_MOM Momentum, SELO_STO Storage
                                       from RS_STABLE_ELO where SNAME in {SNAME_List}
                                       """.format(SNAME_List = SNAME_List))

    #Filling ELO Prior - First Racers with 1500 + Random
    SELO_DF = ELO_DF.merge(Extraction_S, on=['SNAME'], how='left')

    # ran = pd.DataFrame(1500 + np.random.randn(len(ELO_DF.index),1),
    # columns=ELO_DF.Prior, index=ELO_DF.index)
    # ELO_DF.loc[:, 'Prior'].update(ran)

    for row in SELO_DF[SELO_DF['Prior'].isnull()].index.values:
        SELO_DF.loc[row, 'Prior'] = 1500 + np.random.randint(-4,4)

    #Filling Momentum - First Racers with WWWW
    SELO_DF.loc[:, 'Momentum'].fillna('WWWW', inplace = True)

    #Filling StorageFirst Racers with 0
    SELO_DF.loc[:, 'Storage'].fillna(0, inplace = True)

    """
    Posterior ELO Score with Rx = Rx' + k(Ax - Ex)
    #K is used to control the variability of ELO
    """
    Adjustment = (ELO_Actual(SELO_DF,'SNAME').loc[:,'Actual_Score']
                  - ELO_Expected(SELO_DF, 'SNAME').loc[:,'Expected_Score']).reset_index()
    Adjustment.columns = ['SNAME','Adj']
    SELO_DF = SELO_DF.merge(Adjustment, on=['SNAME'])
    SELO_DF.loc[:,'ELO'] = SELO_DF.loc[:,'Prior'] + K * SELO_DF.loc[:,'Adj']

    #One Stable Multiple Horses in Race
    SELO_DF_Prior = SELO_DF.groupby('SNAME').apply(np.mean)
    SELO_DF_Prior.drop(['Prior', 'Storage','Adj'], axis=1, inplace = True)
    SELO_DF_Prior.reset_index(inplace = True)
    SELO_DF.drop(['RESFP','ELO'], axis=1, inplace = True)
    SELO_DF = SELO_DF_Prior.merge(SELO_DF, on = 'SNAME', how='left')
    SELO_DF.drop_duplicates(subset='SNAME', keep='first', inplace = True)

    """
    Momentum
    """
    #Make a Copy before iterating
    df = SELO_DF.copy()
    #Update Momentum according to Last Race's Result
    for index, row in df.iterrows():
        if row['ELO'] > row['Prior']:
            #Won
            SELO_DF.loc[index,'Momentum'] =  SELO_DF.loc[index,'Momentum'][1:]+'W'
        else :
            #Lost
            SELO_DF.loc[index,'Momentum'] =  SELO_DF.loc[index,'Momentum'][1:]+'L'

    #Copy updated Momentum
    df = SELO_DF.copy()
    #Update ELO Score with Momentum
    for index, row in df.iterrows():
        if row['Momentum'] in ['WLLW', 'LLLW', 'WWWL','LWWL']:
            #DO not update score when violating a streak
            SELO_DF.loc[index, 'ELO'] = SELO_DF.loc[index, 'Prior']

        if row['Momentum'] in ['WWLL', 'LLWW']:
            #Accounting back old scores
            SELO_DF.loc[index, 'ELO'] = SELO_DF.loc[index, 'ELO'] + SELO_DF.loc[index, 'Storage']

    #Update Storage
    for index, row in df.iterrows():
        if row['Momentum'] in ['WLLW', 'LLLW', 'WWWL','LWWL']:
            #Store Win as +ve , Lost as -ve
            SELO_DF.loc[index,'Storage'] = row['ELO'] - row['Prior']
        else :
            SELO_DF.loc[index,'Storage'] = 0

    #Rename Columns
    SELO_DF = SELO_DF.loc[:,['SNAME', 'ELO', 'Momentum', 'Storage']]
    SELO_DF.rename(columns={'ELO':'SELO', 'Momentum':'SELO_MOM', 'Storage':'SELO_STO'},inplace=True)

    return SELO_DF







