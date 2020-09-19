#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ivan
"""

"""
Database Mannagement
"""

#importing library
import sqlite3
import pandas as pd

def Load_Dataset_toDatabase(Database, Dataset):

    """
    Loading Dataset to a Given Database
    Parameter
    ---------
    Db : String of Database Name : 'RaceDb'
    Dataset : Dataframe
    """

    #Create Connection
    with sqlite3.connect('Data.db') as db:
        sql = db.cursor()
        db.commit()

    #Loading Dataset to Database
    Dataset.to_sql(name = Database, con = db, if_exists = 'append', index = False)

    #Closing Connection
    sql.close()
    db.close()

    return None  #print("---- Data is Loaded to Database in %s seconds ----" %(str(round((time.time() - start_time),4))))


def Extraction_Database(query, values=[]):

    """
    Extraction of Data from RaceDb
    Parameter
    ---------
    query : SQL Query string statement
    value : values to be passed with the query
    Return
    -------
    Dataframe of retult
    """

    #Create Connection
    with sqlite3.connect('Data.db') as db:
        db.commit()

    Dataframe = pd.DataFrame()

    if values == [] :
        Dataframe = pd.read_sql_query(sql = query, con = db)
    else :
        Dataframe = pd.read_sql_query(sql = query, con = db, params = values)
    db.commit()

    #Close Connection
    db.close()

    return Dataframe


def General_Query_Database(query):

    """
    Update Database or Generl SQL Query
    Parameter
    ---------
    query : SQL Query string statement
    Return
    -------
    None
    """

    #Create Connection
    with sqlite3.connect('Data.db') as db:
        sql = db.cursor()
        db.commit()

    #Execute query
    sql.execute(query)
    db.commit()
    db.close()

    return None


def Create_RaceDb():

    """
    Create RaceDb
    """

    #Create Connection
    with sqlite3.connect('Data.db') as db:
        sql = db.cursor()
        db.commit()

    #Create RaceDb Table
    sql.execute("""
                select name from sqlite_master where name = 'RaceDb'
                """)
    result = sql.fetchall()
    keep_table = True
    if len(result) == 1 :
        response = input("The Table RaceDb already exists, do you wish to recerate it with new data (y/n): ")
        if response == "y":
            keep_table = False
            print("The RaceDb table will be recreated - all previous data will be lost and replaced by new data")
            sql.execute("drop table if exists RaceDb")
            db.commit()
        else :
            keep_table = True
            print("The existing table will be kept and new data will be appended to the end")
    else :
        keep_table = False
    if not keep_table:
        sql.execute("""
                    Create Table RaceDb(
                    HBWEI integer, HAGEI integer, HDRAW integer, HJRAT integer, HNUMI integer, HNAME text,
                    HWEIC integer,JNAME text, RACLS integer, RARAL text, RADAT text, RADIS integer, RAGOG text,
                    RALOC text, RARID integer, RARUN integer, RASEA integer,
                    RATRA text, RESFP integer, RESFT real, RESP1 integer,
                    RESP2 integer, RESP3 integer, RESP4 integer, RESP5 integer, RESP6 integer,
                    RESPB integer,RESS1 real, RESS2 real, RESS3 real, RESS4 real, RESS5 real,
                    RESS6 real, RESSP real, RESWB integer, RESWD real, RESWL integer, RESWT real,
                    SNAME text, RESFO real, ODPLA real)
                    """)
        db.commit()

    #Closing Connection
    sql.close()
    db.close()

    return print("---- RaceDb is Created ----")


def Create_FeatureDb(Feature_List):

    """
    Create FeatureDb
    Parameters
    ----------
    Feature_List = List of Feature Names, Identical with Function Name
    All Features are of Type 'real'
    """
    #Formatting Feature List
    Feature_Str = ""
    for Feature in Feature_List:
        Feature_Str = Feature_Str + Feature + ' real, '
    Feature_Str = 'RARID integer, HNAME text, ' + Feature_Str[:-2]

    query = """ Create Table FeatureDb(%s)""" %(Feature_Str)

    #Create Connection
    with sqlite3.connect('Data.db') as db:
        sql = db.cursor()
        db.commit()

    #Create FeatureDb Table
    sql.execute("""
                select name from sqlite_master where name = 'FeatureDb'
                """)
    result = sql.fetchall()
    keep_table = True
    if len(result) == 1 :
        response = input("The Table FeatureDb already exists, do you wish to recerate it with new data (y/n): ")
        if response == "y":
            keep_table = False
            print("The FeatureDb table will be recreated - all previous data will be lost and replaced by new data")
            sql.execute("Drop table if exists FeatureDb")
            db.commit()
        else :
            keep_table = True
            print("The existing table will be kept and new data will be appended to the end")
    else :
        keep_table = False
    if not keep_table:
        General_Query_Database(query)

    #Closing Connection
    sql.close()
    db.close()

    return print("---- FeatureDb is Created ----")


def Create_Irregular_Record():

    """
    Create Irregular_RecordDb

    """

    query = """ Create Table Irregular_RecordDb(
                HNAME text, INCIDENT_DATE text)"""

    #Create Connection
    with sqlite3.connect('Data.db') as db:
        sql = db.cursor()
        db.commit()

    #Create FeatureDb Table
    sql.execute("""
                select name from sqlite_master where name = 'Irregular_RecordDb'
                """)
    result = sql.fetchall()
    keep_table = True
    if len(result) == 1 :
        response = input("The Table Irregular_RecordDb already exists, do you wish to recerate it with new data (y/n): ")
        if response == "y":
            keep_table = False
            print("The Irregular_RecordDb table will be recreated - all previous data will be lost and replaced by new data")
            sql.execute("Drop table if exists Irregular_RecordDb")
            db.commit()

        else :
            keep_table = True
            print("The existing table will be kept and new data will be appended to the end")
    else :
        keep_table = False
    if not keep_table:
        General_Query_Database(query)

    #Closing Connection
    sql.close()
    db.close()

    return print("---- Irregular_RecordDb is Created ----")


def Create_SNameDb():

    """
    Create SNameDb
    """

    query = """ Create Table SNameDb(
                SNAME text, SNum text)"""

    #Create Connection
    with sqlite3.connect('Data.db') as db:
        sql = db.cursor()
        db.commit()

    #Create FeatureDb Table
    sql.execute("""
                select name from sqlite_master where name = 'SNameDb'
                """)
    result = sql.fetchall()
    keep_table = True
    if len(result) == 1 :
        response = input("The Table SNameDb already exists, do you wish to recerate it with new data (y/n): ")
        if response == "y":
            keep_table = False
            print("The SNameDb table will be recreated - all previous data will be lost and replaced by new data")
            sql.execute("Drop table if exists SNameDb")
            db.commit()

        else :
            keep_table = True
            print("The existing table will be kept and new data will be appended to the end")
    else :
        keep_table = False
    if not keep_table:
        General_Query_Database(query)

    #Closing Connection
    sql.close()
    db.close()

    return print("---- SNameDb is Created ----")


def Create_Race_PosteriorDb():

    """
    Create Race_PosteriorDb
    """

    query = """
            Create Table Race_PosteriorDb(HNAME text, JNAME text, SNAME text, RARID integer, RADAT real, Profile text, RARAL text, RALOC text, HDRAW integer, RADIS integer,
            RATRA text, RAGOG text, RACLS text, RESFP integer, BEYER_SPEED real, EARLY_PACE real, FINAL_FRACTION_PACE	real, AVERAGE_PAGE real,
            SUSTAINED_PACE real, EARLY_ENERGY real, BEATEN_FIGURE real, PACE_S1 real, PACE_S2 real, PACE_S3 real, PACE_S4 real, PACE_S5 real,
            PACE_S6 real, HPRE_DIST_RES real, HPRE_GO_RES real, HPRE_SUR_RES real, HPRE_PFL_RES real, JPRE_DIST_RES real, JPRE_GO_RES real,
            JPRE_SUR_RES real, JPRE_LOC_RES real, JPRE_PFL_RES real, SPRE_DIST_RES real, SPRE_GO_RES real,SPRE_SUR_RES real, SPRE_LOC_RES real,
            SPRE_PFL_RES real)
            """

    #Create Connection
    with sqlite3.connect('Data.db') as db:
        sql = db.cursor()
        db.commit()

    #Create FeatureDb Table
    sql.execute("""
                select name from sqlite_master where name = 'Race_PosteriorDb'
                """)
    result = sql.fetchall()
    keep_table = True
    if len(result) == 1 :
        response = input("The Table Race_PosteriorDb already exists, do you wish to recerate it with new data (y/n): ")
        if response == "y":
            keep_table = False
            print("The Race_PosteriorDb table will be recreated - all previous data will be lost and replaced by new data")
            sql.execute("Drop table if exists Race_PosteriorDb")
            db.commit()

        else :
            keep_table = True
            print("The existing table will be kept and new data will be appended to the end")
    else :
        keep_table = False
    if not keep_table:
        General_Query_Database(query)

    #Closing Connection
    sql.close()
    db.close()

    return print("---- Race_PosteriorDb is Created ----")


def Create_Running_StatDb():

    """
    Create columns for
    RS_ELO_H, RS_ELO_HP, RS_ELO_H_DIST, RS_ELO_HP_DIST, RS_ELO_H_GO, RS_ELO_HP_GO, RS_ELO_H_SUR
    RS_ELO_HP_SUR, RS_ELO_H_PFL, RS_ELO_HP_PFL, RS_ELO_J, RS_ELO_JP, RS_ELO_S, RS_ELO_SP
    """

    with sqlite3.connect('Data.db') as db:
        sql = db.cursor()
        db.commit()
    sql.execute("drop table if exists RS_HORSE_ELO")
    sql.execute("drop table if exists RS_JOCKEY_ELO")
    sql.execute("drop table if exists RS_STABLE_ELO")
    sql.execute("""Create Table RS_HORSE_ELO(HNAME text, HELO real, HELO_MOM real, HELO_STO real, HELO_1000 real,
                HELO_1000_MOM real, HELO_1000_STO real,HELO_1200 real, HELO_1200_MOM real, HELO_1200_STO real,
                HELO_1400 real, HELO_1400_MOM real, HELO_1400_STO real,HELO_1600 real, HELO_1600_MOM real,
                HELO_1600_STO real, HELO_1650 real, HELO_1650_MOM real,
                HELO_1650_STO real, HELO_1800 real, HELO_1800_MOM real, HELO_1800_STO real, HELO_2000 real,
                HELO_2000_MOM real, HELO_2000_STO real, HELO_2200 real,
                HELO_2200_MOM real, HELO_2200_STO real, HELO_2400 real, HELO_2400_MOM real, HELO_2400_STO real,
                HELO_TURF real, HELO_TURF_MOM real, HELO_TURF_STO real, HELO_AW real, HELO_AW_MOM real,
                HELO_AW_STO real, HELO_GF real, HELO_GF_MOM real, HELO_GF_STO real, HELO_G real, HELO_G_MOM real,
                HELO_G_STO real, HELO_GL real, HELO_GL_MOM real, HELO_GL_STO real, HELO_L real, HELO_L_MOM real,
                HELO_L_STO real, HELO_LS real, HELO_LS_MOM real, HELO_LS_STO real, HELO_S real, HELO_S_MOM real,
                HELO_S_STO real, HELO_WF real, HELO_WF_MOM real, HELO_WF_STO real, HELO_MF real, HELO_MF_MOM real,
                HELO_MF_STO real, HELO_MG real, HELO_MG_MOM real, HELO_MG_STO real, HELO_WS real, HELO_WS_MOM real,
                HELO_WS_STO real, HV_1000_T real, HV_1000_T_MOM real, HV_1000_T_STO real, ST_1000_T real,
                ST_1000_T_MOM real, ST_1000_T_STO real, ST_1200_AW real, ST_1200_AW_MOM real, ST_1200_AW_STO real,
                HV_1200_T real, HV_1200_T_MOM real, HV_1200_T_STO real, ST_1200_T real, ST_1200_T_MOM real,
                ST_1200_T_STO real, ST_1400_T real, ST_1400_T_MOM real, ST_1400_T_STO real, ST_1600_T real,
                ST_1600_T_MOM real, ST_1600_T_STO real, ST_1650_AW real, ST_1650_AW_MOM real, ST_1650_AW_STO real,
                HV_1650_T real, HV_1650_T_MOM real, HV_1650_T_STO real, ST_1800_AW real, ST_1800_AW_MOM real,
                ST_1800_AW_STO real, HV_1800_T real, HV_1800_T_MOM real, HV_1800_T_STO real, ST_1800_T real,
                ST_1800_T_MOM real, ST_1800_T_STO real, ST_2000_T real, ST_2000_T_MOM real, ST_2000_T_STO real,
                HV_2200_T real, HV_2200_T_MOM real, HV_2200_T_STO real, ST_2200_T real, ST_2200_T_MOM real,
                ST_2200_T_STO real, ST_2400_T real, ST_2400_T_MOM real, ST_2400_T_STO real)""")

    sql.execute("""Create Table RS_JOCKEY_ELO(JNAME text, JELO real, JELO_MOM real, JELO_STO real)""")
    sql.execute("""Create Table RS_STABLE_ELO(SNAME text, SELO real, SELO_MOM real, SELO_STO real)""")

    db.commit()

    #Closing Connection
    sql.close()
    db.close()

    return print("---- 3 Running Statistic Databases are Created ----")




