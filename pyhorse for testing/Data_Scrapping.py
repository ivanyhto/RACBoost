#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ivan
"""

"""
Data Scrapping
"""

#Import Libraries
import re
import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from pyhorse.Database_Management import Extraction_Database, Load_Dataset_toDatabase

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

========================== Pre - Match Functions ==========================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def Get_Racecard(race):

    #Selenium Options
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--log-level=OFF")
    #Getting HTML
    web_driver = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)

    #Get List of Races on Day
    web_driver.get('https://racing.hkjc.com/racing/Info/meeting/RaceCard/chinese/Local/')
    time.sleep(2)
    raw_html = web_driver.page_source
    page = BeautifulSoup(raw_html, 'html5lib')
    table = page.find('div', attrs={'id':'racecard'}).find_all('table')
    race_list = [i.split('"')[0] for i in  str(table[1]).split('Local/')[1:]]
    race_list.insert(0,race_list[3][:-1]+'1')

    MatchDay_Dataset_List = []

    for race in race_list:
        web_driver.get('https://racing.hkjc.com/racing/Info/meeting/RaceCard/chinese/Local/'+race)
        time.sleep(2)
        raw_html = web_driver.page_source
        page = BeautifulSoup(raw_html, 'html5lib')

        One_Race_Dataset = pd.DataFrame()

        Condition = str(page.find('div', attrs={'id':'racecard'}).find_all('table')[3])
        Result = []
        table = page.find_all('table')[8]
        rows = table.find_all('tr')
        for row in rows:
            cols = row.find_all('td')
            cols = [ele.text.strip() for ele in cols]
            Result.append([ele for ele in cols if ele])
        Result = pd.DataFrame(Result[1:])

        #HNUMI
        One_Race_Dataset['HNUMI'] = Result[0]
        #HNAME
        One_Race_Dataset['HNAME'] = Result[2]
        #HBWEI
        One_Race_Dataset['HBWEI'] = Result[4]
        #JNAME
        One_Race_Dataset['JNAME'] = Result[5].str.replace("\W", '')
        One_Race_Dataset['JNAME'] = One_Race_Dataset['JNAME'].str.replace("\d", '')
        #HDRAW
        One_Race_Dataset['HDRAW'] = Result[6]
        #SNAME
        One_Race_Dataset['SNAME'] = Result[7]
        #HJRAT
        One_Race_Dataset['HJRAT'] = Result[8]
        #HWEIC
        One_Race_Dataset['HWEIC'] = Result[10]
        #RARUN
        One_Race_Dataset['RARUN'] = Result[1].astype(bool).sum(axis=0)
        #RACLS
        One_Race_Dataset['RACLS'] = Condition.split('班')[-2][-1]
        One_Race_Dataset['RACLS'].replace(['一','二','三','四','五'],[1, 2, 3, 4, 5], inplace = True)
        #RADAT
        mon_day = list(map(lambda x: '0'+x if len(x) == 1 else x, \
                  [Condition.split('年')[1].split('月')[0],  Condition.split('月')[1].split('日')[0]]))
        One_Race_Dataset['RADAT'] = Condition.split('年')[0][-4:] + mon_day[0] + mon_day[1]
        #RADIS
        One_Race_Dataset['RADIS'] = Condition.split('米')[0][-4:]
        #RALOC
        One_Race_Dataset['RALOC'] = Condition.split('星期')[1][3:5]
        One_Race_Dataset['RALOC'].replace(['沙田','跑馬'],['ST', 'HV'], inplace = True)
        #RAGOG
        One_Race_Dataset['RAGOG'] = Condition.split('米')[1][2:4]
        #RATRA
        One_Race_Dataset['RATRA'] = Condition.split('地')[1][-1]
        One_Race_Dataset['RATRA'].replace(['草','泥'],['T', 'AW'], inplace = True)
        #RARID
        race_num = pd.DataFrame([race.split('/')[-1]]).replace(['1','2','3','4','5','6','7','8','9'],\
                               ['01','02','03','04','05','06','07','08','09'])
        One_Race_Dataset['RARID'] = Condition.split('年')[0][-4:] + mon_day[0] + mon_day[1] + race_num.values[0][0]

        MatchDay_Dataset_List.append(One_Race_Dataset)


    #Closing Session
    web_driver.close()

    return MatchDay_Dataset_List


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

========================= Post Matchday Functions =========================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Get List of Raceday in a season
RaceDay_Num =  {'2012' : [i for i in range(1866, 1949)], '2013':[i for i in range(1949, 2032)],
                    '2014':[i for i in range(2032, 2115)],'2015':[i for i in range(2115, 2198)],
                    '2016':[i for i in range(2198, 2286)],'2017':[i for i in range(2286, 2374)],
                '2018':[1,2,3,2417,2430,2437,2440,2445,2452,2455,2462,2466,2471,2477,2482,2486,2491,
                        2495,2500,2504,2509,2513,2518,2522,2527,2530,2531,2533,2534,2535,
                        2537,2538,2540,2541,2542,2543,2547,2548,2577,2579]+list(range(2374,2386))\
                        +list(range(2550,2562))+list(range(2563,2565))+list(range(2566,2576))+list(range(2585,2598)),
                '2019': list(range(2598, 2601))+[2602]+list(range(2604,2618))+list(range(2619,2683))}


def Scrapping_Data_Season(Season_list):

    """
    Scrapping Historical Data of a certain Season
    Parameter :
    ------------
    Season_list : List of Season : ['2015','2016']
    Return :
    ------------
    Dataframe
    """
    #Start Timer
    start_time = time.time()

    RaceDay_List = []
    for i in Season_list:
        RaceDay_List += RaceDay_Num[i]
    #Convert to strings
    RaceDays = ["{:02d}".format(x) for x in RaceDay_List]

    #Loop through Days
    for RaceDay in RaceDays:
        Scrapping_Data_Day(RaceDay)
        time.sleep(10)

    print("---- %s Racedays are Scrapped in %s hours ----"
          %(len(RaceDays), str(round((time.time() - start_time)/3600,4))))

    return None

RaceDay = '1890'

def Scrapping_Data_Day(RaceDay):

    """
    Scrapping Historical Data of a certain day
    """
    #Start Timer
    print(RaceDay)
    start_time = time.time()

    #Scrap Data
    All_race_Result = Get_data(RaceDay)

    #Loop through Races
    Dataset = Formatting_Result(All_race_Result[0])
    for i in range(1,len(All_race_Result)):
        data = Formatting_Result(All_race_Result[i])
        Dataset = Dataset.append(data, sort = False)

    #Reset Index
    Dataset.reset_index(inplace = True, drop = True)

    #Loading to Dataset
    Load_Dataset_toDatabase('RaceDb', Dataset)

    print("---- %s Races are Scrapped in %s minutes ----" %(len(All_race_Result), str(round((time.time() - start_time)/60,4))))

    return Dataset

# data = All_race_Result[7]
# df = Formatting_Result(data)

def Formatting_Result(data, Odds):

    """
    Formatting the 3 tables into Dataset
    Patameter
    ---------
    Result : page of Beautiful Soup parsed html
    Return
    -------
    Dataframe of Result
    """
    """
    Set Up
    """
    Dataframe = pd.DataFrame()

    Result = []
    table = data.find('table', attrs={'class':'race_entry'})
    rows = table.find_all('tr')
    for row in rows:
        cols = row.find_all('td')
        cols = [ele.text.strip() for ele in cols]
        Result.append([ele for ele in cols if ele])
    Result = pd.DataFrame(Result[1:-2])

    """
    If last row is missing horse number
    """
    try :
        last_item = int(Result[0].tail(1).values[0])
    except :
        last_item = Result[0].tail(1).values[0]
    try :
        last_last_item = int(Result[0].tail(2).values[0])
    except :
        last_last_item = Result[0].tail(2).values[0]
    if type(last_item) == str:
        Result.iloc[-1] = pd.concat([pd.Series([200]), Result.iloc[-1][:-1]]).reset_index(drop=True)
    if type(last_last_item) == str:
        Result.iloc[-2] = pd.concat([pd.Series([201]), Result.iloc[-2][:-1]]).reset_index(drop=True)

    #HNUMI
    Dataframe['HNUMI'] = Result[0].map(int)

    #RADAT
    Condition = str(data).split('race-day-race__content')[3:]
    mon_day = list(map(lambda x: '0'+x if len(x) == 1 else x, \
                       [Condition[0].split('年')[1].split('月')[0], Condition[0].split('月')[1].split('日')[0]]))
    Date = Condition[0].split('年')[0][-4:] + mon_day[0] + mon_day[1]

    #Delete horses that did not race
    Irregular_List =  ['退出','00.00','-']
    Irr_Horse = Result.index[Result[13].map(lambda x: x in Irregular_List)].tolist() \
    + Result.index[Result[14].map(lambda x: x in Irregular_List)].tolist() \
    + Result.index[Result[11].map(lambda x: x in Irregular_List)].tolist()

    #Load HNAME to Irregular_RecordDb
    Irr_HNAME = pd.DataFrame()
    Irr_HNAME['HNAME'] = Result.loc[Irr_Horse][1].drop_duplicates()
    Irr_HNAME['INCIDENT_DATE'] = Date
    Load_Dataset_toDatabase('Irregular_RecordDb', Irr_HNAME)
    Result.loc[Irr_Horse,1:] = 0

    # #Remove Duplicates
    Irr_Horse = list(set(Irr_Horse))
    # Irr_Horse = Result[0][list(set(Irr_Horse))].tolist()
    # Irr_Horse = [int(x) for x in Irr_Horse]

    #Delete horses that did not race
    try :
        # Irr_HNum = list(Result.loc[Irr_Horse,0].values)
        Dataframe.drop(Irr_Horse, inplace = True)
        Result.drop(Irr_Horse, inplace = True)
    except:
        pass

    """
    Results
    """
    #HBWEI
    Dataframe['HBWEI'] = Result[8].map(int)

    #HAGEI
    Dataframe['HAGEI'] = Result[2].map(int)

    #HDRAW
    Dataframe['HDRAW'] = Result[5].map(int)

    #HJRAT
    try :
        Dataframe['HJRAT'] = Result[6].map(int)
    except :
        try :
            if Condition[1].split('評分 (')[1].split(')<')[0][-2:] in ['',[]]:
                Dataframe['HJRAT'] = 0
            else :
                Dataframe['HJRAT'] =  Condition[1].split('評分 (')[1].split(')<')[0][-2:]
        except :
            Dataframe['HJRAT'] = 0
    #HNAME
    Dataframe['HNAME'] = Result[1]

    #HWEIC
    Dataframe['HWEIC'] = Result[4].map(int)

    #JNAME
    Dataframe['JNAME'] = Result[3]

    #RARUN
    Dataframe['RARUN'] = Result[1].astype(bool).sum(axis=0)

    #RESFP
    def RESFP(x):
        if '併頭馬' in str(x):
            x = x.replace('併頭馬','')
        if '平頭馬' in str(x):
            x = x.replace('平頭馬','')
        return x
    Dataframe['RESFP'] = Result[14].map(RESFP)

    #RESFT
    def finishing_time(x):
        try :
            if len(x)==7:
                return round(int(x.split('.')[0]) * 60 + float(x[2:]),2)
            else :
                return float(x)
        except :
            pass
    Dataframe['RESFT'] = Result[16].map(finishing_time)

    #RESP123456, RESS123456
    #Remove Irr Races
    Res_copy = Result.copy()
    try :
        Res_copy.drop(Irr_Horse, inplace = True)
    except:
        pass

    #Remove front 0.00
    SS = []
    for i in Res_copy[15]:
        i = str(i).split()
        i = [x for x in i if x!= '0.00']
        SS.append(i)
    Res_copy[15] = SS

    #Seperate Sections
    for i in range(1,7):
        Dataframe['RESP'+str(i)] = 0
        Dataframe['RESS'+str(i)] = 0
        try :
            Dataframe['RESS'+str(i)] = Res_copy[15].map(lambda x : x[i-1])
            Dataframe['RESP'+str(i)] = Res_copy[13].map(lambda x : float(str(x).split()[i-1]))
        except:
            pass
    #RESPB
    def Bets(x):
        if ',' in str(x) :
            return int(x.replace(',',''))
        else :
            return int(x) * 10000
    Dataframe['RESPB'] = Result[11].map(Bets)

    #RESWB
    Dataframe['RESWB'] = Result[10].map(Bets)

    #RESWD
    def Winning_Dist(x):
        if x == '頭馬':
            return 0
        elif x == '鼻':
            return 0.1
        elif x == '短頭':
            return 0.2
        elif x == '頭':
            return 0.3
        elif x == '頸':
            return 0.4
        elif x == '多位':
            return 50
        elif '-' in str(x):
            y = str(x).split('-')
            z = y[1].split('/')
            return int(y[0]) + ( int(z[0]) / int(z[1]) )
        elif '/' in str(x):
            y = str(x).split('/')
            return int(y[0]) / int(y[1])
        else:
            try:
                return int(x)
            except :
                return 50
    Dataframe['RESWD'] = Result[17].map(Winning_Dist)
    Dataframe['RESWD'] = Dataframe['RESWD'].replace(0, -1 * min([n for n in Dataframe['RESWD'].tolist() if n>0]))

    #RESWL
    Dataframe['RESWL'] = ( Dataframe['RESFP'].astype(int) < 2 ).astype(int)

    #RESWT
    Dataframe['RESWT'] = Dataframe['RESFT'].min()

    #SNAME
    Stable_List = []
    #Get stable numbers, filter out any & and characters
    Stable_Num  = [re.sub("\D","",Stable[:3]) for Stable in str(data).split('trainer=')[1:]]
    for _, _ in enumerate(Irr_Horse):
        Stable_Num.pop()
    for i in Stable_Num:
        Stable_List.append(Extraction_Database(""" Select SName from SNameDb where SNum = ? """,[i]).values.tolist()[0][0])
    Dataframe['SNAME'] = Stable_List

    """
    Conditions
    """
    #RACLS
    Dataframe['RACLS'] = Condition[1].split('班')[0][-1]
    if Dataframe['HJRAT'][0] == 0:
            Dataframe['RACLS'] = '五'
    Dataframe['RACLS'].replace(['一','二','三','四','五','"'],[1, 2, 3, 4, 5, 0], inplace = True)

    #RARAL
    Dataframe['RARAL'] = Condition[0].split('跑道')[0].split('地')[-1]
    def check_Rail(rail):
        if '賽' in rail:
            return 'NA'
        else :
            return rail
    Dataframe['RARAL'] = Dataframe['RARAL'].map(check_Rail)


    #RADIS
    Dataframe['RADIS'] = Condition[1].split('米')[0][-4:]

    #RADAT
    Dataframe['RADAT'] = Date

    #RESSP
    Dataframe['RESSP'] = Dataframe['RADIS'].astype(float) / Dataframe['RESFT']

    #RAGOG
    Dataframe['RAGOG'] = Condition[2].split('場地:')[1][0:3]

    #RALOC
    Dataframe['RALOC'] = Condition[0].split('日')[1][0]
    Dataframe['RALOC'].replace(['沙','快'],['ST', 'HV'], inplace = True)

    #RARID
    Dataframe['RARID'] = Condition[1].split('第')[1].split('場')[0]
    Dataframe['RARID'].replace(['一','二','三','四','五','六','七','八','九','十','十一','十二','十三','十四']\
             ,['01','02','03','04','05','06','07','08','09','10','11','12','13','14'], inplace = True)
    Dataframe['RARID'] = Date + Dataframe['RARID']

    #RASEA
    def Season_from_RARID(x):
        if int(str(x)[4:6]) < 9: #Matches before September -> Considered Last Season
            return int(str(x)[0:4]) - 1
        else :
            return int(str(x)[0:4])
    Dataframe['RASEA'] = Dataframe['RARID'].map(Season_from_RARID)

    #RATRA
    Dataframe['RATRA'] = Condition[1].split('米')[1].split('(')[1][0]
    Dataframe['RATRA'].replace(['草','泥'],['T', 'AW'], inplace = True)

    """
    Odds
    """
    # try :
    #     Result[0] = Result[0].map(int)
    #     Dataframe.set_index('HNUMI', inplace=True, drop = False)
    #     Result.set_index(0, inplace=True, drop = False)
    #     Dataframe.sort_index(inplace=True)
    #     Result.sort_index(inplace=True)
    #     Dataframe.drop(Irr_Horse)
    #     Dataframe.drop(Irr_Horse, inplace = True)
    #     Result.drop(Irr_Horse, inplace = True)
    # except:
    #     pass
    #Fill avaliable odds data
    Dataframe['RESFO'] = Result[9].map(lambda x : x.split('\n')[1].strip().split(' ')[1]).fillna(0)
    Dataframe['ODPLA'] = Result[12].fillna(0)

    Dataframe.sort_index(inplace=True)

    return Dataframe


def Get_Stable():

    #Start Timer
    start_time = time.time()
    #Selenium Options
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--log-level=OFF")
    web_driver = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)

    #Get Current list of Stables on record
    Data = Extraction_Database("""
                               Select Distinct SNum from SnameDb
                               """)['SNum'].values.tolist()
    To_scrap = list(set(map(str,range(1,1000)))- set(Data))
    Stable_DF = pd.DataFrame()
    for i in To_scrap:
        web_driver.get('https://racing.appledaily.com.hk/search-horse/result?trainer='+i)
        time.sleep(3)
        _Login(web_driver)
        #Wait for the page to load
        time.sleep(3)
        raw_html = web_driver.page_source
        try:
            name = str(raw_html).split('養馬數目')[0].split('<div data-v-6bfbe05a="">')[-2].split('</div>')[0]
            if len(name) < 10:
                Stable_DF = Stable_DF.append(pd.DataFrame([[name,i]]))
        except :
            pass
    #Closing Session
    web_driver.close()

    if len(Stable_DF) != 0:
        #Reset Index
        Stable_DF.reset_index(inplace = True, drop = True)
        Stable_DF.columns=['SNAME','SNum']
        #Loading to Dataset
        Load_Dataset_toDatabase('SNameDb', Stable_DF)
    print("---- %s Stable Names are Scrapped in %s minutes ----"
          %(len(Stable_DF), str(round((time.time() - start_time)/60,4))))

    return None


def Get_data(RaceDay):

    All_race_Result = []
    #Selenium Options
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--log-level=OFF")
    #Getting HTML
    web_driver = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)

    #Get Race
    while True:
        try :
            #Get List of Races on Day
            web_driver.get('https://racing.appledaily.com.hk/race-day/race-position?raceDay='+RaceDay)
            time.sleep(2)
            _Login(web_driver)
            time.sleep(2)
            raw_html = web_driver.page_source
            race_num = [ i.split('race=')[1].split('"')[0] for i in str(raw_html).split('/race-day/race-position?raceDay=')[1:]]
            break
        except :
            pass
#    race_num = [ i[14:19].split('"')[0] for i in str(raw_html).split('/race-day/race-position?raceDay=')[1:]]

    for i in range(len(race_num)):
        race = race_num[i]
        #Cancelled race
        if race not in ['18877']:
            while True:
                web_driver.get('https://racing.appledaily.com.hk/race-day/race-position?raceDay='+RaceDay+'&race='+race)
                time.sleep(2)
                _Login(web_driver)
                time.sleep(2)
                raw_html = web_driver.page_source
                page = BeautifulSoup(raw_html, 'html5lib')
                if 'class="race_entry"' in str(page):
                    All_race_Result.append(page)
                    break
                else :
                    time.sleep(2)
    #Closing Session
    web_driver.close()

    return All_race_Result


def _Login(web_driver):
    try:
        #Login
        web_driver.execute_script("OMOSDK.auth().redirectLogin()")
        username = web_driver.find_element_by_id("email")
        username.clear()
        username.send_keys("2634521@gmail.com")
        password = web_driver.find_element_by_name("password")
        password.clear()
        password.send_keys("26345211Abc123$$")
        web_driver.find_element_by_id("loginButton").click()
    except :
        pass
    return None


#Slice away from training dataset, only kept for feature engineering
Incomplete_Data = ['2019032301', '2019032302''2019032303','2019032304','2019032305']
