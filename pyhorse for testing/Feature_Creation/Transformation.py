#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ivan
"""

"""
Feature Transformation
"""

#Loading Libraries
import numpy as np

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

============================ Current Condition ============================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
CC_AGE_TRS
"""

def CC_AGE_TRS(Dataframe):

    """
    XX Transformation on Age of Horse
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, CC_AGE_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','CC_AGE']]
    Feature_DF.loc[:,'CC_AGE_TRS'] = Feature_DF.loc[:,'CC_AGE'].pow(-2/7)
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_AGE_TRS']]

    return Feature_DF

"""
CC_REC_DAYL_TRS
"""

def CC_REC_DAYL_TRS(Dataframe):

    """
    XX Transformation on Number of Days since last race
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, CC_REC_DAYL_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','CC_REC_DAYL']]
    Feature_DF.loc[:,'CC_REC_DAYL_TRS'] = Feature_DF.loc[:,'CC_REC_DAYL'].apply(lambda x : (1+x)**(-9/7))
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_REC_DAYL_TRS']]

    return Feature_DF

"""
CC_REC_DAYL_DIST_TRS
"""

def CC_REC_DAYL_DIST_TRS(Dataframe):

    """
    XX Transformation on Days since Last Race in relation with distance
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, CC_REC_DAYL_DIST_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','CC_REC_DAYL_DIST']]
    Feature_DF.loc[:,'CC_REC_DAYL_DIST_TRS'] = Feature_DF.loc[:,'CC_REC_DAYL_DIST'].apply(lambda x : (1+x)**(-4/3))
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_REC_DAYL_DIST_TRS']]

    return Feature_DF

"""
CC_REC_DAYL_AGE_TRS
"""

def CC_REC_DAYL_AGE_TRS(Dataframe):

    """
    XX Transformation on Number of Days since Last Race in relation to age
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, CC_REC_DAYL_AGE_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','CC_REC_DAYL_AGE']]
    Feature_DF.loc[:,'CC_REC_DAYL_AGE_TRS'] = Feature_DF.loc[:,'CC_REC_DAYL_AGE'].apply(lambda x : (1+x)**(-8/5))
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_REC_DAYL_AGE_TRS']]

    return Feature_DF

"""
CC_REC_INC_TRS
"""

def CC_REC_INC_TRS(Dataframe):

    """
    XX Transformation on Number of Days since an incident date
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, CC_REC_INC_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','CC_REC_INC']]
    Feature_DF.loc[:,'CC_REC_INC_TRS'] = Feature_DF.loc[:,'CC_REC_INC'].apply(lambda x : (1+x)**(-1/3))
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_REC_INC_TRS']]

    return Feature_DF

"""
CC_REC_DAY_LWIN_TRS
"""

def CC_REC_DAY_LWIN_TRS(Dataframe):

    """
    XX Transformation on Number of days since last win
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, CC_REC_DAY_LWIN_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','CC_REC_DAY_LWIN']]
    Feature_DF.loc[:,'CC_REC_DAY_LWIN_TRS'] = Feature_DF.loc[:,'CC_REC_DAY_LWIN'].apply(lambda x : (1+x)**(-4/3))
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_REC_DAY_LWIN_TRS']]

    return Feature_DF

"""
CC_REC_DAY_PT3_TRS
"""

def CC_REC_DAY_PT3_TRS(Dataframe):

    """
    XX Transformation on Predicted days until next Top 3
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, CC_REC_DAY_PT3_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','CC_REC_DAY_PT3']]
    min_value = min(Feature_DF.loc[:,'CC_REC_DAY_PT3'])
    Feature_DF.loc[:,'CC_REC_DAY_PT3_TRS'] = Feature_DF.loc[:,'CC_REC_DAY_PT3'].apply(lambda x : (1+x-min_value)**(9/2))
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_REC_DAY_PT3_TRS']]

    return Feature_DF

"""
CC_REC_NUM_LT3_TRS
"""

def CC_REC_NUM_LT3_TRS(Dataframe):

    """
    XX Transformation on Predicted races until next Top 3
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, CC_REC_NUM_LT3_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','CC_REC_NUM_LT3']]
    min_value = min(Feature_DF.loc[:,'CC_REC_NUM_LT3'])
    Feature_DF.loc[:,'CC_REC_NUM_LT3_TRS'] = Feature_DF.loc[:,'CC_REC_NUM_LT3'].apply(lambda x : (1+x-min_value)**(-3/8))
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_REC_NUM_LT3_TRS']]

    return Feature_DF

"""
CC_REC_NUM_DAYB_TRS
"""

def CC_REC_NUM_DAYB_TRS(Dataframe):

    """
    XX Transformation on Number of Days since best performance
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, CC_REC_NUM_DAYB_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','CC_REC_NUM_DAYB']]
    Feature_DF.loc[:,'CC_REC_NUM_DAYB_TRS'] = Feature_DF.loc[:,'CC_REC_NUM_DAYB'].apply(lambda x : (1+x)**(-9/5))
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_REC_NUM_DAYB_TRS']]

    return Feature_DF

"""
CC_CLS_TRS
"""

def CC_CLS_TRS(Dataframe):

    """
    XX Transformation on HKJC Rating
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, CC_CLS_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','CC_CLS']]
    Feature_DF.loc[:,'CC_CLS_TRS'] = Feature_DF.loc[:,'CC_CLS'].pow(5/4)
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_CLS_TRS']]

    return Feature_DF

"""
CC_CLS_D_TRS
"""

def CC_CLS_D_TRS(Dataframe):

    """
    XX Transformation on Change in HKJC Rating
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, CC_CLS_D_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','CC_CLS_D']]
    min_value = min(Feature_DF.loc[:,'CC_CLS_D'])
    Feature_DF.loc[:,'CC_CLS_D_TRS'] = Feature_DF.loc[:,'CC_CLS_D'].apply(lambda x : (1+x-min_value)**(7/5))
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_CLS_D_TRS']]

    return Feature_DF

"""
CC_CLS_CC_TRS
"""

def CC_CLS_CC_TRS(Dataframe):

    """
    XX Transformation on Competitive Class
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, CC_CLS_CC_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','CC_CLS_CC']]
    min_value = min(Feature_DF.loc[:,'CC_CLS_CC'])
    Feature_DF.loc[:,'CC_CLS_CC_TRS'] = Feature_DF.loc[:,'CC_CLS_CC'].apply(lambda x : (1+x-min_value)**(9/10))
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_CLS_CC_TRS']]

    return Feature_DF

"""
CC_CLS_CL_TRS
"""

def CC_CLS_CL_TRS(Dataframe):

    """
    XX Transformation on Competition Level
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, CC_CLS_CC_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','CC_CLS_CL']]
    min_value = min(Feature_DF.loc[:,'CC_CLS_CL'])
    Feature_DF.loc[:,'CC_CLS_CL_TRS'] = Feature_DF.loc[:,'CC_CLS_CL'].apply(lambda x : (1+x-min_value)**(-9/8))
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_CLS_CL_TRS']]

    return Feature_DF

"""
CC_BWEI_TRS
"""

def CC_BWEI_TRS(Dataframe):

    """
    XX Transformation on Bodyweight of Horse
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, CC_BWEI_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','CC_BWEI']]
    Feature_DF.loc[:,'CC_BWEI_TRS'] = Feature_DF.loc[:,'CC_BWEI'].pow(3/4)
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_BWEI_TRS']]

    return Feature_DF

"""
CC_BWEI_D_TRS
"""

def CC_BWEI_D_TRS(Dataframe):

    """
    XX Transformation on Change in Bodyweight of Horse
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, CC_BWEI_D_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','CC_BWEI_D']]
    Feature_DF.loc[:,'CC_BWEI_D_TRS'] = Feature_DF.loc[:,'CC_BWEI_D'].apply(lambda x : (1+x)**(-9/2))
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_BWEI_D_TRS']]

    return Feature_DF

"""
CC_BWEI_DWIN_TRS
"""

def CC_BWEI_DWIN_TRS(Dataframe):

    """
    XX Transformation on Bodyweight difference with Winning Performance
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, CC_BWEI_DWIN_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','CC_BWEI_DWIN']]
    Feature_DF.loc[:,'CC_BWEI_DWIN_TRS'] = Feature_DF.loc[:,'CC_BWEI_DWIN'].apply(lambda x : (1+x)**(-6/5))
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_BWEI_DWIN_TRS']]

    return Feature_DF

"""
CC_BWEI_DT3_TRS
"""

def CC_BWEI_DT3_TRS(Dataframe):

    """
    XX Transformation on Bodyweight difference with Top 3 Performance
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, CC_BWEI_DT3_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','CC_BWEI_DT3']]
    Feature_DF.loc[:,'CC_BWEI_DT3_TRS'] = Feature_DF.loc[:,'CC_BWEI_DT3'].apply(lambda x : (1+x)**(-10/9))
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_BWEI_DT3_TRS']]

    return Feature_DF

"""
CC_WEI
"""

def CC_WEI_TRS(Dataframe):

    """
    XX Transformation on Weight Carried
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, CC_WEI_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','CC_WEI']]
    Feature_DF.loc[:,'CC_WEI_TRS'] = Feature_DF.loc[:,'CC_WEI'].pow(4/3)
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_WEI_TRS']]

    return Feature_DF

"""
CC_WEI_DIST_TRS
"""

def CC_WEI_DIST_TRS(Dataframe):

    """
    XX Transformation on Relative weight relative to Distance
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, CC_WEI_DIST_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','CC_WEI_DIST']]
    min_value = min(Feature_DF.loc[:,'CC_WEI_DIST'])
    Feature_DF.loc[:,'CC_WEI_DIST_TRS'] = Feature_DF.loc[:,'CC_WEI_DIST'].apply(lambda x : (1+x-min_value)**(4/3))
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_WEI_DIST_TRS']]

    return Feature_DF

"""
CC_WEI_D_TRS
"""

def CC_WEI_D_TRS(Dataframe):

    """
    XX Transformation on Change in weight carried
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, CC_WEI_D_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','CC_WEI_D']]
    min_value = min(Feature_DF.loc[:,'CC_WEI_D'])
    Feature_DF.loc[:,'CC_WEI_D_TRS'] = Feature_DF.loc[:,'CC_WEI_D'].apply(lambda x : (1+x-min_value)**(-2/5))
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_WEI_D_TRS']]

    return Feature_DF

"""
CC_WEI_SP_TRS
"""

def CC_WEI_SP_TRS(Dataframe):

    """
    XX Transformation on Weight's effect on Speed Figure
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, CC_WEI_SP_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','CC_WEI_SP']]
    min_value = min(Feature_DF.loc[:,'CC_WEI_SP'])
    Feature_DF.loc[:,'CC_WEI_SP_TRS'] = Feature_DF.loc[:,'CC_WEI_SP'].apply(lambda x : (1+x-min_value)**(8/3))
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_WEI_SP_TRS']]

    return Feature_DF

"""
CC_WEI_EXP_TRS
"""

def CC_WEI_EXP_TRS(Dataframe):

    """
    XX Transformation on Weight carrying experience
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, CC_WEI_EXP_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','CC_WEI_EXP']]
    min_value = min(Feature_DF.loc[:,'CC_WEI_EXP'])
    Feature_DF.loc[:,'CC_WEI_EXP_TRS'] = Feature_DF.loc[:,'CC_WEI_EXP'].apply(lambda x : (1+x-min_value)**(9/4))
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_WEI_EXP_TRS']]

    return Feature_DF

"""
CC_WEI_MAX_TRS
"""

def CC_WEI_MAX_TRS(Dataframe):

    """
    XX Transformation on Weight carrying threshold
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, CC_WEI_MAX_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','CC_WEI_MAX']]
    Feature_DF.loc[:,'CC_WEI_MAX_TRS'] = Feature_DF.loc[:,'CC_WEI_MAX'].pow(9/5)
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_WEI_MAX_TRS']]

    return Feature_DF

"""
CC_WEI_BCH_TRS
"""

def CC_WEI_BCH_TRS(Dataframe):

    """
    XX Transformation on Weight carrying over the threshold
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, CC_WEI_BCH_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','CC_WEI_BCH']]
    min_value = min(Feature_DF.loc[:,'CC_WEI_BCH'])
    Feature_DF.loc[:,'CC_WEI_BCH_TRS'] = Feature_DF.loc[:,'CC_WEI_BCH'].apply(lambda x : (1+x-min_value)**(4/3))
    Feature_DF = Feature_DF.loc[:,['HNAME','CC_WEI_BCH_TRS']]

    return Feature_DF

"""
PP_EXP_NRACE_TRS
"""

def PP_EXP_NRACE_TRS(Dataframe):

    """
    XX Transformation on Number of Past Races Ran
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_EXP_NRACE_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_EXP_NRACE']]
    Feature_DF.loc[:,'PP_EXP_NRACE_TRS'] = Feature_DF.loc[:,'PP_EXP_NRACE'].apply(lambda x : (1+x)**(-7/10))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_EXP_NRACE_TRS']]

    return Feature_DF

"""
PP_EXP_NRACE_DIST_TRS
"""

def PP_EXP_NRACE_DIST_TRS(Dataframe):

    """
    XX Transformation on Number of Past Races on Distance
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_EXP_NRACE_DIST_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_EXP_NRACE_DIST']]
    Feature_DF.loc[:,'PP_EXP_NRACE_DIST_TRS'] = Feature_DF.loc[:,'PP_EXP_NRACE_DIST'].apply(lambda x : (1+x)**(-9/5))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_EXP_NRACE_DIST_TRS']]

    return Feature_DF

"""
PP_EXP_NRACE_SIM_DIST_TRS
"""

def PP_EXP_NRACE_SIM_DIST_TRS(Dataframe):

    """
    XX Transformation on Number of Past Races on similar distance
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_EXP_NRACE_SIM_DIST_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_EXP_NRACE_SIM_DIST']]
    Feature_DF.loc[:,'PP_EXP_NRACE_SIM_DIST_TRS'] = Feature_DF.loc[:,'PP_EXP_NRACE_SIM_DIST'].apply(lambda x : (1+x)**(-5/4))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_EXP_NRACE_SIM_DIST_TRS']]

    return Feature_DF

"""
PP_EXP_NRACE_GO_TRS
"""

def PP_EXP_NRACE_GO_TRS(Dataframe):

    """
    XX Transformation on Number of Past Races on Going
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_EXP_NRACE_GO_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_EXP_NRACE_GO']]
    Feature_DF.loc[:,'PP_EXP_NRACE_GO_TRS'] = Feature_DF.loc[:,'PP_EXP_NRACE_GO'].apply(lambda x : (1+x)**(-10/9))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_EXP_NRACE_GO_TRS']]

    return Feature_DF

"""
PP_EXP_NRACE_SUR_TRS
"""

def PP_EXP_NRACE_SUR_TRS(Dataframe):

    """
    XX Transformation on Number of Past Races on Surface
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_EXP_NRACE_SUR_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_EXP_NRACE_SUR']]
    Feature_DF.loc[:,'PP_EXP_NRACE_SUR_TRS'] = Feature_DF.loc[:,'PP_EXP_NRACE_SUR'].apply(lambda x : (1+x)**(-10/9))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_EXP_NRACE_SUR_TRS']]

    return Feature_DF

"""
PP_EXP_NRACE_PFL_TRS
"""

def PP_EXP_NRACE_PFL_TRS(Dataframe):

    """
    XX Transformation on Number of Past Races on Profile
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_EXP_NRACE_PFL_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_EXP_NRACE_PFL']]
    Feature_DF.loc[:,'PP_EXP_NRACE_PFL_TRS'] = Feature_DF.loc[:,'PP_EXP_NRACE_PFL'].apply(lambda x : (1+x)**(-7/3))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_EXP_NRACE_PFL_TRS']]

    return Feature_DF

"""
PP_FH_FP_CLS_TRS
"""

def PP_FH_FP_CLS_TRS(Dataframe):

    """
    XX Transformation on Finishing Position by Class
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_FH_FP_CLS_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_FH_FP_CLS']]
    Feature_DF.loc[:,'PP_FH_FP_CLS_TRS'] = Feature_DF.loc[:,'PP_FH_FP_CLS'].apply(lambda x : (1+x)**(-1/3))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_FP_CLS_TRS']]

    return Feature_DF

"""
PP_FH_FP_AVG_TRS
"""

def PP_FH_FP_AVG_TRS(Dataframe):

    """
    XX Transformation on Average Finishing Position in History
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_FH_FP_AVG_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_FH_FP_AVG']]
    min_value = min(Feature_DF.loc[:,'PP_FH_FP_AVG'])
    Feature_DF.loc[:,'PP_FH_FP_AVG_TRS'] = Feature_DF.loc[:,'PP_FH_FP_AVG'].apply(lambda x : (1+x-min_value)**(-1/4))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_FP_AVG_TRS']]

    return Feature_DF

"""
PP_FH_FP_AVGRW
"""

def PP_FH_FP_AVGRW_TRS(Dataframe):

    """
    XX Transformation on Recency Weighted Avg Finishing Position of Stable
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_FH_FP_AVGRW_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_FH_FP_AVGRW']]
    min_value = min(Feature_DF.loc[:,'PP_FH_FP_AVGRW'])
    Feature_DF.loc[:,'PP_FH_FP_AVGRW_TRS'] = Feature_DF.loc[:,'PP_FH_FP_AVGRW'].apply(lambda x : (1+x-min_value)**(9/10))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_FP_AVGRW_TRS']]

    return Feature_DF

"""
PP_FH_FP_BIN_TRS
"""

def PP_FH_FP_BIN_TRS(Dataframe):

    """
    XX Transformation on Average Binned Finishing Position in History
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_FH_FP_BIN_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_FH_FP_BIN']]
    min_value = min(Feature_DF.loc[:,'PP_FH_FP_BIN'])
    Feature_DF.loc[:,'PP_FH_FP_BIN_TRS'] = Feature_DF.loc[:,'PP_FH_FP_BIN'].apply(lambda x : (1+x-min_value)**(3/2))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_FP_BIN_TRS']]

    return Feature_DF

"""
PP_FH_FP_DIST_TRS
"""

def PP_FH_FP_DIST_TRS(Dataframe):

    """
    XX Transformation on Average Finishing Position in Distance
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_FH_FP_DIST_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_FH_FP_DIST']]
    min_value = min(Feature_DF.loc[:,'PP_FH_FP_DIST'])
    Feature_DF.loc[:,'PP_FH_FP_DIST_TRS'] = Feature_DF.loc[:,'PP_FH_FP_DIST'].apply(lambda x : (1+x-min_value)**(1/5))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_FP_DIST_TRS']]

    return Feature_DF

"""
PP_FH_FP_SIM_DIST_TRS
"""

def PP_FH_FP_SIM_DIST_TRS(Dataframe):

    """
    XX Transformation on Average Finishing Position on Similar Distance
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_FH_FP_SIM_DIST_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_FH_FP_SIM_DIST']]
    min_value = min(Feature_DF.loc[:,'PP_FH_FP_SIM_DIST'])
    Feature_DF.loc[:,'PP_FH_FP_SIM_DIST_TRS'] = Feature_DF.loc[:,'PP_FH_FP_SIM_DIST'].apply(lambda x : (1+x-min_value)**(4/9))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_FP_SIM_DIST_TRS']]

    return Feature_DF

"""
PP_FH_FP_GO_TRS
"""

def PP_FH_FP_GO_TRS(Dataframe):

    """
    XX Transformation on Average Finishing Position on Going
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_FH_FP_GO_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_FH_FP_GO']]
    min_value = min(Feature_DF.loc[:,'PP_FH_FP_GO'])
    Feature_DF.loc[:,'PP_FH_FP_GO_TRS'] = Feature_DF.loc[:,'PP_FH_FP_GO'].apply(lambda x : (1+x-min_value)**(-2/9))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_FP_GO_TRS']]

    return Feature_DF

"""
PP_FH_FP_SUR_TRS
"""

def PP_FH_FP_SUR_TRS(Dataframe):

    """
    XX Transformation on Average Finishing Position on Surface
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_FH_FP_SUR_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_FH_FP_SUR']]
    min_value = min(Feature_DF.loc[:,'PP_FH_FP_SUR'])
    Feature_DF.loc[:,'PP_FH_FP_SUR_TRS'] = Feature_DF.loc[:,'PP_FH_FP_SUR'].apply(lambda x : (1+x-min_value)**(-2/9))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_FP_SUR_TRS']]

    return Feature_DF

"""
PP_FH_FP_PFL_TRS
"""

def PP_FH_FP_PFL_TRS(Dataframe):

    """
    XX Transformation on Average Finishing Position on Profile
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_FH_FP_PFL_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_FH_FP_PFL']]
    min_value = min(Feature_DF.loc[:,'PP_FH_FP_PFL'])
    Feature_DF.loc[:,'PP_FH_FP_PFL_TRS'] = Feature_DF.loc[:,'PP_FH_FP_PFL'].apply(lambda x : (1+x-min_value)**(3/4))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_FP_PFL_TRS']]

    return Feature_DF

"""
PP_FH_FTP_TRS
"""

def PP_FH_FTP_TRS(Dataframe):

    """
    XX Transformation on Finishing Time on Profile
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_FH_FTP_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_FH_FTP']]
    Feature_DF.loc[:,'PP_FH_FTP_TRS'] = Feature_DF.loc[:,'PP_FH_FTP'].pow(1/2)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_FTP_TRS']]

    return Feature_DF

"""
PP_FH_NUMW_TRS
"""

def PP_FH_NUMW_TRS(Dataframe):

    """
    XX Transformation on Number of Wins
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_FH_NUMW_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_FH_NUMW']]
    Feature_DF.loc[:,'PP_FH_NUMW_TRS'] = Feature_DF.loc[:,'PP_FH_NUMW'].apply(lambda x : (1+x)**(-4/3))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_NUMW_TRS']]

    return Feature_DF

"""
PP_FH_HTH_TRS
"""

def PP_FH_HTH_TRS(Dataframe):

    """
    XX Transformation on Head to Head History
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_FH_HTH_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_FH_HTH']]
    min_value = min(Feature_DF.loc[:,'PP_FH_HTH'])
    Feature_DF.loc[:,'PP_FH_HTH_TRS'] = Feature_DF.loc[:,'PP_FH_HTH'].apply(lambda x : (1+x-min_value)**(-2/9))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_HTH_TRS']]

    return Feature_DF

"""
PP_FH_WIN_TRS
"""

def PP_FH_WIN_TRS(Dataframe):

    """
    XX Transformation on Has Won
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_FH_WIN_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_FH_WIN']]
    Feature_DF.loc[:,'PP_FH_WIN_TRS'] = Feature_DF.loc[:,'PP_FH_WIN'].pow(9/7)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_WIN_TRS']]

    return Feature_DF

"""
PP_FH_WINP_TRS
"""

def PP_FH_WINP_TRS(Dataframe):

    """
    XX Transformation on Win Percentage of Horse
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_FH_WINP_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_FH_WINP']]
    Feature_DF.loc[:,'PP_FH_WINP_TRS'] = Feature_DF.loc[:,'PP_FH_WINP'].apply(lambda x : (1+x)**(-9/10))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_WINP_TRS']]

    return Feature_DF

"""
PP_FH_WINPY_TRS
"""

def PP_FH_WINPY_TRS(Dataframe):

    """
    XX Transformation on Win Percentage of Horse in one year
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_FH_WINPY_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_FH_WINPY']]
    Feature_DF.loc[:,'PP_FH_WINPY_TRS'] = Feature_DF.loc[:,'PP_FH_WINPY'].apply(lambda x : (1+x)**(-3))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_WINPY_TRS']]

    return Feature_DF

"""
PP_FH_WINP_W_TRS
"""

def PP_FH_WINP_W_TRS(Dataframe):

    """
    XX Transformation on Win Percentage after Win
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_FH_WINP_W_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_FH_WINP_W']]
    Feature_DF.loc[:,'PP_FH_WINP_W_TRS'] = Feature_DF.loc[:,'PP_FH_WINP_W'].apply(lambda x : (1+x)**(-9/2))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_WINP_W_TRS']]

    return Feature_DF

"""
PP_FH_WINP_DIST_TRS
"""

def PP_FH_WINP_DIST_TRS(Dataframe):

    """
    XX Transformation on Win Percentage of Horse on Distance
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_FH_WINP_DIST_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_FH_WINP_DIST']]
    Feature_DF.loc[:,'PP_FH_WINP_DIST_TRS'] = Feature_DF.loc[:,'PP_FH_WINP_DIST'].apply(lambda x : (1+x)**(-7/2))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_WINP_DIST_TRS']]

    return Feature_DF

"""
PP_FH_WINP_SIM_DIST_TRS
"""

def PP_FH_WINP_SIM_DIST_TRS(Dataframe):

    """
    XX Transformation on Win Percentage of Horse on similar distance
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_FH_WINP_SIM_DIST_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_FH_WINP_SIM_DIST']]
    Feature_DF.loc[:,'PP_FH_WINP_SIM_DIST_TRS'] = Feature_DF.loc[:,'PP_FH_WINP_SIM_DIST'].apply(lambda x : (1+x)**(-9/2))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_WINP_SIM_DIST_TRS']]

    return Feature_DF

"""
PP_FH_WINP_GO_TRS
"""

def PP_FH_WINP_GO_TRS(Dataframe):

    """
    XX Transformation on Win Percentage of Horse on Going
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_FH_WINP_GO_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_FH_WINP_GO']]
    Feature_DF.loc[:,'PP_FH_WINP_GO_TRS'] = Feature_DF.loc[:,'PP_FH_WINP_GO'].apply(lambda x : (1+x)**(-3))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_WINP_GO_TRS']]

    return Feature_DF

"""
PP_FH_WINP_SUR_TRS
"""

def PP_FH_WINP_SUR_TRS(Dataframe):

    """
    XX Transformation on Win Percentage of Horse on Surface
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_FH_WINP_SUR_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_FH_WINP_SUR']]
    Feature_DF.loc[:,'PP_FH_WINP_SUR_TRS'] = Feature_DF.loc[:,'PP_FH_WINP_SUR'].apply(lambda x : (1+x)**(-3/2))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_WINP_SUR_TRS']]

    return Feature_DF

"""
PP_FH_WINP_PFL_TRS
"""

def PP_FH_WINP_PFL_TRS(Dataframe):

    """
    XX Transformation on Win Percentage of Horse on Profile
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_FH_WINP_PFL_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_FH_WINP_PFL']]
    Feature_DF.loc[:,'PP_FH_WINP_PFL_TRS'] = Feature_DF.loc[:,'PP_FH_WINP_PFL'].apply(lambda x : (1+x)**(-9/2))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_WINP_PFL_TRS']]

    return Feature_DF

"""
PP_FH_T3P_TRS
"""

def PP_FH_T3P_TRS(Dataframe):

    """
    XX Transformation on Top 3 Percentage of Horse
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_FH_T3P_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_FH_T3P']]
    Feature_DF.loc[:,'PP_FH_T3P_TRS'] = Feature_DF.loc[:,'PP_FH_T3P'].pow(7/8)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_T3P_TRS']]

    return Feature_DF

"""
PP_FH_T3P_T3_TRS
"""

def PP_FH_T3P_T3_TRS(Dataframe):

    """
    XX Transformation on Top 3 Percentage after T3
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_FH_T3P_T3_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_FH_T3P_T3']]
    Feature_DF.loc[:,'PP_FH_T3P_T3_TRS'] = Feature_DF.loc[:,'PP_FH_T3P_T3'].apply(lambda x : (1+x)**(-9/4))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_T3P_T3_TRS']]

    return Feature_DF

"""
PP_FH_T3P_DIST_TRS
"""

def PP_FH_T3P_DIST_TRS(Dataframe):

    """
    XX Transformation on Top 3 Percentage of Horse on Distance
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_FH_T3P_DIST_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_FH_T3P_DIST']]
    Feature_DF.loc[:,'PP_FH_T3P_DIST_TRS'] = Feature_DF.loc[:,'PP_FH_T3P_DIST'].apply(lambda x : (1+x)**(-2/3))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_T3P_DIST_TRS']]

    return Feature_DF

"""
PP_FH_T3P_SIM_DIST_TRS
"""

def PP_FH_T3P_SIM_DIST_TRS(Dataframe):

    """
    XX Transformation on Top 3 Percentage of Horse on similar distance
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_FH_T3P_SIM_DIST_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_FH_T3P_SIM_DIST']]
    Feature_DF.loc[:,'PP_FH_T3P_SIM_DIST_TRS'] = Feature_DF.loc[:,'PP_FH_T3P_SIM_DIST'].apply(lambda x : (1+x)**(-9/5))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_T3P_SIM_DIST_TRS']]

    return Feature_DF

"""
PP_FH_T3P_GO_TRS
"""

def PP_FH_T3P_GO_TRS(Dataframe):

    """
    XX Transformation on Top 3 Percentage of Horse on Going
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_FH_T3P_GO_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_FH_T3P_GO']]
    Feature_DF.loc[:,'PP_FH_T3P_GO_TRS'] = Feature_DF.loc[:,'PP_FH_T3P_GO'].apply(lambda x : (1+x)**(-1/2))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_T3P_GO_TRS']]

    return Feature_DF

"""
PP_FH_T3P_SUR_TRS
"""

def PP_FH_T3P_SUR_TRS(Dataframe):

    """
    XX Transformation on NTop 3 percentage of Horse on Surface
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_FH_T3P_SUR_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_FH_T3P_SUR']]
    Feature_DF.loc[:,'PP_FH_T3P_SUR_TRS'] = Feature_DF.loc[:,'PP_FH_T3P_SUR'].pow(4/7)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_T3P_SUR_TRS']]

    return Feature_DF

"""
PP_FH_T3P_PFL_TRS
"""

def PP_FH_T3P_PFL_TRS(Dataframe):

    """
    XX Transformation on Top 3 Percentage of Horse on Profile
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_FH_T3P_PFL_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_FH_T3P_PFL']]
    Feature_DF.loc[:,'PP_FH_T3P_PFL_TRS'] = Feature_DF.loc[:,'PP_FH_T3P_PFL'].apply(lambda x : (1+x)**(-9/5))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_FH_T3P_PFL_TRS']]

    return Feature_DF

"""
PP_BL_AVG_TRS
"""

def PP_BL_AVG_TRS(Dataframe):

    """
    XX Transformation on Average beaten length of horse in History
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_BL_AVG_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_BL_AVG']]
    min_value = min(Feature_DF.loc[:,'PP_BL_AVG'])
    Feature_DF.loc[:,'PP_BL_AVG_TRS'] = Feature_DF.loc[:,'PP_BL_AVG'].apply(lambda x : (1+x-min_value)**(-3/4))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_BL_AVG_TRS']]

    return Feature_DF

"""
PP_BL_SUM_TRS
"""

def PP_BL_SUM_TRS(Dataframe):

    """
    XX Transformation on Total beaten lengths of previous races
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_BL_SUM_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_BL_SUM']]
    min_value = min(Feature_DF.loc[:,'PP_BL_SUM'])
    Feature_DF.loc[:,'PP_BL_SUM_TRS'] = Feature_DF.loc[:,'PP_BL_SUM'].apply(lambda x : (1+x-min_value)**(-7/4))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_BL_SUM_TRS']]

    return Feature_DF

"""
PP_BL_AVGF_TRS
"""

def PP_BL_AVGF_TRS(Dataframe):

    """
    XX Transformation on Average Beaten Length Figure
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_BL_AVGF_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_BL_AVGF']]
    min_value = min(Feature_DF.loc[:,'PP_BL_AVGF'])
    Feature_DF.loc[:,'PP_BL_AVGF_TRS'] = Feature_DF.loc[:,'PP_BL_AVGF'].apply(lambda x : (1+x-min_value)**(7/2))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_BL_AVGF_TRS']]

    return Feature_DF

"""
PP_BL_AVGF_SUR_TRS
"""

def PP_BL_AVGF_SUR_TRS(Dataframe):

    """
    XX Transformation on Average Beaten Length Figure on Surface
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_BL_AVGF_SUR_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_BL_AVGF_SUR']]
    min_value = min(Feature_DF.loc[:,'PP_BL_AVGF_SUR'])
    Feature_DF.loc[:,'PP_BL_AVGF_SUR_TRS'] = Feature_DF.loc[:,'PP_BL_AVGF_SUR'].apply(lambda x : (1+x-min_value)**(-9/2))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_BL_AVGF_SUR_TRS']]

    return Feature_DF

"""
PP_SPF_L1_TRS
"""

def PP_SPF_L1_TRS(Dataframe):

    """
    XX Transformation on Last Speed Figure
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_SPF_L1_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_SPF_L1']]
    min_value = min(Feature_DF.loc[:,'PP_SPF_L1'])
    Feature_DF.loc[:,'PP_SPF_L1_TRS'] = Feature_DF.loc[:,'PP_SPF_L1'].apply(lambda x : (1+x-min_value)**(8/3))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_L1_TRS']]

    return Feature_DF

"""
PP_SPF_L2_TRS
"""

def PP_SPF_L2_TRS(Dataframe):

    """
    XX Transformation on Second Last Speed Figure
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_SPF_L2_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_SPF_L2']]
    min_value = min(Feature_DF.loc[:,'PP_SPF_L2'])
    Feature_DF.loc[:,'PP_SPF_L2_TRS'] = Feature_DF.loc[:,'PP_SPF_L2'].apply(lambda x : (1+x-min_value)**(7/3))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_L2_TRS']]

    return Feature_DF

"""
PP_SPF_SEC_TRS
"""

def PP_SPF_SEC_TRS(Dataframe):

    """
    XX Transformation on 2/3 Speed Figure
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_SPF_SEC_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_SPF_SEC']]
    min_value = min(Feature_DF.loc[:,'PP_SPF_SEC'])
    Feature_DF.loc[:,'PP_SPF_SEC_TRS'] = Feature_DF.loc[:,'PP_SPF_SEC'].apply(lambda x : (1+x-min_value)**(8/3))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_SEC_TRS']]

    return Feature_DF

"""
PP_SPF_KNN_PFL_TRS
"""

def PP_SPF_KNN_PFL_TRS(Dataframe):

    """
    XX Transformation on KNN Predict Speed Figure
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_SPF_KNN_PFL_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_SPF_KNN_PFL']]
    min_value = min(Feature_DF.loc[:,'PP_SPF_KNN_PFL'])
    Feature_DF.loc[:,'PP_SPF_KNN_PFL_TRS'] = Feature_DF.loc[:,'PP_SPF_KNN_PFL'].apply(lambda x : (1+x-min_value)**(5/2))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_KNN_PFL_TRS']]

    return Feature_DF

"""
PP_SPF_D1_TRS
"""

def PP_SPF_D1_TRS(Dataframe):

    """
    XX Transformation on Change in Speed Figure
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_SPF_D1_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_SPF_D1']]
    min_value = min(Feature_DF.loc[:,'PP_SPF_D1'])
    Feature_DF.loc[:,'PP_SPF_D1_TRS'] = Feature_DF.loc[:,'PP_SPF_D1'].apply(lambda x : (1+x-min_value)**(7/6))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_D1_TRS']]

    return Feature_DF

"""
PP_SPF_D_TRS
"""

def PP_SPF_D_TRS(Dataframe):

    """
    XX Transformation on Percentage Change in Speed Figure
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_SPF_D_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_SPF_D']]
    min_value = min(Feature_DF.loc[:,'PP_SPF_D'])
    Feature_DF.loc[:,'PP_SPF_D_TRS'] = Feature_DF.loc[:,'PP_SPF_D'].apply(lambda x : (1+x-min_value)**(1/3))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_D_TRS']]

    return Feature_DF

"""
PP_SPF_AVG_TRS
"""

def PP_SPF_AVG_TRS(Dataframe):

    """
    XX Transformation on Average Speed Figure
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_SPF_AVG_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_SPF_AVG']]
    min_value = min(Feature_DF.loc[:,'PP_SPF_AVG'])
    Feature_DF.loc[:,'PP_SPF_AVG_TRS'] = Feature_DF.loc[:,'PP_SPF_AVG'].apply(lambda x : (1+x-min_value)**(8/3))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_AVG_TRS']]

    return Feature_DF

"""
PP_SPF_AVGRW_TRS
"""

def PP_SPF_AVGRW_TRS(Dataframe):

    """
    XX Transformation on Recency Weighted Average Speed Figure
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_SPF_AVGRW_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_SPF_AVGRW']]
    min_value = min(Feature_DF.loc[:,'PP_SPF_AVGRW'])
    Feature_DF.loc[:,'PP_SPF_AVGRW_TRS'] = Feature_DF.loc[:,'PP_SPF_AVGRW'].apply(lambda x : (1+x-min_value)**(8/3))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_AVGRW_TRS']]

    return Feature_DF

"""
PP_SPF_AVG_DIST_TRS
"""

def PP_SPF_AVG_DIST_TRS(Dataframe):

    """
    XX Transformation on Average Speed Figure on Distance
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_SPF_AVG_DIST_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_SPF_AVG_DIST']]
    min_value = min(Feature_DF.loc[:,'PP_SPF_AVG_DIST'])
    Feature_DF.loc[:,'PP_SPF_AVG_DIST_TRS'] = Feature_DF.loc[:,'PP_SPF_AVG_DIST'].apply(lambda x : (1+x-min_value)**(5/2))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_AVG_DIST_TRS']]

    return Feature_DF

"""
PP_SPF_AVGRW_DIST_TRS
"""

def PP_SPF_AVGRW_DIST_TRS(Dataframe):

    """
    XX Transformation on Recency Weighted Average Speed Figure on Distance
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_SPF_AVGRW_DIST_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_SPF_AVGRW_DIST']]
    min_value = min(Feature_DF.loc[:,'PP_SPF_AVGRW_DIST'])
    Feature_DF.loc[:,'PP_SPF_AVGRW_DIST_TRS'] = Feature_DF.loc[:,'PP_SPF_AVGRW_DIST'].apply(lambda x : (1+x-min_value)**(5/2))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_AVGRW_DIST_TRS']]

    return Feature_DF

"""
PP_SPF_AVG_SIM_DIST_TRS
"""

def PP_SPF_AVG_SIM_DIST_TRS(Dataframe):

    """
    XX Transformation on Average Speed Figure on Similar Distance
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_SPF_AVG_SIM_DIST_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_SPF_AVG_SIM_DIST']]
    min_value = min(Feature_DF.loc[:,'PP_SPF_AVG_SIM_DIST'])
    Feature_DF.loc[:,'PP_SPF_AVG_SIM_DIST_TRS'] = Feature_DF.loc[:,'PP_SPF_AVG_SIM_DIST'].apply(lambda x : (1+x-min_value)**(5/2))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_AVG_SIM_DIST_TRS']]

    return Feature_DF

"""
PP_SPF_AVGRW_SIM_DIST_TRS
"""

def PP_SPF_AVGRW_SIM_DIST_TRS(Dataframe):

    """
    XX Transformation on Recency Weighted Average Speed Figure on Similar Distance
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_SPF_AVGRW_SIM_DIST_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_SPF_AVGRW_SIM_DIST']]
    min_value = min(Feature_DF.loc[:,'PP_SPF_AVGRW_SIM_DIST'])
    Feature_DF.loc[:,'PP_SPF_AVGRW_SIM_DIST_TRS'] = Feature_DF.loc[:,'PP_SPF_AVGRW_SIM_DIST'].apply(lambda x : (1+x-min_value)**(7/3))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_AVGRW_SIM_DIST_TRS']]

    return Feature_DF

"""
PP_SPF_AVG_GO_TRS
"""

def PP_SPF_AVG_GO_TRS(Dataframe):

    """
    XX Transformation on Average Speed Figure on Going
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_SPF_AVG_GO_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_SPF_AVG_GO']]
    min_value = min(Feature_DF.loc[:,'PP_SPF_AVG_GO'])
    Feature_DF.loc[:,'PP_SPF_AVG_GO_TRS'] = Feature_DF.loc[:,'PP_SPF_AVG_GO'].apply(lambda x : (1+x-min_value)**(8/3))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_AVG_GO_TRS']]

    return Feature_DF

"""
PP_SPF_AVGRW_GO_TRS
"""

def PP_SPF_AVGRW_GO_TRS(Dataframe):

    """
    XX Transformation on Recency Weighted Average Speed Figure on Going
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_SPF_AVGRW_GO_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_SPF_AVGRW_GO']]
    min_value = min(Feature_DF.loc[:,'PP_SPF_AVGRW_GO'])
    Feature_DF.loc[:,'PP_SPF_AVGRW_GO_TRS'] = Feature_DF.loc[:,'PP_SPF_AVGRW_GO'].apply(lambda x : (1+x-min_value)**(8/3))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_AVGRW_GO_TRS']]

    return Feature_DF

"""
PP_SPF_AVG_SUR_TRS
"""

def PP_SPF_AVG_SUR_TRS(Dataframe):

    """
    XX Transformation on Average Speed Figure on Surface
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_SPF_AVG_SUR_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_SPF_AVG_SUR']]
    min_value = min(Feature_DF.loc[:,'PP_SPF_AVG_SUR'])
    Feature_DF.loc[:,'PP_SPF_AVG_SUR_TRS'] = Feature_DF.loc[:,'PP_SPF_AVG_SUR'].apply(lambda x : (1+x-min_value)**(8/3))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_AVG_SUR_TRS']]

    return Feature_DF

"""
PP_SPF_AVGRW_SUR_TRS
"""

def PP_SPF_AVGRW_SUR_TRS(Dataframe):

    """
    XX Transformation on Recency Weighted Average Speed Figure on Surface
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_SPF_AVGRW_SUR_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_SPF_AVGRW_SUR']]
    min_value = min(Feature_DF.loc[:,'PP_SPF_AVGRW_SUR'])
    Feature_DF.loc[:,'PP_SPF_AVGRW_SUR_TRS'] = Feature_DF.loc[:,'PP_SPF_AVGRW_SUR'].apply(lambda x : (1+x-min_value)**(8/3))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_AVGRW_SUR_TRS']]

    return Feature_DF

"""
PP_SPF_AVG_PFL_TRS
"""

def PP_SPF_AVG_PFL_TRS(Dataframe):

    """
    XX Transformation on Average Speed Figure on Profile
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_SPF_AVG_PFL_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_SPF_AVG_PFL']]
    min_value = min(Feature_DF.loc[:,'PP_SPF_AVG_PFL'])
    Feature_DF.loc[:,'PP_SPF_AVG_PFL_TRS'] = Feature_DF.loc[:,'PP_SPF_AVG_PFL'].apply(lambda x : (1+x-min_value)**(9/5))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_AVG_PFL_TRS']]

    return Feature_DF


"""
PP_SPF_AVGRW_PFL_TRS
"""

def PP_SPF_AVGRW_PFL_TRS(Dataframe):

    """
    XX Transformation on Recency Weighted Average Speed Figure on Profile
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_SPF_AVGRW_PFL_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_SPF_AVGRW_PFL']]
    min_value = min(Feature_DF.loc[:,'PP_SPF_AVGRW_PFL'])
    Feature_DF.loc[:,'PP_SPF_AVGRW_PFL_TRS'] = Feature_DF.loc[:,'PP_SPF_AVGRW_PFL'].apply(lambda x : (1+x-min_value)**(9/5))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_AVGRW_PFL_TRS']]

    return Feature_DF

"""
PP_SPF_TOP_TRS
"""

def PP_SPF_TOP_TRS(Dataframe):

    """
    XX Transformation on Best Speed Figure
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_SPF_TOP_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_SPF_TOP']]
    min_value = min(Feature_DF.loc[:,'PP_SPF_TOP'])
    Feature_DF.loc[:,'PP_SPF_TOP_TRS'] = Feature_DF.loc[:,'PP_SPF_TOP'].apply(lambda x : (1+x-min_value)**(9/5))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_TOP_TRS']]

    return Feature_DF

"""
PP_SPF_TOPY_TRS
"""

def PP_SPF_TOPY_TRS(Dataframe):

    """
    XX Transformation on Best Speed Figure in one year
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_SPF_TOPY_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_SPF_TOPY']]
    min_value = min(Feature_DF.loc[:,'PP_SPF_TOPY'])
    Feature_DF.loc[:,'PP_SPF_TOPY_TRS'] = Feature_DF.loc[:,'PP_SPF_TOPY'].apply(lambda x : (1+x-min_value)**(8/3))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_TOPY_TRS']]

    return Feature_DF

"""
PP_SPF_TOP_DIST_TRS
"""

def PP_SPF_TOP_DIST_TRS(Dataframe):

    """
    XX Transformation on Top Speed Figure on Distance
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_SPF_TOP_DIST_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_SPF_TOP_DIST']]
    min_value = min(Feature_DF.loc[:,'PP_SPF_TOP_DIST'])
    Feature_DF.loc[:,'PP_SPF_TOP_DIST_TRS'] = Feature_DF.loc[:,'PP_SPF_TOP_DIST'].apply(lambda x : (1+x-min_value)**(9/4))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_TOP_DIST_TRS']]

    return Feature_DF

"""
PP_SPF_TOP_SIM_DIST_TRS
"""

def PP_SPF_TOP_SIM_DIST_TRS(Dataframe):

    """
    XX Transformation on Top Speed Figure on similar distance
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_SPF_TOP_SIM_DIST_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_SPF_TOP_SIM_DIST']]
    min_value = min(Feature_DF.loc[:,'PP_SPF_TOP_SIM_DIST'])
    Feature_DF.loc[:,'PP_SPF_TOP_SIM_DIST_TRS'] = Feature_DF.loc[:,'PP_SPF_TOP_SIM_DIST'].apply(lambda x : (1+x-min_value)**(9/4))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_TOP_SIM_DIST_TRS']]

    return Feature_DF

"""
PP_SPF_TOP_GO_TRS
"""

def PP_SPF_TOP_GO_TRS(Dataframe):

    """
    XX Transformation on Top Speed Figure on Going
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_SPF_TOP_GO_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_SPF_TOP_GO']]
    min_value = min(Feature_DF.loc[:,'PP_SPF_TOP_GO'])
    Feature_DF.loc[:,'PP_SPF_TOP_GO_TRS'] = Feature_DF.loc[:,'PP_SPF_TOP_GO'].apply(lambda x : (1+x-min_value)**(5/2))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_TOP_GO_TRS']]

    return Feature_DF

"""
PP_SPF_TOP_SUR_TRS
"""

def PP_SPF_TOP_SUR_TRS(Dataframe):

    """
    XX Transformation on Top Speed Figure on Surface
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_SPF_TOP_SUR_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_SPF_TOP_SUR']]
    min_value = min(Feature_DF.loc[:,'PP_SPF_TOP_SUR'])
    Feature_DF.loc[:,'PP_SPF_TOP_SUR_TRS'] = Feature_DF.loc[:,'PP_SPF_TOP_SUR'].apply(lambda x : (1+x-min_value)**(9/4))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_TOP_SUR_TRS']]

    return Feature_DF

"""
PP_SPF_TOP_PFL_TRS
"""

def PP_SPF_TOP_PFL_TRS(Dataframe):

    """
    XX Transformation on Top Speed Figure on Profile
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_SPF_TOP_PFL_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_SPF_TOP_PFL']]
    min_value = min(Feature_DF.loc[:,'PP_SPF_TOP_PFL'])
    Feature_DF.loc[:,'PP_SPF_TOP_PFL_TRS'] = Feature_DF.loc[:,'PP_SPF_TOP_PFL'].apply(lambda x : (1+x-min_value)**(5/3))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_SPF_TOP_PFL_TRS']]

    return Feature_DF

"""
PP_PAF_SPT_TRS
"""

def PP_PAF_SPT_TRS(Dataframe):

    """
    XX Transformation on Speed Point
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_PAF_SPT_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_PAF_SPT']]
    Feature_DF.loc[:,'PP_PAF_SPT_TRS'] = Feature_DF.loc[:,'PP_PAF_SPT'].apply(lambda x : (1+x)**(-9/2))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_SPT_TRS']]

    return Feature_DF

"""
PP_PAF_SPT_DIST_TRS
"""

def PP_PAF_SPT_DIST_TRS(Dataframe):

    """
    XX Transformation on Speed Poing relative to Distance
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_PAF_SPT_DIST_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_PAF_SPT_DIST']]
    Feature_DF.loc[:,'PP_PAF_SPT_DIST_TRS'] = Feature_DF.loc[:,'PP_PAF_SPT_DIST'].apply(lambda x : (1+x)**(-9/2))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_SPT_DIST_TRS']]

    return Feature_DF

"""
PP_PAF_EP_AVG_TRS
"""

def PP_PAF_EP_AVG_TRS(Dataframe):

    """
    XX Transformation on Average Early Pace
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_PAF_EP_AVG_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_PAF_EP_AVG']]
    Feature_DF.loc[:,'PP_PAF_EP_AVG_TRS'] = Feature_DF.loc[:,'PP_PAF_EP_AVG'].pow(9/7)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_EP_AVG_TRS']]

    return Feature_DF

"""
PP_PAF_EP_AVGRW_TRS
"""

def PP_PAF_EP_AVGRW_TRS(Dataframe):

    """
    XX Transformation on Recency Weighted Average Early Pace
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_PAF_EP_AVGRW_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_PAF_EP_AVGRW']]
    Feature_DF.loc[:,'PP_PAF_EP_AVGRW_TRS'] = Feature_DF.loc[:,'PP_PAF_EP_AVGRW'].pow(5/4)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_EP_AVGRW_TRS']]

    return Feature_DF

"""
PP_PAF_EP_ADV_GOPFL_TRS
"""

def PP_PAF_EP_ADV_GOPFL_TRS(Dataframe):

    """
    XX Transformation on Early Pace Advantage on Going and Profile
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_PAF_EP_ADV_GOPFL_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_PAF_EP_ADV_GOPFL']]
    Feature_DF.loc[:,'PP_PAF_EP_ADV_GOPFL_TRS'] = Feature_DF.loc[:,'PP_PAF_EP_ADV_GOPFL'].apply(lambda x : (1+x)**(-1/5))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_EP_ADV_GOPFL_TRS']]

    return Feature_DF

"""
PP_PAF_EP_ADV_PFL_TRS
"""

def PP_PAF_EP_ADV_PFL_TRS(Dataframe):

    """
    XX Transformation on Early Pace Advantage on Profile
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_PAF_EP_ADV_PFL_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_PAF_EP_ADV_PFL']]
    Feature_DF.loc[:,'PP_PAF_EP_ADV_PFL_TRS'] = Feature_DF.loc[:,'PP_PAF_EP_ADV_PFL'].apply(lambda x : (1+x)**(-5/9))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_EP_ADV_PFL_TRS']]

    return Feature_DF

"""
PP_PAF_EP_WIN_PFL_TRS
"""

def PP_PAF_EP_WIN_PFL_TRS(Dataframe):

    """
    XX Transformation on Distance from winning Early Pace
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_PAF_EP_WIN_PFL_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_PAF_EP_WIN_PFL']]
    Feature_DF.loc[:,'PP_PAF_EP_WIN_PFL_TRS'] = Feature_DF.loc[:,'PP_PAF_EP_WIN_PFL'].apply(lambda x : (1+x)**(-7/9))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_EP_WIN_PFL_TRS']]

    return Feature_DF

"""
PP_PAF_EP_DIST_TRS
"""

def PP_PAF_EP_DIST_TRS(Dataframe):

    """
    XX Transformation on Early Pace relative to Distance
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_PAF_EP_DIST_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_PAF_EP_DIST']]
    Feature_DF.loc[:,'PP_PAF_EP_DIST_TRS'] = Feature_DF.loc[:,'PP_PAF_EP_DIST'].pow(5/4)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_EP_DIST_TRS']]

    return Feature_DF

"""
PP_PAF_SP_AVG_TRS
"""

def PP_PAF_SP_AVG_TRS(Dataframe):

    """
    XX Transformation on Average Sustained Pace
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_PAF_SP_AVG_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_PAF_SP_AVG']]
    min_value = min(Feature_DF.loc[:,'PP_PAF_SP_AVG'])
    Feature_DF.loc[:,'PP_PAF_SP_AVG_TRS'] = Feature_DF.loc[:,'PP_PAF_SP_AVG'].apply(lambda x : (1+x-min_value)**(8/3))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_SP_AVG_TRS']]

    return Feature_DF

"""
PP_PAF_SP_AVGRW_TRS
"""

def PP_PAF_SP_AVGRW_TRS(Dataframe):

    """
    XX Transformation on Recency Weighted Average Sustained Pace
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_PAF_SP_AVGRW_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_PAF_SP_AVGRW']]
    min_value = min(Feature_DF.loc[:,'PP_PAF_SP_AVGRW'])
    Feature_DF.loc[:,'PP_PAF_SP_AVGRW_TRS'] = Feature_DF.loc[:,'PP_PAF_SP_AVGRW'].apply(lambda x : (1+x-min_value)**(8/3))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_SP_AVGRW_TRS']]

    return Feature_DF

"""
PP_PAF_SP_ADV_GOPFL_TRS
"""

def PP_PAF_SP_ADV_GOPFL_TRS(Dataframe):

    """
    XX Transformation on Sustained Pace Advantage on Profile and Going
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_PAF_SP_ADV_GOPFL_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_PAF_SP_ADV_GOPFL']]
    Feature_DF.loc[:,'PP_PAF_SP_ADV_GOPFL_TRS'] = Feature_DF.loc[:,'PP_PAF_SP_ADV_GOPFL'].apply(lambda x : (1+x)**(-4/5))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_SP_ADV_GOPFL_TRS']]

    return Feature_DF

"""
PP_PAF_SP_ADV_PFL_TRS
"""

def PP_PAF_SP_ADV_PFL_TRS(Dataframe):

    """
    XX Transformation on Sustained Pace Advantage on Profile
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_PAF_SP_ADV_PFL_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_PAF_SP_ADV_PFL']]
    Feature_DF.loc[:,'PP_PAF_SP_ADV_PFL_TRS'] = Feature_DF.loc[:,'PP_PAF_SP_ADV_PFL'].apply(lambda x : (1+x)**(-9/10))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_SP_ADV_PFL_TRS']]

    return Feature_DF

"""
PP_PAF_SP_WIN_PFL_TRS
"""

def PP_PAF_SP_WIN_PFL_TRS(Dataframe):

    """
    XX Transformation on Distance from winning Sustained Pace
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_PAF_SP_WIN_PFL_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_PAF_SP_WIN_PFL']]
    Feature_DF.loc[:,'PP_PAF_SP_WIN_PFL_TRS'] = Feature_DF.loc[:,'PP_PAF_SP_WIN_PFL'].apply(lambda x : (1+x)**(-5/8))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_SP_WIN_PFL_TRS']]

    return Feature_DF

"""
PP_PAF_SP_DIST_TRS
"""

def PP_PAF_SP_DIST_TRS(Dataframe):

    """
    XX Transformation on Sustained Pace relative to Distance
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_PAF_SP_DIST_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_PAF_SP_DIST']]
    min_value = min(Feature_DF.loc[:,'PP_PAF_SP_DIST'])
    Feature_DF.loc[:,'PP_PAF_SP_DIST_TRS'] = Feature_DF.loc[:,'PP_PAF_SP_DIST'].apply(lambda x : (1+x-min_value)**(8/3))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_SP_DIST_TRS']]

    return Feature_DF

"""
PP_PAF_AP_AVG_TRS
"""

def PP_PAF_AP_AVG_TRS(Dataframe):

    """
    XX Transformation on Average Average Pace
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_PAF_AP_AVG_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_PAF_AP_AVG']]
    min_value = min(Feature_DF.loc[:,'PP_PAF_AP_AVG'])
    Feature_DF.loc[:,'PP_PAF_AP_AVG_TRS'] = Feature_DF.loc[:,'PP_PAF_AP_AVG'].apply(lambda x : (1+x-min_value)**(8/3))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_AP_AVG_TRS']]

    return Feature_DF

"""
PP_PAF_AP_AVGRW_TRS
"""

def PP_PAF_AP_AVGRW_TRS(Dataframe):

    """
    XX Transformation on Recency Weighted Average Average Pace
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_PAF_AP_AVGRW_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_PAF_AP_AVGRW']]
    min_value = min(Feature_DF.loc[:,'PP_PAF_AP_AVGRW'])
    Feature_DF.loc[:,'PP_PAF_AP_AVGRW_TRS'] = Feature_DF.loc[:,'PP_PAF_AP_AVGRW'].apply(lambda x : (1+x-min_value)**(8/3))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_AP_AVGRW_TRS']]

    return Feature_DF

"""
PP_PAF_AP_ADV_GOPFL_TRS
"""

def PP_PAF_AP_ADV_GOPFL_TRS(Dataframe):

    """
    XX Transformation on Average Pace Advantage on Profile and Going
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_PAF_AP_ADV_GOPFL_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_PAF_AP_ADV_GOPFL']]
    Feature_DF.loc[:,'PP_PAF_AP_ADV_GOPFL_TRS'] = Feature_DF.loc[:,'PP_PAF_AP_ADV_GOPFL'].apply(lambda x : (1+x)**(-2/7))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_AP_ADV_GOPFL_TRS']]

    return Feature_DF

"""
PP_PAF_AP_ADV_PFL_TRS
"""

def PP_PAF_AP_ADV_PFL_TRS(Dataframe):

    """
    XX Transformation on Average Pace Advantage on Profile
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_PAF_AP_ADV_PFL_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_PAF_AP_ADV_PFL']]
    Feature_DF.loc[:,'PP_PAF_AP_ADV_PFL_TRS'] = Feature_DF.loc[:,'PP_PAF_AP_ADV_PFL'].apply(lambda x : (1+x)**(-4/9))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_AP_ADV_PFL_TRS']]

    return Feature_DF

"""
PP_PAF_AP_WIN_PFL_TRS
"""

def PP_PAF_AP_WIN_PFL_TRS(Dataframe):

    """
    XX Transformation on Distance from winning Average Pace
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_PAF_AP_WIN_PFL_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_PAF_AP_WIN_PFL']]
    Feature_DF.loc[:,'PP_PAF_AP_WIN_PFL_TRS'] = Feature_DF.loc[:,'PP_PAF_AP_WIN_PFL'].apply(lambda x : (1+x)**(-1/2))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_AP_WIN_PFL_TRS']]

    return Feature_DF

"""
PP_PAF_FP_AVG_TRS
"""

def PP_PAF_FP_AVG_TRS(Dataframe):

    """
    XX Transformation on Average Final Fraction Pace
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_PAF_FP_AVG_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_PAF_FP_AVG']]
    min_value = min(Feature_DF.loc[:,'PP_PAF_FP_AVG'])
    Feature_DF.loc[:,'PP_PAF_FP_AVG_TRS'] = Feature_DF.loc[:,'PP_PAF_FP_AVG'].apply(lambda x : (1+x-min_value)**(5/2))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_FP_AVG_TRS']]

    return Feature_DF

"""
PP_PAF_FP_AVGRW_TRS
"""

def PP_PAF_FP_AVGRW_TRS(Dataframe):

    """
    XX Transformation on Recency Weighted Average Final Fraction Pace
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_PAF_FP_AVGRW_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_PAF_FP_AVGRW']]
    min_value = min(Feature_DF.loc[:,'PP_PAF_FP_AVGRW'])
    Feature_DF.loc[:,'PP_PAF_FP_AVGRW_TRS'] = Feature_DF.loc[:,'PP_PAF_FP_AVGRW'].apply(lambda x : (1+x-min_value)**(5/2))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_FP_AVGRW_TRS']]

    return Feature_DF

"""
PP_PAF_FP_ADV_GOPFL_TRS
"""

def PP_PAF_FP_ADV_GOPFL_TRS(Dataframe):

    """
    XX Transformation on Final Fraction Pace Advantage on Profile and Going
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_PAF_FP_ADV_GOPFL_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_PAF_FP_ADV_GOPFL']]
    Feature_DF.loc[:,'PP_PAF_FP_ADV_GOPFL_TRS'] = Feature_DF.loc[:,'PP_PAF_FP_ADV_GOPFL'].apply(lambda x : (1+x)**(-5/6))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_FP_ADV_GOPFL_TRS']]

    return Feature_DF

"""
PP_PAF_FP_ADV_PFL_TRS
"""

def PP_PAF_FP_ADV_PFL_TRS(Dataframe):

    """
    XX Transformation on Final Fraction Pace Advantage on Profile
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_PAF_FP_ADV_PFL_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_PAF_FP_ADV_PFL']]
    Feature_DF.loc[:,'PP_PAF_FP_ADV_PFL_TRS'] = Feature_DF.loc[:,'PP_PAF_FP_ADV_PFL'].apply(lambda x : (1+x)**(-6/7))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_FP_ADV_PFL_TRS']]

    return Feature_DF

"""
PP_PAF_FP_WIN_PFL_TRS
"""

def PP_PAF_FP_WIN_PFL_TRS(Dataframe):

    """
    XX Transformation on Distance from winning Final Fraction Pace
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_PAF_FP_WIN_PFL_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_PAF_FP_WIN_PFL']]
    Feature_DF.loc[:,'PP_PAF_FP_WIN_PFL_TRS'] = Feature_DF.loc[:,'PP_PAF_FP_WIN_PFL'].apply(lambda x : (1+x)**(-2/3))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_FP_WIN_PFL_TRS']]

    return Feature_DF

"""
PP_PAF_EDW_DIST_TRS
"""

def PP_PAF_EDW_DIST_TRS(Dataframe):

    """
    XX Transformation on Winning Energy Distribution on Distance
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_PAF_EDW_DIST_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_PAF_EDW_DIST']]
    Feature_DF.loc[:,'PP_PAF_EDW_DIST_TRS'] = Feature_DF.loc[:,'PP_PAF_EDW_DIST'].apply(lambda x : (1+x)**(-7/5))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_EDW_DIST_TRS']]

    return Feature_DF

"""
PP_PAF_EDW_PFL_TRS
"""

def PP_PAF_EDW_PFL_TRS(Dataframe):

    """
    XX Transformation on Winning Energy Distribution on Profile
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_PAF_EDW_PFL_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_PAF_EDW_PFL']]
    Feature_DF.loc[:,'PP_PAF_EDW_PFL_TRS'] = Feature_DF.loc[:,'PP_PAF_EDW_PFL'].apply(lambda x : (1+x)**(-10/7))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_EDW_PFL_TRS']]

    return Feature_DF

"""
PP_PAF_EDL_DIST_TRS
"""

def PP_PAF_EDL_DIST_TRS(Dataframe):

    """
    XX Transformation on Winning Energy Distribution Limit on Distance
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_PAF_EDL_DIST_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_PAF_EDL_DIST']]
    Feature_DF.loc[:,'PP_PAF_EDL_DIST_TRS'] = Feature_DF.loc[:,'PP_PAF_EDL_DIST'].pow(9/2)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_EDL_DIST_TRS']]

    return Feature_DF

"""
PP_PAF_EDL_PFL_TRS
"""

def PP_PAF_EDL_PFL_TRS(Dataframe):

    """
    XX Transformation on Winning Energy Distribution Limit on Profile
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_PAF_EDL_PFL_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_PAF_EDL_PFL']]
    min_value = min(Feature_DF.loc[:,'PP_PAF_EDL_PFL'])
    Feature_DF.loc[:,'PP_PAF_EDL_PFL_TRS'] = Feature_DF.loc[:,'PP_PAF_EDL_PFL'].apply(lambda x : (1+x-min_value)**(-7/5))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_EDL_PFL_TRS']]

    return Feature_DF

"""
PP_PAF_STL_AVG_PFL_TRS
"""

def PP_PAF_STL_AVG_PFL_TRS(Dataframe):

    """
    XX Transformation on Average Final Straight Line Speed on Distance
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_PAF_STL_AVG_PFL_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_PAF_STL_AVG_PFL']]
    min_value = min(Feature_DF.loc[:,'PP_PAF_STL_AVG_PFL'])
    Feature_DF.loc[:,'PP_PAF_STL_AVG_PFL_TRS'] = Feature_DF.loc[:,'PP_PAF_STL_AVG_PFL'].apply(lambda x : (1+x-min_value)**(9/4))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_STL_AVG_PFL_TRS']]

    return Feature_DF

"""
PP_PAF_STL_B_PFL_TRS
"""

def PP_PAF_STL_B_PFL_TRS(Dataframe):

    """
    XX Transformation on Best Final Straight Line Speed
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_PAF_STL_B_PFL_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_PAF_STL_B_PFL']]
    min_value = min(Feature_DF.loc[:,'PP_PAF_STL_B_PFL'])
    Feature_DF.loc[:,'PP_PAF_STL_B_PFL_TRS'] = Feature_DF.loc[:,'PP_PAF_STL_B_PFL'].apply(lambda x : (1+x-min_value)**(9/4))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_STL_B_PFL_TRS']]

    return Feature_DF

"""
PP_PAF_BEST_TRS
"""

def PP_PAF_BEST_TRS(Dataframe):

    """
    XX Transformation on Sum of fastest section time
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_PAF_BEST_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_PAF_BEST']]
    min_value = min(Feature_DF.loc[:,'PP_PAF_BEST'])
    Feature_DF.loc[:,'PP_PAF_BEST_TRS'] = Feature_DF.loc[:,'PP_PAF_BEST'].apply(lambda x : (1+x-min_value)**(9/5))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_BEST_TRS']]

    return Feature_DF

"""
PP_PAF_BEST_GOPFL_TRS
"""

def PP_PAF_BEST_GOPFL_TRS(Dataframe):

    """
    XX Transformation on Best Sectional Time
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_PAF_BEST_GOPFL_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_PAF_BEST_GOPFL']]
    Feature_DF.loc[:,'PP_PAF_BEST_GOPFL_TRS'] = Feature_DF.loc[:,'PP_PAF_BEST_GOPFL'].pow(4/5)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_PAF_BEST_GOPFL_TRS']]

    return Feature_DF

"""
PP_EPM_TRS
"""

def PP_EPM_TRS(Dataframe):

    """
    XX Transformation on Cumulative Price Money earnings in History
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_EPM_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_EPM']]
    Feature_DF.loc[:,'PP_EPM_TRS'] = Feature_DF.loc[:,'PP_EPM'].apply(lambda x : (1+x)**(-7/4))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_EPM_TRS']]

    return Feature_DF

"""
PP_EPM_AVG_TRS
"""

def PP_EPM_AVG_TRS(Dataframe):

    """
    XX Transformation on Average Price money per race in history
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_EPM_AVG_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_EPM_AVG']]
    Feature_DF.loc[:,'PP_EPM_AVG_TRS'] = Feature_DF.loc[:,'PP_EPM_AVG'].apply(lambda x : (1+x)**(-4/3))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_EPM_AVG_TRS']]

    return Feature_DF

"""
PP_EMP_AVG_WIN_TRS
"""

def PP_EMP_AVG_WIN_TRS(Dataframe):

    """
    XX Transformation on Average Win Earnings
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_EMP_AVG_WIN_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_EMP_AVG_WIN']]
    Feature_DF.loc[:,'PP_EMP_AVG_WIN_TRS'] = Feature_DF.loc[:,'PP_EMP_AVG_WIN'].apply(lambda x : (1+x)**(-9/7))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_EMP_AVG_WIN_TRS']]

    return Feature_DF

"""
PP_EMP_AVG_PLA_TRS
"""

def PP_EMP_AVG_PLA_TRS(Dataframe):

    """
    XX Transformation on Average Place Earnings
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_EMP_AVG_PLA_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_EMP_AVG_PLA']]
    Feature_DF.loc[:,'PP_EMP_AVG_PLA_TRS'] = Feature_DF.loc[:,'PP_EMP_AVG_PLA'].pow(1/2)
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_EMP_AVG_PLA_TRS']]

    return Feature_DF

"""
PP_EMP_YR_TRS
"""

def PP_EMP_YR_TRS(Dataframe):

    """
    XX Transformation on Prize Money in last year
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_EMP_YR_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','PP_EMP_YR']]
    Feature_DF.loc[:,'PP_EMP_YR_TRS'] = Feature_DF.loc[:,'PP_EMP_YR'].apply(lambda x : (1+x)**(-3))
    Feature_DF = Feature_DF.loc[:,['HNAME','PP_EMP_YR_TRS']]

    return Feature_DF

"""
JS_J_FP_TRS
"""

def JS_J_FP_TRS(Dataframe):

    """
    XX Transformation on Average Finishing Position of Jockey
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, JS_J_FP_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JS_J_FP']]
    min_value = min(Feature_DF.loc[:,'JS_J_FP'])
    Feature_DF.loc[:,'JS_J_FP_TRS'] = Feature_DF.loc[:,'JS_J_FP'].apply(lambda x : (1+x-min_value)**(9/5))
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_FP_TRS']]

    return Feature_DF

"""
JS_J_FPRW_TRS
"""

def JS_J_FPRW_TRS(Dataframe):

    """
    XX Transformation on Recency Weighted Avg Finishing Position of Jockey
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, JS_J_FPRW_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JS_J_FPRW']]
    min_value = min(Feature_DF.loc[:,'JS_J_FPRW'])
    Feature_DF.loc[:,'JS_J_FPRW_TRS'] = Feature_DF.loc[:,'JS_J_FPRW'].apply(lambda x : (1+x-min_value)**(5/3))
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_FPRW_TRS']]

    return Feature_DF

"""
JS_J_WINP_TRS
"""

def JS_J_WINP_TRS(Dataframe):

    """
    XX Transformation on Jockey Win Percentage
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, JS_J_WINP_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JS_J_WINP']]
    Feature_DF.loc[:,'JS_J_WINP_TRS'] = Feature_DF.loc[:,'JS_J_WINP'].apply(lambda x : (1+x)**(-3/2))
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_WINP_TRS']]

    return Feature_DF

"""
JS_J_WINP_JDIST_TRS
"""

def JS_J_WINP_JDIST_TRS(Dataframe):

    """
    XX Transformation on Win Percentage of Jockey on Distance
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, JS_J_WINP_JDIST_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JS_J_WINP_JDIST']]
    Feature_DF.loc[:,'JS_J_WINP_JDIST_TRS'] = Feature_DF.loc[:,'JS_J_WINP_JDIST'].apply(lambda x : (1+x)**(-8/7))
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_WINP_JDIST_TRS']]

    return Feature_DF

"""
JS_J_WINP_JGO_TRS
"""

def JS_J_WINP_JGO_TRS(Dataframe):

    """
    XX Transformation on Win Percentage of Jockey on Going
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, JS_J_WINP_JGO_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JS_J_WINP_JGO']]
    Feature_DF.loc[:,'JS_J_WINP_JGO_TRS'] = Feature_DF.loc[:,'JS_J_WINP_JGO'].apply(lambda x : (1+x)**(-7/4))
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_WINP_JGO_TRS']]

    return Feature_DF

"""
JS_J_WINP_JSUR_TRS
"""

def JS_J_WINP_JSUR_TRS(Dataframe):

    """
    XX Transformation on Win Percentage of Jockey on Surface
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, JS_J_WINP_JSUR_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JS_J_WINP_JSUR']]
    Feature_DF.loc[:,'JS_J_WINP_JSUR_TRS'] = Feature_DF.loc[:,'JS_J_WINP_JSUR'].apply(lambda x : (1+x)**(-4/3))
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_WINP_JSUR_TRS']]

    return Feature_DF

"""
JS_J_WINP_JLOC_TRS
"""

def JS_J_WINP_JLOC_TRS(Dataframe):

    """
    XX Transformation on Win Percentage of Jockey on Location
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, JS_J_WINP_JLOC_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JS_J_WINP_JLOC']]
    Feature_DF.loc[:,'JS_J_WINP_JLOC_TRS'] = Feature_DF.loc[:,'JS_J_WINP_JLOC'].apply(lambda x : (1+x)**(-9/7))
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_WINP_JLOC_TRS']]

    return Feature_DF

"""
JS_J_WINP_JPFL_TRS
"""

def JS_J_WINP_JPFL_TRS(Dataframe):

    """
    XX Transformation on Win Percentage of Jockey on Profile
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, JS_J_WINP_JPFL_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JS_J_WINP_JPFL']]
    Feature_DF.loc[:,'JS_J_WINP_JPFL_TRS'] = Feature_DF.loc[:,'JS_J_WINP_JPFL'].apply(lambda x : (1+x)**(-9/10))
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_WINP_JPFL_TRS']]

    return Feature_DF

"""
JS_J_T3P_TRS
"""

def JS_J_T3P_TRS(Dataframe):

    """
    XX Transformation on Jocke Top 3 Percentage
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, JS_J_T3P_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JS_J_T3P']]
    Feature_DF.loc[:,'JS_J_T3P_TRS'] = Feature_DF.loc[:,'JS_J_T3P'].apply(lambda x : (1+x)**(-5/8))
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_T3P_TRS']]

    return Feature_DF

"""
JS_J_T3P_JDIST_TRS
"""

def JS_J_T3P_JDIST_TRS(Dataframe):

    """
    XX Transformation on Jockey Top 3 Percentage on Distance
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, JS_J_T3P_JDIST_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JS_J_T3P_JDIST']]
    Feature_DF.loc[:,'JS_J_T3P_JDIST_TRS'] = Feature_DF.loc[:,'JS_J_T3P_JDIST'].apply(lambda x : (1+x)**(-2/7))
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_T3P_JDIST_TRS']]

    return Feature_DF

"""
JS_J_T3P_JGO_TRS
"""

def JS_J_T3P_JGO_TRS(Dataframe):

    """
    XX Transformation on Jockey Top 3 Percentage on Going
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, JS_J_T3P_JGO_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JS_J_T3P_JGO']]
    Feature_DF.loc[:,'JS_J_T3P_JGO_TRS'] = Feature_DF.loc[:,'JS_J_T3P_JGO'].apply(lambda x : (1+x)**(-1/10))
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_T3P_JGO_TRS']]

    return Feature_DF

"""
JS_J_T3P_JSUR_TRS
"""

def JS_J_T3P_JSUR_TRS(Dataframe):

    """
    XX Transformation on Jockey Top 3 Percentage on Surface
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, JS_J_T3P_JSUR_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JS_J_T3P_JSUR']]
    Feature_DF.loc[:,'JS_J_T3P_JSUR_TRS'] = Feature_DF.loc[:,'JS_J_T3P_JSUR'].apply(lambda x : (1+x)**(-4/7))
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_T3P_JSUR_TRS']]

    return Feature_DF

"""
JS_J_T3P_JLOC_TRS
"""

def JS_J_T3P_JLOC_TRS(Dataframe):

    """
    XX Transformation on Jockey Top 3 Percentage on Location
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, JS_J_T3P_JLOC_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JS_J_T3P_JLOC']]
    Feature_DF.loc[:,'JS_J_T3P_JLOC_TRS'] = Feature_DF.loc[:,'JS_J_T3P_JLOC'].apply(lambda x : (1+x)**(-1/2))
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_T3P_JLOC_TRS']]

    return Feature_DF

"""
JS_J_T3P_JPFL_TRS
"""

def JS_J_T3P_JPFL_TRS(Dataframe):

    """
    XX Transformation on Jockey Top 3 Percentage on Profile
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, JS_J_T3P_JPFL_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JS_J_T3P_JPFL']]
    Feature_DF.loc[:,'JS_J_T3P_JPFL_TRS'] = Feature_DF.loc[:,'JS_J_T3P_JPFL'].apply(lambda x : (1+x)**(-1/7))
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_T3P_JPFL_TRS']]

    return Feature_DF

"""
JS_J_NUMR_TRS
"""

def JS_J_NUMR_TRS(Dataframe):

    """
    XX Transformation on Number of Races by Jockey
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, JS_J_NUMR_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JS_J_NUMR']]
    Feature_DF.loc[:,'JS_J_NUMR_TRS'] = Feature_DF.loc[:,'JS_J_NUMR'].pow(7/2)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_NUMR_TRS']]

    return Feature_DF

"""
JS_J_HJ_NUM_TRS
"""

def JS_J_HJ_NUM_TRS(Dataframe):

    """
    XX Transformation on Jockey Horse Number of Runs
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, JS_J_HJ_NUM_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JS_J_HJ_NUM']]
    Feature_DF.loc[:,'JS_J_HJ_NUM_TRS'] = Feature_DF.loc[:,'JS_J_HJ_NUM'].apply(lambda x : (1+x)**(-10/3))
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_HJ_NUM_TRS']]

    return Feature_DF

"""
JS_J_HJ_NWIN_TRS
"""

def JS_J_HJ_NWIN_TRS(Dataframe):

    """
    XX Transformation on Jockey Horse Number of Wins
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, JS_J_HJ_NWIN_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JS_J_HJ_NWIN']]
    Feature_DF.loc[:,'JS_J_HJ_NWIN_TRS'] = Feature_DF.loc[:,'JS_J_HJ_NWIN'].apply(lambda x : (1+x)**(-9/2))
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_HJ_NWIN_TRS']]

    return Feature_DF

"""
JS_J_HJ_NT3_TRS
"""

def JS_J_HJ_NT3_TRS(Dataframe):

    """
    XX Transformation on Jockey Horse Number of Top 3
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, JS_J_HJ_NT3_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JS_J_HJ_NT3']]
    Feature_DF.loc[:,'JS_J_HJ_NT3_TRS'] = Feature_DF.loc[:,'JS_J_HJ_NT3'].apply(lambda x : (1+x)**(-9/2))
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_HJ_NT3_TRS']]

    return Feature_DF

"""
JS_J_HJ_SPAVG_TRS
"""

def JS_J_HJ_SPAVG_TRS(Dataframe):

    """
    XX Transformation on Jockey Horse Speed Rating
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, JS_J_HJ_SPAVG_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JS_J_HJ_SPAVG']]
    min_value = min(Feature_DF.loc[:,'JS_J_HJ_SPAVG'])
    Feature_DF.loc[:,'JS_J_HJ_SPAVG_TRS'] = Feature_DF.loc[:,'JS_J_HJ_SPAVG'].apply(lambda x : (1+x-min_value)**(-9/5))
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_HJ_SPAVG_TRS']]

    return Feature_DF

"""
JS_J_HJ_CON_TRS
"""

def JS_J_HJ_CON_TRS(Dataframe):

    """
    XX Transformation on Jockey Contribution to Horse
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, JS_J_HJ_CON_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JS_J_HJ_CON']]
    min_value = min(Feature_DF.loc[:,'JS_J_HJ_CON'])
    Feature_DF.loc[:,'JS_J_HJ_CON_TRS'] = Feature_DF.loc[:,'JS_J_HJ_CON'].apply(lambda x : (1+x-min_value)**(-1/6))
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_HJ_CON_TRS']]

    return Feature_DF

"""
JS_J_SJ_WIN_TRS
"""

def JS_J_SJ_WIN_TRS(Dataframe):

    """
    XX Transformation on Jockey Stable Win Percentage
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, JS_J_SJ_WIN_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JS_J_SJ_WIN']]
    Feature_DF.loc[:,'JS_J_SJ_WIN_TRS'] = Feature_DF.loc[:,'JS_J_SJ_WIN'].apply(lambda x : (1+x)**(-5/7))
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_SJ_WIN_TRS']]

    return Feature_DF

"""
JS_J_SJ_T3_TRS
"""

def JS_J_SJ_T3_TRS(Dataframe):

    """
    XX Transformation on Jockey Stable Top 3 Percentage
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, JS_J_SJ_T3_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JS_J_SJ_T3']]
    Feature_DF.loc[:,'JS_J_SJ_T3_TRS'] = Feature_DF.loc[:,'JS_J_SJ_T3'].pow(1/6)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_J_SJ_T3_TRS']]

    return Feature_DF

"""
JS_S_FP_TRS
"""

def JS_S_FP_TRS(Dataframe):

    """
    XX Transformation on Average Finishing Position of Stable
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, JS_S_FP_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JS_S_FP']]
    min_value = min(Feature_DF.loc[:,'JS_S_FP'])
    Feature_DF.loc[:,'JS_S_FP_TRS'] = Feature_DF.loc[:,'JS_S_FP'].apply(lambda x : (1+x-min_value)**(9/5))
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_S_FP_TRS']]

    return Feature_DF

"""
JS_S_FPRW_TRS
"""

def JS_S_FPRW_TRS(Dataframe):

    """
    XX Transformation on Recency Weighted Avg Finishing Position of Stable
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, JS_S_FPRW_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JS_S_FPRW']]
    min_value = min(Feature_DF.loc[:,'JS_S_FPRW'])
    Feature_DF.loc[:,'JS_S_FPRW_TRS'] = Feature_DF.loc[:,'JS_S_FPRW'].apply(lambda x : (1+x-min_value)**(3/2))
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_S_FPRW_TRS']]

    return Feature_DF

"""
JS_S_WINP_TRS
"""

def JS_S_WINP_TRS(Dataframe):

    """
    XX Transformation on Stable Win Percentage
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, JS_S_WINP_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JS_S_WINP']]
    Feature_DF.loc[:,'JS_S_WINP_TRS'] = Feature_DF.loc[:,'JS_S_WINP'].pow(2/7)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_S_WINP_TRS']]

    return Feature_DF

"""
JS_S_WINP_SDIST_TRS
"""

def JS_S_WINP_SDIST_TRS(Dataframe):

    """
    XX Transformation on Win Percentage of Stable on Distance
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, JS_S_WINP_SDIST_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JS_S_WINP_SDIST']]
    Feature_DF.loc[:,'JS_S_WINP_SDIST_TRS'] = Feature_DF.loc[:,'JS_S_WINP_SDIST'].pow(1/4)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_S_WINP_SDIST_TRS']]

    return Feature_DF

"""
JS_S_WINP_SGO_TRS
"""

def JS_S_WINP_SGO_TRS(Dataframe):

    """
    XX Transformation on Win Percentage of Stable on Going
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_FH_NUMW_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JS_S_WINP_SGO']]
    Feature_DF.loc[:,'JS_S_WINP_SGO_TRS'] = Feature_DF.loc[:,'JS_S_WINP_SGO'].apply(lambda x : (1+x)**(-9/10))
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_S_WINP_SGO_TRS']]

    return Feature_DF

"""
JS_S_WINP_SSUR
"""

def JS_S_WINP_SSUR_TRS(Dataframe):

    """
    XX Transformation on Win Percentage of Stable on Surface
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, JS_S_WINP_SSUR_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JS_S_WINP_SSUR']]
    Feature_DF.loc[:,'JS_S_WINP_SSUR_TRS'] = Feature_DF.loc[:,'JS_S_WINP_SSUR'].pow(2/5)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_S_WINP_SSUR_TRS']]

    return Feature_DF

"""
JS_S_WINP_SLOC_TRS
"""

def JS_S_WINP_SLOC_TRS(Dataframe):

    """
    XX Transformation on Win Percentage of Stable on Location
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, PP_FH_NUMW_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JS_S_WINP_SLOC']]
    Feature_DF.loc[:,'JS_S_WINP_SLOC_TRS'] = Feature_DF.loc[:,'JS_S_WINP_SLOC'].apply(lambda x : (1+x)**(-1/6))
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_S_WINP_SLOC_TRS']]

    return Feature_DF

"""
JS_S_WINP_SPFL_TRS
"""

def JS_S_WINP_SPFL_TRS(Dataframe):

    """
    XX Transformation on Win Percentage of Stable on Profile
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, JS_S_WINP_SPFL_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JS_S_WINP_SPFL']]
    min_value = min(Feature_DF.loc[:,'JS_S_WINP_SPFL'])
    Feature_DF.loc[:,'JS_S_WINP_SPFL_TRS'] = Feature_DF.loc[:,'JS_S_WINP_SPFL'].apply(lambda x : np.log(1+x-min_value))
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_S_WINP_SPFL_TRS']]

    return Feature_DF

"""
JS_S_T3P_TRS
"""

def JS_S_T3P_TRS(Dataframe):

    """
    XX Transformation on Stable Top 3 Percentage
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, JS_S_T3P_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JS_S_T3P']]
    Feature_DF.loc[:,'JS_S_T3P_TRS'] = Feature_DF.loc[:,'JS_S_T3P'].pow(1/4)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_S_T3P_TRS']]

    return Feature_DF

"""
JS_S_T3P_SDIST_TRS
"""

def JS_S_T3P_SDIST_TRS(Dataframe):

    """
    XX Transformation on Stable Top 3 Percentage on Distance
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, JS_S_T3P_SDIST_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JS_S_T3P_SDIST']]
    Feature_DF.loc[:,'JS_S_T3P_SDIST_TRS'] = Feature_DF.loc[:,'JS_S_T3P_SDIST'].pow(3/7)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_S_T3P_SDIST_TRS']]

    return Feature_DF

"""
JS_S_T3P_SGO_TRS
"""

def JS_S_T3P_SGO_TRS(Dataframe):

    """
    XX Transformation on Stable Top 3 Percentage on Going
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, JS_S_T3P_SGO_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JS_S_T3P_SGO']]
    Feature_DF.loc[:,'JS_S_T3P_SGO_TRS'] = Feature_DF.loc[:,'JS_S_T3P_SGO'].apply(lambda x : (1+x)**(-4/3))
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_S_T3P_SGO_TRS']]

    return Feature_DF

"""
JS_S_T3P_SSUR_TRS
"""

def JS_S_T3P_SSUR_TRS(Dataframe):

    """
    XX Transformation on Stable Top 3 Percentage on Surface
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, JS_S_T3P_SSUR_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JS_S_T3P_SSUR']]
    Feature_DF.loc[:,'JS_S_T3P_SSUR_TRS'] = Feature_DF.loc[:,'JS_S_T3P_SSUR'].pow(4/9)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_S_T3P_SSUR_TRS']]

    return Feature_DF

"""
JS_S_T3P_SLOC_TRS
"""

def JS_S_T3P_SLOC_TRS(Dataframe):

    """
    XX Transformation on Stable Top 3 Percentage on Location
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, JS_S_T3P_SLOC_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JS_S_T3P_SLOC']]
    Feature_DF.loc[:,'JS_S_T3P_SLOC_TRS'] = Feature_DF.loc[:,'JS_S_T3P_SLOC'].apply(lambda x : (1+x)**(-1/6))
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_S_T3P_SLOC_TRS']]

    return Feature_DF

"""
JS_S_T3P_SPFL_TRS
"""

def JS_S_T3P_SPFL_TRS(Dataframe):

    """
    XX Transformation on Stable Top 3 Percentage on Profile
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, JS_S_T3P_SPFL_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','JS_S_T3P_SPFL']]
    Feature_DF.loc[:,'JS_S_T3P_SPFL_TRS'] = Feature_DF.loc[:,'JS_S_T3P_SPFL'].pow(3/10)
    Feature_DF = Feature_DF.loc[:,['HNAME','JS_S_T3P_SPFL_TRS']]

    return Feature_DF

"""
RC_DIST_DAY_DIST_TRS
"""

def RC_DIST_DAY_DIST_TRS(Dataframe):

    """
    XX Transformation on Days since Distance
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RC_DIST_DAY_DIST_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RC_DIST_DAY_DIST']]
    Feature_DF.loc[:,'RC_DIST_DAY_DIST_TRS'] = Feature_DF.loc[:,'RC_DIST_DAY_DIST'].pow(9/5)
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_DIST_DAY_DIST_TRS']]

    return Feature_DF

"""
RC_DIST_DAY_SIM_DIST_TRS
"""

def RC_DIST_DAY_SIM_DIST_TRS(Dataframe):

    """
    XX Transformation on Days since similar Distance
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RC_DIST_DAY_SIM_DIST_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RC_DIST_DAY_SIM_DIST']]
    Feature_DF.loc[:,'RC_DIST_DAY_SIM_DIST_TRS'] = Feature_DF.loc[:,'RC_DIST_DAY_SIM_DIST'].apply(lambda x : (1+x)**(-5/9))
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_DIST_DAY_SIM_DIST_TRS']]

    return Feature_DF

"""
RC_DIST_AVG_DIST_TRS
"""

def RC_DIST_AVG_DIST_TRS(Dataframe):

    """
    XX Transformation on Average Recent Distance
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RC_DIST_AVG_DIST_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RC_DIST_AVG_DIST']]
    Feature_DF.loc[:,'RC_DIST_AVG_DIST_TRS'] = Feature_DF.loc[:,'RC_DIST_AVG_DIST'].apply(lambda x : (1+x)**(-5/6))
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_DIST_AVG_DIST_TRS']]

    return Feature_DF

"""
RC_DIST_RANGE_DIST_TRS
"""

def RC_DIST_RANGE_DIST_TRS(Dataframe):

    """
    XX Transformation on T3 Distribution Limit on Distance
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RC_DIST_RANGE_DIST_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RC_DIST_RANGE_DIST']]
    min_value = min(Feature_DF.loc[:,'RC_DIST_RANGE_DIST'])
    Feature_DF.loc[:,'RC_DIST_RANGE_DIST_TRS'] = Feature_DF.loc[:,'RC_DIST_RANGE_DIST'].apply(lambda x : (1+x-min_value)**(-9/4))
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_DIST_RANGE_DIST_TRS']]

    return Feature_DF

"""
RC_DIST_HPRE_TRS
"""

def RC_DIST_HPRE_TRS(Dataframe):

    """
    XX Transformation on Distance Preference of Horse
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RC_DIST_HPRE_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RC_DIST_HPRE']]
    min_value = min(Feature_DF.loc[:,'RC_DIST_HPRE'])
    Feature_DF.loc[:,'RC_DIST_HPRE_TRS'] = Feature_DF.loc[:,'RC_DIST_HPRE'].apply(lambda x : (1+x-min_value)**(9/10))
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_DIST_HPRE_TRS']]

    return Feature_DF

"""
RC_DIST_JPRE_TRS
"""

def RC_DIST_JPRE_TRS(Dataframe):

    """
    XX Transformation on Distance Preference of Jockey
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RC_DIST_JPRE_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RC_DIST_JPRE']]
    min_value = min(Feature_DF.loc[:,'RC_DIST_JPRE'])
    Feature_DF.loc[:,'RC_DIST_JPRE_TRS'] = Feature_DF.loc[:,'RC_DIST_JPRE'].apply(lambda x : (1+x-min_value)**(1/2))
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_DIST_JPRE_TRS']]

    return Feature_DF

"""
RC_DIST_SPRE_TRS
"""

def RC_DIST_SPRE_TRS(Dataframe):

    """
    XX Transformation on Distance Preference of Stable
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RC_DIST_SPRE_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RC_DIST_SPRE']]
    min_value = min(Feature_DF.loc[:,'RC_DIST_SPRE'])
    Feature_DF.loc[:,'RC_DIST_SPRE_TRS'] = Feature_DF.loc[:,'RC_DIST_SPRE'].apply(lambda x : (1+x-min_value)**(2/3))
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_DIST_SPRE_TRS']]

    return Feature_DF

"""
RC_GO_DAY_GO_TRS
"""

def RC_GO_DAY_GO_TRS(Dataframe):

    """
    XX Transformation on Days since Going
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RC_GO_DAY_GO_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RC_GO_DAY_GO']]
    Feature_DF.loc[:,'RC_GO_DAY_GO_TRS'] = Feature_DF.loc[:,'RC_GO_DAY_GO'].apply(lambda x : (1+x)**(-9/7))
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_GO_DAY_GO_TRS']]

    return Feature_DF

"""
RC_GO_AVG_GO_TRS
"""

def RC_GO_AVG_GO_TRS(Dataframe):

    """
    XX Transformation on Average Recent Going
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RC_GO_AVG_GO_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RC_GO_AVG_GO']]
    min_value = min(Feature_DF.loc[:,'RC_GO_AVG_GO'])
    Feature_DF.loc[:,'RC_GO_AVG_GO_TRS'] = Feature_DF.loc[:,'RC_GO_AVG_GO'].apply(lambda x : (1+x-min_value)**(5/4))
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_GO_AVG_GO_TRS']]

    return Feature_DF

"""
RC_GO_JPRE_TRS
"""

def RC_GO_JPRE_TRS(Dataframe):

    """
    XX Transformation on Going Preference of Jockey
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RC_GO_JPRE_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RC_GO_JPRE']]
    min_value = min(Feature_DF.loc[:,'RC_GO_JPRE'])
    Feature_DF.loc[:,'RC_GO_JPRE_TRS'] = Feature_DF.loc[:,'RC_GO_JPRE'].apply(lambda x : (1+x-min_value)**(9/5))
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_GO_JPRE_TRS']]

    return Feature_DF

"""
RC_GO_SPRE_TRS
"""

def RC_GO_SPRE_TRS(Dataframe):

    """
    XX Transformation on Going Preference of Stable
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RC_GO_SPRE_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RC_GO_SPRE']]
    min_value = min(Feature_DF.loc[:,'RC_GO_SPRE'])
    Feature_DF.loc[:,'RC_GO_SPRE_TRS'] = Feature_DF.loc[:,'RC_GO_SPRE'].apply(lambda x : (1+x-min_value)**(9/10))
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_GO_SPRE_TRS']]

    return Feature_DF

"""
RC_SUR_DAY_SUR_TRS
"""

def RC_SUR_DAY_SUR_TRS(Dataframe):

    """
    XX Transformation on Days since Surface
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RC_SUR_DAY_SUR_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RC_SUR_DAY_SUR']]
    Feature_DF.loc[:,'RC_SUR_DAY_SUR_TRS'] = Feature_DF.loc[:,'RC_SUR_DAY_SUR'].apply(lambda x : (1+x)**(-9/5))
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_SUR_DAY_SUR_TRS']]

    return Feature_DF

"""
RC_SUR_AVG_SUR_TRS
"""

def RC_SUR_AVG_SUR_TRS(Dataframe):

    """
    XX Transformation on Average Recent Surface
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RC_SUR_AVG_SUR_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RC_SUR_AVG_SUR']]
    Feature_DF.loc[:,'RC_SUR_AVG_SUR_TRS'] = Feature_DF.loc[:,'RC_SUR_AVG_SUR'].pow(4)
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_SUR_AVG_SUR_TRS']]

    return Feature_DF

"""
RC_SUR_HPRE_TRS
"""

def RC_SUR_HPRE_TRS(Dataframe):

    """
    XX Transformation on Surface Preference of Horse
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RC_SUR_HPRE_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RC_SUR_HPRE']]
    min_value = min(Feature_DF.loc[:,'RC_SUR_HPRE'])
    Feature_DF.loc[:,'RC_SUR_HPRE_TRS'] = Feature_DF.loc[:,'RC_SUR_HPRE'].apply(lambda x : (1+x-min_value)**(5/6))
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_SUR_HPRE_TRS']]

    return Feature_DF

"""
RC_SUR_JPRE_TRS
"""

def RC_SUR_JPRE_TRS(Dataframe):

    """
    XX Transformation on Surface Preference of Jockey
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RC_SUR_JPRE_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RC_SUR_JPRE']]
    min_value = min(Feature_DF.loc[:,'RC_SUR_JPRE'])
    Feature_DF.loc[:,'RC_SUR_JPRE_TRS'] = Feature_DF.loc[:,'RC_SUR_JPRE'].apply(lambda x : (1+x-min_value)**(4/5))
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_SUR_JPRE_TRS']]

    return Feature_DF

"""
RC_SUR_SPRE_TRS
"""

def RC_SUR_SPRE_TRS(Dataframe):

    """
    XX Transformation on Surface Preference of Stable
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RC_SUR_SPRE_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RC_SUR_SPRE']]
    min_value = min(Feature_DF.loc[:,'RC_SUR_SPRE'])
    Feature_DF.loc[:,'RC_SUR_SPRE_TRS'] = Feature_DF.loc[:,'RC_SUR_SPRE'].apply(lambda x : (1+x-min_value)**(6/5))
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_SUR_SPRE_TRS']]

    return Feature_DF

"""
RC_LOC_JPRE_TRS
"""

def RC_LOC_JPRE_TRS(Dataframe):

    """
    XX Transformation on Location Preference of Jockey
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RC_LOC_JPRE_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RC_LOC_JPRE']]
    min_value = min(Feature_DF.loc[:,'RC_LOC_JPRE'])
    Feature_DF.loc[:,'RC_LOC_JPRE_TRS'] = Feature_DF.loc[:,'RC_LOC_JPRE'].apply(lambda x : (1+x-min_value)**(5/6))
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_LOC_JPRE_TRS']]

    return Feature_DF

"""
RC_LOC_SPRE_TRS
"""

def RC_LOC_SPRE_TRS(Dataframe):

    """
    XX Transformation on Location Preference of Stable
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RC_LOC_SPRE_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RC_LOC_SPRE']]
    min_value = min(Feature_DF.loc[:,'RC_LOC_SPRE'])
    Feature_DF.loc[:,'RC_LOC_SPRE_TRS'] = Feature_DF.loc[:,'RC_LOC_SPRE'].apply(lambda x : (1+x-min_value)**(2/3))
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_LOC_SPRE_TRS']]

    return Feature_DF

"""
RC_PFL_HPRE_TRS
"""

def RC_PFL_HPRE_TRS(Dataframe):

    """
    XX Transformation on Profile Preference of Horse
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RC_PFL_HPRE_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RC_PFL_HPRE']]
    min_value = min(Feature_DF.loc[:,'RC_PFL_HPRE'])
    Feature_DF.loc[:,'RC_PFL_HPRE_TRS'] = Feature_DF.loc[:,'RC_PFL_HPRE'].apply(lambda x : (1+x-min_value)**(7/4))
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_PFL_HPRE_TRS']]

    return Feature_DF

"""
RC_PFL_JPRE_TRS
"""

def RC_PFL_JPRE_TRS(Dataframe):

    """
    XX Transformation on Profile Preference of Jockey
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RC_PFL_JPRE_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RC_PFL_JPRE']]
    min_value = min(Feature_DF.loc[:,'RC_PFL_JPRE'])
    Feature_DF.loc[:,'RC_PFL_JPRE_TRS'] = Feature_DF.loc[:,'RC_PFL_JPRE'].apply(lambda x : (1+x-min_value)**(6/5))
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_PFL_JPRE_TRS']]

    return Feature_DF

"""
RC_PP_TRS
"""

def RC_PP_TRS(Dataframe):

    """
    XX Transformation on Post Position
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RC_PP_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RC_PP']]
    Feature_DF.loc[:,'RC_PP_TRS'] = Feature_DF.loc[:,'RC_PP'].pow(3)
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_PP_TRS']]

    return Feature_DF

"""
RC_PP_W_TRS
"""

def RC_PP_W_TRS(Dataframe):

    """
    XX Transformation on Post Position Win Percentage
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RC_PP_W_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RC_PP_W']]
    Feature_DF.loc[:,'RC_PP_W_TRS'] = Feature_DF.loc[:,'RC_PP_W'].pow(-9/5)
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_PP_W_TRS']]

    return Feature_DF

"""
RC_PP_GOA_TRS
"""

def RC_PP_GOA_TRS(Dataframe):

    """
    XX Transformation on Post Position Advantage of Going
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RC_PP_GOA_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RC_PP_GOA']]
    Feature_DF.loc[:,'RC_PP_GOA_TRS'] = Feature_DF.loc[:,'RC_PP_GOA'].apply(lambda x : (1+x)**(-7/2))
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_PP_GOA_TRS']]

    return Feature_DF

"""
RC_PP_PFLA_TRS
"""

def RC_PP_PFLA_TRS(Dataframe):

    """
    XX Transformation on Post Position Advantage of Profile
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RC_PP_PFLA_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RC_PP_PFLA']]
    Feature_DF.loc[:,'RC_PP_PFLA_TRS'] = Feature_DF.loc[:,'RC_PP_PFLA'].pow(1/10)
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_PP_PFLA_TRS']]

    return Feature_DF

"""
RC_PP_GOPFLA_TRS
"""

def RC_PP_GOPFLA_TRS(Dataframe):

    """
    XX Transformation on Post Position Advantage of Going Profile
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RC_PP_GOPFLA_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RC_PP_GOPFLA']]
    Feature_DF.loc[:,'RC_PP_GOPFLA_TRS'] = Feature_DF.loc[:,'RC_PP_GOPFLA'].pow(-4/9)
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_PP_GOPFLA_TRS']]

    return Feature_DF

"""
RC_PP_PFLEP_TRS
"""

def RC_PP_PFLEP_TRS(Dataframe):

    """
    XX Transformation on Average Early Pace of Post Position on Profile
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RC_PP_PFLEP_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RC_PP_PFLEP']]
    Feature_DF.loc[:,'RC_PP_PFLEP_TRS'] = Feature_DF.loc[:,'RC_PP_PFLEP'].pow(6/7)
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_PP_PFLEP_TRS']]

    return Feature_DF

"""
RC_PP_HPRE_TRS
"""

def RC_PP_HPRE_TRS(Dataframe):

    """
    XX Transformation on Horse's Post Position Advantage
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RC_PP_HPRE_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RC_PP_HPRE']]
    Feature_DF.loc[:,'RC_PP_HPRE_TRS'] = Feature_DF.loc[:,'RC_PP_HPRE'].apply(lambda x : (1+x)**(-9/5))
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_PP_HPRE_TRS']]

    return Feature_DF

"""
RC_PP_JPRE_JPFL_TRS
"""

def RC_PP_JPRE_JPFL_TRS(Dataframe):

    """
    XX Transformation on Jockey's Post Position Advantage
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RC_PP_JPRE_JPFL_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RC_PP_JPRE_JPFL']]
    Feature_DF.loc[:,'RC_PP_JPRE_JPFL_TRS'] = Feature_DF.loc[:,'RC_PP_JPRE_JPFL'].apply(lambda x : (1+x)**(-1/5))
    Feature_DF = Feature_DF.loc[:,['HNAME','RC_PP_JPRE_JPFL_TRS']]

    return Feature_DF

"""
OD_CR_FAVT_TRS
"""

def OD_CR_FAVT_TRS(Dataframe):

    """
    XX Transformation on The Favourite
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, OD_CR_FAVT_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','OD_CR_FAVT']]
    Feature_DF.loc[:,'OD_CR_FAVT_TRS'] = Feature_DF.loc[:,'OD_CR_FAVT'].pow(10)
    Feature_DF = Feature_DF.loc[:,['HNAME','OD_CR_FAVT_TRS']]

    return Feature_DF

"""
OD_CR_FAVO_TRS
"""

def OD_CR_FAVO_TRS(Dataframe):

    """
    XX Transformation on Not Favourite
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, OD_CR_FAVO_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','OD_CR_FAVO']]
    Feature_DF.loc[:,'OD_CR_FAVO_TRS'] = Feature_DF.loc[:,'OD_CR_FAVO'].pow(-9/2)
    Feature_DF = Feature_DF.loc[:,['HNAME','OD_CR_FAVO_TRS']]

    return Feature_DF

"""
OD_PR_LPAVG_TRS
"""

def OD_PR_LPAVG_TRS(Dataframe):

    """
    XX Transformation on Average Log Odds Implied Probability
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, OD_PR_LPAVG_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','OD_PR_LPAVG']]
    min_value = min(Feature_DF.loc[:,'OD_PR_LPAVG'])
    Feature_DF.loc[:,'OD_PR_LPAVG_TRS'] = Feature_DF.loc[:,'OD_PR_LPAVG'].apply(lambda x : (1+x-min_value)**(9/5))
    Feature_DF = Feature_DF.loc[:,['HNAME','OD_PR_LPAVG_TRS']]

    return Feature_DF

"""
OD_PR_LPW_TRS
"""

def OD_PR_LPW_TRS(Dataframe):

    """
    XX Transformation on Average Winning Odds
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, OD_PR_LPW_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','OD_PR_LPW']]
    Feature_DF.loc[:,'OD_PR_LPW_TRS'] = Feature_DF.loc[:,'OD_PR_LPW'].apply(lambda x : (1+x)**(-9/5))
    Feature_DF = Feature_DF.loc[:,['HNAME','OD_PR_LPW_TRS']]

    return Feature_DF

"""
OD_PR_FAVB_TRS
"""

def OD_PR_FAVB_TRS(Dataframe):

    """
    XX Transformation on Number of favourite beaten
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, OD_PR_FAVB_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','OD_PR_FAVB']]
    Feature_DF.loc[:,'OD_PR_FAVB_TRS'] = Feature_DF.loc[:,'OD_PR_FAVB'].apply(lambda x : (1+x)**(-2/5))
    Feature_DF = Feature_DF.loc[:,['HNAME','OD_PR_FAVB_TRS']]

    return Feature_DF

"""
OD_PR_BFAV_TRS
"""

def OD_PR_BFAV_TRS(Dataframe):

    """
    XX Transformation on Number of beaten favouriate
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, OD_PR_BFAV_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','OD_PR_BFAV']]
    Feature_DF.loc[:,'OD_PR_BFAV_TRS'] = Feature_DF.loc[:,'OD_PR_BFAV'].pow(3)
    Feature_DF = Feature_DF.loc[:,['HNAME','OD_PR_BFAV_TRS']]

    return Feature_DF

"""
RS_ELO_H_TRS
"""

def RS_ELO_H_TRS(Dataframe):

    """
    XX Transformation on Horse's ELO Score
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RS_ELO_H_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RS_ELO_H']]
    Feature_DF.loc[:,'RS_ELO_H_TRS'] = Feature_DF.loc[:,'RS_ELO_H'].pow(4/5)
    Feature_DF = Feature_DF.loc[:,['HNAME','RS_ELO_H_TRS']]

    return Feature_DF

"""
RS_ELO_HP_TRS
"""

def RS_ELO_HP_TRS(Dataframe):

    """
    XX Transformation on Horse's ELO Score Implied Probability
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RS_ELO_HP_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RS_ELO_HP']]
    Feature_DF.loc[:,'RS_ELO_HP_TRS'] = Feature_DF.loc[:,'RS_ELO_HP'].pow(4/5)
    Feature_DF = Feature_DF.loc[:,['HNAME','RS_ELO_HP_TRS']]

    return Feature_DF

"""
RS_ELO_H_DIST_TRS
"""

def RS_ELO_H_DIST_TRS(Dataframe):

    """
    XX Transformation on Horse’s ELO Score on Distance
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RS_ELO_H_DIST_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RS_ELO_H_DIST']]
    Feature_DF.loc[:,'RS_ELO_H_DIST_TRS'] = Feature_DF.loc[:,'RS_ELO_H_DIST'].pow(1/2)
    Feature_DF = Feature_DF.loc[:,['HNAME','RS_ELO_H_DIST_TRS']]

    return Feature_DF

"""
RS_ELO_HP_DIST_TRS
"""

def RS_ELO_HP_DIST_TRS(Dataframe):

    """
    XX Transformation on Horse’s ELO Score Implied Probability on Distance
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RS_ELO_HP_DIST_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RS_ELO_HP_DIST']]
    Feature_DF.loc[:,'RS_ELO_HP_DIST_TRS'] = Feature_DF.loc[:,'RS_ELO_HP_DIST'].pow(1/2)
    Feature_DF = Feature_DF.loc[:,['HNAME','RS_ELO_HP_DIST_TRS']]

    return Feature_DF

"""
RS_ELO_H_GO_TRS
"""

def RS_ELO_H_GO_TRS(Dataframe):

    """
    XX Transformation on Horse’s ELO Score on Going
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RS_ELO_H_GO_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RS_ELO_H_GO']]
    Feature_DF.loc[:,'RS_ELO_H_GO_TRS'] = Feature_DF.loc[:,'RS_ELO_H_GO'].pow(5/8)
    Feature_DF = Feature_DF.loc[:,['HNAME','RS_ELO_H_GO_TRS']]

    return Feature_DF

"""
RS_ELO_HP_GO_TRS
"""

def RS_ELO_HP_GO_TRS(Dataframe):

    """
    XX Transformation on Horse’s ELO Score Implied Probability on Going
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RS_ELO_HP_GO_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RS_ELO_HP_GO']]
    Feature_DF.loc[:,'RS_ELO_HP_GO_TRS'] = Feature_DF.loc[:,'RS_ELO_HP_GO'].pow(2/3)
    Feature_DF = Feature_DF.loc[:,['HNAME','RS_ELO_HP_GO_TRS']]

    return Feature_DF

"""
RS_ELO_H_SUR_TRS
"""

def RS_ELO_H_SUR_TRS(Dataframe):

    """
    XX Transformation on Horse’s ELO Score on Surface
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RS_ELO_H_SUR_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RS_ELO_H_SUR']]
    Feature_DF.loc[:,'RS_ELO_H_SUR_TRS'] = Feature_DF.loc[:,'RS_ELO_H_SUR'].pow(4/9)
    Feature_DF = Feature_DF.loc[:,['HNAME','RS_ELO_H_SUR_TRS']]

    return Feature_DF

"""
RS_ELO_HP_SUR_TRS
"""

def RS_ELO_HP_SUR_TRS(Dataframe):

    """
    XX Transformation on Horse’s ELO Score Implied Probability on Surface
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RS_ELO_HP_SUR_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RS_ELO_HP_SUR']]
    Feature_DF.loc[:,'RS_ELO_HP_SUR_TRS'] = Feature_DF.loc[:,'RS_ELO_HP_SUR'].pow(4/9)
    Feature_DF = Feature_DF.loc[:,['HNAME','RS_ELO_HP_SUR_TRS']]

    return Feature_DF

"""
RS_ELO_H_PFL_TRS
"""

def RS_ELO_H_PFL_TRS(Dataframe):

    """
    XX Transformation on Horse’s ELO Score on Profile
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RS_ELO_H_PFL_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RS_ELO_H_PFL']]
    Feature_DF.loc[:,'RS_ELO_H_PFL_TRS'] = Feature_DF.loc[:,'RS_ELO_H_PFL'].pow(-9/10)
    Feature_DF = Feature_DF.loc[:,['HNAME','RS_ELO_H_PFL_TRS']]

    return Feature_DF

"""
RS_ELO_HP_PFL_TRS
"""

def RS_ELO_HP_PFL_TRS(Dataframe):

    """
    XX Transformation on Horse’s ELO Score Implied Probability on Profile
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RS_ELO_HP_PFL_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RS_ELO_HP_PFL']]
    Feature_DF.loc[:,'RS_ELO_HP_PFL_TRS'] = Feature_DF.loc[:,'RS_ELO_HP_PFL'].pow(-8/9)
    Feature_DF = Feature_DF.loc[:,['HNAME','RS_ELO_HP_PFL_TRS']]

    return Feature_DF

"""
RS_ELO_J_TRS
"""

def RS_ELO_J_TRS(Dataframe):

    """
    XX Transformation on Jockey’s ELO Score
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RS_ELO_J_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RS_ELO_J']]
    Feature_DF.loc[:,'RS_ELO_J_TRS'] = Feature_DF.loc[:,'RS_ELO_J'].pow(5/7)
    Feature_DF = Feature_DF.loc[:,['HNAME','RS_ELO_J_TRS']]

    return Feature_DF

"""
RS_ELO_JP_TRS
"""

def RS_ELO_JP_TRS(Dataframe):

    """
    XX Transformation on Jockey’s ELO Score Implied Probability
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RS_ELO_JP_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RS_ELO_JP']]
    Feature_DF.loc[:,'RS_ELO_JP_TRS'] = Feature_DF.loc[:,'RS_ELO_JP'].pow(5/7)
    Feature_DF = Feature_DF.loc[:,['HNAME','RS_ELO_JP_TRS']]

    return Feature_DF

"""
RS_ELO_S_TRS
"""

def RS_ELO_S_TRS(Dataframe):

    """
    XX Transformation on Stable’s ELO Score
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RS_ELO_S_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RS_ELO_S']]
    Feature_DF.loc[:,'RS_ELO_S_TRS'] = Feature_DF.loc[:,'RS_ELO_S'].pow(5/6)
    Feature_DF = Feature_DF.loc[:,['HNAME','RS_ELO_S_TRS']]

    return Feature_DF

"""
RS_ELO_SP_TRS
"""

def RS_ELO_SP_TRS(Dataframe):

    """
    XX Transformation on Stable’s ELO Score Implied Probability
    Parameter
    ---------
    Dataframe : Dataframe of Base Features
    Return
    ------
    Dataframe [HNAME, RS_ELO_SP_TRS]
    """

    Feature_DF = Dataframe.loc[:,['HNAME','RS_ELO_SP']]
    Feature_DF.loc[:,'RS_ELO_SP_TRS'] = Feature_DF.loc[:,'RS_ELO_SP'].pow(5/6)
    Feature_DF = Feature_DF.loc[:,['HNAME','RS_ELO_SP_TRS']]

    return Feature_DF
