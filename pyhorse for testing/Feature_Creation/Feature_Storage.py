#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ivan
"""

"""
Feature Storage
"""

from pyhorse.Feature_Creation.Odds import *
from pyhorse.Feature_Creation.Running_Stat import *
from pyhorse.Feature_Creation.Jockey_Stable import *
from pyhorse.Feature_Creation.Past_Performance import *
from pyhorse.Feature_Creation.Current_Condition import *
from pyhorse.Feature_Creation.Racetrack_Condition import *
from pyhorse.Feature_Creation.Transformation import *
from pyhorse.Feature_Creation.Race_Posterior_Figures import Update_Race_PosteriroDb

"""
All Feature Functions accepts a Matchday Dataframe
then return a dataframe for a race,
"""

#List of Features to Create
Curent_Condition = ['CC_AGE','CC_FRB','CC_REC_DAYL','CC_REC_DAYL_DIST','CC_REC_DAYL_AGE','CC_REC_INC','CC_REC_NUMM','CC_REC_DAY_LWIN',
                    'CC_REC_DAY_PT3','CC_REC_NUM_LT3','CC_REC_NUM_DAYB','CC_CLS','CC_CLS_D','CC_CLS_CC','CC_CLS_CL','CC_BWEI','CC_BWEI_D',
                    'CC_BWEI_DWIN','CC_BWEI_DT3','CC_WEI','CC_WEI_DIST','CC_WEI_PER','CC_WEI_D','CC_WEI_SP','CC_WEI_EXP','CC_WEI_MAX',
                    'CC_WEI_BCH']


Past_Performance = ['PP_EXP_NRACE','PP_EXP_NRACE_DIST','PP_EXP_NRACE_SIM_DIST','PP_EXP_NRACE_GO','PP_EXP_NRACE_SUR','PP_EXP_NRACE_PFL',
                    'PP_FH_FP_CLS','PP_FH_FP_AVG','PP_FH_FP_AVGRW','PP_FH_FP_BIN','PP_FH_FP_DIST','PP_FH_FP_SIM_DIST','PP_FH_FP_GO','PP_FH_FP_SUR',
                    'PP_FH_FP_PFL','PP_FH_FTP','PP_FH_NUMW','PP_FH_HTH','PP_FH_WIN','PP_FH_WINP','PP_FH_WINPY','PP_FH_WINP_W','PP_FH_WINP_DIST',
                    'PP_FH_WINP_SIM_DIST','PP_FH_WINP_GO','PP_FH_WINP_SUR','PP_FH_WINP_PFL','PP_FH_T3P','PP_FH_T3P_T3','PP_FH_T3P_DIST',
                    'PP_FH_T3P_SIM_DIST','PP_FH_T3P_GO','PP_FH_T3P_SUR','PP_FH_T3P_PFL','PP_BL_AVG','PP_BL_SUM','PP_BL_AVGF','PP_BL_AVGF_SUR',
                    'PP_SPF_L1','PP_SPF_L2','PP_SPF_SEC','PP_SPF_KNN_PFL','PP_SPF_D1','PP_SPF_D','PP_SPF_AVG','PP_SPF_AVGRW',
                    'PP_SPF_AVG_DIST','PP_SPF_AVGRW_DIST','PP_SPF_AVG_SIM_DIST','PP_SPF_AVGRW_SIM_DIST','PP_SPF_AVG_GO','PP_SPF_AVGRW_GO',
                    'PP_SPF_AVG_SUR','PP_SPF_AVGRW_SUR','PP_SPF_AVG_PFL','PP_SPF_AVGRW_PFL','PP_SPF_TOP','PP_SPF_TOPY','PP_SPF_TOP_DIST',
                    'PP_SPF_TOP_SIM_DIST','PP_SPF_TOP_GO','PP_SPF_TOP_SUR','PP_SPF_TOP_PFL','PP_PAF_SPT','PP_PAF_SPT_DIST','PP_PAF_EP_AVG',
                    'PP_PAF_EP_AVGRW','PP_PAF_EP_ADV_GOPFL','PP_PAF_EP_ADV_PFL','PP_PAF_EP_WIN_PFL','PP_PAF_EP_DIST','PP_PAF_SP_AVG',
                    'PP_PAF_SP_AVGRW','PP_PAF_SP_ADV_GOPFL','PP_PAF_SP_ADV_PFL','PP_PAF_SP_WIN_PFL','PP_PAF_SP_DIST','PP_PAF_AP_AVG',
                    'PP_PAF_AP_AVGRW','PP_PAF_AP_ADV_GOPFL','PP_PAF_AP_ADV_PFL','PP_PAF_AP_WIN_PFL','PP_PAF_FP_AVG','PP_PAF_FP_AVGRW',
                    'PP_PAF_FP_ADV_GOPFL','PP_PAF_FP_ADV_PFL','PP_PAF_FP_WIN_PFL','PP_PAF_EDW_DIST','PP_PAF_EDW_PFL','PP_PAF_EDL_DIST',
                    'PP_PAF_EDL_PFL','PP_PAF_STL_AVG_PFL','PP_PAF_STL_B_PFL','PP_PAF_BEST','PP_PAF_BEST_GOPFL','PP_EPM','PP_EPM_AVG',
                    'PP_EMP_AVG_WIN','PP_EMP_AVG_PLA','PP_EMP_YR']


Jocky_Stable = ['JS_J_FP','JS_J_FPRW','JS_J_WINP','JS_J_WINP_JDIST','JS_J_WINP_JGO','JS_J_WINP_JSUR','JS_J_WINP_JLOC','JS_J_WINP_JPFL',
                'JS_J_T3P','JS_J_T3P_JDIST','JS_J_T3P_JGO','JS_J_T3P_JSUR','JS_J_T3P_JLOC','JS_J_T3P_JPFL','JS_J_NUMR','JS_J_HJ_NUM',
                'JS_J_HJ_NWIN','JS_J_HJ_NT3','JS_J_HJ_SPAVG','JS_J_HJ_CON','JS_J_SJ_WIN','JS_J_SJ_T3','JS_S_FP','JS_S_FPRW','JS_S_WINP',
                'JS_S_WINP_SDIST','JS_S_WINP_SGO','JS_S_WINP_SSUR','JS_S_WINP_SLOC','JS_S_WINP_SPFL','JS_S_T3P','JS_S_T3P_SDIST',
                'JS_S_T3P_SGO','JS_S_T3P_SSUR','JS_S_T3P_SLOC','JS_S_T3P_SPFL','JS_S_PRE']


Racetrack_Condition = ['RC_DIST_DAY_DIST','RC_DIST_DAY_SIM_DIST','RC_DIST_AVG_DIST','RC_DIST_RANGE_DIST','RC_DIST_HPRE','RC_DIST_JPRE',
                       'RC_DIST_SPRE','RC_GO_DAY_GO','RC_GO_AVG_GO','RC_GO_HPRE','RC_GO_JPRE','RC_GO_SPRE','RC_SUR_DAY_SUR',
                       'RC_SUR_AVG_SUR','RC_SUR_HPRE','RC_SUR_JPRE','RC_SUR_SPRE','RC_LOC_JPRE','RC_LOC_SPRE','RC_PFL_HPRE',
                       'RC_PFL_JPRE','RC_PFL_SPRE','RC_PP','RC_PP_W','RC_PP_GOA','RC_PP_PFLA','RC_PP_GOPFLA','RC_PP_PFLEP','RC_PP_HPRE',
                       'RC_PP_JPRE_JPFL']


Odds = ['OD_CR_LP','OD_CR_FAVT','OD_CR_FAVO','OD_CR_FAVW','OD_PR_LPAVG','OD_PR_LPW','OD_PR_FAVB','OD_PR_BFAV']


Composite_Stat = ['RS_ELO_H','RS_ELO_HP','RS_ELO_H_DIST','RS_ELO_HP_DIST','RS_ELO_H_GO','RS_ELO_HP_GO','RS_ELO_H_SUR','RS_ELO_HP_SUR','RS_ELO_H_PFL',
                  'RS_ELO_HP_PFL','RS_ELO_J','RS_ELO_JP','RS_ELO_S','RS_ELO_SP']

Feature_List = Curent_Condition + Past_Performance + Jocky_Stable + Racetrack_Condition + Odds + Composite_Stat

Transformation_List = ['CC_AGE_TRS','CC_REC_DAYL_TRS','CC_REC_DAYL_DIST_TRS','CC_REC_DAYL_AGE_TRS','CC_REC_INC_TRS',
                       'CC_REC_DAY_LWIN_TRS','CC_REC_DAY_PT3_TRS','CC_REC_NUM_LT3_TRS','CC_REC_NUM_DAYB_TRS','CC_CLS_TRS','CC_CLS_D_TRS',
                       'CC_CLS_CC_TRS','CC_CLS_CL_TRS','CC_BWEI_TRS','CC_BWEI_D_TRS','CC_BWEI_DWIN_TRS','CC_BWEI_DT3_TRS','CC_WEI_TRS',
                       'CC_WEI_DIST_TRS','CC_WEI_D_TRS','CC_WEI_SP_TRS','CC_WEI_EXP_TRS','CC_WEI_MAX_TRS','CC_WEI_BCH_TRS',
                       'PP_EXP_NRACE_TRS','PP_EXP_NRACE_DIST_TRS','PP_EXP_NRACE_SIM_DIST_TRS','PP_EXP_NRACE_GO_TRS','PP_EXP_NRACE_SUR_TRS',
                       'PP_EXP_NRACE_PFL_TRS','PP_FH_FP_CLS_TRS','PP_FH_FP_AVG_TRS','PP_FH_FP_AVGRW_TRS','PP_FH_FP_BIN_TRS','PP_FH_FP_DIST_TRS',
                       'PP_FH_FP_SIM_DIST_TRS','PP_FH_FP_GO_TRS','PP_FH_FP_SUR_TRS','PP_FH_FP_PFL_TRS','PP_FH_FTP_TRS','PP_FH_NUMW_TRS','PP_FH_HTH_TRS',
                       'PP_FH_WIN_TRS','PP_FH_WINP_TRS','PP_FH_WINPY_TRS','PP_FH_WINP_W_TRS','PP_FH_WINP_DIST_TRS','PP_FH_WINP_SIM_DIST_TRS',
                       'PP_FH_WINP_GO_TRS','PP_FH_WINP_SUR_TRS','PP_FH_WINP_PFL_TRS','PP_FH_T3P_TRS','PP_FH_T3P_T3_TRS','PP_FH_T3P_DIST_TRS',
                       'PP_FH_T3P_SIM_DIST_TRS','PP_FH_T3P_GO_TRS','PP_FH_T3P_SUR_TRS','PP_FH_T3P_PFL_TRS','PP_BL_AVG_TRS','PP_BL_SUM_TRS',
                       'PP_BL_AVGF_TRS','PP_BL_AVGF_SUR_TRS','PP_SPF_L1_TRS','PP_SPF_L2_TRS','PP_SPF_SEC_TRS','PP_SPF_KNN_PFL_TRS','PP_SPF_D1_TRS',
                       'PP_SPF_D_TRS','PP_SPF_AVG_TRS','PP_SPF_AVGRW_TRS','PP_SPF_AVG_DIST_TRS','PP_SPF_AVGRW_DIST_TRS','PP_SPF_AVG_SIM_DIST_TRS',
                       'PP_SPF_AVGRW_SIM_DIST_TRS','PP_SPF_AVG_GO_TRS','PP_SPF_AVGRW_GO_TRS','PP_SPF_AVG_SUR_TRS','PP_SPF_AVGRW_SUR_TRS',
                       'PP_SPF_AVG_PFL_TRS','PP_SPF_AVGRW_PFL_TRS','PP_SPF_TOP_TRS','PP_SPF_TOPY_TRS','PP_SPF_TOP_DIST_TRS','PP_SPF_TOP_SIM_DIST_TRS',
                       'PP_SPF_TOP_GO_TRS','PP_SPF_TOP_SUR_TRS','PP_SPF_TOP_PFL_TRS','PP_PAF_SPT_TRS','PP_PAF_SPT_DIST_TRS','PP_PAF_EP_AVG_TRS',
                       'PP_PAF_EP_AVGRW_TRS','PP_PAF_EP_ADV_GOPFL_TRS','PP_PAF_EP_ADV_PFL_TRS','PP_PAF_EP_WIN_PFL_TRS','PP_PAF_EP_DIST_TRS',
                       'PP_PAF_SP_AVG_TRS','PP_PAF_SP_AVGRW_TRS','PP_PAF_SP_ADV_GOPFL_TRS','PP_PAF_SP_ADV_PFL_TRS','PP_PAF_SP_WIN_PFL_TRS',
                       'PP_PAF_SP_DIST_TRS','PP_PAF_AP_AVG_TRS','PP_PAF_AP_AVGRW_TRS','PP_PAF_AP_ADV_GOPFL_TRS','PP_PAF_AP_ADV_PFL_TRS',
                       'PP_PAF_AP_WIN_PFL_TRS','PP_PAF_FP_AVG_TRS','PP_PAF_FP_AVGRW_TRS','PP_PAF_FP_ADV_GOPFL_TRS','PP_PAF_FP_ADV_PFL_TRS',
                       'PP_PAF_FP_WIN_PFL_TRS','PP_PAF_EDW_DIST_TRS','PP_PAF_EDW_PFL_TRS','PP_PAF_EDL_DIST_TRS','PP_PAF_EDL_PFL_TRS',
                       'PP_PAF_STL_AVG_PFL_TRS','PP_PAF_STL_B_PFL_TRS','PP_PAF_BEST_TRS','PP_PAF_BEST_GOPFL_TRS','PP_EPM_TRS','PP_EPM_AVG_TRS',
                       'PP_EMP_AVG_WIN_TRS','PP_EMP_AVG_PLA_TRS','PP_EMP_YR_TRS','JS_J_FP_TRS','JS_J_FPRW_TRS','JS_J_WINP_TRS','JS_J_WINP_JDIST_TRS',
                       'JS_J_WINP_JGO_TRS','JS_J_WINP_JSUR_TRS','JS_J_WINP_JLOC_TRS','JS_J_WINP_JPFL_TRS','JS_J_T3P_TRS','JS_J_T3P_JDIST_TRS',
                       'JS_J_T3P_JGO_TRS','JS_J_T3P_JSUR_TRS','JS_J_T3P_JLOC_TRS','JS_J_T3P_JPFL_TRS','JS_J_NUMR_TRS','JS_J_HJ_NUM_TRS','JS_J_HJ_NWIN_TRS',
                       'JS_J_HJ_NT3_TRS','JS_J_HJ_SPAVG_TRS','JS_J_HJ_CON_TRS','JS_J_SJ_WIN_TRS','JS_J_SJ_T3_TRS','JS_S_FP_TRS','JS_S_FPRW_TRS','JS_S_WINP_TRS',
                       'JS_S_WINP_SDIST_TRS','JS_S_WINP_SGO_TRS','JS_S_WINP_SSUR_TRS','JS_S_WINP_SLOC_TRS','JS_S_WINP_SPFL_TRS','JS_S_T3P_TRS',
                       'JS_S_T3P_SDIST_TRS','JS_S_T3P_SGO_TRS','JS_S_T3P_SSUR_TRS','JS_S_T3P_SLOC_TRS','JS_S_T3P_SPFL_TRS','RC_DIST_DAY_DIST_TRS',
                       'RC_DIST_DAY_SIM_DIST_TRS','RC_DIST_AVG_DIST_TRS','RC_DIST_RANGE_DIST_TRS','RC_DIST_HPRE_TRS','RC_DIST_JPRE_TRS','RC_DIST_SPRE_TRS',
                       'RC_GO_DAY_GO_TRS','RC_GO_AVG_GO_TRS','RC_GO_JPRE_TRS','RC_GO_SPRE_TRS','RC_SUR_DAY_SUR_TRS','RC_SUR_AVG_SUR_TRS',
                       'RC_SUR_HPRE_TRS','RC_SUR_JPRE_TRS','RC_SUR_SPRE_TRS','RC_LOC_JPRE_TRS','RC_LOC_SPRE_TRS','RC_PFL_HPRE_TRS','RC_PFL_JPRE_TRS',
                       'RC_PP_TRS','RC_PP_W_TRS','RC_PP_GOA_TRS','RC_PP_PFLA_TRS','RC_PP_GOPFLA_TRS','RC_PP_PFLEP_TRS','RC_PP_HPRE_TRS',
                       'RC_PP_JPRE_JPFL_TRS','OD_CR_FAVT_TRS','OD_CR_FAVO_TRS','OD_PR_LPAVG_TRS','OD_PR_LPW_TRS','OD_PR_FAVB_TRS',
                       'OD_PR_BFAV_TRS','RS_ELO_H_TRS','RS_ELO_HP_TRS','RS_ELO_H_DIST_TRS','RS_ELO_HP_DIST_TRS','RS_ELO_H_GO_TRS','RS_ELO_HP_GO_TRS',
                       'RS_ELO_H_SUR_TRS','RS_ELO_HP_SUR_TRS','RS_ELO_H_PFL_TRS','RS_ELO_HP_PFL_TRS','RS_ELO_J_TRS','RS_ELO_JP_TRS','RS_ELO_S_TRS','RS_ELO_SP_TRS']

FeatureDb_List = Feature_List + Transformation_List
