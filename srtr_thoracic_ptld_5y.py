import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn import metrics
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import imblearn
import sys
import lightgbm as lgb
from lightgbm import LGBMClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from hyperopt import tpe, STATUS_OK, Trials, hp, fmin
from fasterrisk.fasterrisk import RiskScoreOptimizer, RiskScoreClassifier
from fasterrisk.binarization_util import convert_continuous_df_to_binary_df
from imodels import HSTreeClassifier
import joblib
from hmeasure import h_score
import scipy
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import logging
logging.captureWarnings(True)

path_srtr = "C:\\Users\\Henry Johnston\\Documents\\SRTR_Files_Heart\\"
PTLDHR = pd.read_csv(path_srtr+"SRTR_Heart\\PTLD_Data_SRTR_Heart.csv")
PTLDLU = pd.read_csv(path_srtr+"SRTR_Lung\\PTLD_Data_SRTR_LUNG.csv")
PTLDHL = pd.read_csv(path_srtr+"SRTR_HeartLung\\PTLD_Data_SRTR_HEARTLUNG.csv")
PTLDTH = pd.concat([PTLDHR,PTLDLU,PTLDHL]).reset_index(drop=True)
PTLDTH.to_csv("PTLD_Data_SRTR_Thoracic.csv",index=False)

# Imputation Class

from sklearn.base import TransformerMixin

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == 'category' else X[c].median() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

def metrics_results(metrics_results, y_train, y_true, y_pred_proba, y_pred, model_num, model_name):
    metrics_results.loc[model_num,'Model'] = model_name
    metrics_results.loc[model_num,'Prevalence'] = np.mean(y_train)
    metrics_results.loc[model_num,'AUROC'] = metrics.roc_auc_score(y_true, y_pred_proba)
    metrics_results.loc[model_num,'H-measure'] = h_score(y_true.to_numpy(),  y_pred_proba)
    metrics_results.loc[model_num,'Average precision'] = metrics.average_precision_score(y_true, y_pred_proba)
    metrics_results.loc[model_num,'Log loss'] = metrics.log_loss(y_true, y_pred_proba)
    metrics_results.loc[model_num,'Brier score'] = metrics.brier_score_loss(y_true, y_pred_proba)
    metrics_results.loc[model_num,'Sensitivity'] = imblearn.metrics.sensitivity_score(y_true, y_pred)
    metrics_results.loc[model_num,'Specificity'] = imblearn.metrics.specificity_score(y_true, y_pred)
    metrics_results.loc[model_num,'Balanced accuracy'] = metrics.balanced_accuracy_score(y_true, y_pred)
    metrics_results.loc[model_num,'Precision'] = metrics.precision_score(y_true, y_pred)
    metrics_results.loc[model_num,'F1 score'] = metrics.f1_score(y_true, y_pred)

def calibration_results(calibration_results, y_true, y_pred_proba, model_name):
    if (model_name=='FASTRISK5') or (model_name=='HSDT'):
        current_n_bin=6
        for current_n_bin in np.arange(6,500):
            fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred_proba, n_bins=current_n_bin,strategy='quantile')
            if len(fraction_of_positives)>=6:
                break
        calibration_results['Y_TRUE_'+model_name]= np.append(fraction_of_positives,(11-len(fraction_of_positives))*[np.nan])
        calibration_results['Y_PRED_'+model_name]= np.append(mean_predicted_value,(11-len(mean_predicted_value))*[np.nan])
    else:
        current_n_bin=10
        for current_n_bin in np.arange(10,500):
            fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred_proba, n_bins=current_n_bin,strategy='quantile')
            if len(fraction_of_positives)>=10:
                break
        if len(fraction_of_positives)<11:
            calibration_results['Y_TRUE_'+model_name]= np.append(fraction_of_positives,(11-len(fraction_of_positives))*[np.nan])
            calibration_results['Y_PRED_'+model_name]= np.append(mean_predicted_value,(11-len(mean_predicted_value))*[np.nan])
        else:
            calibration_results['Y_TRUE_'+model_name]= fraction_of_positives
            calibration_results['Y_PRED_'+model_name]= mean_predicted_value

### READ SRTR Data for training ###

PTLD5YHR = pd.read_csv(path_srtr+"SRTR_Heart\\PTLD5Y_Data_Heart.csv")
PTLD5YLU = pd.read_csv(path_srtr+"SRTR_Lung\\PTLD5Y_Data_LUNG.csv")
PTLD5YHL = pd.read_csv(path_srtr+"SRTR_HeartLung\\PTLD5Y_Data_HEARTLUNG.csv")
PTLD5Y = pd.concat([PTLD5YHR,PTLD5YLU,PTLD5YHL]).reset_index(drop=True)
PTLD5Y.to_csv("PTLD5Y_Data_Thoracic.csv")

### READ UNOS Data matched to SRTR Data for testing ###
PTLD5Y_TH_UNOS = pd.read_csv("C:\\Users\\Henry Johnston\\Documents\\UNOS_Files_Heart\\PTLD5Y_Data_HRLUHL_UNOS_SRTR_MATCHED.csv")

metrics_results_train_all = pd.DataFrame()
metrics_results_test_all = pd.DataFrame()
calibration_results_train_all = pd.DataFrame()
calibration_results_test_all = pd.DataFrame() 

for test_year in np.arange(2008,2018):
    print("Minimum year of PTLD5Y: "+str(PTLD5Y['REC_TX_DT'].min()))
    print("Maximum year of PTLD5Y: "+str(PTLD5Y['REC_TX_DT'].max()))
    ########################################## TRAINING ##################################################
    # ######################################## For sklearn models ########################################
    X = PTLD5Y.copy()
    y = X['EVENT']
    X = X.drop(columns=["PERS_ID","TRR_ID","REC_TX_DT_2008","DONOR_ID","REC_CTR_CD","EVENT"])

    # ######################################## For LGB and XGB models ########################################
    PTLD5Y_LGBM = PTLD5Y.copy()
    cat_cols_all = PTLD5Y_LGBM.columns[8:len(PTLD5Y_LGBM.columns)-1].tolist()
    PTLD5Y_LGBM[cat_cols_all] = PTLD5Y_LGBM[cat_cols_all].astype('category')
    X_LGBM = PTLD5Y_LGBM.copy()
    y_LGBM = X_LGBM['EVENT']
    X_LGBM = X_LGBM.drop(columns=["PERS_ID","TRR_ID","REC_TX_DT_2008","DONOR_ID","REC_CTR_CD","EVENT"])

    X_train = X.loc[X['REC_TX_DT']<test_year]
    y_train = y.loc[X['REC_TX_DT']<test_year]
    REC_TX_DT_TRAIN = X_train['REC_TX_DT']
    X_train = X_train.drop(columns=["REC_TX_DT"])

    X_train_LGBM = X_LGBM.loc[X_LGBM['REC_TX_DT']<test_year]
    y_train_LGBM = y_LGBM.loc[X_LGBM['REC_TX_DT']<test_year]
    X_train_LGBM = X_train_LGBM.drop(columns=["REC_TX_DT"])

    print(X_train.shape)
    print(y_train.value_counts())
    print(X_train_LGBM.shape)
    print(y_train_LGBM.value_counts())
    print("Minimum year of X_train: "+str(REC_TX_DT_TRAIN.min()))
    print("Maximum year of X_train: "+str(REC_TX_DT_TRAIN.max()))
    
    ########################################## TESTING ###################################################
    ######################################### For sklearn models ########################################
    X = PTLD5Y_TH_UNOS.copy()
    y = X['EVENT']
    X = X.drop(columns=["PERS_ID","TRR_ID_CODE","REC_TX_DT_2008","DONOR_ID","REC_CTR_CD","EVENT"])

    # ######################################## For LGB and XGB models ########################################
    PTLD5Y_TH_UNOS_LGBM = PTLD5Y_TH_UNOS.copy()
    cat_cols_all_unos = PTLD5Y_TH_UNOS_LGBM.columns[8:len(PTLD5Y_TH_UNOS_LGBM.columns)-1].tolist()
    PTLD5Y_TH_UNOS_LGBM[cat_cols_all_unos] = PTLD5Y_TH_UNOS_LGBM[cat_cols_all_unos].astype('category')
    X_LGBM = PTLD5Y_TH_UNOS_LGBM.copy()
    y_LGBM = X_LGBM['EVENT']
    X_LGBM = X_LGBM.drop(columns=["PERS_ID","TRR_ID_CODE","REC_TX_DT_2008","DONOR_ID","REC_CTR_CD","EVENT"])

    X_test = X.loc[(X['REC_TX_DT']==test_year)]
    y_test = y.loc[(X['REC_TX_DT']==test_year)]
    REC_TX_DT_TEST = X_test['REC_TX_DT']
    X_test = X_test.drop(columns=["REC_TX_DT"])

    X_test_LGBM = X_LGBM.loc[(X_LGBM['REC_TX_DT']==test_year)]
    y_test_LGBM = y_LGBM.loc[(X_LGBM['REC_TX_DT']==test_year)]
    X_test_LGBM = X_test_LGBM.drop(columns=["REC_TX_DT"])

    print(X_test.shape)
    print(y_test.value_counts())
    print(X_test_LGBM.shape)
    print(y_test_LGBM.value_counts())
    print("Minimum year of X_test: "+str(REC_TX_DT_TEST.min()))
    print("Maximum year of X_test: "+str(REC_TX_DT_TEST.max()))
    
    saved = DataFrameImputer()
    Ximp_train_LGBM = saved.fit_transform(X_train_LGBM)
    Ximp_test_LGBM = saved.transform(X_test_LGBM)
    Ximp_train_LGBM['REC_VENTILATOR'] = Ximp_train_LGBM['REC_VENTILATOR'].astype('int32')
    Ximp_test_LGBM['REC_VENTILATOR'] = Ximp_test_LGBM['REC_VENTILATOR'].astype('int32')

    print(Ximp_train_LGBM.shape)
    print(Ximp_test_LGBM.shape)


    num_cols = X_train_LGBM.select_dtypes('number').columns
    cat_cols = X_train_LGBM.select_dtypes(exclude=['number']).columns

    # Sklearn numerical variables pipeline for lasso
    num_pipe_lasso = Pipeline([('imputer', SimpleImputer(strategy='median', missing_values=np.nan,add_indicator=False)),
                               ('scaler', StandardScaler())])

    # Sklearn numerical variables pipeline for all other models
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median', missing_values=np.nan,add_indicator=False))])

    # Sklearn categorical variables pipeline for Logistic Regression
    cat_pipe_lr = Pipeline([('imputer', SimpleImputer(strategy='most_frequent', missing_values=np.nan)),
                         ('encoder', OneHotEncoder(drop='first',handle_unknown='ignore'))])

    # Sklearn categorical variables pipeline for all other models
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent', missing_values=np.nan)),
                         ('encoder', OneHotEncoder(drop='if_binary',handle_unknown='ignore'))])

    #########

    # Sklearn pipeline for random forest
    preprocess = ColumnTransformer([("num_pipe", num_pipe, num_cols),
                                    ("cat_pipe", cat_pipe, cat_cols)])

    # Sklearn pipeline for lasso, ridge, elasticnet
    preprocess_lasso = ColumnTransformer([("num_pipe", num_pipe_lasso, num_cols),
                                          ("cat_pipe", cat_pipe_lr, cat_cols)])

    # Sklearn pipeline for unregularized logistic regression
    preprocess_lr = ColumnTransformer([("num_pipe", num_pipe, num_cols),
                                      ("cat_pipe", cat_pipe_lr, cat_cols)])
    n_evals=30
    
    #############################################################################################################
    ######################### FasterRisk Training and Testing Data Pre-processing ###############################
    #############################################################################################################

    nrows,ncols=Ximp_train_LGBM.shape
    Xnum_train_LGBM = convert_continuous_df_to_binary_df(Ximp_train_LGBM.iloc[:,np.arange(0,2)].astype('int32'))
    Xmulti_train_LGBM = pd.get_dummies(Ximp_train_LGBM.iloc[:,np.arange(2,12)],drop_first=False,dummy_na=False).reset_index(drop=True)
    Xbin_train_LGBM = pd.get_dummies(Ximp_train_LGBM.iloc[:,np.arange(12,ncols)],drop_first=True,dummy_na=False).reset_index(drop=True)
    X_train_interpretable = pd.concat([Xnum_train_LGBM,Xmulti_train_LGBM,Xbin_train_LGBM],axis=1)
    y_train_interpretable = y_train_LGBM.copy()
    y_train_interpretable[y_train_interpretable==0]=-1

    Xnum_test_LGBM = convert_continuous_df_to_binary_df(Ximp_test_LGBM.iloc[:,np.arange(0,2)].astype('int32'))
    Xmulti_test_LGBM = pd.get_dummies(Ximp_test_LGBM.iloc[:,np.arange(2,12)],drop_first=False,dummy_na=False).reset_index(drop=True)
    Xbin_test_LGBM = pd.get_dummies(Ximp_test_LGBM.iloc[:,np.arange(12,ncols)],drop_first=True,dummy_na=False).reset_index(drop=True)
    X_test_interpretable = pd.concat ([Xnum_test_LGBM,Xmulti_test_LGBM,Xbin_test_LGBM],axis=1)
    y_test_interpretable = y_test_LGBM.copy()
    y_test_interpretable[y_test_interpretable==0]=-1

    X_train_interpretable2,X_test_interpretable2 = X_train_interpretable.align(X_test_interpretable, join='outer', axis=1, fill_value=0)
    X_train_interpretable2 = X_train_interpretable2.astype('int32')
    X_test_interpretable2 = X_test_interpretable2.astype('int32')
    
    #############################################################################################################
    #############################################################################################################
    #############################################################################################################  
    
    metrics_results_train = pd.DataFrame()
    metrics_results_test = pd.DataFrame()
    calibration_results_train = pd.DataFrame()
    calibration_results_test = pd.DataFrame()
    total_models=9
    metrics_results_train['Train years'] = total_models*['Before '+str(test_year)]
    metrics_results_test['Test years'] = total_models*[str(test_year)+' and later']  
    calibration_results_train['Train years'] = 11*['Before '+str(test_year)]
    calibration_results_train['Bin'] = np.arange(0,11)
    calibration_results_test['Test years'] = 11*[str(test_year)+' and later']  
    calibration_results_test['Bin'] = np.arange(0,11)
    model_num = 0

    #######################################################################
    ############################ FasterRisk ###############################
    #######################################################################
    scores = pd.DataFrame()
    for counter, coef_num in enumerate(np.arange(6,11)):
        print(coef_num)
        saved = DataFrameImputer()
        xtrain = saved.fit_transform(X_train_LGBM[REC_TX_DT_TRAIN<test_year-1])
        xval = saved.transform(X_train_LGBM[(REC_TX_DT_TRAIN==test_year-1)])
        ytrain = y_train_LGBM[REC_TX_DT_TRAIN<test_year-1]
        yval = y_train_LGBM[(REC_TX_DT_TRAIN==test_year-1)]        
        sparsity = coef_num # produce a risk score model with coef_num nonzero coefficients 
        scores.loc[counter,'Sparsity']=sparsity
        
        nrows,ncols=xtrain.shape
        xtrain_num = convert_continuous_df_to_binary_df(xtrain.iloc[:,np.arange(0,2)].astype('int32'))
        xtrain_multi = pd.get_dummies(xtrain.iloc[:,np.arange(2,12)],drop_first=False,dummy_na=False).reset_index(drop=True)
        xtrain_bin = pd.get_dummies(xtrain.iloc[:,np.arange(12,ncols)],drop_first=True,dummy_na=False).reset_index(drop=True)
        xtrain_interpretable = pd.concat([xtrain_num,xtrain_multi,xtrain_bin],axis=1)
        ytrain_interpretable = ytrain.copy()
        ytrain_interpretable[ytrain_interpretable==0]=-1

        xval_num = convert_continuous_df_to_binary_df(xval.iloc[:,np.arange(0,2)].astype('int32'))
        xval_multi = pd.get_dummies(xval.iloc[:,np.arange(2,12)],drop_first=False,dummy_na=False).reset_index(drop=True)
        xval_bin = pd.get_dummies(xval.iloc[:,np.arange(12,ncols)],drop_first=True,dummy_na=False).reset_index(drop=True)
        xval_interpretable = pd.concat ([xval_num,xval_multi,xval_bin],axis=1)
        yval_interpretable = yval.copy()
        yval_interpretable[yval_interpretable==0]=-1

        xtrain_interpretable2, xval_interpretable2 = xtrain_interpretable.align(xval_interpretable, join='outer', axis=1, fill_value=0)
        xtrain_interpretable2 = xtrain_interpretable2.astype('int32')
        xval_interpretable2 = xval_interpretable2.astype('int32')
        
        # initialize a risk score optimizer
        m = RiskScoreOptimizer(X = np.asarray(xtrain_interpretable2), y = np.asarray(ytrain_interpretable), k = sparsity)

        # perform optimization
        m.optimize()

        # get all top m solutions from the final diverse pool
        arr_multiplier, arr_intercept, arr_coefficients = m.get_models() # get m solutions from the diverse pool; Specifically, arr_multiplier.shape=(m, ), arr_intercept.shape=(m, ), arr_coefficients.shape=(m, p)

        featureNames = xtrain_interpretable2.columns.tolist()
        RiskScoreClassifier_m = RiskScoreClassifier(multiplier = arr_multiplier[0],
                                                    intercept = arr_intercept[0], 
                                                    coefficients = arr_coefficients[0],
                                                    X_train = np.asarray(xtrain_interpretable2), featureNames = featureNames)
        y_val_pred_proba = RiskScoreClassifier_m.predict_prob(X = np.asarray(xval_interpretable2))
        scores.loc[counter,'Val loss']= metrics.log_loss(yval, y_val_pred_proba)
         
    print(scores)    
    # initialize a risk score optimizer
    optimal_sparsity = scores.loc[np.argmin(scores['Val loss']),'Sparsity'].astype('int32')
    m = RiskScoreOptimizer(X = np.asarray(X_train_interpretable2), y = np.asarray(y_train_interpretable), k = optimal_sparsity)

    # perform optimization
    m.optimize()

    # get all top m solutions from the final diverse pool
    arr_multiplier, arr_intercept, arr_coefficients = m.get_models() # get m solutions from the diverse pool; Specifically, arr_multiplier.shape=(m, ), arr_intercept.shape=(m, ), arr_coefficients.shape=(m, p)

    featureNames = X_train_interpretable2.columns.tolist()
    RiskScoreClassifier_m = RiskScoreClassifier(multiplier = arr_multiplier[0],
                                                intercept = arr_intercept[0], 
                                                coefficients = arr_coefficients[0],
                                                X_train = np.asarray(X_train_interpretable2), featureNames = featureNames)
    RiskScoreClassifier_m.print_model_card()

    y_train_pred_proba = RiskScoreClassifier_m.predict_prob(X = np.asarray(X_train_interpretable2))
    y_test_pred_proba = RiskScoreClassifier_m.predict_prob(X = np.asarray(X_test_interpretable2))
    y_train_pred = (y_train_pred_proba > np.mean(y_train)).astype(bool)
    y_test_pred = (y_test_pred_proba > np.mean(y_train)).astype(bool)
    
    y_train.to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Labels preds\\Labels preds 5Y\\y_train_5Y_'+str(test_year)+'.csv',index=False)
    y_test.to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Labels preds\\Labels preds 5Y\\y_test_5Y_'+str(test_year)+'.csv',index=False)
    pd.Series(y_train_pred_proba,name='TRAIN_PRED_PROBA').to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Labels preds\\Labels preds 5Y\\y_train_pred_proba_fastrisk'+str(optimal_sparsity)+'_5Y_'+str(test_year)+'.csv',index=False)
    pd.Series(y_test_pred_proba,name='TEST_PRED_PROBA').to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Labels preds\\Labels preds 5Y\\y_test_pred_proba_fastrisk'+str(optimal_sparsity)+'_5Y_'+str(test_year)+'.csv',index=False)
    pd.Series(y_train_pred,name='TRAIN_PRED').to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Labels preds\\Labels preds 5Y\\y_train_pred_fastrisk'+str(optimal_sparsity)+'_5Y_'+str(test_year)+'.csv',index=False)
    pd.Series(y_test_pred,name='TEST_PRED').to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Labels preds\\Labels preds 5Y\\y_test_pred_fastrisk'+str(optimal_sparsity)+'_5Y_'+str(test_year)+'.csv',index=False)

    metrics_results(metrics_results_train, y_train, y_train, y_train_pred_proba, y_train_pred, model_num, 'FASTRISK')
    metrics_results(metrics_results_test, y_train, y_test, y_test_pred_proba, y_test_pred, model_num, 'FASTRISK') 
    calibration_results(calibration_results_train, y_train, y_train_pred_proba, 'FASTRISK')
    calibration_results(calibration_results_test, y_test, y_test_pred_proba, 'FASTRISK')

    # save model
    joblib.dump(RiskScoreClassifier_m, path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Pickle files\\Pickle files 5Y\\model_fastrisk'+str(optimal_sparsity)+'_5Y_'+str(test_year)+'.pkl')
    joblib.dump(m, path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Pickle files\\Pickle files 5Y\\opt_fastrisk'+str(optimal_sparsity)+'_5Y_'+str(test_year)+'.pkl')
    model_num = model_num+1
    del m, RiskScoreClassifier_m, y_train_pred_proba, y_train_pred, y_test_pred_proba, y_test_pred, 
    pd.DataFrame(featureNames,columns=['Feature']).to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Feature names FASTRISK\\Feature names 5Y\\feature_names_fastrisk_5Y_'+str(test_year)+'.csv',index=False)

    #######################################################################
    ################ Hierarchical Shrinkage Decision Tree #################
    #######################################################################

    space_hstree = {'max_leaf_nodes': hp.quniform('max_leaf_nodes',6, 24, 1), 
                    'reg_param': hp.uniform('reg_param',0.1, 100)} 

    def objective(args):
        
        xtrain = X_train[REC_TX_DT_TRAIN<test_year-1]
        xval = X_train[(REC_TX_DT_TRAIN==test_year-1)]
        ytrain = y_train[REC_TX_DT_TRAIN<test_year-1]
        yval = y_train[(REC_TX_DT_TRAIN==test_year-1)]
        pipe_hstree = Pipeline(steps=[('preprocess',preprocess),
                                      ('model',HSTreeClassifier(random_state=9700, 
                                                                max_leaf_nodes = int(args['max_leaf_nodes']),
                                                                reg_param = args['reg_param']))])
        pipe_hstree.fit(xtrain, ytrain)    
        y_pred_val = pipe_hstree.predict_proba(xval) 

        return {'loss': metrics.log_loss(yval, y_pred_val[:,1]), 'params': args, 'status': STATUS_OK}

    best = fmin(fn = objective, space = space_hstree, algo = tpe.suggest, max_evals = n_evals, trials = Trials(), rstate=np.random.default_rng(89), return_argmin=False)
    pipe_hstree = Pipeline(steps=[('preprocess',preprocess),
                                  ('model',HSTreeClassifier(random_state=9700, 
                                                            max_leaf_nodes = int(best['max_leaf_nodes']), 
                                                            reg_param = best['reg_param']))])
    pipe_hstree.fit(X_train, y_train)
    y_train_pred_proba = pipe_hstree.predict_proba(X_train)
    y_test_pred_proba = pipe_hstree.predict_proba(X_test)
    y_train_pred = (y_train_pred_proba[:,1] > np.mean(y_train)).astype(bool)
    y_test_pred = (y_test_pred_proba[:,1] > np.mean(y_train)).astype(bool)
    
    pd.Series(y_train_pred_proba[:,1],name='TRAIN_PRED_PROBA').to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Labels preds\\Labels preds 5Y\\y_train_pred_proba_pipe_hstree_5Y_'+str(test_year)+'.csv',index=False)
    pd.Series(y_test_pred_proba[:,1],name='TEST_PRED_PROBA').to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Labels preds\\Labels preds 5Y\\y_test_pred_proba_pipe_hstree_5Y_'+str(test_year)+'.csv',index=False)
    pd.Series(y_train_pred,name='TRAIN_PRED').to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Labels preds\\Labels preds 5Y\\y_train_pred_pipe_hstree_5Y_'+str(test_year)+'.csv',index=False)
    pd.Series(y_test_pred,name='TEST_PRED').to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Labels preds\\Labels preds 5Y\\y_test_pred_pipe_hstree_5Y_'+str(test_year)+'.csv',index=False)

    metrics_results(metrics_results_train, y_train, y_train, y_train_pred_proba[:,1], y_train_pred, model_num, 'HSDT') 
    metrics_results(metrics_results_test, y_train, y_test, y_test_pred_proba[:,1], y_test_pred, model_num, 'HSDT') 
    calibration_results(calibration_results_train, y_train, y_train_pred_proba[:,1], 'HSDT')
    calibration_results(calibration_results_test, y_test, y_test_pred_proba[:,1], 'HSDT')
    model_num = model_num+1
    # save model
    joblib.dump(pipe_hstree, path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Pickle files\\Pickle files 5Y\\pipe_hstree_5Y_'+str(test_year)+'.pkl')

    #######################################################################
    ########################## LightGBM ###################################
    #######################################################################

    del pipe_hstree, best, y_train_pred_proba, y_train_pred, y_test_pred_proba, y_test_pred
    boosting_list = ['gbdt','dart']
    space_lgb = {
                'boosting' : hp.choice('boosting',boosting_list),
                'learning_rate': hp.loguniform('learning_rate', -8, -1),    
                'max_depth' : hp.quniform('max_depth', 2, 10, 1),
                'n_estimators': hp.quniform('n_estimators', 100, 1000,1), # Previous 50, 400,1),
                'feature_fraction': hp.uniform('feature_fraction', 0.55, 0.85),
                'subsample':  hp.uniform('subsample', 0.55, 0.85),
                'lambda_l2':  hp.loguniform('lambda_l2', -5, 5),
                'lambda_l1': hp.loguniform('lambda_l1', -5, 5),
                'path_smooth': hp.loguniform('path_smooth', -5, 5),
                'min_gain_to_split': hp.loguniform('min_gain_to_split', -5, 2), 
                'min_child_weight': hp.loguniform('min_child_weight', -5, 5), 
                }

    def objective(params):    
        saved = DataFrameImputer()
        xtrain = saved.fit_transform(X_train_LGBM[REC_TX_DT_TRAIN<test_year-1])
        xval = saved.transform(X_train_LGBM[(REC_TX_DT_TRAIN==test_year-1)])
        ytrain = y_train_LGBM[REC_TX_DT_TRAIN<test_year-1]
        yval = y_train_LGBM[(REC_TX_DT_TRAIN==test_year-1)]
        model_lgb = lgb.LGBMClassifier(
                                   objective= 'binary', 
                                   boosting= params['boosting'],  
                                   learning_rate = params['learning_rate'],  
                                   max_depth = int(params['max_depth']), 
                                   n_estimators = int(params['n_estimators']),
                                   subsample = params['subsample'],
                                   feature_fraction = params['feature_fraction'],
                                   lambda_l2 = params['lambda_l2'],
                                   lambda_l1 = params['lambda_l1'],
                                   path_smooth = params['path_smooth'],
                                   min_gain_to_split = params['min_gain_to_split'], 
                                   min_child_weight = params['min_child_weight'],
                                   random_state = 9700, 
                                   verbose=-1,
                                   importance_type='gain')
        
        init_score =  scipy.special.logit(np.mean(ytrain))
        model_lgb.fit(xtrain, ytrain,init_score=np.full_like(ytrain, init_score, dtype=float))
        y_pred_val = scipy.special.expit(init_score+scipy.special.logit(model_lgb.predict_proba(xval)))

        return {'loss': metrics.log_loss(yval, y_pred_val[:,1]), 'params': params, 'status': STATUS_OK}

    # Optimize
    best = fmin(fn = objective, space = space_lgb, algo = tpe.suggest, max_evals = n_evals, trials = Trials(), rstate=np.random.default_rng(89),return_argmin=False)

    model_lgb = lgb.LGBMClassifier(
                                   objective= 'binary', 
                                   boosting= best['boosting'], 
                                   learning_rate = best['learning_rate'], 
                                   max_depth = int(best['max_depth']),
                                   n_estimators = int(best['n_estimators']), 
                                   subsample = best['subsample'],
                                   feature_fraction = best['feature_fraction'],
                                   lambda_l2 = best['lambda_l2'],
                                   lambda_l1 = best['lambda_l1'],
                                   path_smooth = best['path_smooth'],
                                   min_gain_to_split = best['min_gain_to_split'],
                                   min_child_weight = best['min_child_weight'],
                                   random_state = 9700, 
                                   verbose=-1,
                                   importance_type='gain')

    init_score =  scipy.special.logit(np.mean(y_train_LGBM))
    model_lgb.fit(Ximp_train_LGBM, y_train_LGBM,init_score=np.full_like(y_train_LGBM, init_score, dtype=float))
    y_train_pred_proba = scipy.special.expit(init_score+scipy.special.logit(model_lgb.predict_proba(Ximp_train_LGBM)))
    y_train_pred = (y_train_pred_proba[:,1] > np.mean(y_train)).astype(bool)

    y_test_pred_proba = scipy.special.expit(init_score+scipy.special.logit(model_lgb.predict_proba(Ximp_test_LGBM)))
    y_test_pred = (y_test_pred_proba[:,1] > np.mean(y_train)).astype(bool)
    
    pd.Series(y_train_pred_proba[:,1],name='TRAIN_PRED_PROBA').to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Labels preds\\Labels preds 5Y\\y_train_pred_proba_model_lgb_5Y_'+str(test_year)+'.csv',index=False)
    pd.Series(y_test_pred_proba[:,1],name='TEST_PRED_PROBA').to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Labels preds\\Labels preds 5Y\\y_test_pred_proba_model_lgb_5Y_'+str(test_year)+'.csv',index=False)
    pd.Series(y_train_pred,name='TRAIN_PRED').to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Labels preds\\Labels preds 5Y\\y_train_pred_model_lgb_5Y_'+str(test_year)+'.csv',index=False)
    pd.Series(y_test_pred,name='TEST_PRED').to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Labels preds\\Labels preds 5Y\\y_test_pred_model_lgb_5Y_'+str(test_year)+'.csv',index=False)

    metrics_results(metrics_results_train, y_train, y_train, y_train_pred_proba[:,1], y_train_pred, model_num, 'LGB') 
    metrics_results(metrics_results_test, y_train, y_test, y_test_pred_proba[:,1], y_test_pred, model_num, 'LGB') 
    calibration_results(calibration_results_train, y_train, y_train_pred_proba[:,1], 'LGB')
    calibration_results(calibration_results_test, y_test, y_test_pred_proba[:,1], 'LGB')
    model_num = model_num+1

    # Variable importance
    lgb_feature_imp = pd.DataFrame(sorted(zip(model_lgb.feature_importances_,X_train_LGBM.columns)), columns=['Value','Feature'])
    lgb_feature_imp = lgb_feature_imp.sort_values(by=['Value'],ascending=False).reset_index(drop=True)
    lgb_feature_imp.to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Variable importance\\Variable importance 5Y\\lgb_total_gain_5Y_'+str(test_year)+'.csv',index=False)
    # Save model
    joblib.dump(model_lgb, path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Pickle files\\Pickle files 5Y\\model_lgb_5Y_'+str(test_year)+'.pkl')

    #######################################################################
    ########################### XGBoost ###################################
    #######################################################################

    del model_lgb, best, y_train_pred_proba, y_train_pred, y_test_pred_proba, y_test_pred
    def objective(params):
        scores = []
        saved = DataFrameImputer()
        xtrain = saved.fit_transform(X_train_LGBM[REC_TX_DT_TRAIN<test_year-1])
        xval = saved.transform(X_train_LGBM[(REC_TX_DT_TRAIN==test_year-1)])
        ytrain = y_train_LGBM[REC_TX_DT_TRAIN<test_year-1]
        yval = y_train_LGBM[(REC_TX_DT_TRAIN==test_year-1)]
        model_xgb = XGBClassifier(grow_policy = params['grow_policy'],
                                  learning_rate = params['learning_rate'],
                                  max_depth = int(params['max_depth']),
                                  n_estimators = int(params['n_estimators']),
                                  subsample = params['subsample'], 
                                  colsample_bytree = params['colsample_bytree'],
                                  reg_lambda = params['reg_lambda'],  
                                  reg_alpha = params['reg_alpha'], 
                                  gamma = params['gamma'],
                                  min_child_weight = params['min_child_weight'],
                                  random_state=9700, 
                                  tree_method="hist",
                                  enable_categorical=True,
                                  objective='binary:logistic',
                                  base_score=np.mean(ytrain))

        model_xgb.fit(xtrain, ytrain)
        y_pred_val = model_xgb.predict_proba(xval) 

        return {'loss': metrics.log_loss(yval, y_pred_val[:,1]), 'params': params, 'status': STATUS_OK}

    space_xgb = {   
                    'grow_policy': hp.choice('grow_policy',['depthwise','lossguide']),
                    'learning_rate': hp.loguniform('learning_rate', -8, -1),
                    'max_depth' : hp.quniform('max_depth', 2, 10, 1),
                    'n_estimators': hp.quniform('n_estimators',100, 1000,1), # Previous 50, 400,1),
                    'subsample':  hp.uniform('subsample', 0.55, 0.85),
                    'colsample_bytree': hp.uniform('colsample_bytree', 0.55, 0.85),
                    'reg_lambda': hp.loguniform('reg_lambda', -5, 5),
                    'reg_alpha': hp.loguniform('reg_alpha', -5, 5),
                    'gamma': hp.loguniform('gamma', -5, 2),
                    'min_child_weight': hp.loguniform('min_child_weight', -5, 5),
        
    }


    # Optimize
    best = fmin(fn = objective, space = space_xgb, algo = tpe.suggest, max_evals = n_evals, trials = Trials(), rstate=np.random.default_rng(89),return_argmin=False)
    model_xgb = XGBClassifier(    grow_policy = best['grow_policy'],
                                  learning_rate = best['learning_rate'],
                                  max_depth = int(best['max_depth']),
                                  n_estimators = int(best['n_estimators']),
                                  subsample = best['subsample'], 
                                  colsample_bytree = best['colsample_bytree'],
                                  reg_lambda = best['reg_lambda'],  
                                  reg_alpha = best['reg_alpha'],
                                  gamma = best['gamma'],
                                  min_child_weight = best['min_child_weight'],
                                  random_state=9700, 
                                  tree_method="hist",
                                  enable_categorical=True,
                                  objective='binary:logistic',
                                  base_score=np.mean(y_train_LGBM))
    model_xgb.fit(Ximp_train_LGBM, y_train_LGBM)
    y_train_pred_proba = model_xgb.predict_proba(Ximp_train_LGBM)
    y_train_pred = (y_train_pred_proba[:,1] > np.mean(y_train)).astype(bool)

    y_test_pred_proba = model_xgb.predict_proba(Ximp_test_LGBM)
    y_test_pred = (y_test_pred_proba[:,1] > np.mean(y_train)).astype(bool)
    
    pd.Series(y_train_pred_proba[:,1],name='TRAIN_PRED_PROBA').to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Labels preds\\Labels preds 5Y\\y_train_pred_proba_model_xgb_5Y_'+str(test_year)+'.csv',index=False)
    pd.Series(y_test_pred_proba[:,1],name='TEST_PRED_PROBA').to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Labels preds\\Labels preds 5Y\\y_test_pred_proba_model_xgb_5Y_'+str(test_year)+'.csv',index=False)
    pd.Series(y_train_pred,name='TRAIN_PRED').to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Labels preds\\Labels preds 5Y\\y_train_pred_model_xgb_5Y_'+str(test_year)+'.csv',index=False)
    pd.Series(y_test_pred,name='TEST_PRED').to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Labels preds\\Labels preds 5Y\\y_test_pred_model_xgb_5Y_'+str(test_year)+'.csv',index=False)

    metrics_results(metrics_results_train, y_train, y_train, y_train_pred_proba[:,1], y_train_pred, model_num, 'XGB') 
    metrics_results(metrics_results_test, y_train, y_test, y_test_pred_proba[:,1], y_test_pred, model_num, 'XGB')
    calibration_results(calibration_results_train, y_train, y_train_pred_proba[:,1], 'XGB')
    calibration_results(calibration_results_test, y_test, y_test_pred_proba[:,1], 'XGB')
    model_num = model_num+1
    feature_importance = model_xgb.get_booster().get_score(importance_type='total_gain')
    keys = list(feature_importance.keys())
    values = list(feature_importance.values())

    xgb_feature_imp = pd.DataFrame(data=values, index=keys, columns=["Value"]).sort_values(by = "Value", ascending=False)
    xgb_feature_imp = xgb_feature_imp.reset_index()
    xgb_feature_imp = xgb_feature_imp.rename(columns={"index": "Feature"})
    xgb_feature_imp.to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Variable importance\\Variable importance 5Y\\xgb_total_gain_5Y_'+str(test_year)+'.csv',index=False)
    # save model
    joblib.dump(model_xgb, path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Pickle files\\Pickle files 5Y\\model_xgb_5Y_'+str(test_year)+'.pkl')

    #######################################################################
    ######################### Random Forest ###############################
    #######################################################################

    del model_xgb, best, y_train_pred_proba, y_train_pred, y_test_pred_proba, y_test_pred
    def objective(args):

        xtrain = X_train[REC_TX_DT_TRAIN<test_year-1]
        xval = X_train[(REC_TX_DT_TRAIN==test_year-1)]
        ytrain = y_train[REC_TX_DT_TRAIN<test_year-1]
        yval = y_train[(REC_TX_DT_TRAIN==test_year-1)]
        pipe_rfc = Pipeline(steps=[('preprocess',preprocess),('model',RandomForestClassifier(random_state=9700))]) # min_impurity_decrease= 0.0004
        pipe_rfc.set_params(**{'model__criterion': args['model__criterion'],
                               'model__max_depth': int(args['model__max_depth']),
                               'model__max_features': args['model__max_features'],
                               'model__n_estimators': int(args['model__n_estimators']),
                               'model__min_samples_split': int(args['model__min_samples_split']),
                               'model__min_impurity_decrease': args['model__min_impurity_decrease'],
                               'model__max_samples': args['model__max_samples']})

        pipe_rfc.fit(xtrain, ytrain)
        y_pred_val = pipe_rfc.predict_proba(xval) 

        return {'loss': metrics.log_loss(yval, y_pred_val[:,1]), 'params': args, 'status': STATUS_OK}

    space_rfc = {
            'model__criterion': hp.choice('model__criterion',['gini','log_loss']),
            'model__max_depth': hp.quniform('model__max_depth',5,25,1),
            'model__max_features': hp.uniform('model__max_features', 0.55, 0.85), 
            'model__n_estimators': hp.quniform('model__n_estimators', 100, 1000,1), # Previous 50, 400,1),
            'model__min_samples_split': hp.quniform('model__min_samples_split', 30, 150, 1),
            'model__min_impurity_decrease': hp.loguniform('model__min_impurity_decrease', -8, 2),
            'model__max_samples':  hp.uniform('model__max_samples', 0.55, 0.85)}
    # Optimize
    best = fmin(fn = objective, space = space_rfc, algo = tpe.suggest, max_evals = n_evals, trials = Trials(), rstate=np.random.default_rng(89), return_argmin=False)

    pipe_rfc = Pipeline(steps=[('preprocess',preprocess),('model',RandomForestClassifier(random_state=9700))]) # min_impurity_decrease= 0.0004
    best_params = {'model__criterion': best['model__criterion'],
                   'model__max_depth': int(best['model__max_depth']),
                   'model__max_features': best['model__max_features'],
                   'model__n_estimators': int(best['model__n_estimators']),               
                   'model__min_samples_split': int(best['model__min_samples_split']),
                   'model__min_impurity_decrease': best['model__min_impurity_decrease'],
                   'model__max_samples': best['model__max_samples']} 
    pipe_rfc.set_params(**best_params)
    pipe_rfc.fit(X_train, y_train)
    y_train_pred_proba = pipe_rfc.predict_proba(X_train) 
    y_test_pred_proba = pipe_rfc.predict_proba(X_test)
    y_train_pred = (y_train_pred_proba[:,1] > np.mean(y_train)).astype(bool)
    y_test_pred = (y_test_pred_proba[:,1] > np.mean(y_train)).astype(bool)
    
    pd.Series(y_train_pred_proba[:,1],name='TRAIN_PRED_PROBA').to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Labels preds\\Labels preds 5Y\\y_train_pred_proba_pipe_rfc_5Y_'+str(test_year)+'.csv',index=False)
    pd.Series(y_test_pred_proba[:,1],name='TEST_PRED_PROBA').to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Labels preds\\Labels preds 5Y\\y_test_pred_proba_pipe_rfc_5Y_'+str(test_year)+'.csv',index=False)
    pd.Series(y_train_pred,name='TRAIN_PRED').to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Labels preds\\Labels preds 5Y\\y_train_pred_pipe_rfc_5Y_'+str(test_year)+'.csv',index=False)
    pd.Series(y_test_pred,name='TEST_PRED').to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Labels preds\\Labels preds 5Y\\y_test_pred_pipe_rfc_5Y_'+str(test_year)+'.csv',index=False)

    metrics_results(metrics_results_train, y_train, y_train, y_train_pred_proba[:,1], y_train_pred, model_num, 'RF') 
    metrics_results(metrics_results_test, y_train, y_test, y_test_pred_proba[:,1], y_test_pred, model_num, 'RF') 
    calibration_results(calibration_results_train, y_train, y_train_pred_proba[:,1], 'RF')
    calibration_results(calibration_results_test, y_test, y_test_pred_proba[:,1], 'RF')
    model_num = model_num+1
    # save model
    joblib.dump(pipe_rfc, path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Pickle files\\Pickle files 5Y\\pipe_rfc_5Y_'+str(test_year)+'.pkl')

    #######################################################################
    ######################### Logistic Regression #########################
    #######################################################################

    del pipe_rfc, best_params, best, y_train_pred_proba, y_train_pred, y_test_pred_proba, y_test_pred
    pipe_lr = Pipeline(steps=[('preprocess',preprocess_lr),('model',LogisticRegression(penalty='none',solver='saga',max_iter=400,random_state=9700))])
    pipe_lr.fit(X_train, y_train)
    y_train_pred_proba = pipe_lr.predict_proba(X_train) 
    y_test_pred_proba = pipe_lr.predict_proba(X_test)
    y_train_pred = (y_train_pred_proba[:,1] > np.mean(y_train)).astype(bool)
    y_test_pred = (y_test_pred_proba[:,1] > np.mean(y_train)).astype(bool)
    
    pd.Series(y_train_pred_proba[:,1],name='TRAIN_PRED_PROBA').to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Labels preds\\Labels preds 5Y\\y_train_pred_proba_pipe_lr_5Y_'+str(test_year)+'.csv',index=False)
    pd.Series(y_test_pred_proba[:,1],name='TEST_PRED_PROBA').to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Labels preds\\Labels preds 5Y\\y_test_pred_proba_pipe_lr_5Y_'+str(test_year)+'.csv',index=False)
    pd.Series(y_train_pred,name='TRAIN_PRED').to_csv('C:\\Users\\Henry Johnston\Documents\\SRTR_Files_Heart\\SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Labels preds\\Labels preds 5Y\\y_train_pred_pipe_lr_5Y_'+str(test_year)+'.csv',index=False)
    pd.Series(y_test_pred,name='TEST_PRED').to_csv('C:\\Users\\Henry Johnston\Documents\\SRTR_Files_Heart\\SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Labels preds\\Labels preds 5Y\\y_test_pred_pipe_lr_5Y_'+str(test_year)+'.csv',index=False)

    metrics_results(metrics_results_train, y_train, y_train, y_train_pred_proba[:,1], y_train_pred, model_num, 'LR') 
    metrics_results(metrics_results_test, y_train, y_test, y_test_pred_proba[:,1], y_test_pred, model_num, 'LR') 
    calibration_results(calibration_results_train, y_train, y_train_pred_proba[:,1], 'LR')
    calibration_results(calibration_results_test, y_test, y_test_pred_proba[:,1], 'LR')
    model_num = model_num+1

    # save model
    joblib.dump(pipe_lr, path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Pickle files\\Pickle files 5Y\\pipe_lr_5Y_'+str(test_year)+'.pkl')

    #######################################################################
    ######################### ElasticNet Logistic Regression ##############
    #######################################################################

    del pipe_lr, y_train_pred_proba, y_train_pred, y_test_pred_proba, y_test_pred
    def objective(args):
        xtrain = X_train[REC_TX_DT_TRAIN<test_year-1]
        xval = X_train[(REC_TX_DT_TRAIN==test_year-1)]
        ytrain = y_train[REC_TX_DT_TRAIN<test_year-1]
        yval = y_train[(REC_TX_DT_TRAIN==test_year-1)]
        pipe_elasticnet = Pipeline(steps=[('preprocess',preprocess_lasso),('model',LogisticRegression(penalty='elasticnet',solver='saga',max_iter=400,random_state=9700))])
        pipe_elasticnet.set_params(**{
                                     'model__l1_ratio': args['model__l1_ratio'],
                                     'model__C': args['model__C']})
        pipe_elasticnet.fit(xtrain, ytrain)
        y_pred_val = pipe_elasticnet.predict_proba(xval) 

        return {'loss': metrics.log_loss(yval, y_pred_val[:,1]), 'params': args, 'status': STATUS_OK}

    space_elasticnet = {

                        'model__l1_ratio': hp.loguniform('model__l1_ratio', -3, 0),
                        'model__C': hp.loguniform('model__C',-4, 2)}

    # Optimize
    best = fmin(fn = objective, space = space_elasticnet, algo = tpe.suggest, max_evals = n_evals, trials = Trials(), rstate=np.random.default_rng(89), return_argmin=False)
    pipe_elasticnet = Pipeline(steps=[('preprocess',preprocess_lasso),('model',LogisticRegression(penalty='elasticnet',solver='saga',max_iter=400,random_state=9700))])
    best_params = {
                   'model__l1_ratio': best['model__l1_ratio'],
                   'model__C': best['model__C']}
    pipe_elasticnet.set_params(**best_params)
    pipe_elasticnet.fit(X_train, y_train)
    y_train_pred_proba = pipe_elasticnet.predict_proba(X_train)
    y_train_pred = (y_train_pred_proba[:,1] > np.mean(y_train)).astype(bool)
    y_test_pred_proba = pipe_elasticnet.predict_proba(X_test)
    y_test_pred = (y_test_pred_proba[:,1] > np.mean(y_train)).astype(bool)
    
    pd.Series(y_train_pred_proba[:,1],name='TRAIN_PRED_PROBA').to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Labels preds\\Labels preds 5Y\\y_train_pred_proba_pipe_elasticnet_5Y_'+str(test_year)+'.csv',index=False)
    pd.Series(y_test_pred_proba[:,1],name='TEST_PRED_PROBA').to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Labels preds\\Labels preds 5Y\\y_test_pred_proba_pipe_elasticnet_5Y_'+str(test_year)+'.csv',index=False)
    pd.Series(y_train_pred,name='TRAIN_PRED').to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Labels preds\\Labels preds 5Y\\y_train_pred_pipe_elasticnet_5Y_'+str(test_year)+'.csv',index=False)
    pd.Series(y_test_pred,name='TEST_PRED').to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Labels preds\\Labels preds 5Y\\y_test_pred_pipe_elasticnet_5Y_'+str(test_year)+'.csv',index=False)

    metrics_results(metrics_results_train, y_train, y_train, y_train_pred_proba[:,1], y_train_pred, model_num, 'ELASTICNET') 
    metrics_results(metrics_results_test, y_train, y_test, y_test_pred_proba[:,1], y_test_pred, model_num, 'ELASTICNET') 
    calibration_results(calibration_results_train, y_train, y_train_pred_proba[:,1], 'ELASTICNET')
    calibration_results(calibration_results_test, y_test, y_test_pred_proba[:,1], 'ELASTICNET')
    model_num = model_num+1
    # save model
    joblib.dump(pipe_elasticnet, path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Pickle files\\Pickle files 5Y\\pipe_elasticnet_5Y_'+str(test_year)+'.pkl')

    #######################################################################
    ######################### LASSO Logistic Regression ###################
    #######################################################################

    del pipe_elasticnet, best_params, best, y_train_pred_proba, y_train_pred, y_test_pred_proba, y_test_pred
    def objective(args):
        
        xtrain = X_train[REC_TX_DT_TRAIN<test_year-1]
        xval = X_train[(REC_TX_DT_TRAIN==test_year-1)]
        ytrain = y_train[REC_TX_DT_TRAIN<test_year-1]
        yval = y_train[(REC_TX_DT_TRAIN==test_year-1)]
        pipe_lasso = Pipeline(steps=[('preprocess',preprocess_lasso),('model',LogisticRegression(penalty='l1',random_state=9700))])
        pipe_lasso.set_params(**{'model__solver': args['model__solver'],
                                 'model__C': args['model__C']})

        pipe_lasso.fit(xtrain, ytrain)
        y_pred_val = pipe_lasso.predict_proba(xval) 


        return {'loss': metrics.log_loss(yval, y_pred_val[:,1]), 'params': args, 'status': STATUS_OK}


    space_lasso = {'model__solver': hp.choice('model__solver',['liblinear']),
                   'model__C': hp.loguniform('model__C',-4, 2)}

    # Optimize
    best = fmin(fn = objective, space = space_lasso, algo = tpe.suggest, max_evals = n_evals, trials = Trials(), rstate=np.random.default_rng(89), return_argmin=False)
    pipe_lasso = Pipeline(steps=[('preprocess',preprocess_lasso),('model',LogisticRegression(penalty='l1',random_state=9700))])
    best_params = {'model__solver': best['model__solver'],
                   'model__C': best['model__C']}
    pipe_lasso.set_params(**best_params)
    pipe_lasso.fit(X_train, y_train)
    y_train_pred_proba = pipe_lasso.predict_proba(X_train)
    y_train_pred = (y_train_pred_proba[:,1] > np.mean(y_train)).astype(bool)
    y_test_pred_proba = pipe_lasso.predict_proba(X_test)
    y_test_pred = (y_test_pred_proba[:,1] > np.mean(y_train)).astype(bool)
    
    pd.Series(y_train_pred_proba[:,1],name='TRAIN_PRED_PROBA').to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Labels preds\\Labels preds 5Y\\y_train_pred_proba_pipe_lasso_5Y_'+str(test_year)+'.csv',index=False)
    pd.Series(y_test_pred_proba[:,1],name='TEST_PRED_PROBA').to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Labels preds\\Labels preds 5Y\\y_test_pred_proba_pipe_lasso_5Y_'+str(test_year)+'.csv',index=False)
    pd.Series(y_train_pred,name='TRAIN_PRED').to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Labels preds\\Labels preds 5Y\\y_train_pred_pipe_lasso_5Y_'+str(test_year)+'.csv',index=False)
    pd.Series(y_test_pred,name='TEST_PRED').to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Labels preds\\Labels preds 5Y\\y_test_pred_pipe_lasso_5Y_'+str(test_year)+'.csv',index=False)

    metrics_results(metrics_results_train, y_train, y_train, y_train_pred_proba[:,1], y_train_pred, model_num, 'LASSO') 
    metrics_results(metrics_results_test, y_train, y_test, y_test_pred_proba[:,1], y_test_pred, model_num, 'LASSO') 
    calibration_results(calibration_results_train, y_train, y_train_pred_proba[:,1], 'LASSO')
    calibration_results(calibration_results_test, y_test, y_test_pred_proba[:,1], 'LASSO')

    model_num = model_num+1
    # save model
    joblib.dump(pipe_lasso, path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Pickle files\\Pickle files 5Y\\pipe_lasso_5Y_'+str(test_year)+'.pkl')

    #######################################################################
    ######################### Ridge Logistic Regression ###################
    #######################################################################

    del pipe_lasso, best_params, best, y_train_pred_proba, y_train_pred, y_test_pred_proba, y_test_pred
    def objective(args):
        
        xtrain = X_train[REC_TX_DT_TRAIN<test_year-1]
        xval = X_train[(REC_TX_DT_TRAIN==test_year-1)]
        ytrain = y_train[REC_TX_DT_TRAIN<test_year-1]
        yval = y_train[(REC_TX_DT_TRAIN==test_year-1)]
        pipe_ridge = Pipeline(steps=[('preprocess',preprocess_lasso),('model',LogisticRegression(penalty='l2',random_state=9700))])
        pipe_ridge.set_params(**{                    
                                 'model__solver': args['model__solver'],
                                 'model__C': args['model__C']})

        pipe_ridge.fit(xtrain, ytrain)
        y_pred_val = pipe_ridge.predict_proba(xval) 


        return {'loss': metrics.log_loss(yval, y_pred_val[:,1]), 'params': args, 'status': STATUS_OK}

    space_ridge = {'model__solver': hp.choice('model__solver',['lbfgs']),
                   'model__C': hp.loguniform('model__C',-4, 2)}

    # Optimize
    best = fmin(fn = objective, space = space_ridge, algo = tpe.suggest, max_evals = n_evals, trials = Trials(), rstate=np.random.default_rng(89), return_argmin=False)
    pipe_ridge = Pipeline(steps=[('preprocess',preprocess_lasso),('model',LogisticRegression(penalty='l2',random_state=9700))])
    best_params = {'model__solver': best['model__solver'],
                   'model__C': best['model__C']}
    pipe_ridge.set_params(**best_params)
    pipe_ridge.fit(X_train, y_train)
    y_train_pred_proba = pipe_ridge.predict_proba(X_train)
    y_train_pred = (y_train_pred_proba[:,1] > np.mean(y_train)).astype(bool)
    y_test_pred_proba = pipe_ridge.predict_proba(X_test)
    y_test_pred = (y_test_pred_proba[:,1] > np.mean(y_train)).astype(bool)
    
    pd.Series(y_train_pred_proba[:,1],name='TRAIN_PRED_PROBA').to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Labels preds\\Labels preds 5Y\\y_train_pred_proba_pipe_ridge_5Y_'+str(test_year)+'.csv',index=False)
    pd.Series(y_test_pred_proba[:,1],name='TEST_PRED_PROBA').to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Labels preds\\Labels preds 5Y\\y_test_pred_proba_pipe_ridge_5Y_'+str(test_year)+'.csv',index=False)
    pd.Series(y_train_pred,name='TRAIN_PRED').to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Labels preds\\Labels preds 5Y\\y_train_pred_pipe_ridge_5Y_'+str(test_year)+'.csv',index=False)
    pd.Series(y_test_pred,name='TEST_PRED').to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Labels preds\\Labels preds 5Y\\y_test_pred_pipe_ridge_5Y_'+str(test_year)+'.csv',index=False)

    metrics_results(metrics_results_train, y_train, y_train, y_train_pred_proba[:,1], y_train_pred, model_num, 'RIDGE') 
    metrics_results(metrics_results_test, y_train, y_test, y_test_pred_proba[:,1], y_test_pred, model_num, 'RIDGE') 
    calibration_results(calibration_results_train, y_train, y_train_pred_proba[:,1], 'RIDGE')
    calibration_results(calibration_results_test, y_test, y_test_pred_proba[:,1], 'RIDGE')
    model_num = model_num+1
    # save model
    joblib.dump(pipe_ridge, path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Pickle files\\Pickle files 5Y\\pipe_ridge_5Y_'+str(test_year)+'.pkl')
     
    
    metrics_results_train.to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Metrics\\Metrics 5Y\\metrics_results_train_5Y_'+str(test_year)+'.csv',index=False)
    metrics_results_test.to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Metrics\\Metrics 5Y\\metrics_results_test_5Y_'+str(test_year)+'.csv',index=False)
    calibration_results_train.to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Calibration\\Calibration 5Y\\calibration_results_train_5Y_'+str(test_year)+'.csv',index=False)
    calibration_results_test.to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Calibration\\Calibration 5Y\\calibration_results_test_5Y_'+str(test_year)+'.csv',index=False)
    
    metrics_results_train_all = pd.concat([metrics_results_train_all,metrics_results_train],ignore_index=True)
    metrics_results_test_all = pd.concat([metrics_results_test_all,metrics_results_test],ignore_index=True)
    calibration_results_train_all = pd.concat([calibration_results_train_all,calibration_results_train],ignore_index=True)
    calibration_results_test_all = pd.concat([calibration_results_test_all,calibration_results_test],ignore_index=True)

metrics_results_train_all.to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Metrics\\Metrics 5Y\\metrics_results_train_5Y.csv',index=False)
metrics_results_test_all.to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Metrics\\Metrics 5Y\\metrics_results_test_5Y.csv',index=False)
calibration_results_train_all.to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Calibration\\Calibration 5Y\\calibration_results_train_5Y.csv',index=False)
calibration_results_test_all.to_csv(path_srtr+'SRTR_Heart\\Results_THORACIC_PTLD1_v3\\Calibration\\Calibration 5Y\\calibration_results_test_5Y.csv',index=False)
