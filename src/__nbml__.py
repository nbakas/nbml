


########################################################################################################################################
# More instructions in README.md and __nbml__.pdf
########################################################################################################################################



########################################################################################################################################
# Define the problem
# Simply define the main parameters here. The code will automatically produce the corresponding graphs and tables.
ROOT_DIR = ""
# A *****directory without the filename***** with only one excel file.
# The *.xlsx file shhould comprise with all the independent variables at the first $n$ columns, followed by the target variable as the last column. 
# For Windows, please use *****\\***** separators and remeber to *****add the \\ at the end*****. 
# For Linux please use /.../.../ format
LOGISTIC_REGR = False # If True do classification
PERMUTE_TRAIN_TEST = True # If True split the data into training/testing sets randomly, after shuffling. 
# If False, Top rows are train and bottom test, which is helpfull for time series data.
########################################################################################################################################



########################################################################################################################################
# import Libraries
from import_libraries import *
import import_libraries, misc_functions, descriptive_statistics, ml_linear_regression, ml_nlregr, ml_xgboost
import ml_ANNBN, ml_random_forests, ml_DANN
reload(import_libraries); reload(misc_functions); reload(descriptive_statistics)
reload(ml_linear_regression); reload(ml_nlregr); reload(ml_xgboost); reload(ml_ANNBN); reload(ml_random_forests); reload(ml_DANN)
########################################################################################################################################



########################################################################################################################################
# Open the Dataset and Split to Train & Test
test_ratio = 0.3 #The ratio of the data to be used for testing (0-1). Usually 0.2-0.3
random_seed = 0 #The random seed to be used for the random number generator.
df, features_names, target_name = misc_functions.read_the_dataset_dropna(ROOT_DIR)
df, features_names, target_name = misc_functions.make_features_and_target_categorical_if_so(df, target_name)
Xtr, Xte, ytr, yte = misc_functions.split_train_test(PERMUTE_TRAIN_TEST, test_ratio, random_seed, df)
misc_functions.create_all_directories(ROOT_DIR, PERMUTE_TRAIN_TEST)
########################################################################################################################################



########################################################################################################################################
# Clean the Data
Xte,yte = misc_functions.delete_missing_and_inf_values_rows(Xte,yte)
Xtr,ytr = misc_functions.delete_missing_and_inf_values_rows(Xtr,ytr)
# Xtr,ytr = misc_functions.delete_identical_rows(Xtr,ytr)
Xtr, Xte, features_names = misc_functions.check_fix_multicolinearity(Xtr, Xte, features_names, ROOT_DIR)
# Xtr, ytr, features_names = misc_functions.make_lags(Xtr, ytr, features_names, target_name, lags=51)
# Xte, yte, features_names = misc_functions.make_lags(Xte, yte, features_names, target_name, lags=51)
########################################################################################################################################



########################################################################################################################################
# Descriptive Statistics 
descriptive_statistics.descriptive_statistics(Xtr, Xte, features_names, ROOT_DIR)
descriptive_statistics.plot_short_tree(Xtr, ytr, features_names, target_name, ROOT_DIR)
descriptive_statistics.plot_pdf_cdf_all(Xtr, ytr, features_names, target_name, ROOT_DIR)
descriptive_statistics.plot_all_by_all_correlation_matrix(Xtr, ytr, features_names, target_name, ROOT_DIR)
# descriptive_statistics.export_descriptive_per_bin(Xtr,Xte,ytr,yte,features_names, target_name, ROOT_DIR)
# descriptive_statistics.plot_all_timeseries(Xtr, features_names, ytr, target_name, "Train", ROOT_DIR)
# descriptive_statistics.plot_all_timeseries(Xte, features_names, yte, target_name, "Test", ROOT_DIR)
########################################################################################################################################



########################################################################################################################################
# Machine Learning
########################################################################################################################################


## Linear Regression
ml_linear_regression.do_regression(Xtr, Xte, ytr, yte, features_names, target_name, ROOT_DIR, LOGISTIC_REGR)


## Polynomial Regression
ml_nlregr.do_nlregr(Xtr, Xte, ytr, yte, features_names, target_name, LOGISTIC_REGR, PERMUTE_TRAIN_TEST, ROOT_DIR)


## XGBoost
__thres_early_stop__ = 1e-2
__thres_min_tune_rounds__ = 100
ml_xgboost.do_xgboost(Xtr,Xte,ytr,yte,features_names,target_name,__thres_early_stop__,__thres_min_tune_rounds__,
                      PERMUTE_TRAIN_TEST,LOGISTIC_REGR,ROOT_DIR)
ml_xgboost.do_QuantileGradientBoostingRegressor(ROOT_DIR, Xtr, Xte, ytr, yte, target_name)


## Random Forests
__thres_early_stop__ = 1e-2
__thres_min_tune_rounds__ = 100
ml_random_forests.do_random_forests(Xtr,Xte,ytr,yte,features_names,target_name,__thres_early_stop__,
                                    __thres_min_tune_rounds__,PERMUTE_TRAIN_TEST,LOGISTIC_REGR,ROOT_DIR)


## ANNBN
ml_ANNBN.do_ANNBN(Xtr, Xte, ytr, yte, features_names, target_name,
                  PERMUTE_TRAIN_TEST, LOGISTIC_REGR, ROOT_DIR, min_obs_over_neurons=10)


## Deep Learning
__thres_early_stop__ = 1e-2
__thres_min_tune_rounds__ = 100
ml_DANN.do_DANN(Xtr,ytr,Xte,yte,features_names,target_name,__thres_early_stop__,
                __thres_min_tune_rounds__,PERMUTE_TRAIN_TEST,LOGISTIC_REGR,ROOT_DIR)


########################################################################################################################################
# Sensitivity Analysis
misc_functions.gather_all_sensitivity_curves(Xtr, features_names, target_name, ROOT_DIR)
########################################################################################################################################


########################################################################################################################################
# Predict: Please run the first #####2 blocs##### at the top of this file, these are:
# ####Define the problem#####
# ####import Libraries#####
# The *.xlsx file to be used for prediction, should be in the #####"Predict"##### folder, inside #####ROOT_DIR##### directory.
########################################################################################################################################
Xout, yout, features_names, target_name = misc_functions.read_out_of_sample_data(ROOT_DIR)
Xout,yout = misc_functions.delete_missing_values_rows(Xout,yout)
########################################################################################################################################
ml_linear_regression.predict_linregr(Xout, yout, target_name, LOGISTIC_REGR, ROOT_DIR)
########################################################################################################################################
ml_xgboost.predict_xgboost(Xout, yout, target_name, LOGISTIC_REGR, ROOT_DIR)
ml_xgboost.predict_quantile_gb(ROOT_DIR, Xout)
########################################################################################################################################
ml_random_forests.predict_rf(Xout, yout, target_name, LOGISTIC_REGR, ROOT_DIR)
########################################################################################################################################
ml_ANNBN.predict_ANNBN(Xout, yout, target_name, LOGISTIC_REGR, ROOT_DIR)
########################################################################################################################################
ml_DANN.predict_DANN(Xout, yout, target_name, LOGISTIC_REGR, ROOT_DIR)
########################################################################################################################################

