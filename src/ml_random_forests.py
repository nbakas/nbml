
# Description: This file contains the code for training and evaluating rf models.


from import_libraries import *
from misc_functions import *


def do_random_forests(Xtr,Xte,ytr,yte,features_names,target_name,__thres_early_stop__,__thres_min_tune_rounds__,PERMUTE_TRAIN_TEST,LOGISTIC_REGR,ROOT_DIR):
    __n_jobs__ = 4
    try:
        t0=time()
        # This parameter determines the number of decision trees to build. Increasing the number of trees typically 
        # leads to better performance, but also increases the computational complexity and training time.
        n_estimators_ = list(concatenate((arange(1,11),arange(20, 101, 10),arange(200, 1001, 100))))

        # This parameter controls the maximum depth of each decision tree. Increasing the maximum depth can improve 
        # the model's ability to fit complex patterns in the data, but can also lead to overfitting.
        max_depth_ = list(arange(1, 11))

        # This parameter controls the minimum number of samples required to split an internal node. 
        # Increasing this parameter can prevent overfitting by requiring more samples to be present before a split is considered.
        min_samples_split_ = list(concatenate((arange(2,11),arange(20, 101, 10))))

        # This parameter determines the number of features to consider when looking for the best split. 
        # Increasing this parameter can improve the model's ability to capture complex patterns in the data, but can also increase overfitting.
        max_features_ = list(np_round(arange(0.1, 1.01, 0.05), 2))

        # This parameter controls the minimum number of samples required to be at a leaf node. 
        # Increasing this parameter can prevent overfitting by requiring each leaf to have more samples.
        min_samples_leaf_ = list(concatenate((arange(1,11),arange(20, 101, 10))))


        combinations = list(product(n_estimators_, max_depth_, min_samples_split_, max_features_, min_samples_leaf_)) 
        # randomly permute the combinations
        combinations = shuffle(combinations, random_state=0)


        # split train to train and validation
        ___perc_cv___ = 0.8; nof_folds = 5; obs = len(ytr)
        tr_inds, vl_inds = split_tr_vl(obs,___perc_cv___,nof_folds,PERMUTE_TRAIN_TEST)

        # train rf on each combination and evaluate on validation set
        acc_tr_all = array([]); acc_vl_all = array([]); acc_te_all = array([])
        opti_tr = -1; opti_vl = -1; opti_te= -1  
        best_combination = None
        for i, (n_estimators_, max_depth_, min_samples_split_, max_features_, min_samples_leaf_) in enumerate(combinations):
            # catch exception if the combination is not valid
            try:
                print("Training Random Forests Model",i+1,"/>",__thres_min_tune_rounds__,"/",len(combinations),":",
                                            (n_estimators_, max_depth_, min_samples_split_, max_features_, min_samples_leaf_))
                if LOGISTIC_REGR:
                    rf = RandomForestClassifier(random_state=0, n_estimators = n_estimators_, max_depth = max_depth_, 
                                                min_samples_split = min_samples_split_, max_features = max_features_, 
                                                min_samples_leaf = min_samples_leaf_, n_jobs = __n_jobs__)
                else: 
                    rf = RandomForestRegressor(random_state=0, n_estimators = n_estimators_, max_depth = max_depth_, 
                            min_samples_split = min_samples_split_, max_features = max_features_, 
                            min_samples_leaf = min_samples_leaf_, n_jobs = __n_jobs__)
                acc_tr = 0; acc_vl = 0; acc_te = 0
                for fold in range(nof_folds):
                    rf.fit(Xtr[tr_inds[fold],:], ytr[tr_inds[fold]])
                    pred_tr = rf.predict(Xtr[tr_inds[fold],:])
                    pred_vl = rf.predict(Xtr[vl_inds[fold],:])
                    pred_te = rf.predict(Xte)
                    acc_tr += pearsonr(ytr[tr_inds[fold]], pred_tr).correlation
                    acc_vl += pearsonr(ytr[vl_inds[fold]], pred_vl).correlation
                    acc_te += pearsonr(yte, pred_te).correlation
                acc_tr /= nof_folds; acc_vl /= nof_folds; acc_te /= nof_folds
                acc_tr_all = np_append(acc_tr_all, acc_tr)
                acc_vl_all = np_append(acc_vl_all, acc_vl)
                acc_te_all = np_append(acc_te_all, acc_te)
                # print all the r2 scores
                vl_75 = percentile(acc_vl_all, 75)
                vl_max = max(acc_vl_all)
                slope = (vl_max - vl_75)/(0.25*(i+1))
                print(datetime.datetime.now().strftime("%H:%M:%S"),"Comb:",i,"R2 Score:: Train:",acc_tr,"Validation:",acc_vl,"Test:",acc_te, 
                    "\nMax Val:",vl_max,"Q75 Val:",vl_75,"Slope:",slope)
                imax = argmax(acc_vl_all)
                best_combination = combinations[imax]
                print(best_combination)
                if i>__thres_min_tune_rounds__ and slope <__thres_early_stop__:
                    print("Early Stopping, vl_max:",vl_max,"vl_95:",vl_75)
                    break
            except:
                break

        # find minimum length among acc_tr_all, acc_vl_all, acc_te_all, in case of exception
        max_len = min([len(acc_tr_all), len(acc_vl_all), len(acc_te_all)])
        # make all the arrays of same length
        acc_tr_all = acc_tr_all[:max_len]
        acc_vl_all = acc_vl_all[:max_len]
        acc_te_all = acc_te_all[:max_len]

        imax = argmax(acc_vl_all)
        best_combination = combinations[imax]
        print("Best Combination:",best_combination)

        ttr = time()-t0
        __method__ = "RF"
        path_ = ROOT_DIR+"ML_Models"+os.sep+__method__+os.sep
        path_err = ROOT_DIR+"ML_Models"+os.sep+__method__+os.sep+"Error_Analysis"+os.sep
        path_sens = ROOT_DIR+"ML_Models"+os.sep+__method__+os.sep+"Sensitivity_Analysis"+os.sep

        (n_estimators_, max_depth_, min_samples_split_, max_features_, min_samples_leaf_) = best_combination
        if LOGISTIC_REGR:
            rf = RandomForestClassifier(random_state=0, n_estimators = n_estimators_, max_depth = max_depth_, 
                                        min_samples_split = min_samples_split_, max_features = max_features_, 
                                        min_samples_leaf = min_samples_leaf_, n_jobs = __n_jobs__)
        else: 
            rf = RandomForestRegressor(random_state=0, n_estimators = n_estimators_, max_depth = max_depth_, 
                    min_samples_split = min_samples_split_, max_features = max_features_, 
                    min_samples_leaf = min_samples_leaf_, n_jobs = __n_jobs__)
        rf.fit(Xtr, ytr)
        with open(path_ + "best_estimator_rf.pkl", 'wb') as f:
            pickle.dump(rf, f)

        pred_tr = rf.predict(Xtr)
        t0=time()
        for i in range(10):
            pred_te = rf.predict(Xte)
        tte=(time()-t0)/10
        # get the feature importance from best_xgb and plot it
        feature_importance = rf.feature_importances_
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = argsort(feature_importance)
        pos = arange(sorted_idx.shape[0]) + .5
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.barh(pos, feature_importance[sorted_idx], align='center')
        plt.yticks(pos, features_names[sorted_idx])
        plt.savefig(path_ + "features_importance.png")
        plt.close()

        iso = argsort(acc_vl_all)
        plt.plot(acc_tr_all[iso], 'x', label='Train')
        plt.plot(acc_vl_all[iso], 'x', label='Validation')
        plt.plot(acc_te_all[iso], 'x', label='Test')
        plt.legend()
        plt.xlabel('Model')
        plt.ylabel('R2 Score')
        plt.savefig(path_ + "rf_tune_cv_history.png")
        plt.close()
        
        iso = argsort(acc_vl_all)[int(0.5*len(acc_vl_all)):]
        plt.plot(acc_tr_all[iso], 'x', label='Train')
        plt.plot(acc_vl_all[iso], 'x', label='Validation')
        plt.plot(acc_te_all[iso], 'x', label='Test')
        plt.legend()
        plt.xlabel('Model')
        plt.ylabel('R2 Score')
        plt.savefig(path_ + "rf_tune_cv_history_50perc.png")
        plt.close()

        do_sensitivity(Xtr, features_names, target_name, rf.predict, __method__, path_sens)

        plot_mae_per_bin(ytr, yte, pred_te, target_name, __method__, path_)

        error_analysis(ytr,pred_tr,target_name,__method__,"Train",path_err)
        error_analysis(yte,pred_te,target_name,__method__,"Test",path_err)    

        plot_target_vs_predicted(ytr, pred_tr, target_name, __method__, "Train",path_)  
        plot_target_vs_predicted(yte, pred_te, target_name, __method__, "Test",path_) 
        export_metrics(ytr, pred_tr, yte, pred_te, __method__, ttr, tte, LOGISTIC_REGR, path_)  
        
        print("See results in folder:", path_)
        
        # export_notebook_to_html()
        gather_all_ML_metrics(ROOT_DIR)
        
    except Exception as ex1:
        print(ex1)
 


def predict_rf(Xout, yout, target_name, LOGISTIC_REGR, ROOT_DIR):
    __method__ = "RF"
    path_ = ROOT_DIR+"ML_Models"+os.sep+__method__+os.sep
    path_pred = ROOT_DIR+"Predict"+os.sep+__method__+os.sep
    
    try:
        with open(path_ + "best_estimator_rf.pkl", 'rb') as f:
            best_estimator_rf = pickle.load(f)
    except Exception as e:
        print("Error: ", e)
        return

    pred_out = best_estimator_rf.predict(Xout)

    #  save predictions to file
    with open(path_pred+"Predictions_"+__method__+".csv", "w") as file:
        for yi in pred_out:
            file.write(str(yi) + '\n')

    plot_target_vs_predicted(yout, pred_out, target_name, __method__, "Out", path_pred) 
    export_metrics_out(yout, pred_out, path_pred + __method__ + "_Out", LOGISTIC_REGR)
    error_analysis(yout, pred_out, target_name, __method__, "Out", path_pred)    
    
    print("See results in folder: ", path_pred)