
# Description: This file contains the code for training and evaluating XGBoost models.


from import_libraries import *
from misc_functions import *


def do_xgboost(Xtr,Xte,ytr,yte,features_names,target_name,__thres_early_stop__,__thres_min_tune_rounds__,PERMUTE_TRAIN_TEST,LOGISTIC_REGR,ROOT_DIR):
    try:
        t0=time()
        max_depth = list(arange(1, 11))
        learning_rate = list(np_round(arange(0.01, 0.51, 0.02), 2))
        n_estimators = list(concatenate((arange(1,11),arange(20, 101, 10),arange(200, 1001, 100))))
        colsample_bytree = list(np_round(arange(0.25, 1.01, 0.05), 2))
        subsample = list(np_round(arange(0.25, 1.01, 0.05), 2))
        combinations = list(product(max_depth, learning_rate, n_estimators, colsample_bytree, subsample)) 
        # randomly permute the combinations
        combinations = shuffle(combinations, random_state=0)

        # split train to train and validation
        ___perc_cv___ = 0.8; nof_folds = 5; obs = len(ytr)
        tr_inds, vl_inds = split_tr_vl(obs,___perc_cv___,nof_folds,PERMUTE_TRAIN_TEST)

        # train xgboost on each combination and evaluate on validation set
        acc_tr_all = array([]); acc_vl_all = array([]); acc_te_all = array([])
        opti_tr = -1; opti_vl = -1; opti_te= -1  
        best_combination = None
        for i, (max_depth, learning_rate, n_estimators, colsample_bytree, subsample) in enumerate(combinations):
            # catch exception if the combination is not valid
            try:
                print("Training XGBoost Model",i+1,"/>",__thres_min_tune_rounds__,"/",len(combinations),":",
                                            (max_depth, learning_rate, n_estimators, colsample_bytree, subsample))
                if LOGISTIC_REGR:
                    xgboost = xgb.XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, 
                                            colsample_bytree=colsample_bytree, subsample=subsample, objective='binary:logistic')
                else: 
                    xgboost = xgb.XGBRegressor(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, 
                        colsample_bytree=colsample_bytree, subsample=subsample, objective='reg:squarederror')
                acc_tr = 0; acc_vl = 0; acc_te = 0
                for fold in range(nof_folds):
                    xgboost.fit(Xtr[tr_inds[fold],:], ytr[tr_inds[fold]])
                    pred_tr = xgboost.predict(Xtr[tr_inds[fold],:])
                    pred_vl = xgboost.predict(Xtr[vl_inds[fold],:])
                    pred_te = xgboost.predict(Xte)
                    acc_tr += pearsonr(ytr[tr_inds[fold]], pred_tr).correlation
                    acc_vl += pearsonr(ytr[vl_inds[fold]], pred_vl).correlation
                    acc_te += pearsonr(yte, pred_te).correlation
                acc_tr /= nof_folds; acc_vl /= nof_folds; acc_te /= nof_folds
                if isnan(acc_tr):
                    acc_tr = 0
                if isnan(acc_vl):
                    acc_vl = 0
                if isnan(acc_te):
                    acc_te = 0
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
        __method__ = "XGBoost"
        path_ = ROOT_DIR+"ML_Models"+os.sep+__method__+os.sep
        path_err = ROOT_DIR+"ML_Models"+os.sep+__method__+os.sep+"Error_Analysis"+os.sep
        path_sens = ROOT_DIR+"ML_Models"+os.sep+__method__+os.sep+"Sensitivity_Analysis"+os.sep
        
        with open(path_ + "Best_Combination.txt", "w") as f:
            f.write(str(best_combination) + "\n")
            f.write("max_depth, learning_rate, n_estimators, colsample_bytree, subsample" + "\n")

        (max_depth, learning_rate, n_estimators, colsample_bytree, subsample) = best_combination
        if LOGISTIC_REGR:
            xgboost = xgb.XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, 
                                    colsample_bytree=colsample_bytree, subsample=subsample, objective='binary:logistic')
        else: 
            xgboost = xgb.XGBRegressor(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, 
                colsample_bytree=colsample_bytree, subsample=subsample, objective='reg:squarederror')
        xgboost.fit(Xtr, ytr)
        with open(path_ + "best_estimator_xgb.pkl", 'wb') as f:
            pickle.dump(xgboost, f)

        pred_tr = xgboost.predict(Xtr)
        t0=time()
        for i in range(10):
            pred_te = xgboost.predict(Xte)
        tte=(time()-t0)/10
        # get the feature importance from best_xgb and plot it
        feature_importance = xgboost.feature_importances_
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
        plt.savefig(path_ + "XGBoost_tune_cv_history.png")
        plt.close()
        
        iso = argsort(acc_vl_all)[int(0.5*len(acc_vl_all)):]
        plt.plot(acc_tr_all[iso], 'x', label='Train')
        plt.plot(acc_vl_all[iso], 'x', label='Validation')
        plt.plot(acc_te_all[iso], 'x', label='Test')
        plt.legend()
        plt.xlabel('Model')
        plt.ylabel('R2 Score')
        plt.savefig(path_ + "XGBoost_tune_cv_history_50perc.png")
        plt.close()

        do_sensitivity(Xtr, features_names, target_name, xgboost.predict, __method__, path_sens)

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

    

def predict_xgboost(Xout, yout, target_name, LOGISTIC_REGR, ROOT_DIR):
    __method__ = "XGBoost"
    path_ = ROOT_DIR+"ML_Models"+os.sep+__method__+os.sep
    path_pred = ROOT_DIR+"Predict"+os.sep+__method__+os.sep
    try:
        with open(path_ + "best_estimator_xgb.pkl", 'rb') as f:
            best_estimator_xgb = pickle.load(f)
    except Exception as e:
        print("Error: ", e)
        return

    pred_out = best_estimator_xgb.predict(Xout)

    # save predictions to file
    with open(path_pred + "Predictions_"+__method__+".csv", "w") as file:
        for yi in pred_out:
            file.write(str(yi) + '\n')

    plot_target_vs_predicted(yout, pred_out, target_name, __method__, "Out", path_pred) 
    export_metrics_out(yout, pred_out, path_pred + __method__ + "_Out", LOGISTIC_REGR)
    error_analysis(yout, pred_out, target_name, __method__, "Out", path_pred)    
      
    
    print("See results in folder: ", path_pred)


def do_QuantileGradientBoostingRegressor(ROOT_DIR, Xtr, Xte, ytr, yte, target_name):
    # open the text file
    __method__ = "XGBoost"
    path_ = ROOT_DIR+"ML_Models"+os.sep+__method__+os.sep
    with open(path_ + 'Best_Combination.txt') as file:
        values = file.read()
    values = values.split("\n")[0]
    values = tuple(map(float, values.strip('()').split(', ')))
    print("Best_Combination:",values)
    # extract the parameters from the values
    max_depth__ = int(values[0])
    learning_rate__ = float(values[1])
    n_estimators__ = int(values[2])
    colsample_bytree__ = float(values[3])
    subsample__ = float(values[4])

    common_params = dict(
        learning_rate=learning_rate__,
        n_estimators=n_estimators__,
        max_depth=max_depth__,
        subsample = subsample__,
        max_features = colsample_bytree__
    )
    
    alpha_low = 0.05
    print('Training Q'+str(round(100*alpha_low,0))+'%')
    gbr = GradientBoostingRegressor(loss="quantile", alpha=alpha_low, **common_params)
    model_low = gbr.fit(Xtr, ytr)
    
    alpha_med = 0.5
    print('Training Q'+str(round(100*alpha_med,0))+'%')
    gbr = GradientBoostingRegressor(loss="quantile", alpha=alpha_med, **common_params)
    model_med = gbr.fit(Xtr, ytr)
    
    alpha_high = 0.95
    print('Training Q'+str(round(100*alpha_high,0))+'%')
    gbr = GradientBoostingRegressor(loss="quantile", alpha=alpha_high, **common_params)
    model_high = gbr.fit(Xtr, ytr)
    
    plt.figure(figsize=(200, 20))
    plt.plot(yte, label=target_name + ' Test Set', marker='x')
    plt.plot(model_low.predict(Xte), label='Q'+str(round(100*alpha_low,0))+'%', marker='v')
    plt.plot(model_med.predict(Xte), label='Q'+str(round(100*alpha_med,0))+'%', marker='x')
    plt.plot(model_high.predict(Xte), label='Q'+str(round(100*alpha_high,0))+'%', marker='^')
    plt.grid()
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
    plt.legend(fontsize=50)
    plt.tight_layout()
    plt.savefig(path_ + "QuantileGradientBoostingRegressor_Test.png")
    plt.close()

    # Save the trained model to a file
    with open(path_ + "model_low.pkl", "wb") as f:
        pickle.dump(model_low, f)
    # Save the trained model to a file
    with open(path_ + "model_med.pkl", "wb") as f:
        pickle.dump(model_med, f)
    # Save the trained model to a file
    with open(path_ + "model_high.pkl", "wb") as f:
        pickle.dump(model_high, f)
        
def predict_quantile_gb(ROOT_DIR, Xout):
    __method__ = "XGBoost"
    path_ = ROOT_DIR+"ML_Models"+os.sep+__method__+os.sep
    path_pred = ROOT_DIR+"Predict"+os.sep+__method__+os.sep
    # Load the trained model_low from the file
    with open(path_ + "model_low.pkl", "rb") as f:
        model_low = pickle.load(f)
    # Use the loaded model to make predictions on new data
    y_pred_low = model_low.predict(Xout)
    
    # Load the trained model_med from the file
    with open(path_ + "model_med.pkl", "rb") as f:
        model_med = pickle.load(f)
    # Use the loaded model to make predictions on new data
    y_pred_med = model_med.predict(Xout)
    
    # Load the trained model_high from the file
    with open(path_ + "model_high.pkl", "rb") as f:
        model_high = pickle.load(f)
    # Use the loaded model to make predictions on new data
    y_pred_high = model_high.predict(Xout)
    
    # Create a DataFrame with the predictions
    df = pd.DataFrame({ 'y_pred_low': y_pred_low,
                        'y_pred_med': y_pred_med,
                        'y_pred_high': y_pred_high})
    # Save the DataFrame to an Excel file
    df.to_excel(path_pred + 'predict_quantile_gb.xlsx', index=False)