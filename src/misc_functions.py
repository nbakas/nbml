import os
from import_libraries import *

def create_ml_dir(ROOT_DIR,__method__):
    if not os.path.exists(ROOT_DIR+"ML_Models"+os.sep+__method__):
        os.makedirs(ROOT_DIR+"ML_Models"+os.sep+__method__)
    if not os.path.exists(ROOT_DIR+"ML_Models"+os.sep+__method__+os.sep+"Error_Analysis"):
        os.makedirs(ROOT_DIR+"ML_Models"+os.sep+__method__+os.sep+"Error_Analysis")
    if not os.path.exists(ROOT_DIR+"ML_Models"+os.sep+__method__+os.sep+"Sensitivity_Analysis"):
        os.makedirs(ROOT_DIR+"ML_Models"+os.sep+__method__+os.sep+"Sensitivity_Analysis")

def create_all_directories(ROOT_DIR, PERMUTE_TRAIN_TEST):
    if not os.path.exists(ROOT_DIR+"Descriptive_Statistics"):
        os.makedirs(ROOT_DIR+"Descriptive_Statistics")
    if not os.path.exists(ROOT_DIR+"Descriptive_Statistics"+os.path.sep+"PDF_CDF"):
        os.makedirs(ROOT_DIR+"Descriptive_Statistics"+os.path.sep+"PDF_CDF")
    if not os.path.exists(ROOT_DIR+"Descriptive_Statistics"+os.path.sep+"Tree"):
        os.makedirs(ROOT_DIR+"Descriptive_Statistics"+os.path.sep+"Tree")
    if not os.path.exists(ROOT_DIR+"Descriptive_Statistics"+os.path.sep+"TimeSeries"):
        os.makedirs(ROOT_DIR+"Descriptive_Statistics"+os.path.sep+"TimeSeries")
    if not PERMUTE_TRAIN_TEST:
        if not os.path.exists(ROOT_DIR + "Descriptive_Statistics" + os.sep + "TimeSeries"):
            os.makedirs(ROOT_DIR + "Descriptive_Statistics" + os.sep + "TimeSeries")
        
    __methods__ = ["LinRegr", "NLRegr", "XGBoost", "RF", "ANNBN", "DANN"]
    for __method__ in __methods__:
        create_ml_dir(ROOT_DIR,__method__)
        
    if not os.path.exists(ROOT_DIR + "ML_Models" + os.sep + "ALL_Sensitivity"):
        os.makedirs(ROOT_DIR + "ML_Models" + os.sep + "ALL_Sensitivity")
        
    if not os.path.exists(ROOT_DIR+"Predict"):
        os.makedirs(ROOT_DIR+"Predict")
    for __method__ in __methods__:   
        if not os.path.exists(ROOT_DIR+"Predict"+os.sep+__method__):
            os.makedirs(ROOT_DIR+"Predict"+os.sep+__method__)


def regression_bak(X, Y):
    Result = namedtuple("Result", ["aa", "p_vals", "rr", "mean_resid", "t_statistic", "standard_errors", "residuals", "flag"])
    nn = len(Y)
    varss = X.shape[1]
    Xt = X.transpose()
    XtX = dot(Xt, X)
    try:
        invXtX = inv(XtX)
        aa = dot(dot(invXtX, Xt), Y)
        mean_y = mean(Y)
        sstot = sum((Y - mean_y) ** 2)
        predicted = dot(X, aa)
        residuals = Y - predicted
        ssreg = sum((predicted - mean_y) ** 2)
        ssres = sum(residuals ** 2)
        rr = 1 - ssres / sstot
        if rr < 1e-10000:
            return Result(aa, 0, rr, 0, 0, 0, 0, "rr")
        mean_resid = mean(residuals)
        ss_resid = sum((residuals - mean_resid) ** 2) / (nn - varss)
        var_covar = dot(invXtX, ss_resid)
        if min(diag(var_covar)) < 0:
            print("minimum(diag(var_covar))<0", min(diag(var_covar)))
            return Result(aa, 0, rr, 0, 0, 0, 0, "var_covar")
        standard_errors = sqrt(diag(var_covar))
        t_statistic = aa / standard_errors
        p_vals = 1 - t.cdf(abs(t_statistic), nn) + t.cdf(-abs(t_statistic), nn)
        return Result(aa, p_vals, rr, mean_resid, t_statistic, standard_errors, residuals, "OK")
    except Exception as ex1:
        print(ex1)
        return Result(0, 0, 0, 0, 0, 0, 0, ex1)
    

def read_the_dataset_dropna(ROOT_DIR):
    s_all = []
    for file in os.listdir(ROOT_DIR):
        if file.endswith(".xlsx"):
            s_all.append(ROOT_DIR + file)
            
    if len(s_all)>1:
        print("More than 1 files exists in the directory. Execution Stopped.")
        return
    else:
        my_file = s_all[0]
        print(my_file+" is being used for training and testing.")
    df = pd.read_excel(my_file)
    
    cols = df.columns
    
    # Replace empty string values with NaN
    df.replace(' ', nan, inplace=True)
    df = df.dropna(how='any')
    
    return df, cols[:-1], cols[-1]
    
def make_features_and_target_categorical_if_so(df, target_name):
    
    # Select categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns

    if len(cat_cols)>0:
        # One-hot encode categorical columns
        # It creates 2 columns, one er category. Multicoliearity will remove the second, later.
        onehot_encoded = pd.get_dummies(df[cat_cols], prefix=cat_cols)

        # If Target is categorical select only one of 2 categories.
        if target_name == cat_cols[-1]:
            onehot_encoded = onehot_encoded.iloc[:,:-1]

        # Combine one-hot encoded features with original numerical columns
        df = pd.concat([df.select_dtypes(include=['number']), onehot_encoded], axis=1)

        print(df.head())
    
    cols = df.columns 
    
    return df, cols[:-1], cols[-1] 
    

def split_train_test(PERMUTE_TRAIN_TEST, test_ratio, random_seed, df):

    XY = df.to_numpy()
    XY = XY.astype(float64)
    obs = XY.shape[0]

    if PERMUTE_TRAIN_TEST:
        rng = random.Random()
        rng.seed(random_seed)
        shuffled_indices = rng.sample(list(range(obs)),obs)
    else:
        shuffled_indices = list(range(obs))

    test_set_size = int(obs * test_ratio)
    train_set_size = obs - test_set_size
    train_indices = shuffled_indices[:train_set_size]
    test_indices = shuffled_indices[train_set_size:]
    
    # export_notebook_to_html()
    print("Training set size: ", train_set_size, "Test set size: ", test_set_size, "Random Perumte =", PERMUTE_TRAIN_TEST, ", Random Seed =", random_seed)  

    return XY[train_indices,:-1], XY[test_indices,:-1], XY[train_indices,-1], XY[test_indices,-1]


def plot_target_vs_predicted(ytr, pred, target_name, __method__, tag, path_):
    plt.scatter(ytr, pred)
    plt.xlabel(target_name)
    plt.ylabel(__method__.replace("_", " ") + " Prediction " + tag)
    plt.title("Target vs Predicted - "+tag+" Set")
    rr = linspace(min(ytr), max(ytr), 100)
    plt.plot(rr, rr, 'k', label="Ideal Correlation")
    plt.plot(rr, rr*1.1, 'k--', label="Ideal +10%")
    plt.plot(rr, rr*0.9, 'k:', label="Ideal -10%")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(path_ + __method__ + "_Target_vs_Predicted_"+tag+".png")
    plt.close()

def error_metrics(ACT, PRED, LOGISTIC_REGR):
    _cor_ = corrcoef(ACT, PRED)[0,1]
    _mape_ = mean(abs(ACT-PRED)/abs(ACT))
    _mamte_ = mean(abs(ACT-PRED))/mean(abs(ACT))
    _mae_ = mean(abs(ACT-PRED))
    _rmse_ = sqrt(mean((ACT-PRED)**2))
    _beta_, _alpha_ = lstsq(column_stack((ones(len(ACT)), ACT)), PRED, rcond=None)[0]
    
    _accuracy_ = mean(ACT==round(PRED,0))
    # false positive rate
    _FP_ = mean((ACT==0) & (round(PRED,0)==1))
    _TP_ = mean((ACT==1) & (round(PRED,0)==1))   
    _FN_ = mean((ACT==1) & (round(PRED,0)==0))
    _TN_ = mean((ACT==0) & (round(PRED,0)==0))
    
    if LOGISTIC_REGR:
        # plot the confusion matrix
        cm = confusion_matrix(ACT, round(PRED,0))
        plt.imshow(cm, cmap=plt.cm.Blues)
        plt.colorbar()
        # add numbers to the confusion matrix
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(x=j, y=i, s=cm[i, j], va='center', ha='center')
        # xticks and yticks only 0-1
        plt.xticks([0,1], [0,1])
        plt.yticks([0,1], [0,1])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig("Confusion_Matrix.png")
        plt.close()

    return [_cor_, _mape_, _mamte_, _mae_, _rmse_, _alpha_, _beta_, _accuracy_, _FP_, _TP_, _FN_, _TN_]

def export_metrics(ytr, pred_tr, yte, pred_te, __method__, ttr, tte, LOGISTIC_REGR, path_):
    errTr = error_metrics(ytr, pred_tr, LOGISTIC_REGR)
    errTe = error_metrics(yte, pred_te, LOGISTIC_REGR) 
    errTr.append(ttr)
    errTe.append(tte)
    df = pd.DataFrame([errTr, errTe])
    df.columns = ["R", "MAPE", "MAMPE", "MAE", "RMSE", "alpha", "beta", "Acc", "FP", "TP", "FN", "TN", "Sec"]
    df.index = ["Train", "Test"]
    pd.options.display.float_format = '{:,.2f}'.format
    df.to_excel(path_+__method__+"_error_metrics.xlsx", index=True)
    df[abs(df) > 1e10] = Inf
    print(df[["R", "MAPE", "MAMPE", "MAE", "RMSE", "Acc", "Sec"]])

def export_metrics_out(yout, pred_out, __method__, LOGISTIC_REGR):
    errOut = error_metrics(yout, pred_out, LOGISTIC_REGR)
    df = pd.DataFrame([errOut])
    df.columns = ["R", "MAPE", "MAMPE", "MAE", "RMSE", "alpha", "beta", "Acc", "FP", "TP", "FN", "TN"]
    df.index = ["Out of Sample Metrics"]
    pd.options.display.float_format = '{:,.3f}'.format
    print(df)
    df.to_excel(__method__+"_error_metrics.xlsx", index=True)

def create_change_ml_dir(__method__):
    cwd = os.getcwd()
    if not os.path.exists("ML_Models"):
        os.makedirs("ML_Models")
    os.chdir("ML_Models")
    if not os.path.exists(__method__):
        os.makedirs(__method__)
    os.chdir(__method__)
    if not os.path.exists("Error_Analysis"):
        os.makedirs("Error_Analysis")
    if not os.path.exists("Sensitivity_Analysis"):
        os.makedirs("Sensitivity_Analysis")
    return cwd

def __pdf__(_x_, bins):
    rr = linspace(min(_x_)-1e-10, max(_x_)+1e-10, bins)
    ret = [len(where((rr[i-1] <= _x_) & (_x_ < rr[i]))[0]) for i in range(1, bins)]
    rr = (rr[1:] + rr[:-1])/2
    return divide(ret, sum(ret)), rr

def cdf_pdf_plot(varVals,var_,path_):
    bin = 20
    #bin = int(len(var1)/10)
    _pdf_,_range_ = __pdf__(varVals,bin)
    fig, ax = plt.subplots(ncols=1)
    ax.set_xlabel(var_)
    ax.set_ylabel("Frequency")
    ax.set_title("bins="+str(bin)+", samples="+str(len(varVals)))
    ax1 = ax.twinx()
    ax.bar(_range_,_pdf_,width=(_range_[1]-_range_[0])*0.9)
    
    # Add the number of occurrences on the vertical bars
    for i, v in enumerate(_pdf_):
        ax.text(_range_[i], v+0.00, str(int(_pdf_[i]*len(varVals))), ha='center', va='bottom')
        
    cdf = _pdf_.cumsum()
    cdf /= max(cdf)
    ax1.plot(_range_,cdf)
    ax1.set_ylabel("Cumulative Probability")
    ax1.grid(which='both')
    plt.savefig(path_+var_+"_PDF_CDF.png")
    plt.close()

def error_analysis(yy,pred,target_name,__method__,tag,path_err):
    err = yy - pred
    plt.scatter(yy, err)
    plt.xlabel(target_name)
    plt.ylabel(__method__.replace("_", " ") + " Residual Errors "+tag)
    cor = pearsonr(err, yy)[0]
    plt.title("Pearson (Errors-"+target_name+") = " + "{:.5f}".format(cor) + ", Average Error = " + "{:.5f}".format(mean(err)))
    rr = linspace(min(yy), max(yy), len(yy))
    plt.plot(rr, percentile(err, 5)*ones(len(rr)), color='black', label='5% Percentile', linestyle='--')
    plt.plot(rr, percentile(err, 50)*ones(len(rr)), color='black', label='Median', linestyle='-')
    plt.plot(rr, percentile(err, 95)*ones(len(rr)), color='black', label='95% Percentile', linestyle=':')
    plt.legend(loc='best')
    plt.savefig(path_err+__method__ + "_Errors_"+tag+".png")
    plt.close()

    cdf_pdf_plot(err,__method__ + "_Errors_"+tag,path_err)


def check_fix_multicolinearity(Xtr, Xte, features_names, ROOT_DIR):
    try:
        k = matrix_rank(Xtr)
        print("Rank of Xtr: ", k, " out of ", Xtr.shape[1], " features")
        if k<Xtr.shape[1]:
            q, r, p = linalg.qr(Xtr, pivoting=True)
            idx = p[:k]
            inds_exclude = delete(arange(Xtr.shape[1]), idx)
            istd = where(std(Xtr,axis=0)==0)[0]
            istd = np_append(istd, where(isnan(std(Xtr,axis=0)))[0])
            if len(istd)>0:
                print(len(istd),"features with zeros std:",features_names[istd])
            inds_exclude = np_append(inds_exclude, istd)
            inds_exclude = sort(unique(inds_exclude))
            print("Multicolinearity detected, excluding features: ", [i for i in features_names[inds_exclude]])
            Xtr = delete(Xtr, inds_exclude, axis=1)
            Xte = delete(Xte, inds_exclude, axis=1)
            features_names = delete(features_names, inds_exclude)
            with open(ROOT_DIR + "excluded_features.txt", "w") as f:
                for i in inds_exclude:
                    f.write(str(i) + "\n")
            print("Xtr.shape=",Xtr.shape)
        else:
            print("No multicolinearity detected.")
    except Exception as ex1:
        print(ex1)
    return Xtr, Xte, features_names
    

def delete_missing_and_inf_values_rows(XX, Y):
    try:
        missing_values = array([]).astype(int)
        for i in range(XX.shape[0]):
            for j in range(XX.shape[1]):
                if isnan(XX[i,j]) or isinf(XX[i,j]):
                    missing_values = np_append(missing_values, i)
            if isnan(Y[i]) or isinf(Y[i]):
                missing_values = np_append(missing_values, i)
        missing_values = unique(missing_values) # remove duplicates
        print("Number of missing values: ", len(missing_values),"::", missing_values)
        # remove rows with missing values
        XX = delete(XX, missing_values, axis=0)
        Y = delete(Y, missing_values, axis=0)
        # export_notebook_to_html()
    except Exception as ex1:
        print(ex1)

    return XX, Y

def compute_distances(XY):
    D=XY@XY.T
    g=diag(D)
    return -2*D+g+g.T


def delete_identical_rows(XX, Y):
    cwd = os.getcwd()
    try:
        XY = c_[XX, Y]
        i_del = array([])
        obs = XY.shape[0]
        
        D = compute_distances(XY)
        for i in range(obs):
            for j in range(i):
                if D[i,j] == 0:
                    i_del = np_append(i_del, i)
        D = 0
        print(len(i_del),"Identical rows found (XX, Y).")
        
        iok = []
        for i in range(obs):
            if i not in i_del:
                iok.append(i)
        print("NOF samples in new dataset =",len(iok),"in initial dataset = ",obs)
                
        XX = XX[iok,:]
        Y = Y[iok]
    
    except Exception as ex1:
        print(ex1)
        os.chdir(cwd)
    return XX, Y
    
def read_out_of_sample_data(ROOT_DIR):
    s_all = []
    for file in os.listdir(ROOT_DIR+"Predict"):
        if file.endswith(".xlsx"):
            s_all.append(ROOT_DIR + "Predict" + os.sep + file)
            
    if len(s_all)>1:
        print("More than 1 files exists in the directory")
    elif len(s_all)==0:
        print("No files exist in the directory")
        return None
    else:
        my_file = s_all[0]
        print("######################################################################")
        print(my_file)
        print("######################################################################")
    df = pd.read_excel(my_file)

    cols = df.columns
    df = df.to_numpy()

    inds_exclude = []
    if os.path.isfile(ROOT_DIR + "excluded_features.txt"):
        with open(ROOT_DIR + "excluded_features.txt", 'r') as f:
            inds_exclude = [int(line.strip()) for line in f]
        print("Multicolinearity was detected, excluding features: ", [i for i in cols[inds_exclude]])

    # export_notebook_to_html()
    
    return  delete(df[:,:-1], inds_exclude, axis=1), df[:,-1], delete(cols[:-1], inds_exclude), cols[-1]



def split_tr_vl(obs,___perc_cv___,nof_folds,PERMUTE_TRAIN_TEST):
    obs_fold = int(math.ceil(___perc_cv___*obs))
    obs_valid = obs - obs_fold

    i = 1
    if PERMUTE_TRAIN_TEST:
        obs_scanned = sorted(shuffle(range(obs), random_state=i)[:obs_valid])
    else:
        obs_scanned = sorted(range(obs)[:obs_valid])
    vl_inds = [obs_scanned]
    __restart__ = False
    while i < nof_folds:
        i += 1
        obs_to_add = []
        if obs - len(obs_scanned) <= obs_valid + obs % obs_valid:
            __restart__ = True
            obs_to_add = list(set(range(obs)) - set(obs_scanned))
        else:
            __restart__ = False
            ii = list(set(range(obs)) - set(obs_scanned))
            if PERMUTE_TRAIN_TEST:
                obs_to_add = sorted(shuffle(ii, random_state=i)[:obs_valid])
            else:
                obs_to_add = sorted(ii[:obs_valid])
            obs_scanned = sorted(obs_scanned + obs_to_add)
        vl_inds.append(obs_to_add)
        if __restart__ and i < nof_folds:
            obs_scanned = sorted(shuffle(range(obs), random_state=i)[:obs_valid])
            vl_inds.append(obs_scanned)
            i += 1
    print("Lengths Validation: ", [len(ir) for ir in vl_inds])

    intersect_all = zeros((len(vl_inds), len(vl_inds)))
    for i in range(len(vl_inds)):
        for j in range(i+1, len(vl_inds)):
            intersect_all[i,j] = len(set(vl_inds[i]).intersection(vl_inds[j]))
    print("intersect_all Validation:\n", intersect_all)


    tr_inds = []
    for i in range(nof_folds):
        i_tr_all = list(set(range(obs)) - set(vl_inds[i]))
        tr_inds.append(i_tr_all)
    print("Lengths Train: ", [len(ir) for ir in tr_inds])

    lvalid = []
    ltrain = []
    for i in range(nof_folds):
        lvalid += vl_inds[i]
        ltrain += tr_inds[i]

    print("Total Validation Samples=", len(lvalid), " Total Train Samples=", len(ltrain), " out of:", obs)
    print("Unique Validation Samples=", len(set(lvalid)), " Unique Train Samples=", len(set(ltrain)), " out of:", obs)

    return tr_inds, vl_inds



def export_notebook_to_html():
    notebook_name = '__nbml__.py'
    output_file_name = 'output.html'

    exporter = HTMLExporter()
    output_notebook = nbformat.read(
                        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    notebook_name),
                                    as_version=4)

    output, resources = exporter.from_notebook_node(output_notebook)
    codecs.open(output_file_name, 'w', encoding='utf-8').write(output)


def do_sensitivity(Xtr, features_names, target_name, PREDICTOR, __method__, path_sens):
    vars = Xtr.shape[1]
    _med_ = median(Xtr, axis=0).reshape(1, vars)
    _75q_ = quantile(Xtr, 0.75, axis=0).reshape(1, vars)
    _25q_ = quantile(Xtr, 0.25, axis=0).reshape(1, vars)
    sens_med_all = zeros((Xtr.shape[0], vars))
    sens_75q_all = zeros((Xtr.shape[0], vars))
    sens_25q_all = zeros((Xtr.shape[0], vars))

    for i in range(vars):
        xx_tr_sens = repeat(_med_, Xtr.shape[0], axis=0)
        xx_tr_sens[:, i] = Xtr[:, i]
        sens_med_all[:, i] = PREDICTOR(xx_tr_sens)

        xx_tr_sens = repeat(_75q_, Xtr.shape[0], axis=0)
        xx_tr_sens[:, i] = Xtr[:, i]
        sens_75q_all[:, i] = PREDICTOR(xx_tr_sens)
        
        xx_tr_sens = repeat(_25q_, Xtr.shape[0], axis=0)
        xx_tr_sens[:, i] = Xtr[:, i]
        sens_25q_all[:, i] = PREDICTOR(xx_tr_sens)
        print(f"Preparing Sensitivity Curves for feature: {i+1} of {vars}.")

    df = pd.DataFrame(sens_med_all, columns=features_names)
    df.to_excel(path_sens+"sens_med_all_"+__method__+".xlsx", index=False)

    df = pd.DataFrame(sens_75q_all, columns=features_names)
    df.to_excel(path_sens+"sens_75q_all_"+__method__+".xlsx", index=False)

    df = pd.DataFrame(sens_25q_all, columns=features_names)
    df.to_excel(path_sens+"sens_25q_all_"+__method__+".xlsx", index=False)

    # scatter plot the feature values vs the sensitivity curves
    for i in range(vars):
        plt.plot(Xtr[:, i], sens_med_all[:, i], 'x', label='Median')
        plt.plot(Xtr[:, i], sens_75q_all[:, i], 'x', label='75%')
        plt.plot(Xtr[:, i], sens_25q_all[:, i], 'x', label='25%')
        plt.xlabel(features_names[i])
        plt.ylabel(target_name)
        plt.legend()
        plt.savefig(path_sens+f"sensitivity_curve_{features_names[i]}.png")
        plt.close()
        print(f"Plotting Sensitivity Curves for feature: {i+1} of {vars}.")


def gather_all_sensitivity_curves(Xtr, features_names, target_name, ROOT_DIR):
    try:
        sens_all = zeros(len(features_names))
        for i in range(len(features_names)):
            plt.figure(figsize=(10, 5))
            for root, dirs, files in os.walk(os.path.join(ROOT_DIR, "ML_Models")):
                for dir in dirs:
                    for root1, dirs1, files1 in os.walk(os.path.join(ROOT_DIR, "ML_Models", dir)):
                        for dir1 in dirs1:
                            if dir1.startswith("Sensitivity_Analysis"):
                                for root2, dirs2, files2 in os.walk(os.path.join(ROOT_DIR, "ML_Models", dir, dir1)):
                                    for file in files2:
                                        if file.startswith("sens_med_all"):
                                            print(f"Reading Sensitivity Curves for: {file}")
                                            df = pd.read_excel(os.path.join(ROOT_DIR, "ML_Models", dir, dir1, file))
                                            iso = argsort(Xtr[:, i])
                                            plt.plot(Xtr[:, i][iso], df[features_names[i]][iso], marker='x', label=file[13:])
                                            plt.xlabel(features_names[i])
                                            if file.find("XGBoost") != -1:
                                                sens_all[i] = df[features_names[i]].max()-df[features_names[i]].min()
            plt.ylabel(target_name)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.tight_layout()
            plt.savefig(ROOT_DIR + "ML_Models" + os.sep + "ALL_Sensitivity" + os.sep + f"sensitivity_curve_{features_names[i]}.png")
            plt.close()
            print(f"Plotting Sensitivity Curves for feature: {i+1} of {len(features_names)}.")
        
        iso = argsort(sens_all)
        plt.barh(range(len(features_names)), sens_all[iso], align='center')
        plt.yticks(range(len(features_names)), array(features_names)[iso])
        for i, v in enumerate(sens_all[iso]):
            plt.text(v + 0.01, i + 0.25, str(round(v, 3)), color='blue', fontweight='bold')
        plt.xlabel("Sensitivity")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.savefig(ROOT_DIR + "ML_Models" + os.sep + "ALL_Sensitivity" + os.sep + "All_Sensitivity_XGBoost.png")
        plt.close()
    except Exception as ex1:
        print(ex1)

# fit an function to the data
def fit_func(x, a, b):
    # return a * log(x) + b
    return a * x + b

#  a function to count the number of samples of train target per bin
# and the mae of the test set per bin and plot results
def plot_mae_per_bin(ytr, yte, yte_pred, target_name, __method__, path_):
    # count the number of samples of train target per bin
    __nof_nins__ = 20
    min_y = min(min(ytr), min(yte))-1e-10
    max_y = max(max(ytr), max(yte))+1e-10
    bins = linspace(min_y, max_y, __nof_nins__)
    ytr_bins = digitize(ytr, bins)
    ytr_bins_count = zeros(len(bins))
    for i in range(1, len(bins) + 1):
        ytr_bins_count[i-1] = sum(ytr_bins == i)

    # calculate the mae of the test set per bin
    yte_bins = digitize(yte, bins)
    yte_mae = zeros(len(bins))
    for i in range(1, len(bins) + 1):
        if sum(yte_bins == i) > 0:
            yte_mae[i-1] = mean(abs(yte[yte_bins == i] - yte_pred[yte_bins == i]))

    inz = ytr_bins_count > 0
    ytr_bins_count = ytr_bins_count[inz]
    yte_mae = yte_mae[inz]
    iso = argsort(ytr_bins_count)
    ytr_bins_count = ytr_bins_count[iso]
    yte_mae = yte_mae[iso]
    # scatter plot the number of samples of train target per bin vs the mae of the test set per bin
    #  plot only where the number of samples of train target per bin is greater than 0
    plt.plot(ytr_bins_count, yte_mae, 'x', color='blue')
    plt.xlabel("Number of samples of train target per bin (NSTB)")
    plt.ylabel("MAE of the test set per bin (MAET)")

    popt, pcov = curve_fit(fit_func, ytr_bins_count, yte_mae)
    pred_fit = fit_func(ytr_bins_count, *popt)
    corr = corrcoef(yte_mae, pred_fit)[0,1]
  
    xx = (ytr_bins_count[:-1] + ytr_bins_count[1:]) / 2
    xx = np_append(xx, xx[-1] + 2*(ytr_bins_count[-1]-xx[-1]))
    xx = np_append(xx[0] - 2*(xx[0]-ytr_bins_count[0]), xx)
    if xx[0]<=0:
        xx[0]=1
    pred_fit = fit_func(xx, *popt)
    # plt.plot(xx, pred_fit, 'o-', label="Curve Fit: MAET=%.3f * log(NSTB) + %.3f \nPearson Correlation (Fit-MAET): %.3f" % (popt[0], popt[1], corr))
    plt.plot(xx, pred_fit, 'o-', label="Curve Fit: MAET=%.3f * NSTB + %.3f \nPearson Correlation (Fit-MAET): %.3f" % (popt[0], popt[1], corr))
    # add the number of samples of train target per bin to the plot
    for i, v in enumerate(ytr_bins_count):
        plt.text(v + 0.01, yte_mae[i] + 0.01, str(int(v))+(";%.3f"%yte_mae[i]), color='blue')
    plt.legend()
    plt.savefig(path_ + f"MAE_per_bin_{__method__}.png")
    plt.close()

    

def gather_all_ML_metrics(ROOT_DIR):    
    for root, dirs, files in os.walk(ROOT_DIR + "ML_Models"):
        for dir in dirs:
            for root1, dirs1, files1 in os.walk(ROOT_DIR + "ML_Models" + os.sep + dir):
                for file in files1:
                    # find all xlsx files containing the word "error_metrics", read them and concantenate them
                    if file.endswith(".xlsx") and "error_metrics" in file:
                        df = pd.read_excel(ROOT_DIR + "ML_Models" + os.sep + dir + os.sep + file)
                        # rename Unnamed: 0 column to "Dataset"
                        df.rename(columns={"Unnamed: 0": "Dataset"}, inplace=True)
                        # add a first column with the name of the model
                        df.insert(0, "Model", file.split("_")[0])                        
                        if "df_all" in locals():
                            df_all = pd.concat([df_all, df], ignore_index=True)
                        else:
                            df_all = df
                            
    # Select even and odd rows using iloc
    odd_df = df_all.iloc[::2]  # select tr rows
    even_df = df_all.iloc[1::2]  # select te rows
    # Concatenate the two DataFrames in the desired order
    df_all = pd.concat([odd_df, even_df])
    # save the concantinated dataframe to a xlsx file
    df_all.to_excel(ROOT_DIR + "ML_Models" + os.sep + "All_ML_metrics.xlsx", index=False)
    
    # bar plot of the MAE of the test set for all models
    df_all = df_all[df_all["Dataset"] == "Test"]
    df_all = df_all[["Model","MAE"]]  
    # sort the dataframe by MAE
    df_all = df_all.sort_values(by="MAE")
    # plot the bar plot
    plt.bar(df_all["Model"], df_all["MAE"]) 
    plt.xticks(rotation=45)
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.savefig(ROOT_DIR + "ML_Models" + os.sep + "MAE_test_set.png")
    plt.close()
    
    
def make_lags(X, Y, features_names, target_name, lags):
    XX = Y[:-lags].copy()
    XX_names = [target_name+"_LAG_"+str(lags)]
    for i in range(1,lags):
        XX = c_[XX,Y[i:-lags+i]]
        XX_names.append(target_name+"_LAG_"+str(lags-i))
    Y = Y[lags:]
    for j in range(X.shape[1]):
        XX = c_[XX, X[:,j][:-lags]]
        XX_names.append(features_names[j]+"_LAG_"+str(lags))
        for i in range(1,lags):
            XX = c_[XX, X[:,j][i:-lags+i]]
            XX_names.append(features_names[j]+"_LAG_"+str(lags-i))
        XX = c_[XX, X[:,j][lags:]]
        XX_names.append(features_names[j])
    return XX, Y, array(XX_names)