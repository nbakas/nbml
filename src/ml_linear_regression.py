
from import_libraries import *
from misc_functions import *



def plot_pvalues(p_vals, features_names, path_):
    iso = argsort(p_vals)
    scf = int(len(features_names)/10)+1
    plt.figure(figsize=(10*scf, 10))
    plt.rcParams.update({'font.size': 7*scf})
    tik = arange(len(p_vals))/7
    plt.bar(tik, p_vals[iso], width=0.05)
    plt.xticks(tik, features_names, rotation=45, ha='right')
    plt.ylabel("p-values")
    plt.tight_layout()
    plt.savefig(path_ + "p_Values.png")
    plt.close()
    plt.rcParams.update({'font.size': 11})

def plot_normalised_weights(Xtr, ytr, features_names, LOGISTIC_REGR, path_):
    
    Xtr_norm = (Xtr - Xtr.min(axis=0)) / (Xtr.max(axis=0) - Xtr.min(axis=0))
    ytr_norm = (ytr - ytr.min(axis=0)) / (ytr.max(axis=0) - ytr.min(axis=0))
    
    if LOGISTIC_REGR:
        res = regression_bak(Xtr_norm, logit((ytr_norm*0.98)+0.01))
    else:
        res = regression_bak(Xtr_norm, ytr_norm)
        
    aa = res.aa

    iso = argsort(abs(aa))[::-1]
    scf = int(len(features_names)/10)+1
    plt.figure(figsize=(10*scf, 10))
    plt.rcParams.update({'font.size': 7*scf})
    tik = arange(len(aa))/7
    plt.bar(tik, aa[iso], width=0.05)
    plt.xticks(tik, features_names, rotation=45, ha='right')
    plt.title("Normalised Regression Weights")
    plt.xlabel("Features")
    plt.ylabel("Weights")
    plt.tight_layout()
    plt.savefig(path_ + "Normalised_Regression_Weights.png")
    plt.close()
    plt.rcParams.update({'font.size': 11})

def export_equation_in_excel(ww, LOGISTIC_REGR, path_, ROOT_DIR):
    col_excel = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    col_excel0 = list(col_excel)
    for i in range(len(col_excel0)):
        col_excel = col_excel + [col_excel0[i]+c for c in col_excel0]
    col_excel = array(col_excel)

    inds_exclude = []
    if os.path.isfile(ROOT_DIR + "excluded_features.txt"):
        with open(ROOT_DIR + "excluded_features.txt", 'r') as f:
            inds_exclude = [int(line.strip()) for line in f]
        inds_exclude = array(inds_exclude)
        print("Multicolinearity was detected, excluding columns from excel: ", [i for i in col_excel[inds_exclude]])
    col_excel = delete(col_excel, inds_exclude, axis=0)
    
    equation_excel = "="
    if LOGISTIC_REGR:
        equation_excel += "expit("
    for i in range(len(ww)):
        str_ = "{:.15E}".format(ww[i])
        if ww[i] >= 0:
            str_ = "+" + str_
        equation_excel += "{}*{}2".format(str_, col_excel[i])
    if LOGISTIC_REGR:
        equation_excel += ")"

    # write equation in txt file
    with open(path_ + "Equation.txt", "w") as file:
        file.write(equation_excel)
    
    
def do_regression(Xtr, Xte, ytr, yte, features_names, target_name, ROOT_DIR, LOGISTIC_REGR):
    global pred_tr, pred_te, res, do_logistic
    do_logistic = LOGISTIC_REGR
    __method__ = "LinRegr"
    path_ = ROOT_DIR+"ML_Models"+os.sep+__method__+os.sep
    path_err = ROOT_DIR+"ML_Models"+os.sep+__method__+os.sep+"Error_Analysis"+os.sep
    path_sens = ROOT_DIR+"ML_Models"+os.sep+__method__+os.sep+"Sensitivity_Analysis"+os.sep
    try:
        t0=time()
        if LOGISTIC_REGR:
            res = regression_bak(Xtr, logit((ytr*0.98)+0.01))
        else:
            res = regression_bak(Xtr, ytr)
        ttr = time()-t0
        with open(path_ + "LinRegrResults.csv", "w") as file:
            for field in res._fields:
                vector = getattr(res, field)
                file.write(field + ',')
                # write each element of the vector on a separate column
                # if is string
                if type(vector) == str:
                    file.write(vector + ',')
                else:
                    # if is vector
                    if isinstance(vector, ndarray):
                        for element in vector:
                            file.write(str(element) + ',')
                    # if is scalar
                    else:
                        file.write(str(vector) + ',')
                file.write('\n')
                      
        pred_tr = predict_lin_regr(Xtr)
        t0=time()
        for i in range(10):
            pred_te = predict_lin_regr(Xte)
        tte=(time()-t0)/10
        
        plot_pvalues(res.p_vals, features_names, path_)
        
        plot_normalised_weights(Xtr, ytr, features_names, LOGISTIC_REGR, path_)
        
        export_equation_in_excel(res.aa, LOGISTIC_REGR, path_, ROOT_DIR)
        
        do_sensitivity(Xtr, features_names, target_name, predict_lin_regr, __method__, path_sens)

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


def predict_lin_regr(Xtr):
    global res, do_logistic
    if do_logistic:
        return expit(dot(Xtr, res.aa))
    else:
        return dot(Xtr, res.aa)


def predict_linregr(Xout, yout, target_name, LOGISTIC_REGR, ROOT_DIR):
    global res, cwd
    __method__ = "LinRegr"
    path_ = ROOT_DIR+"ML_Models"+os.sep+__method__+os.sep
    path_pred = ROOT_DIR+"Predict"+os.sep+__method__+os.sep
    # try to load the results from the CSV file
    try:
        with open(path_ + "LinRegrResults.csv", 'r') as f:
            reader = csv.reader(f)
            row = next(reader)

        for i in range(1, len(row)-1):
            row[i] = float(row[i])
        aa = array(row[1:-1])

        if LOGISTIC_REGR:
            pred_out = expit(dot(Xout, aa))
        else:
            pred_out = dot(Xout, aa)
    except Exception as e:
        print("Error: ", e)
        return

    # save predictions to file
    with open(path_pred+"Predictions_"+__method__+".csv", "w") as file:
        for yi in pred_out:
            file.write(str(yi) + '\n')

    plot_target_vs_predicted(yout, pred_out, target_name, __method__, "Out", path_pred) 
    export_metrics_out(yout, pred_out, path_pred + __method__ + "_Out", LOGISTIC_REGR)
    error_analysis(yout, pred_out, target_name, __method__, "Out", path_pred)    
    
    print("See results in folder: ", path_pred)
