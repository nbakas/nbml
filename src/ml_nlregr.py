
from import_libraries import *
from misc_functions import *


def export_equation_txt(opti_aa, feat_names, poly_degree, target_name, combins_list, best_ira, path_):
    __formula__ = ""
    for i in range(len(opti_aa)):
        str_val = "{:.5E}".format(opti_aa[i])
        if opti_aa[i] > 0:
            str_val = "+" + str_val
        __formula__ = __formula__ + str_val
        for j in range(poly_degree):
            __formula__ = __formula__ + "*" + feat_names[combins_list[best_ira[i], j]]
    __formula__ = target_name + "=" + __formula__
    print(__formula__)
    with open(path_ + "__formula__.txt", "w") as f:
        f.write(__formula__)


def export_equation_in_excel(opti_aa, LOGISTIC_REGR, poly_degree, best_ira, vars, path_, ROOT_DIR):
    equation_excel = ""
    col_excel = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    col_excel += [f"A{col}" for col in col_excel]
    col_excel = array(col_excel)
    
    inds_exclude = []
    if os.path.isfile(ROOT_DIR + "excluded_features.txt"):
        with open(ROOT_DIR + "excluded_features.txt", 'r') as f:
            inds_exclude = [int(line.strip()) for line in f]
        inds_exclude = array(inds_exclude)
        print("Multicolinearity was detected, excluding columns from excel: ", [i for i in col_excel[inds_exclude]])
    col_excel = delete(col_excel, inds_exclude, axis=0)

    for i in range(len(opti_aa)):
        str_val = "{:.15E}".format(opti_aa[i])
        if opti_aa[i] >= 0:
            str_val = "+" + str_val
        equation_excel = equation_excel + str_val
        for j in range(poly_degree):
            ij = combins_list[best_ira[i], j]
            if ij == 0:
                str_ij = "1"
            else:
                str_ij = col_excel[ij - 1] + "2"
            equation_excel = equation_excel + "*" + str_ij

    equation_excel = "=" + equation_excel
    print(equation_excel)
    with open(path_ + "__equation_excel__.txt", "w") as f:
        f.write(equation_excel)

    
def nl_features_mat(XX, INDS, NOFNL):
    XXX = XX[:, INDS[:, 0]].copy()
    for i in range(1, NOFNL):
        XXX *= XX[:, INDS[:, i]]
    return XXX


def do_nlregr(Xtr, Xte, ytr, yte, features_names, target_name, LOGISTIC_REGR, PERMUTE_TRAIN_TEST, ROOT_DIR):
    global pred_tr, pred_te, combins_list, best_ira, poly_degree, aa, do_logistic
    
    do_logistic = LOGISTIC_REGR

    __method__ = "NLRegr"
    path_ = ROOT_DIR+"ML_Models"+os.sep+__method__+os.sep
    path_err = ROOT_DIR+"ML_Models"+os.sep+__method__+os.sep+"Error_Analysis"+os.sep
    path_sens = ROOT_DIR+"ML_Models"+os.sep+__method__+os.sep+"Sensitivity_Analysis"+os.sep
    
    try:
        t0=time()
        
        # split train to train and validation
        ___perc_cv___ = 0.8; nof_folds = 5; obs = len(ytr)
        tr_inds, vl_inds = split_tr_vl(obs,___perc_cv___,nof_folds,PERMUTE_TRAIN_TEST)
        
        Xtr = c_[ones(Xtr.shape[0]), Xtr]
        Xte = c_[ones(Xte.shape[0]), Xte]
        feat_names = features_names.insert(0, '1')  
        
        max_n_rounds = 1_000
        poly_degree = 3
        vars = Xtr.shape[1]
        combins_list = array(list(combinations_with_replacement(range(vars), poly_degree)))
        n_comb = combins_list.shape[0]
        print("Total Combinations:",n_comb,", for",vars,"variables and polynomial degree",poly_degree,".")
        
        ira = array(arange(vars))
        err_vl_all1 = []
        err_tr_all1 = []
        pred_te = zeros(len(yte))
        MNL_te = nl_features_mat(Xte, combins_list[ira,:], poly_degree)
        for j in range(nof_folds):
            if LOGISTIC_REGR:
                aa = lstsq(nl_features_mat(Xtr[tr_inds[j], :], combins_list[ira, :], poly_degree), 
                        logit((ytr[tr_inds[j]]*0.98)+0.01), rcond=None)[0]
            else:
                aa = lstsq(nl_features_mat(Xtr[tr_inds[j], :], combins_list[ira, :], poly_degree),
                        ytr[tr_inds[j]], rcond=None)[0]
            pred_vl = nl_features_mat(Xtr[vl_inds[j], :], combins_list[ira, :], poly_degree) @ aa
            err_vl_all1.append(corrcoef(pred_vl, ytr[vl_inds[j]])[0,1])
            pred_tr = nl_features_mat(Xtr[tr_inds[j], :], combins_list[ira, :], poly_degree) @ aa
            err_tr_all1.append(corrcoef(pred_tr, ytr[tr_inds[j]])[0,1])
            pred_te += MNL_te @ aa
        pred_te /= nof_folds
        err_te = corrcoef(pred_te, yte)[1,0]
        err_vl_all1 = array(err_vl_all1)
        err_vl_all1[isnan(err_vl_all1)] = 0
        err_tr_all1 = array(err_tr_all1)
        err_tr_all1[isnan(err_tr_all1)] = 0
        err_vl = quantile(err_vl_all1, 0.05)
        err_tr = quantile(err_tr_all1, 0.05)
        best_acc = err_vl.copy()
        if isnan(best_acc):
            best_acc = 0
        print("Starting Accuracy = ", best_acc)
        best_ira = ira.copy()

        acc_tr_all = [err_tr]
        acc_vl_all = [err_vl]
        acc_te_all = [err_te]
        i_all = [0]
        
        for iter in range(max_n_rounds):
            try:
                rr = rand()
                if rr < 0.35:
                    ira[randint(0,len(ira))] = choice([x for x in range(n_comb) if x not in ira])
                elif rr < 0.7:
                    ira = np_append(ira, choice([x for x in range(n_comb) if x not in ira]))
                else:
                    # delete a random index of ira
                    ira = np_delete(ira, randint(0,len(ira)))
                    
                err_vl_all1 = []
                err_tr_all1 = []
                pred_te = zeros(len(yte))
                MNL_te = nl_features_mat(Xte, combins_list[ira,:], poly_degree)
                for j in range(nof_folds):
                    if LOGISTIC_REGR:
                        aa = lstsq(nl_features_mat(Xtr[tr_inds[j], :], combins_list[ira, :], poly_degree), 
                                logit((ytr[tr_inds[j]]*0.98)+0.01), rcond=None)[0]
                    else:
                        aa = lstsq(nl_features_mat(Xtr[tr_inds[j], :], combins_list[ira, :], poly_degree),
                                ytr[tr_inds[j]], rcond=None)[0]
                    pred_vl = nl_features_mat(Xtr[vl_inds[j], :], combins_list[ira, :], poly_degree) @ aa
                    err_vl_all1.append(corrcoef(pred_vl, ytr[vl_inds[j]])[0,1])
                    pred_tr = nl_features_mat(Xtr[tr_inds[j], :], combins_list[ira, :], poly_degree) @ aa
                    err_tr_all1.append(corrcoef(pred_tr, ytr[tr_inds[j]])[0,1])
                    pred_te += MNL_te @ aa
                pred_te /= nof_folds
                
                err_te = corrcoef(pred_te, yte)[1,0]
                err_vl_all1 = array(err_vl_all1)
                err_vl_all1[isnan(err_vl_all1)] = 0
                err_tr_all1 = array(err_tr_all1)
                err_tr_all1[isnan(err_tr_all1)] = 0
                err_vl = quantile(err_vl_all1, 0.05)
                err_tr = quantile(err_tr_all1, 0.05)
                
                if err_vl > best_acc:
                    best_acc = err_vl.copy()
                    best_ira = ira.copy()
                    print(time()-t0,iter, err_tr, best_acc, err_te, len(best_ira))
                    acc_tr_all.append(err_tr)
                    acc_vl_all.append(err_vl)
                    acc_te_all.append(err_te)
                    i_all.append(iter+1)                
                else:
                    ira = best_ira.copy()
            except KeyboardInterrupt:
                print('KeyboardInterrupt: Stopped by user')
                break
            
        plt.plot(i_all, acc_tr_all, label='train')
        plt.plot(i_all, acc_vl_all, label='validation')
        plt.plot(i_all, acc_te_all, label='test')
        plt.legend()
        plt.savefig(path_ + "acc.png")
        plt.close()

        ipl = int(len(i_all)/2)
        plt.plot(i_all[ipl:], acc_tr_all[ipl:], label='train')
        plt.plot(i_all[ipl:], acc_vl_all[ipl:], label='validation')
        plt.plot(i_all[ipl:], acc_te_all[ipl:], label='test')
        plt.legend()
        plt.savefig(path_ + "acc50perc.png")
        plt.close()
        
                    
        aa = lstsq(nl_features_mat(Xtr, combins_list[best_ira, :], poly_degree), ytr, rcond=None)[0]
        pred_tr = nl_features_mat(Xtr, combins_list[best_ira, :], poly_degree) @ aa
        ttr = time()-t0
        t0=time()
        for i in range(10):
            pred_te = nl_features_mat(Xte, combins_list[best_ira,:], poly_degree) @ aa
        tte=(time()-t0)/10
        
        Xtr = Xtr[:,1:]
        Xte = Xte[:,1:]
        
        export_equation_txt(aa, feat_names, poly_degree, target_name, combins_list, best_ira, path_)
        export_equation_in_excel(aa, LOGISTIC_REGR, poly_degree, best_ira, Xtr.shape[1], path_, ROOT_DIR)

        do_sensitivity(Xtr, features_names, target_name, predict_nlregr, __method__, path_sens)

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


def predict_nlregr(XX):
    global combins_list, best_ira, poly_degree, aa, do_logistic
    
    pred = nl_features_mat(c_[ones(XX.shape[0]), XX], combins_list[best_ira, :], poly_degree) @ aa
    
    if do_logistic:
        return expit(pred)
    else:
        return pred

