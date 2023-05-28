# from IPython import get_ipython
from import_libraries import *
from misc_functions import *


def solve_bak(x, y, a):
    e = x@a - y
    err_all = array([])
    for j in range(10):
        for i in range(x.shape[1]):
            da = sum(e*x[:,i])/dot(x[:,i],x[:,i])
            e -= da*x[:,i]
            a[i] -= da
        err_all = np_append(err_all, mean(abs(e)))
    return a

# normalize ytr to have minimum -0.99 and maximum 0.99
def normalize_y(ytr):
    min_y = min(ytr)
    ytr = ytr - min_y
    max_y = max(ytr)
    ytr = ytr / max_y
    ytr = ytr * 0.98
    ytr = ytr + 0.01
    ytr = ytr * 2
    ytr = ytr - 1
    return ytr, min_y, max_y

#  denormalize ytr to their initial values
def denormalize_y(ytr, min_y, max_y):
    ytr = ytr + 1
    ytr = ytr / 2
    ytr -= 0.01
    ytr /= 0.98
    ytr *= max_y
    ytr += min_y
    return ytr

def compute_w(Xtr,ytr,neurons,min_obs_over_neurons): 
    i_train = Xtr.shape[0]
    layer1 = zeros((i_train, neurons))
    w_all = []
    i_err = []
    ii_all = []
    n_train_internal = int(2*Xtr.shape[0]/min_obs_over_neurons)
    
    # k means clustering of Xtr 
    # print("K-means clustering running for",neurons,"neurons.")
    #  normalize Xtr
    Xclust = (Xtr-mean(Xtr,axis=0))
    Xclust = Xclust/std(Xclust,axis=0)
    kmeans = KMeans(n_clusters=neurons,random_state=0,init='k-means++',n_init='auto').fit(Xclust)
    # print("K-means clustering done.")

    # print("Computing internal neurons' weights...")
    iiXtr = arange(Xtr.shape[0])
    for i in range(neurons):
        # ii = permutation(Xtr.shape[0])[:n_train_internal]
        ii = iiXtr[kmeans.labels_==i]
        ii_all.append(ii)
        try:
            aa = lstsq(Xtr[ii],arctanh(ytr[ii]), rcond=None)[0]
            w_all.append(aa)
            layer1[:, i] = tanh(dot(Xtr, aa))
        except Exception as ex:
            i_err.append(i)
            print("Error in neuron ", i, ":", ex)
        # if i in range(0, neurons, int(math.floor(neurons/10))) or i == neurons:
        #     print("Training ", i, " of ", neurons, " internal neurons.")
     

    i_keep = delete(arange(neurons), i_err)
    layer1 = layer1[:, i_keep]
    neurons = len(i_keep)
    ii_all = [ii_all[i] for i in i_keep]

    idx = arange(neurons)
    # print("SVD running...")
    # u, s, v = svd(layer1)
    # k = sum(s > 1e-10)
    # print("SVD done. k =", k,"of",layer1.shape[1],"neurons.")
    # print("QR-Factorization running...")
    # q, r, p = linalg.qr(layer1, pivoting=True)
    # idx = p[:k]
    # layer1 = layer1[:, idx]
    # neurons = len(idx)
    # w_all = [w_all[i] for i in idx]
    # ii_all = [ii_all[i] for i in idx]

    return layer1, w_all, ii_all, neurons, idx

def compute_layer1_te(Xte):
    global w_all, V
    # neurons = len(w_all)
    # layer1_te = zeros((Xte.shape[0], neurons))
    # for i in range(neurons):
    #     layer1_te[:,i] = tanh(dot(Xte, w_all[i]))
    layer1_te = tanh(dot(Xte, w_all.T))
    return layer1_te

def pred_annbn(Xpred):
    global w_all, V
    return dot(tanh(dot(Xpred, w_all.T)), V)


def do_ANNBN(Xtr, Xte, ytr, yte, features_names, target_name, PERMUTE_TRAIN_TEST, LOGISTIC_REGR, ROOT_DIR, min_obs_over_neurons):
    global w_all, V
    
    try:
        neurons=int(Xtr.shape[0]/min_obs_over_neurons)

        t0=time()
        # ytr, min_y, max_y = normalize_y(ytr)
        # layer1, w_all, ii_all, neurons, idx = compute_w(Xtr,ytr,neurons,min_obs_over_neurons)
        # w_all = array(w_all)
        # ytr = denormalize_y(ytr, min_y, max_y)
        # layer1_te = compute_layer1_te(Xte)

        ___perc_cv___ = 0.8; nof_folds = 5; obs = len(ytr)
        tr_inds, vl_inds = split_tr_vl(obs,___perc_cv___,nof_folds,PERMUTE_TRAIN_TEST)
        opti_idx = None
        acc_tr_all = array([]); acc_vl_all = array([]); acc_te_all = array([])
        i_all = array([]).astype(int)
        # step = int(Xtr.shape[0]/50)
        step = int(neurons/20)
        ki= 'rank'
        list_run = list(range(10, neurons, step))
        list_run.append(neurons)
        list_run = unique(list_run)[::-1]
        for i in list_run:    
            try:
                ytr, min_y, max_y = normalize_y(ytr)
                layer1, w_all, ii_all, _, idx = compute_w(Xtr,ytr,i,min_obs_over_neurons)
                w_all = array(w_all)
                ytr = denormalize_y(ytr, min_y, max_y)
                layer1_te = compute_layer1_te(Xte)
                
                inds_keep = range(i)#idx[:i]
                acc_tr = 0; acc_vl = 0; acc_te = 0
                V = zeros(i)
                print("Solving for",i,"neurons.")
                for fold in range(nof_folds):
                    # res = regression_bak(layer1[tr_inds[fold],:][:,inds_keep], ytr[tr_inds[fold]])
                    # if res.flag != "OK":
                    #     raise Exception("Regression failed at",i,"neurons.")
                    # V = res.aa
                    V,_,ki,_ = lstsq(layer1[tr_inds[fold],:][:,inds_keep], ytr[tr_inds[fold]], rcond=None)
                    # X = layer1[tr_inds[fold],:][:,inds_keep]
                    # XX = X.T@X
                    # V = solve(XX, X.T@ytr[tr_inds[fold]])
                    # V = solve_bak(layer1[tr_inds[fold],:][:,inds_keep], ytr[tr_inds[fold]], V)
                    pred_tr = dot(layer1[tr_inds[fold],:][:,inds_keep], V)
                    pred_vl = dot(layer1[vl_inds[fold],:][:,inds_keep], V)
                    pred_te = dot(layer1_te[:,inds_keep], V)
                    acc_tr += pearsonr(ytr[tr_inds[fold]], pred_tr).correlation
                    acc_vl += pearsonr(ytr[vl_inds[fold]], pred_vl).correlation
                    acc_te += pearsonr(yte, pred_te).correlation
                acc_tr /= nof_folds; acc_vl /= nof_folds; acc_te /= nof_folds
                acc_tr_all = np_append(acc_tr_all, acc_tr)
                acc_vl_all = np_append(acc_vl_all, acc_vl)
                acc_te_all = np_append(acc_te_all, acc_te)
                i_all = np_append(i_all, i)
                print(datetime.datetime.now().strftime("%H:%M:%S"),"CombNeurons:",i,"of",neurons,". Rank:",ki,
                    "\nR2 Score:: Train:",acc_tr,"Validation:",acc_vl,"Test:",acc_te)
            except KeyboardInterrupt:
                print('KeyboardInterrupt: Stopped by user')
                break

        # find minimum length among acc_tr_all, acc_vl_all, acc_te_all, in case of exception
        max_len = min([len(acc_tr_all), len(acc_vl_all), len(acc_te_all)])
        # make all the arrays of same length
        acc_tr_all = acc_tr_all[:max_len]
        acc_vl_all = acc_vl_all[:max_len]
        acc_te_all = acc_te_all[:max_len]

        imax = argmax(acc_vl_all)
        imax = i_all[imax]
        print(">>>>>>>>>>>>>>>>>>>>Optimal number of neurons:",imax,"<<<<<<<<<<<<<<<<<<<")
        # opti_idx = range(imax)
        # layer1 = layer1[:, opti_idx]
        # layer1_te = layer1_te[:, opti_idx]
        # neurons = len(opti_idx)
        # w_all = w_all[opti_idx,:]
        # ii_all = [ii_all[i] for i in opti_idx] 
        
        neurons = imax.copy()
        ytr, min_y, max_y = normalize_y(ytr)
        layer1, w_all, ii_all, neurons, idx = compute_w(Xtr,ytr,imax,min_obs_over_neurons)
        w_all = array(w_all)
        ytr = denormalize_y(ytr, min_y, max_y)
        layer1_te = compute_layer1_te(Xte)
                
        
        ttr = time()-t0    
        __method__ = "ANNBN"
        path_ = ROOT_DIR+"ML_Models"+os.sep+__method__+os.sep
        path_err = ROOT_DIR+"ML_Models"+os.sep+__method__+os.sep+"Error_Analysis"+os.sep
        path_sens = ROOT_DIR+"ML_Models"+os.sep+__method__+os.sep+"Sensitivity_Analysis"+os.sep
        
        cdf_pdf_plot(layer1.flatten(),"Layer1Histogram",path_)
        
        # res = regression_bak(layer1, ytr)
        # V = res.aa
        V,_,ki,_ = lstsq(layer1, ytr, rcond=None)
        pred_tr = dot(layer1, V)
        t0=time()
        for i in range(10):
            # layer1_te = compute_layer1_te(Xte)#####
            # pred_te = dot(layer1_te, V)
            pred_annbn(Xte)
        tte=(time()-t0)/10

        try:
            export_equation_in_excel(Xtr, ytr, features_names, target_name, min_y, max_y, w_all, V, path_, ROOT_DIR)
        except Exception as ex:
            print("Error in exporting equation:", ex)
            return

        with open(path_ + "NeuronsWeights.csv", "w") as file:
            for wi in w_all:
                for wij in wi:
                    file.write(str(wij) + ',')
                file.write('\n')
        with open(path_ + "ExternalLayerWeights.csv", "w") as file:
            for Vi in V:
                file.write(str(Vi) + ',')

        iso = argsort(acc_vl_all)
        plt.plot(i_all[iso], acc_tr_all[iso], 'x', label='Train')
        plt.plot(i_all[iso], acc_vl_all[iso], 'x', label='Validation')
        plt.plot(i_all[iso], acc_te_all[iso], 'x', label='Test')
        plt.legend()
        plt.xlabel('Neurons')
        plt.ylabel('R2 Score')
        plt.savefig(path_ + "ANNBN_tune_cv_history.png")
        plt.close()
        
        iso = argsort(acc_vl_all)[:argmax(acc_vl_all)]
        plt.plot(i_all[iso], acc_tr_all[iso], 'x', label='Train')
        plt.plot(i_all[iso], acc_vl_all[iso], 'x', label='Validation')
        plt.plot(i_all[iso], acc_te_all[iso], 'x', label='Test')
        plt.legend()
        plt.xlabel('Neurons')
        plt.ylabel('R2 Score')
        plt.savefig(path_ + "ANNBN_tune_cv_history_imax.png")
        plt.close()

        do_sensitivity(Xtr, features_names, target_name, pred_annbn, __method__, path_sens)

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




    

def export_equation_in_excel(Xtr, ytr, features_names, target_name, min_y, max_y, w_all, V, path_, ROOT_DIR):
    col_excel = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    col_excel0 = list(col_excel)
    col_excel = col_excel + [f'A{c}' for c in col_excel0]
    col_excel = col_excel + [f'B{c}' for c in col_excel0]
    col_excel = col_excel + [f'C{c}' for c in col_excel0]
    col_excel = col_excel + [f'D{c}' for c in col_excel0]
    col_excel = array(col_excel)
    
    inds_exclude = []
    if os.path.isfile(ROOT_DIR + "excluded_features.txt"):
        with open(ROOT_DIR + "excluded_features.txt", 'r') as f:
            inds_exclude = [int(line.strip()) for line in f]
        inds_exclude = array(inds_exclude)
        print("Multicolinearity was detected, excluding columns from excel: ", [i for i in col_excel[inds_exclude]])
    col_excel = delete(col_excel, inds_exclude, axis=0)

    equation_excel = "="
    minYstr = str(min_y)
    if min_y > 0:
        minYstr = "+" + minYstr
    nof_files = 1
    equations_all = []
    neurons = len(w_all)
    for i in range(neurons):
        new_feature = ""
        for j in range(Xtr.shape[1]):
            str_ = "{:.15E}".format(w_all[i][j])
            if w_all[i][j] >= 0:
                str_ = "+" + str_
            new_feature = "{}{}*{}2".format(new_feature, str_, col_excel[j])

        str00 = "{:.15E}".format(V[i])
        if V[i] >= 0:
            str00 = "+" + str00
        new_feature = "{}*tanh({})".format(str00, new_feature)

        if len(equation_excel) + len(new_feature) > 8000:
            equations_all.append(equation_excel)
            nof_files += 1
            equation_excel = "="
            print("Equation split at neuron:", i)

        equation_excel = "{}{}".format(equation_excel, new_feature)

    equations_all.append(equation_excel)
    print(">>>>>>>>>>>>>>>>>>equation_excel Done in", nof_files, "parts.<<<<<<<<<<<<<<")

    i_med = argsort(ytr)[int(round(len(ytr) / 2))]
    xx_tr_med_given_init = c_[Xtr[i_med, newaxis], array([ytr[i_med]])]
    # Number of columns in the original matrix
    n_cols = xx_tr_med_given_init.shape[1]
    # Number of new columns to insert
    n_new_cols = len(inds_exclude)
    # Initialize a matrix of zeros with the new shape
    new_matrix = zeros((1, n_cols + n_new_cols))
    # Copy the original matrix into the new matrix, excluding the columns to be excluded
    new_matrix[:, [i for i in range(n_cols + n_new_cols) if i not in inds_exclude]] = xx_tr_med_given_init
    xx_tr_med_given_init = new_matrix

    equations_all_mat = empty((1, len(equations_all)), dtype=object)
    pred_str = "="
    for i in range(len(equations_all)):
        equations_all_mat[0, i] = equations_all[i]
        pred_str = pred_str + col_excel[xx_tr_med_given_init.shape[1] - len(inds_exclude) + i] + "2+"
    pred_str = pred_str[:-1]
    equations_all_mat = column_stack([equations_all_mat, pred_str])
    equations_all_mat = column_stack([xx_tr_med_given_init, equations_all_mat])
    df = pd.DataFrame(equations_all_mat)
    
    # New label to insert
    new_label = 'collinear'
    # Initialize a new Index object with the same labels as features_names
    new_index = features_names.copy()
    # Insert the new label at the specified positions
    for i in sorted(inds_exclude):
        new_index = new_index.insert(i, new_label)
    excel_col_pred_names = np_append(new_index, [target_name, *[f"pred-{i}" for i in range(1, len(equations_all) + 1)], "Prediction"])
    df.rename(columns=dict(zip(df.columns, excel_col_pred_names)), inplace=True)
    df.to_excel(path_ + "equation_excel.xlsx", index=False, engine="openpyxl")


def predict_ANNBN(Xout, yout, target_name, LOGISTIC_REGR, ROOT_DIR):
    global w_all, V, cwd
    __method__ = "ANNBN"
    path_ = ROOT_DIR+"ML_Models"+os.sep+__method__+os.sep
    path_pred = ROOT_DIR+"Predict"+os.sep+__method__+os.sep
    # try to load the results from the CSV file
    try:
        with open(path_ + "NeuronsWeights.csv", 'r') as f:
            reader = csv.reader(f)
            w_all = list(reader)
        # delete the last element, of each row, which is an empty string
        for wi in w_all:
            wi.pop()
        w_all = [[float(wij) for wij in wi] for wi in w_all]
        w_all = array(w_all)
        
        with open(path_ + "ExternalLayerWeights.csv", 'r') as f:
            reader = csv.reader(f)
            V = list(reader)
        V[0].pop()
        V = [float(Vi) for Vi in V[0]]

        neurons = len(w_all)
        print("Loaded ANNBN weights from file")
        
        pred_out = pred_annbn(Xout)
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