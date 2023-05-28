

from import_libraries import *
from misc_functions import *

class torch_set_dataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data   
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
    def __len__ (self):
        return len(self.X_data)
    

class MultipleRegression(nn.Module):
    def __init__(self, num_features, num_targets, neurons, dropout, layers):
        super(MultipleRegression, self).__init__()
        self.num_features = num_features
        self.num_targets = num_targets
        self.layers = layers
        self.dropout = dropout
        self.linear_layers = nn.ModuleList()
        # Add input layer
        self.linear_layers.append(nn.Linear(num_features, neurons))
        torch.nn.init.xavier_uniform_(self.linear_layers[-1].weight)
        self.linear_layers.append(nn.ReLU())
        self.linear_layers.append(nn.Dropout(p=dropout))
        # Add hidden layers
        for i in range(layers-1):
            self.linear_layers.append(nn.Linear(neurons, neurons))
            torch.nn.init.xavier_uniform_(self.linear_layers[-1].weight)
            self.linear_layers.append(nn.ReLU())
            self.linear_layers.append(nn.Dropout(p=dropout))
        # Add output layer
        self.linear_layers.append(nn.Linear(neurons, num_targets))
        torch.nn.init.xavier_uniform_(self.linear_layers[-1].weight)
    def forward(self, x):
        for layer in self.linear_layers:
            x = layer(x)
        return x


def plot_all_losses(all_losses_tr,all_losses_vl):
    # convert all_losses_vl to numpy array
    all_losses_vl = array(all_losses_vl)
    all_losses_tr = array(all_losses_tr)
    iso = argsort(all_losses_vl)[::-1]
    plt.plot(all_losses_tr[iso], label='Train Losses')
    plt.plot(all_losses_vl[iso], label='Validation Losses')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("all_losses_tuning.png")
    plt.close()


def plot_history(history,perc_of_epochs_to_plot,tag1,tag2,path_):
    nn = int(len(history['train'])*(100-perc_of_epochs_to_plot)/100)
    # plot train and validation loss
    plt.plot((array(history['train'][nn:]))**0.5, label=tag1)
    plt.plot((array(history['val'][nn:]))**0.5, label=tag2)
    plt.title('RMSE')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(path_ + "loss_history_percentage_kept_"+str(perc_of_epochs_to_plot)+".png")
    plt.close()

def pred_DANN(XX):
    global model

    # Set the model to evaluation mode
    model.eval()
    # Convert the input data to a PyTorch tensor
    XXX = torch.tensor(XX).float()
    # XXX = XXX.to('cuda')
    # Use the model to make predictions on the input data
    with torch.no_grad():
        pred_tr = model(XXX)
    XXX=0
    model.train()
    # Convert the predicted values to a NumPy array
    pred_tr = pred_tr.cpu().numpy()
    return pred_tr.reshape(-1)

def train_pytorch(XTR, YTR, XVL, YVL, params, path_, save_model=False):
    global model

    # params = combinations[0]
    # fold = 0
    # XTR = Xtr[tr_inds[fold],:]; YTR = ytr[tr_inds[fold]]
    # XVL = Xtr[vl_inds[fold],:]; YVL = ytr[vl_inds[fold]]
    
    YTR = YTR.reshape(-1,1)
    YVL = YVL.reshape(-1,1)
    layers = params[0]; neurons = params[1]; epochs = params[2]
    learning_rate = params[3]; dropout = params[4]; batch = params[5]; moment_um = params[6]
    
    train_dataset = torch_set_dataset(torch.from_numpy(XTR).float(), torch.from_numpy(YTR).float())
    val_dataset = torch_set_dataset(torch.from_numpy(XVL).float(), torch.from_numpy(YVL).float())

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch, shuffle=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch, shuffle=False)

    model = MultipleRegression(XTR.shape[1], YTR.shape[1], neurons, dropout, layers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
    # if torch.cuda.is_available():
    #     for i in range(torch.cuda.device_count()):
            # print(torch.cuda.get_device_name(i))
    
    criterion = nn.MSELoss()
    
    optimizer = optim.NAdam(model.parameters(), lr=learning_rate, momentum_decay=moment_um)
    
    loss_stats = {'train': [], "val": []}

    # print("Begin training.")
    # t1=time()
    for e in range(1, epochs+1):
        # TRAINING
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()
            y_train_pred = model(X_train_batch)
            train_loss = criterion(y_train_pred, y_train_batch)
            train_loss.backward()
            optimizer.step()
            
        loss_stats['train'].append(sqrt(mean((pred_DANN(XTR)-YTR.reshape(-1))**2)))
        loss_stats['val'].append(sqrt(mean((pred_DANN(XVL)-YVL.reshape(-1))**2)))                              
        # if e in range (1,epochs+1,1):
        #     print("==========",e,loss_stats['train'][-1],loss_stats['val'][-1])

    # t2=time()
    # t2_t1 = "%.1f" % (t2-t1)
    # print("training got ",t2_t1," seconds")

    if save_model:
        print("saving model...")
        torch.save(model, path_ + "model.pth")
    
    return loss_stats, model
        
def do_DANN(Xtr, ytr, Xte, yte, features_names, target_name, __thres_early_stop__, __thres_min_tune_rounds__, PERMUTE_TRAIN_TEST, LOGISTIC_REGR, ROOT_DIR):
    global model
    __method__ = "DANN"
    path_ = ROOT_DIR+"ML_Models"+os.sep+__method__+os.sep
    path_err = ROOT_DIR+"ML_Models"+os.sep+__method__+os.sep+"Error_Analysis"+os.sep
    path_sens = ROOT_DIR+"ML_Models"+os.sep+__method__+os.sep+"Sensitivity_Analysis"+os.sep
    try:
        t0=time()
        
        scaler = MinMaxScaler()
        scaler.fit(Xtr)
        Xtr = scaler.transform(Xtr)
        Xte = scaler.transform(Xte)
        obs = Xtr.shape[0]
        
        torch.manual_seed(0)
        seed(0)
        random.seed(0)
        
        layers = [2, 5, 10] 
        neurons = [10, 50, 100] 
        epochs = list(concatenate((arange(1,11),arange(20, 101, 10),arange(200, 1001, 100))))
        learning_rate = [0.01, 0.001, 0.0001] 
        dropout = [0.01, 0.05] 
        batch = [int(obs/8), int(obs/4), int(obs/2)] 
        moment_um = [0.25, 0.5, 0.75]
        combinations = list(product(layers, neurons, epochs, learning_rate, dropout, batch, moment_um)) 
        PARS = "layers, neurons, epochs, learning_rate, dropout, batch, moment_um"
        # randomly permute the combinations
        combinations = shuffle(combinations, random_state=0)
        
        # split train to train and validation
        ___perc_cv___ = 0.8; nof_folds = 5; obs = len(ytr)
        tr_inds, vl_inds = split_tr_vl(obs,___perc_cv___,nof_folds,PERMUTE_TRAIN_TEST)
        
        acc_tr_all = array([]); acc_vl_all = array([]); acc_te_all = array([])
        opti_tr = -1; opti_vl = -1; opti_te= -1  
        best_combination = None
        for i,combination in enumerate(combinations):
            # catch exception if the combination is not valid
            try:
                print(datetime.datetime.now().strftime("%H:%M:%S"),"====================",
                    "Training DANN Model",i+1,"/>",__thres_min_tune_rounds__,"/",len(combinations),":",combination)
                acc_tr = 0; acc_vl = 0; acc_te = 0
                for fold in range(nof_folds):
                    history, model = train_pytorch(Xtr[tr_inds[fold],:], ytr[tr_inds[fold]], 
                                                Xtr[vl_inds[fold],:], ytr[vl_inds[fold]], combination, path_, False)
                    pred_tr = pred_DANN(Xtr[tr_inds[fold],:])
                    pred_vl = pred_DANN(Xtr[vl_inds[fold],:])
                    pred_te = pred_DANN(Xte)
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
                
                print(PARS, combination,
                        "\nR2 Score:: Train:",acc_tr,"Validation:",acc_vl,"Test:",acc_te, 
                        "\nMax Val:",vl_max,"Q75 Val:",vl_75,"Slope:",slope)
                
                imax = argmax(acc_vl_all)
                best_combination = combinations[imax]
                print(datetime.datetime.now().strftime("%H:%M:%S"),"====================",
                    "Best Combination:",best_combination)
                if i>__thres_min_tune_rounds__ and slope <__thres_early_stop__:
                    print("Early Stopping, vl_max:",vl_max,"vl_95:",vl_75)
                    break
            except KeyboardInterrupt:
                print('KeyboardInterrupt: Stopped by user')
                break

        print("Best Combination:",best_combination)
        # find minimum length among acc_tr_all, acc_vl_all, acc_te_all, in case of exception
        max_len = min([len(acc_tr_all), len(acc_vl_all), len(acc_te_all)])
        # make all the arrays of same length
        acc_tr_all = acc_tr_all[:max_len]
        acc_vl_all = acc_vl_all[:max_len]
        acc_te_all = acc_te_all[:max_len]

        ttr = time()-t0
        # Save the scaler to a file
        with open(path_ + "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        history, model = train_pytorch(Xtr, ytr, Xte, yte, best_combination, path_, True)
        plot_history(history,100,'Train','Test',path_)
        plot_history(history,50,'Train','Test',path_)
        pred_tr = pred_DANN(Xtr)
        t0=time()
        for i in range(10):
            pred_te = pred_DANN(Xte)
        tte=(time()-t0)/10
        iso = argsort(acc_vl_all)
        plt.plot(acc_tr_all[iso], 'x', label='Train')
        plt.plot(acc_vl_all[iso], 'x', label='Validation')
        plt.plot(acc_te_all[iso], 'x', label='Test')
        plt.legend()
        plt.xlabel('Model')
        plt.ylabel('R2 Score')
        plt.savefig(path_ + "DANN_tune_cv_history.png")
        plt.close()
        
        iso = argsort(acc_vl_all)[int(0.5*len(acc_vl_all)):]
        plt.plot(acc_tr_all[iso], 'x', label='Train')
        plt.plot(acc_vl_all[iso], 'x', label='Validation')
        plt.plot(acc_te_all[iso], 'x', label='Test')
        plt.legend()
        plt.xlabel('Model')
        plt.ylabel('R2 Score')
        plt.savefig(path_ + "DANN_tune_cv_history_50perc.png")
        plt.close()

        do_sensitivity(Xtr, features_names, target_name, pred_DANN, __method__, path_sens)

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


def predict_DANN(Xout, yout, target_name, LOGISTIC_REGR, ROOT_DIR):
    global cwd, model
    __method__ = "DANN"
    path_ = ROOT_DIR+"ML_Models"+os.sep+__method__+os.sep
    path_pred = ROOT_DIR+"Predict"+os.sep+__method__+os.sep
    # try to load the results from the CSV file
    try:
        # Load the scaler from the file
        with open(path_ + "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        Xout = scaler.transform(Xout)

        # load model
        model = torch.load(path_ + "model.pth")

        pred_out = pred_DANN(Xout)
        
    except Exception as e:
        print("Error: ", e)
        return

    # save predictions to file
    with open(path_pred + "Predictions_"+__method__+".csv", "w") as file:
        for yi in pred_out:
            file.write(str(yi) + '\n')

    plot_target_vs_predicted(yout, pred_out, target_name, __method__, "Out", path_pred) 
    export_metrics_out(yout, pred_out, path_pred + __method__ + "_Out", LOGISTIC_REGR)
    error_analysis(yout, pred_out, target_name, __method__, "Out", path_pred)    
    
    print("See results in folder: ", path_pred)