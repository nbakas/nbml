
from import_libraries import *
from misc_functions import *


def descriptive_statistics(Xtr, Xte, features_names, ROOT_DIR):
    try:
        mean1 = mean(Xtr, axis=0)
        median1 = median(Xtr, axis=0)
        std1 = std(Xtr, axis=0)
        min1 = Xtr.min(axis=0)
        max1 = Xtr.max(axis=0)
        skewness1 = skew(Xtr, axis=0)
        kurtosis1 = kurtosis(Xtr, axis=0)
        # df = pd.DataFrame({'mean':mean1, 'median':median1, 'std':std1, 'min':min1, 'max':max1, 'skewness':skewness1, 'kurtosis':kurtosis1}, index=features_names)
        # df.to_excel("descriptive_statistics_train.xlsx", index=True)
        # print("Descriptive statistics saved in Descriptive_Statistics/descriptive_statistics_train.xlsx")
        
        # test set descriptive statistics
        mean2 = mean(Xte, axis=0)
        median2 = median(Xte, axis=0)
        std2 = std(Xte, axis=0)
        min2 = Xte.min(axis=0)
        max2 = Xte.max(axis=0)
        skewness2 = skew(Xte, axis=0)
        kurtosis2 = kurtosis(Xte, axis=0)
        # df = pd.DataFrame({'mean':mean2, 'median':median2, 'std':std2, 'min':min2, 'max':max2, 'skewness':skewness2, 'kurtosis':kurtosis2}, index=features_names)
        # df.to_excel("descriptive_statistics_test.xlsx", index=True)
        # print("Descriptive statistics saved in Descriptive_Statistics/descriptive_statistics_test.xlsx")

        # deferrences between train and test
        # df = pd.DataFrame({'mean':mean1-mean2, 'median':median1-median2, 'std':std1-std2, 'min':min1-min2, 'max':max1-max2, 'skewness':skewness1-skewness2, 'kurtosis':kurtosis1-kurtosis2}, index=features_names)
        # df.to_excel("descriptive_statistics_train_test_differences.xlsx", index=True)
        # print("Descriptive statistics saved in Descriptive_Statistics/descriptive_statistics_train_test_differences.xlsx")
        
        # train, test and differences in one exls file with 3 sheets
        file_exp = ROOT_DIR + "Descriptive_Statistics" + os.path.sep + "descriptive_statistics_train_test_and_differences.xlsx"
        writer = pd.ExcelWriter(file_exp, engine='openpyxl')
        df = pd.DataFrame({'mean':mean1, 'median':median1, 'std':std1, 'min':min1, 'max':max1, 'skewness':skewness1, 
                           'kurtosis':kurtosis1}, index=features_names)
        df.to_excel(writer, sheet_name='train')
        df = pd.DataFrame({'mean':mean2, 'median':median2, 'std':std2, 'min':min2, 'max':max2, 'skewness':skewness2, 
                           'kurtosis':kurtosis2}, index=features_names)
        df.to_excel(writer, sheet_name='test')
        df = pd.DataFrame({'mean':100*(mean1-mean2)/mean1, 'median':100*(median1-median2)/median1, 'std':100*(std1-std2)/std1, 
                           'min':100*(min1-min2)/min1, 'max':100*(max1-max2)/max1, 'skewness':100*(skewness1-skewness2)/skewness1, 
                           'kurtosis':100*(kurtosis1-kurtosis2)/kurtosis1}, index=features_names)
        df.to_excel(writer, sheet_name='%differences')
        writer.book.save(file_exp)
        writer.close()
        print("Descriptive statistics saved in Descriptive_Statistics/descriptive_statistics_train_test_and_differences.xlsx")

        
    except Exception as ex1:
        print(ex1)


def plot_pdf_cdf_all(Xtr, ytr, features_names, target_name, ROOT_DIR):
    try:
        for i in range(Xtr.shape[1]):
            cdf_pdf_plot(Xtr[:,i],features_names[i]+"_Train",ROOT_DIR+"Descriptive_Statistics"+os.path.sep+"PDF_CDF"+os.path.sep)
        cdf_pdf_plot(ytr,target_name+"_Train",ROOT_DIR+"Descriptive_Statistics"+os.path.sep+"PDF_CDF"+os.path.sep)
        print("PDF and CDF saved in Descriptive_Statistics/PDF_CDF")
    except Exception as ex1:
        print(ex1)

def plot_all_timeseries(XX, features_names, YY, target_name, plot_type, ROOT_DIR):
    try:
        ############################
        window_size = 30
        ############################
        for i in range(XX.shape[1]):
            rolling_correlation = zeros(len(YY))
            for j in range(window_size,len(YY)):
                rolling_correlation[j] = corrcoef(YY[j-window_size:j],XX[j-window_size:j,i])[0,1]
            rolling_correlation[isnan(rolling_correlation)] = 0

            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(50, 12))
            ax1.plot(rolling_correlation)
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Rolling Correlation')
            ax1.grid(True)
            ax1.xaxis.set_major_locator(plt.MultipleLocator(100))
            ax2.plot(XX[:,i], color='blue')
            ax2.set_xlabel('Time')
            ax2.set_ylabel(features_names[i], color='blue')
            ax2.grid(True, which='both', axis='both')
            ax2.xaxis.set_major_locator(plt.MultipleLocator(100))
            ax3 = ax2.twinx()
            ax3.plot(YY, color='red')
            ax3.set_ylabel(target_name, color='red')
            fig.savefig(ROOT_DIR + "Descriptive_Statistics" + os.sep + "TimeSeries" + os.sep + plot_type + "_" + features_names[i] + ".png", bbox_inches='tight')
            plt.close()
            
            moving_average = []
            hor = 365
            for ii in range(len(XX[:,i]), hor, -1):
                moving_average.append(mean(XX[:,i][ii:ii-hor:-1]))
            plt.figure(figsize=(50, 12))
            plt.plot(XX[:,i][:len(XX[:,i])-hor])
            plt.title(plot_type + "_" + features_names[i])
            plt.plot(moving_average)
            plt.title(plot_type + "_" + features_names[i]+"_moving_average"+str(hor))
            plt.savefig(ROOT_DIR + "Descriptive_Statistics" + os.sep + "TimeSeries" + os.sep + plot_type + "_" + features_names[i]+"_moving_average"+str(hor)+".png", bbox_inches='tight')
            plt.close()
            
            print("Ploting time series for " + features_names[i])
            
        moving_average = []
        hor = 365
        for i in range(len(YY), hor, -1):
            moving_average.append(mean(YY[i:i-hor:-1]))
        plt.figure(figsize=(50, 12))
        plt.plot(YY[:len(YY)-hor])
        plt.title(plot_type + "_" + target_name)
        plt.plot(moving_average)
        plt.title(plot_type + "_" + target_name+"_moving_average"+str(hor))
        plt.savefig(ROOT_DIR + "Descriptive_Statistics" + os.sep + "TimeSeries" + os.sep + plot_type + "_" + target_name+"_moving_average"+str(hor)+".png", bbox_inches='tight')
        plt.close()
        
        print("Time series saved in Descriptive_Statistics/TimeSeries")
    except Exception as ex1:
        print(ex1)

def plot_all_by_all_correlation_matrix(Xtr, ytr, features_names, target_name, ROOT_DIR):
    try:
        corr = corrcoef(Xtr, ytr, rowvar=False)
        names_all = features_names.copy().tolist()
        names_all.append(target_name)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(corr, vmin=-1, vmax=1)
        fig.colorbar(cax)
        ticks = arange(0,len(names_all),1)
        ax.set_xticks(ticks)
        plt.xticks(rotation=90)
        ax.set_yticks(ticks)
        ax.set_xticklabels(names_all)
        ax.set_yticklabels(names_all)
        plt.savefig(ROOT_DIR + "Descriptive_Statistics"+ os.path.sep + "All_by_All_Correlation_Matrix.png", bbox_inches='tight')
        plt.close()
        # df = pd.DataFrame(c_[Xtr, ytr], columns = features_names.union([target_name]))
        # sns.pairplot(df, diag_kind='kde')
        # df=0
        # plt.savefig(ROOT_DIR + "Descriptive_Statistics"+ os.path.sep + "All_by_All_Correlation_Matrix_Full.png", bbox_inches='tight')
        # plt.close()
        print("All by All Correlation Matrix saved in Descriptive_Statistics/All_by_All_Correlation_Matrix.png")
        
        # scatter plot of all features vs target in one plot
        fig, ax = plt.subplots(Xtr.shape[1], 1, figsize=(7, 5*Xtr.shape[1]))
        iso = argsort(abs(corr[:-1,-1]))
        for k,i in enumerate(iso):
            ax[k].scatter(Xtr[:,i], ytr)
            ax[k].set_xlabel(features_names[i])
            ax[k].set_ylabel(target_name)
            ax[k].set_title("Pearson Correlation: "+str(round(corr[i,-1],3)))
            # add trendline
            z = polyfit(Xtr[:,i], ytr, 1)
            p = poly1d(z)
            ax[k].plot(Xtr[:,i],p(Xtr[:,i]),"r--")
                    
        fig.tight_layout()
        plt.savefig(ROOT_DIR + "Descriptive_Statistics"+ os.path.sep + "All_Features_vs_Target.png", bbox_inches='tight')
        plt.close()
        print("All Features vs Target saved in Descriptive_Statistics/All_Features_vs_Target.png")
        

        # Generate Map
        nof_obj = len(names_all)
        similarity = copy(corr)
        for i in range(nof_obj):
            for j in range(nof_obj): 
                similarity[i,j] /= corr[i,i]+corr[j,j]-similarity[i,j]

        func_evals = 2*nof_obj*500
        lb=-10.0*ones((nof_obj,2))
        ub=copy(-lb)
        xa=lb+(ub-lb)*rand(nof_obj,2)
        opti_xa=copy(xa)
        iter_opti = []; all_opti = []
        opti_fu = -Inf
        inz = logical_and((0.0<similarity), (similarity<1))
        opti_D = 0
        for iter in range(func_evals):
            i=randint(0,high=nof_obj)
            j=randint(0,2)
            xa[i,j]=lb[i,j] + rand()*(ub[i,j]-lb[i,j])
            G = xa@transpose(xa)
            g = diag(G).reshape(-1,1)
            D = sqrt(-2*G+g+transpose(g))
            D /= D.max()
            
            fu = corrcoef(D[inz],-log(similarity[inz]))[0,1]
            if fu>opti_fu:
                opti_xa=copy(xa)
                opti_fu=copy(fu)
                opti_D = copy(D)
                iter_opti.append(iter)
                all_opti.append(opti_fu)
            else:
                xa=copy(opti_xa)
            if iter==func_evals-1:
                print(iter, "Map is ready, Optimal Objective: ", opti_fu)

        plt.scatter(iter_opti, all_opti)
        plt.savefig(ROOT_DIR + "Descriptive_Statistics"+ os.path.sep + "Convergence History.png")  
        plt.close()
        plt.scatter(opti_D[inz],-log(similarity[inz]))
        plt.savefig(ROOT_DIR + "Descriptive_Statistics"+ os.path.sep + "Distances vs Similarity.png")  
        plt.close()

        corry = corr[:,-1]
        ss = [int(100*abs(corry[i])) for i in range(len(corry))]
        fig, ax = plt.subplots()
        plt.scatter(opti_xa[:,0],opti_xa[:,1],s=ss)
        texts = [plt.text(opti_xa[i,0], opti_xa[i,1], names_all[i], 
                fontsize=10) for i in range(nof_obj)]
        plt.axis('off')
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
        # , arrowprops=dict(arrowstyle='->', color='red')
        plt.savefig(ROOT_DIR + "Descriptive_Statistics"+ os.path.sep + "map.png")  
        plt.close()
    except Exception as ex1:
        print(ex1)

    # export_notebook_to_html()
    
def export_descriptive_per_bin(Xtr,Xte,ytr,yte,features_names,target_name,ROOT_DIR):

    file_exp = ROOT_DIR + "Descriptive_Statistics" + os.path.sep + "descriptive_statistics_per_bin.xlsx"
    writer = pd.ExcelWriter(file_exp, engine='openpyxl')
                
    percentiles_ytr = []
    percentiles_yte = []
    qstart = [0,0, 0, 0, 50, 75, 95, 99]
    qend =   [1,5,25,50,100,100,100,100]
    for i in range(len(qstart)):
        percentiles_ytr.append((percentile(ytr,qstart[i]), percentile(ytr,qend[i])))
        percentiles_yte.append((percentile(yte,qstart[i]), percentile(yte,qend[i])))
        

    significant_names = []
    significant_perc = []
    str_min_max = []
    for i in range(len(percentiles_ytr)):
        ii = where((percentiles_ytr[i][0]<=ytr) & (ytr<=percentiles_ytr[i][1]))[0]
        str_= '{:.5e}'.format(percentiles_ytr[i][0])+"<=ytr<="+'{:.5e}'.format(percentiles_ytr[i][1])+"|"+str(len(ii))+"_ytr_values"
        DX = (Xtr[ii,:].max(axis=0) - Xtr[ii,:].min(axis=0))/(Xtr.max(axis=0) - Xtr.min(axis=0))
        iso = argsort(DX)
        str_min_max.append(features_names[iso[0]] + "_in_" + '{:.2e}'.format(Xtr[ii,iso[0]].min(axis=0))+"~"+'{:.2e}'.format(Xtr[ii,iso[0]].max(axis=0)))
        mean1 = mean(Xtr[ii,:], axis=0)[iso]
        median1 = median(Xtr[ii,:], axis=0)[iso]
        std1 = std(Xtr[ii,:], axis=0)[iso]
        min1 = Xtr[ii,:].min(axis=0)[iso]
        max1 = Xtr[ii,:].max(axis=0)[iso]
        skewness1 = skew(Xtr[ii,:], axis=0)[iso]
        kurtosis1 = kurtosis(Xtr[ii,:], axis=0)[iso]
        df1 = pd.DataFrame({'Dataset':"Train", 'mean':mean1, 'median':median1, 'std':std1, 'min':min1, 'max':max1, 'skewness':skewness1, 
                            'kurtosis':kurtosis1}, index=features_names[iso])
        significant_names.append(features_names[iso[0]])
        significant_perc.append(DX[iso[0]])
        
        ii = where((percentiles_yte[i][0]<=yte) & (yte<=percentiles_yte[i][1]))[0]
        str_ += "|"+str(len(ii))+"_yte_values"
        mean2 = mean(Xte[ii,:], axis=0)[iso]
        median2 = median(Xte[ii,:], axis=0)[iso]
        std2 = std(Xte[ii,:], axis=0)[iso]
        min2 = Xte[ii,:].min(axis=0)[iso]
        max2 = Xte[ii,:].max(axis=0)[iso]
        skewness2 = skew(Xte[ii,:], axis=0)[iso]
        kurtosis2 = kurtosis(Xte[ii,:], axis=0)[iso]
        df2 = pd.DataFrame({'Dataset':"Test", 'mean':mean2, 'median':median2, 'std':std2, 'min':min2, 'max':max2, 'skewness':skewness2, 
                            'kurtosis':kurtosis2}, index=features_names[iso])
        new_df = pd.concat([df1, df2])
        new_row = pd.DataFrame({'Dataset':[None], 'mean':[None], 'median':[None], 'std':[None], 'min':[None], 'max':[None], 'skewness':[None], 
                            'kurtosis':[None]}, index=[str_])
        new_df = pd.concat([new_df, new_row])
        new_df.to_excel(writer, sheet_name="q"+str(qstart[i])+"-q"+str(qend[i]))

    writer.book.save(file_exp)
    writer.close()


    plt.barh(range(len(significant_names)), significant_perc, align='center')
    plt.yticks(range(len(significant_names)), str_min_max)
    for i, v in enumerate(significant_perc):
        plt.text(v + 0.01, i + 0.25, target_name+" in Q"+str(qstart[i])+"-Q"+str(qend[i]), color='blue', fontweight='bold')
    plt.xlabel("Percentage of Xmax-Xmin in Quantile")
    plt.tight_layout()
    # Get the current axis object
    ax = plt.gca()
    # Remove the top and right spines of the axis
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(ROOT_DIR + "Descriptive_Statistics"+ os.path.sep + "Percentage_of_Xmax-Xmin_in_Quantile.png")
    plt.close()


def plot_short_tree(Xtr, ytr, features_names, target_name, ROOT_DIR):
        
    leaf_nodes_all = []
    unique_leaf_nodes_all = []
    for __DEPTH__ in range(1,4):
        print("Computing short tree, with depth:",__DEPTH__)
        # create decision tree classifier with max_depth=3
        dtree = DecisionTreeRegressor(max_depth=__DEPTH__, criterion='absolute_error', min_samples_leaf = 20, random_state=0)

        # fit the model with iris data
        dtree.fit(Xtr, ytr)

        # print out the tree structure, thresholds, and leaves
        tree_rules = export_text(dtree, feature_names=list(features_names))
        print(tree_rules)

        # create subplots grid
        # fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20,5))
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(21,10), gridspec_kw={'width_ratios': [2, 1]})
        # plot decision tree on left-hand side
        plot_tree(dtree, feature_names=list(features_names), filled=True, ax=ax1, node_ids = False, proportion= True)
        # plot PDF and CDF of y_train on right-hand side
        sns.kdeplot(ytr, ax=ax2, fill=True, color='r', label="PDF-ALL")
        # sns.kdeplot(ytr, ax=ax2, cumulative=True, color='r', label="CDF-ALL")
        ax2.set_ylabel('PDF for all samples and for each leaf node')
        ax2.set_xlabel(target_name)
        ax2.grid()

        ##############################################
        leaf_nodes = dtree.apply(Xtr)
        unique_leaf_nodes = unique(leaf_nodes)
        leaf_nodes_all.append(leaf_nodes)
        unique_leaf_nodes_all.append(unique_leaf_nodes)
        for i in range(len(unique_leaf_nodes)):
            node = unique_leaf_nodes[i]
            indices = where(leaf_nodes == node)[0]
            sns.kdeplot(ytr[indices], ax=ax2, fill=False, label="LEAF-"+str(i+1))
        ##############################################
        plt.tight_layout()
        plt.legend()
        # save figure and close plot object
        plt.savefig(ROOT_DIR+"Descriptive_Statistics"+os.sep+"Tree"+os.sep +target_name+'_tree_with_pdf_for_all_leaves_depth_'+str(__DEPTH__)+'.png')
        plt.close()
        
        feature_importance = dtree.tree_.compute_feature_importances()
        iso = argsort(feature_importance)[::-1]
        iPos = where(feature_importance[iso]>0.001)[0]
        plt.barh(range(len(iPos)),feature_importance[iso][iPos])
        plt.yticks(range(len(iPos)), features_names[iso][iPos])
        plt.tight_layout()
        plt.savefig(ROOT_DIR+"Descriptive_Statistics"+os.sep+"Tree"+os.sep +target_name+'_feature_importance_depth_'+str(__DEPTH__)+'.png')
        plt.close()
        
    cdf_pdf_plot(ytr,target_name+"_ALL",ROOT_DIR+"Descriptive_Statistics"+os.sep+"Tree"+os.sep)
    for j in range(len(leaf_nodes_all)):
        leaf_nodes = leaf_nodes_all[j]
        unique_leaf_nodes = unique_leaf_nodes_all[j]
        for i in range(len(unique_leaf_nodes)):
            node = unique_leaf_nodes[i]
            indices = where(leaf_nodes == node)[0]
            cdf_pdf_plot(ytr[indices],
                         "Level-"+str(j+1)+"_LEAF-"+str(i+1),
                         ROOT_DIR+"Descriptive_Statistics"+os.sep+"Tree"+os.sep)
