import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import seaborn as sns 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve, auc
import statistics
scaler = StandardScaler()
import random
import os 

def  train_model(df,outdir,df_DTA,df_N,df_k, eeg, emg,gglobal):
    ########################## mkdir if ~ exist ##########################
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    #pattern = r'slowoscillations|slowdelta'
    #df = df.loc[:, ~df.columns.str.contains(pattern)]
    #df_DTA = df_DTA.loc[:, ~df_DTA.columns.str.contains(pattern)]
    #df_N   = df_N.loc[:, ~df_N.columns.str.contains(pattern)]
    #df_k   = df_k.loc[:, ~df_k.columns.str.contains(pattern)]

    X  = df.iloc[:,:-2]

    if eeg==True: 
        X  = X.iloc[:,0:48] # EEG
    elif emg==True:  
        X  = X.iloc[:,48:]
    elif gglobal==True: 
        X  = X.iloc[:,np.array([0,9,10,11,12,13,14,15])] 

    y  = df.iloc[:,-2:]
    y['group'] = 1 - y['group']
    X.fillna(0, inplace=True)
    df["subjects"] = df.aid
    df.iloc[[19,31],-1] = "M16"
    df.iloc[[26,27],-1] = "M88"
    df.iloc[[29,30],-1] = "M96"
    df.iloc[[21,32],-1] = "M121"
    subjects   = np.unique(df["subjects"])
    y.aid      = df["subjects"] 

    
    ########################## Train model LOSO ##########################
    print(len(subjects))
    y_proba    = []
    y_pred_all = []
    best_C     = []

    for train_subject in subjects: 
        X_train = X[train_subject != y.aid]
        y_train = y.group[train_subject != y.aid]
        X_test  = X[train_subject == y.aid]
        y_test  = np.array(y.group[train_subject == y.aid])

        scaler.fit(X_train) 

        X_trainn = scaler.transform(X_train)
        X_testn  = scaler.transform(X_test)

        X_train, y_train = shuffle(X_trainn, y_train, random_state=42)
        model = LogisticRegressionCV(penalty="l1",solver="liblinear",cv=5, random_state=0)
        model.fit(X_train, y_train)
        best_C.append(model.C_[0])
        coefficients = model.coef_[0]
    
        y_pred     = model.predict_proba(X_testn)
        y_pred_all.append(y_pred[:,1])
            
    y_test      = np.array(y.group)
    accuracy    = accuracy_score(y_test, np.concatenate(y_pred_all)>0.5)
    print(round(accuracy,2))
    df_acc = pd.DataFrame({'accuracy': [round(accuracy,2)]})
    df_acc.to_csv(outdir+"LOSO.csv")
    
    model_pred = np.concatenate(y_pred_all)>0.5
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    from matplotlib.colors import LinearSegmentedColormap

    cm = confusion_matrix(y_test, np.concatenate(y_pred_all)>0.5)
    rgb_color = [21/255, 96/255, 130/255]

    # Create a custom colormap
    cmap = LinearSegmentedColormap.from_list('custom_cmap', [[1, 1, 1], rgb_color])

    # Plot CM
    width_cm = 5
    height_cm = 5
    dpi = 600
    width_in_inches = width_cm / 2.54 
    height_in_inches = height_cm / 2.54 

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['WT', 'NT'])
    fig, ax = plt.subplots(figsize=(height_in_inches,width_in_inches))
    disp.plot(cmap=cmap, ax=ax, colorbar=False)
    for text in disp.text_.ravel():
        text.set_fontsize(12)
    # Increase the font size
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title("Accuracy:" + str(round(accuracy,2)),fontsize=10)
    plt.tight_layout()
    plt.savefig(outdir+"CM.pdf",dpi=600)

    # Plot ROC curve
    width_cm = 8.98
    height_cm = 12.5
    dpi = 600
    width_in_inches = width_cm / 2.54 
    height_in_inches = height_cm / 2.54 

    plt.figure(figsize=(height_in_inches,width_in_inches))
    fpr, tpr, _ = roc_curve(y_test, np.concatenate(y_pred_all))
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='black', lw=4, label='AUC %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='lightgrey', lw=2, linestyle='--')
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.tight_layout()
    plt.legend(loc="lower right")
    plt.savefig(outdir+"roc.png",dpi=600)

    ########################## Train full model ##########################

    mode_candidates = [x for x in set(best_C) if best_C.count(x) == max([best_C.count(y) for y in set(best_C)])]
    mode = random.choice(mode_candidates)
    print(mode)
    mode = mode_candidates[1]
    width_cm = 7.47
    height_cm = 22.87
    dpi = 600
    width_in_inches = width_cm / 2.54 
    height_in_inches = height_cm / 2.54 

    X_train = X
    X_train = scaler.fit_transform(X_train) 
    y_train = y.group
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    model = LogisticRegression(penalty="l1",solver="liblinear",verbose=True,C=mode,random_state=1)
    model.fit(X_train, y_train)
    coefficients = model.coef_[0]
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': np.abs(coefficients)})
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    feature_importance.iloc[0:13,:].plot(x='Feature', y='Importance', kind='barh', color=[21/255, 96/255, 130/255], figsize=(height_in_inches,width_in_inches))
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(outdir+"features_importance.png",dpi=600)
    
    ########################## Test full model in DTA ##########################
    X_DTA  = df_DTA.iloc[:,:-2]
    if eeg==True: 
        X_DTA  = X_DTA.iloc[:,0:48] # EEG
    elif emg==True:  
        X_DTA  = X_DTA.iloc[:,48:]
    elif gglobal==True: 
        X_DTA  = X_DTA.iloc[:,np.array([0,9,10,11,12,13,14,15])] 

    y_DTA  = df_DTA.iloc[:,-2:]
    X_DTA.fillna(0, inplace=True)

    st_id = [0,7,14,21]
    sl_id = [7,14,21,28]

    Xs_DTA = scaler.fit_transform(X_DTA)

    prob_plot = []  
    group3    = []
    aid_all   = []

    for j in range(len(st_id)):
        y_test     = y_DTA.iloc[st_id[j]:sl_id[j],:].group
        X_test     = Xs_DTA[st_id[j]:sl_id[j],:]
        y_pred     = model.predict_proba(X_test)
        prob_plot.append(y_pred[:,1])
        group3.append(y_test)
        aid_all.append(y_DTA.iloc[st_id[j]:sl_id[j],:].aid.apply(lambda x: x.split('_')[0]))

    ## DTA over time 
    df_DTA["GT"] = np.where(df_DTA["group"]=="Ryan_cleaned_w0","week0",
                            np.where(df_DTA["group"]=="Ryan_cleaned_w2","week2",
                                    np.where(df_DTA["group"]=="Ryan_cleaned_w4","week4","week6")))    

    pdta = pd.DataFrame(np.array(prob_plot)).T.melt()
    pdta["NT"] = pdta.value>0.5
    pdta["Week"] = pdta.variable*2
    pdta["mouse"] = np.concatenate(aid_all)
    palette = "dark"

    width_cm = 8.13
    height_cm = 15.09
    dpi = 600

    width_in_inches = width_cm / 2.54 
    height_in_inches = height_cm / 2.54 

    plt.figure(figsize=(height_in_inches,width_in_inches))
    sns.lineplot(x='Week', y='value', hue='mouse', data=pdta, marker='o', palette=['black'] * len(pdta['mouse']),legend=False)
    plt.ylabel('P(NT)', fontsize=10)
    plt.xlabel("Week", fontsize=10)
    plt.axhline(0.5, color="lightgrey", linestyle="dashed")
    plt.tight_layout()
    plt.savefig(outdir + "DTA_test.png", dpi=600, bbox_inches='tight')  # Save the figure with adjusted legend position

    ########################## Test full model i Noriaki data ##########################

    df_N['experiment'] = df_N['aid'].str.split('_').str[1].astype(float)
    df_N = df_N.iloc[np.where((df_N.experiment==5)|(df_N.experiment==4))[0],:] 
    X_N  = df_N.iloc[:,:-3]
    if eeg==True: 
        X_N  = X_N.iloc[:,0:48] # EEG
    elif emg==True:  
        X_N  = X_N.iloc[:,48:] 
    elif gglobal==True: 
        X_N  = X_N.iloc[:,np.array([0,9,10,11,12,13,14,15])] 
    
    y_N  = df_N.iloc[:,-3:]
    X_N.fillna(0, inplace=True)

    Xs_N = scaler.fit_transform(X_N)

    y_test     = y_N.group
    X_test     = Xs_N
    y_pred     = model.predict_proba(X_test)
    prob_plot  = y_pred[:,1]

    pN = pd.DataFrame(np.array(prob_plot)).T.melt()
    pN["NT"] = pN.value>0.5
    pN["mouse"] = y_N.iloc[:,0].reset_index().aid
    pN["y_true"] =y_N.iloc[:,1].reset_index().group
    pN.to_csv(outdir+"outdir_preds_Noriaki.csv")

    ########################## Test full model i Ceremedy data ##########################

    X_k  = df_k.iloc[:,:-2]
    if eeg==True: 
        X_k  = X_k.iloc[:,0:48] # EEG
    elif emg==True:  
        X_k  = X_k.iloc[:,48:]  # EMG
    elif gglobal==True: 
        X_k  = X_k.iloc[:,np.array([0,9,10,11,12,13,14,15])] 
    y_k  = df_k.iloc[:,-2:]
    y_k['group'] = 1 - y_k['group']

    X_k.fillna(0, inplace=True)

    Xs_k = scaler.fit_transform(X_k)

    y_test     = y_k.group
    X_test     = Xs_k
    y_pred     = model.predict_proba(X_test)
    prob_plot  = y_pred[:,1]

    pk = pd.DataFrame(np.array(prob_plot)).T.melt()
    pk["NT"] = pk.value>0.5
    pk["mouse"] = y_k.iloc[:,0].reset_index().aid
    pk["y_true"] =y_k.iloc[:,1].reset_index().group
    pk.to_csv(outdir+"outdir_preds_ceremedy.csv")

    plt.close("all")

    return model_pred


def mcnemartest_(model_a_predictions,model_b_predictions,true_labels):
    from statsmodels.stats.contingency_tables import mcnemar 

    # Calculate the contingency table
    a = np.sum((model_a_predictions == true_labels) & (model_b_predictions == true_labels))
    b = np.sum((model_a_predictions == true_labels) & (model_b_predictions != true_labels))
    c = np.sum((model_a_predictions != true_labels) & (model_b_predictions == true_labels))
    d = np.sum((model_a_predictions != true_labels) & (model_b_predictions != true_labels))

    contingency_table = [[a, b], [c, d]]

    # Perform McNemar's test
    print(contingency_table)
    print(mcnemar(contingency_table, exact=True)) 



################################# Global model ####################################
# outdir  = "/Users/qgf169/Documents/python/Usleep/results/experimentA_global/"
# df      = pd.read_csv("/Users/qgf169/Documents/python/Usleep/results/kornum/model_global.csv")
# df_DTA  = pd.read_csv("/Users/qgf169/Documents/python/Usleep/results/DTA_v2/model_global.csv")
# df_N    = pd.read_csv("/Users/qgf169/Documents/python/Usleep/results/noriaki_v2/model_global.csv")
# df_C    = pd.read_csv("/Users/qgf169/Documents/python/Usleep/results/ceremedy/model_global.csv")

# model_global = train_model(df,outdir,df_DTA,df_N,df_C,eeg=False,emg=False,gglobal=True)

################################# Ensemble model ####################################
outdir  = "/Users/qgf169/Documents/python/Usleep/results/experimentA_ensemble/"
df      = pd.read_csv("/Users/qgf169/Documents/python/Usleep/results/kornum/model_baseline.csv")
df_DTA  = pd.read_csv("/Users/qgf169/Documents/python/Usleep/results/DTA_v2/model_baseline.csv")
df_N    = pd.read_csv("/Users/qgf169/Documents/python/Usleep/results/noriaki_v2/model_baseline.csv")
df_C    = pd.read_csv("/Users/qgf169/Documents/python/Usleep/results/HCRTKO_K/model_baseline.csv")

model_ensemble = train_model(df,outdir,df_DTA,df_N,df_C,eeg=False,emg=False,gglobal=False)

#################################   EEG model   ####################################
outdir  = "/Users/qgf169/Documents/python/Usleep/results/experimentA_EEG/"
df      = pd.read_csv("/Users/qgf169/Documents/python/Usleep/results/kornum/model_baseline.csv")
df_DTA  = pd.read_csv("/Users/qgf169/Documents/python/Usleep/results/DTA_v2/model_baseline.csv")
df_N    = pd.read_csv("/Users/qgf169/Documents/python/Usleep/results/noriaki_v2/model_baseline.csv")
df_C    = pd.read_csv("/Users/qgf169/Documents/python/Usleep/results/ceremedy/model_baseline.csv")

model_EEG = train_model(df,outdir,df_DTA,df_N,df_C,eeg=True,emg=False,gglobal=False)

#################################   EMG model   ####################################
outdir  = "/Users/qgf169/Documents/python/Usleep/results/experimentA_EMG/"
df      = pd.read_csv("/Users/qgf169/Documents/python/Usleep/results/kornum/model_baseline.csv")
df_DTA  = pd.read_csv("/Users/qgf169/Documents/python/Usleep/results/DTA_v2/model_baseline.csv")
df_N    = pd.read_csv("/Users/qgf169/Documents/python/Usleep/results/noriaki_v2/model_baseline.csv")
df_C    = pd.read_csv("/Users/qgf169/Documents/python/Usleep/results/ceremedy/model_baseline.csv")

model_EMG = train_model(df,outdir,df_DTA,df_N,df_C,eeg=False,emg=True,gglobal=False)

#################################   both model   ####################################
outdir  = "/Users/qgf169/Documents/python/Usleep/results/experimentA_both/"
df      = pd.read_csv("/Users/qgf169/Documents/python/Usleep/results/kornum/model_both.csv")
df_DTA  = pd.read_csv("/Users/qgf169/Documents/python/Usleep/results/DTA_v2/model_both.csv")
df_N    = pd.read_csv("/Users/qgf169/Documents/python/Usleep/results/noriaki_v2/model_both.csv")
df_C    = pd.read_csv("/Users/qgf169/Documents/python/Usleep/results/ceremedy/model_both.csv")

model_both = train_model(df,outdir,df_DTA,df_N,df_C,eeg=False,emg=False,gglobal=False)

#### McNemar test ####

true_labels = 1-df.group

print("(1) Model global == model EEG")
mcnemartest_(model_global,model_EEG,true_labels)

print("(2) Model global == model EMG")
mcnemartest_(model_global,model_EMG,true_labels)

print("(3) Model global == model EEG and EMG")
mcnemartest_(model_global,model_both,true_labels)

print("# (4) Model global == model ensemble")
mcnemartest_(model_global,model_ensemble,true_labels)

print("# (5) Model EEG == model EMG")
mcnemartest_(model_EEG,model_EMG,true_labels)

print("(6) Model EEG == model EMG + EMG")
mcnemartest_(model_EEG,model_both,true_labels)

print("# (7) Model EEG == model ensemble")
mcnemartest_(model_EEG,model_ensemble,true_labels)

print("(8) Model EMG == model both") 
mcnemartest_(model_EMG,model_both,true_labels)

print("(9) Model EMG == model ensemble ")
mcnemartest_(model_EMG,model_ensemble,true_labels)

print("(10) Model EEG + EMG == model ensemble")
mcnemartest_(model_both,model_ensemble,true_labels)
