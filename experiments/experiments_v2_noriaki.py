import numpy as np
import os
import usleep
import typing
from psg_utils.dataset.sleep_study import SleepStudy
from utime.bin.predict_one import get_sleep_study
import h5py 
import pandas as pd 
import matplotlib.pyplot as plt 
import glob 
import sys
import seaborn as sns 
import matplotlib.cm as cm  # Import Matplotlib's colormap module
from scipy.signal import welch
from matplotlib.colors import LogNorm
import scipy.signal
import matplotlib.colors as colors
from utime.bin.evaluate import get_and_load_model, get_sequencer
from utime.hyperparameters import YAMLHParams
from utime import Defaults
from utime.bin.predict import get_datasets, run_pred
from scipy.stats import iqr
from scipy import signal

def calc_IQR(eeg_signal):
    signal_iqr     = iqr(eeg_signal)
    signal_median  = np.median(eeg_signal)
    upper_bound    = signal_median + 20 * signal_iqr
    lower_bound    = signal_median - 20 * signal_iqr
    extreme_values = np.where((eeg_signal > upper_bound) | (eeg_signal < lower_bound))
    
    return extreme_values

def process_batch(X, y,n_channels,batch_shape):
        # Cast and reshape arrays
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("Expected numpy array inputs.")
        X = np.squeeze(X).astype(Defaults.PSG_DTYPE)

        if n_channels == 1:
            X = np.expand_dims(X, -1)
        y = np.expand_dims(y.astype(Defaults.HYP_DTYPE).squeeze(), -1)

        expected_dim = len(batch_shape)
        if X.ndim == expected_dim-1:
            if batch_shape[1] == 1:
                X = np.expand_dims(X, 1)
            else:
                X, y = np.expand_dims(X, 0), np.expand_dims(y, 0)
        elif X.ndim != expected_dim:
            raise RuntimeError("Dimensionality of X is {} (shape {}), but "
                               "expected {}".format(X.ndim,
                                                    X.shape,
                                                    expected_dim))
       
        return X,y


def bout_analysis(ss,n_classes):
    dfss         = pd.DataFrame(ss)
    dfss.columns = ["hyp"]
    df         = dfss.copy()
    df['bout'] = (df.hyp!=df.hyp.shift(1)).cumsum() 
    df = df.merge(df.groupby('bout').hyp.count().reset_index().rename(columns={'hyp':'len_bout'}), how='left')   
    df["len_bout_sec"]    = df["len_bout"]*4

    f1        = []
    f2        = []
    f3        = []
    bout_int1 = []
    bout_int2 = []
    bout_int3 = []
    bout_int4 = []
    bout_int5 = []

    # bouts 
    for j in range(n_classes):
        bouts      = 0 
        x1         = 0
        bouts_bar  = 0
        ss = pd.DataFrame({"ss": np.where(dfss.hyp == j, 1, 0)})
        ss['bout'] = (ss.ss!=ss.ss.shift(1)).cumsum() 
        ss = ss.merge(ss.groupby('bout').ss.count().reset_index().rename(columns={'ss':'len_bout'}), how='left')   
        ss["len_bout_sec"]    = ss["len_bout"]*4
        ss['inact_bouts_cat'] = np.where(ss.len_bout_sec.between(1,4), '4sec', 
                                        np.where(ss.len_bout_sec.between(4,32), '4sec-32sec',
                                            np.where(ss.len_bout_sec.between(32,60), '32sec-1min',
                                                np.where(ss.len_bout_sec.between(60,5*60), '1min-5min',
                                                        np.where(ss.len_bout_sec>60*5, '>5min',np.nan)))))
        
        bouts  = ss[["ss","bout","len_bout","len_bout_sec","inact_bouts_cat"]].drop_duplicates()

        if len(bouts.ss.unique())>1:                                    
            x1 = bouts.groupby("inact_bouts_cat").apply(lambda x: np.sum(x['ss']*x['len_bout_sec'])) # sums across all cats 
            bouts_bar = (100*x1/np.sum(x1)).reset_index().rename(columns={0:'norm_frequency'})
            bouts = bouts[bouts.ss==1].drop_duplicates()
            
            f1.append(bouts.len_bout_sec.mean())
            f2.append(bouts.len_bout.sum()/df.shape[0]*100)
            f3.append(bouts.bout.count()/(df.shape[0]*4/3600)) #counts pr. h 

            if '4sec' in bouts_bar.inact_bouts_cat.unique(): 
                bout_int1.append(float(bouts_bar[bouts_bar.inact_bouts_cat=='4sec'].norm_frequency))
            else: 
                bout_int1.append(0)

            if '4sec-32sec' in bouts_bar.inact_bouts_cat.unique(): 
                bout_int2.append(float(bouts_bar[bouts_bar.inact_bouts_cat=='4sec-32sec'].norm_frequency))
            else: 
                bout_int2.append(0)

            if '32sec-1min' in bouts_bar.inact_bouts_cat.unique(): 
                bout_int3.append(float(bouts_bar[bouts_bar.inact_bouts_cat=='32sec-1min'].norm_frequency))
            else: 
                bout_int3.append(0)

            if '1min-5min' in bouts_bar.inact_bouts_cat.unique(): 
                bout_int4.append(float(bouts_bar[bouts_bar.inact_bouts_cat=='1min-5min'].norm_frequency))
            else: 
                bout_int4.append(0)
            
            if '>5min' in bouts_bar.inact_bouts_cat.unique(): 
                bout_int5.append(float(bouts_bar[bouts_bar.inact_bouts_cat=='>5min'].norm_frequency))
            else: 
                bout_int5.append(0)

        else: 
            f1.append(np.nan)
            f2.append(np.nan)
            f3.append(np.nan)
            bout_int1.append(np.nan)
            bout_int2.append(np.nan)
            bout_int3.append(np.nan)
            bout_int4.append(np.nan)
            bout_int5.append(np.nan)
                
    return np.array(f1),np.array(f2),np.array(f3),np.array(bout_int1),np.array(bout_int2),np.array(bout_int3),np.array(bout_int4),np.array(bout_int5) 


def power_(x,low,high):
    from scipy import signal
    from scipy.integrate import simps

    if x.shape[0]>0:
        # Define window length (4 seconds)
        sf  = 128 # confirm that its 128
        win = 4 * sf
        freqs, psd = signal.welch(x, sf, nperseg=win)

        # Find intersecting values in frequency vector
        idx_ = np.logical_and(freqs >= low, freqs <= high)
        freq_res = freqs[1] - freqs[0]  # = 1 / 4 = 0.25
        
        # Compute the absolute power by approximating the area under the curve
        delta_power = simps(psd[:,idx_], dx=freq_res)
        total_power = simps(psd, dx=freq_res)
        delta_rel_power = delta_power / total_power
    else: 
        delta_rel_power = np.nan    
    return delta_rel_power 

def power_analysis(ss,xEEG,n_classes):
    bands = {'slowoscillations': [0.5, 1.5], 
         'slowdelta': [1, 2.25],
         'fastdelta': [2.5, 4],
         'slowtheta': [5, 8],
         'fasttheta': [8, 10],
         'alpha': [9, 14],
         'beta': [14, 30]}#,
         #'gamma': [30, 49]}
    
    
    #calculate rms from EMG in the different states: 
    xEEG             = xEEG.reshape(xEEG.shape[0]*xEEG.shape[1],xEEG.shape[2],xEEG.shape[3])
    fs               = 128 
    slowoscillations = []
    slowdelta        = []
    fastdelta        = []
    slowtheta        = []
    fasttheta        = []
    alpha            = []
    beta             = []
    #gamma            = []
    

    for jj in range(n_classes): 
      
        idx  = np.where(ss==jj)[0]
        p   = power_(xEEG[idx,:,0],bands["slowoscillations"][0],bands["slowoscillations"][1])
        slowoscillations.append(np.mean(p))
    
        p    = power_(xEEG[idx,:,0],bands["slowdelta"][0],bands["slowdelta"][1])
        slowdelta.append(np.mean(p))

        p    = power_(xEEG[idx,:,0],bands["fastdelta"][0],bands["fastdelta"][1])
        fastdelta.append(np.mean(p))

        p   = power_(xEEG[idx,:,0],bands["slowtheta"][0],bands["slowtheta"][1])
        slowtheta.append(np.mean(p))
            
        p    = power_(xEEG[idx,:,0],bands["fasttheta"][0],bands["fasttheta"][1])
        fasttheta.append(np.mean(p))
        
        p    = power_(xEEG[idx,:,0],bands["alpha"][0],bands["alpha"][1])
        alpha.append(np.mean(p))

        p    = power_(xEEG[idx,:,0],bands["beta"][0],bands["beta"][1])
        beta.append(np.mean(p))
   
    return np.array(slowoscillations), np.array(slowdelta),np.array(fastdelta), np.array(slowtheta), np.array(fasttheta),np.array(alpha), np.array(beta)#, np.array(gamma)



def rms_fcn(xEMG,ss,n_classes):
    #calculate rms from EMG in the different states: 
    xEMG = xEMG.reshape(xEMG.shape[0]*xEMG.shape[1],xEMG.shape[2],xEMG.shape[3])
    
    rms_vec = np.empty(n_classes)
    rms_vec.fill(np.nan)
    for k in range(0,n_classes): 
        idx = np.where(ss==k)[0]
        if idx.size != 0:
            rms_vec[k] = np.sqrt(np.mean(xEMG[idx,:,0] **2, axis=1)).mean()
        
    return rms_vec


def features(hyp,xEEG,xEMG,n_classes,mask):
    rms_vec = rms_fcn(xEMG,hyp,n_classes)
    f1,f2,f3,f4,f5,f6,f7,f8 = bout_analysis(hyp,n_classes)
    #slow_o, slow_d,fast_d,slow_t,fast_t,alpha,beta,gamma = power_analysis(hyp,xEEG,n_classes)
    slow_o, slow_d,fast_d,slow_t,fast_t,alpha,beta = power_analysis(hyp,xEEG,n_classes)

    
    assert len(rms_vec)==n_classes 
    assert len(f1)==n_classes 
    assert len(f2)==n_classes 
    assert len(f3)==n_classes 
    assert len(f4)==n_classes 
    assert len(f5)==n_classes 
    assert len(f6)==n_classes 
    assert len(f7)==n_classes 
    assert len(f8)==n_classes 
    assert len(slow_o)==n_classes 
    assert len(slow_d)==n_classes 
    assert len(fast_d)==n_classes 
    assert len(slow_t)==n_classes 
    assert len(fast_t)==n_classes 
    assert len(alpha)==n_classes 
    assert len(beta)==n_classes 
    #assert len(gamma)==n_classes 

    mask = mask.astype(bool)
    feature_vec = np.concatenate([rms_vec[mask],f1[mask],f2[mask],f3[mask],f4[mask],f5[mask],
                            f6[mask],f7[mask],f8[mask],slow_o[mask],slow_d[mask],fast_d[mask],
                            slow_t[mask],fast_t[mask],alpha[mask],beta[mask]])#,gamma[mask]])
    
    rms_n = ["rms_" + str(i) for i in np.where(mask==True)[0]]
    f1_n  = ["avr_bout_length_" + str(i) for i in np.where(mask==True)[0]]
    f2_n  = ["total_time_" + str(i) for i in np.where(mask==True)[0]]
    f3_n  = ["counts_h_" + str(i) for i in np.where(mask==True)[0]]
    f4_n  = ["4sec_" + str(i) for i in np.where(mask==True)[0]]
    f5_n  = ["4sec_32sec_" + str(i) for i in np.where(mask==True)[0]]
    f6_n  = ["32sec_1min_" + str(i) for i in np.where(mask==True)[0]]
    f7_n  = ["1min_5min_" + str(i) for i in np.where(mask==True)[0]]
    f8_n  = [">5min_" + str(i) for i in np.where(mask==True)[0]]
    f9_n  = ["slowoscillations_" + str(i) for i in np.where(mask==True)[0]]
    f10_n  = ["slowdelta_" + str(i) for i in np.where(mask==True)[0]]
    f11_n  = ["fastdelta_" + str(i) for i in np.where(mask==True)[0]]
    f12_n  = ["slowtheta_" + str(i) for i in np.where(mask==True)[0]]
    f13_n  = ["fasttheta_" + str(i) for i in np.where(mask==True)[0]]
    f14_n  = ["alpha_" + str(i) for i in np.where(mask==True)[0]]
    f15_n  = ["beta_" + str(i) for i in np.where(mask==True)[0]]
   # f16_n  = ["gamma_" + str(i) for i in np.where(mask==True)[0]]

    combined_list = (rms_n + f1_n + f2_n + f3_n + f4_n + 
                     f5_n + f6_n + f7_n + f8_n + f9_n + 
                     f10_n + f11_n + f12_n + f13_n + f14_n+ 
                     f15_n )#+ f16_n)
    
    return feature_vec,combined_list

def post_pred_mix(pred):
    pred = pred.reshape(-1, 3)
    
    p_ss  = []
    for k in range(len(pred)):
        if pred[k,0] > 0.8: # wake  
            p_ss.append(0)
        elif pred[k,1] > 0.8: # nrem
            p_ss.append(1)
        elif pred[k,2] > 0.8: # rem
            p_ss.append(2)
        elif min(pred[k,:]) > 0.1: #triplet  
            p_ss.append(6)
        elif pred[k,:].argmin()==2: # duplet (p_wn)
            p_ss.append(3)
        elif pred[k,:].argmin()==0: # duplet (p_nr)
            p_ss.append(5)
        elif pred[k,:].argmin()==1: # duplet (p_wr)
            p_ss.append(4)
        else:
            raise ValueError("mixing state error") 
            
    p_ss = np.array(p_ss)
    return p_ss

def post_pred_dis(pred_eeg,pred_emg,n_classes):

    ss_joint = np.empty(len(pred_eeg))
    ss_joint.fill(np.nan)

    if n_classes == 3 or n_classes == 9:
        ints = 3 
    else:
        ints = 7
   
    # Loop through each combination of ss_EEG and ss_EMG
    for i in range(ints):
        for j in range(ints):
            # Calculate the index for the current combination
            index = i * ints + j
            # Assign the corresponding value to ss_joint
            ss_joint[np.where((pred_eeg==i)&(pred_emg==j))[0]] = index
    
    #df = pd.DataFrame([pred_eeg,pred_emg,ss_joint]).T
    #df.columns = ["pEEG","pEMG","pEEG_EMG"]
    return ss_joint 

def run_exp(n_classes,mixed,dis,pred_EEG_all,pred_EMG_all,agreement_exp,disagreement_exp,output_dir,x_EEG_all,x_EMG_all,group_all,aid_all,ss_n,both):
    print("here")
    if (mixed==0) & (dis==0) & (disagreement_exp==0) & (agreement_exp==0) & (ss_n==False) & (both==True):  
        features_all = []

        for j in range(len(x_EEG_all)): 
            hyp_EEG = pred_EEG_all[j].reshape(-1, 3).argmax(axis=1)
            mask = np.ones(n_classes)
            feature_combined,combined_list = features(hyp_EEG,x_EEG_all[j],x_EMG_all[j],n_classes,mask)
            features_all.append(feature_combined)
        # save as a dataframe  
        df = pd.DataFrame(features_all)
        df.columns  = ['eeg_emg' + item for item in combined_list]
        df["aid"]   = aid_all
        df["group"] = group_all 
        df.to_csv(output_dir + "model_both.csv", index=False)

    if (mixed==0) & (dis==0) & (disagreement_exp==0) & (agreement_exp==0) & (ss_n==True):  
        features_all = []

        for j in range(len(x_EEG_all)): 
            hyp_EEG = pred_EEG_all[j].reshape(-1, 3).argmax(axis=1)
            hyp_EEG = np.zeros(len(hyp_EEG))
            mask = np.ones(n_classes)
            feature_combined,combined_list = features(hyp_EEG,x_EEG_all[j],x_EMG_all[j],n_classes,mask)
            features_all.append(feature_combined)

        # save as a dataframe  
        df = pd.DataFrame(features_all)
        df.columns  = ['eeg_' + item for item in combined_list]
        df["aid"]   = aid_all
        df["group"] = group_all 
        df.to_csv(output_dir + "model_global.csv", index=False)


    if (mixed==0) & (dis==0) & (disagreement_exp==0) & (agreement_exp==0) & (ss_n==False) & (both==False): 
        
        features_all = []

        for j in range(len(x_EEG_all)): 
            hyp_EEG = pred_EEG_all[j].reshape(-1, 3).argmax(axis=1)
            hyp_EMG = pred_EMG_all[j].reshape(-1, 3).argmax(axis=1)
            mask = np.ones(n_classes)
            feature_eeg,combined_list = features(hyp_EEG,x_EEG_all[j],x_EMG_all[j],n_classes,mask)
            feature_emg,combined_list = features(hyp_EMG,x_EEG_all[j],x_EMG_all[j],n_classes,mask)
            feature_combined = np.concatenate((feature_eeg, feature_emg))
            features_all.append(feature_combined)

        # save as a dataframe  
        df = pd.DataFrame(features_all)
        df.columns  = ['eeg_' + item for item in combined_list]+['emg_' + item for item in combined_list]
        df["aid"]   = aid_all
        df["group"] = group_all 
        df.to_csv(output_dir + "model_baseline.csv", index=False)

    elif (mixed==1) & (dis==0) & (disagreement_exp==0) & (agreement_exp==0) & (ss_n==False) & (both==False):
        
        features_all = []

        for j in range(len(x_EEG_all)): 
            post_pred_EEG = post_pred_mix(pred_EEG_all[j])
            post_pred_EMG = post_pred_mix(pred_EMG_all[j])
            mask = np.ones(n_classes)
            feature_eeg,combined_list = features(post_pred_EEG,x_EEG_all[j],x_EMG_all[j],n_classes,mask)
            feature_emg,combined_list = features(post_pred_EMG,x_EEG_all[j],x_EMG_all[j],n_classes,mask)
            feature_combined = np.concatenate((feature_eeg, feature_emg))
            features_all.append(feature_combined)

        # save as a dataframe  
        df = pd.DataFrame(features_all)
        df.columns  = ['eeg_' + item for item in combined_list]+['emg_' + item for item in combined_list]
        df["aid"]   = aid_all
        df["group"] = group_all 
        df.to_csv(output_dir + "model1.csv", index=False)
        

    elif (mixed==0) & (dis==1) & (disagreement_exp==0) & (agreement_exp==0) & (ss_n==False)& (both==False):
        
        features_all = []

        for j in range(len(x_EEG_all)): 
            hyp_EEG = pred_EEG_all[j].reshape(-1, 3).argmax(axis=1)
            hyp_EMG = pred_EMG_all[j].reshape(-1, 3).argmax(axis=1)
            joint_ss = post_pred_dis(hyp_EEG,hyp_EMG,n_classes)
            mask = np.ones(n_classes)
            assert ~np.isnan(joint_ss).any()
            features_,combined_list = features(joint_ss,x_EEG_all[j],x_EMG_all[j],n_classes,mask)
            features_all.append(features_)
            
        # save as a dataframe  
        df = pd.DataFrame(features_all)
        df.columns  = combined_list
        df["aid"]   = aid_all
        df["group"] = group_all 
        df.to_csv(output_dir + "model2.csv", index=False)

    elif (mixed==1) & (dis==1) & (disagreement_exp==0) & (agreement_exp==0) & (ss_n==False)& (both==False):
        features_all = []

        for j in range(len(x_EEG_all)): 
            post_pred_EEG = post_pred_mix(pred_EEG_all[j])
            post_pred_EMG = post_pred_mix(pred_EMG_all[j])
            joint_ss = post_pred_dis(post_pred_EEG,post_pred_EMG,n_classes)
            assert ~np.isnan(joint_ss).any()
            mask = np.ones(n_classes)
            features_,combined_list = features(joint_ss,x_EEG_all[j],x_EMG_all[j],n_classes,mask)
            features_all.append(features_)
        
        # save as a dataframe  
        df = pd.DataFrame(features_all)
        df.columns  = combined_list
        df["aid"]   = aid_all
        df["group"] = group_all 
        df.to_csv(output_dir + "model3.csv", index=False)


    elif (mixed==1) & (dis==1) & (disagreement_exp==0) & (agreement_exp==1) & (ss_n==False)& (both==False):
        features_all = []

        for j in range(len(x_EEG_all)): 
            post_pred_EEG = post_pred_mix(pred_EEG_all[j])
            post_pred_EMG = post_pred_mix(pred_EMG_all[j])
            joint_ss = post_pred_dis(post_pred_EEG,post_pred_EMG,n_classes)
            assert ~np.isnan(joint_ss).any()
            # filter away disagreement 
            nums = np.array([0, 8, 16, 24,32,40,48])
            mask = np.zeros(n_classes)
            mask[nums] = 1
            features_,combined_list = features(joint_ss,x_EEG_all[j],x_EMG_all[j],n_classes,mask)
            features_all.append(features_)
       
        # save as a dataframe  
        df = pd.DataFrame(features_all)
        df.columns  = combined_list
        df["aid"]   = aid_all
        df["group"] = group_all 
        df.to_csv(output_dir + "model4.csv", index=False)

    elif (mixed==1) & (dis==1) & (disagreement_exp==1) & (agreement_exp==0) & (ss_n==False)& (both==False):
        features_all = []

        for j in range(len(x_EEG_all)): 
            post_pred_EEG = post_pred_mix(pred_EEG_all[j])
            post_pred_EMG = post_pred_mix(pred_EMG_all[j])
            joint_ss = post_pred_dis(post_pred_EEG,post_pred_EMG,n_classes)
            assert ~np.isnan(joint_ss).any()
             # filter away disagreement 
            nums = np.array([0, 8, 16, 24,32,40,48])
            mask = np.ones(n_classes)
            mask[nums] = 0
            features_,combined_list = features(joint_ss,x_EEG_all[j],x_EMG_all[j],n_classes,mask)
            features_all.append(features_)

        # save as a dataframe  
        df = pd.DataFrame(features_all)
        df.columns  = combined_list
        df["aid"]   = aid_all
        df["group"] = group_all 
        df.to_csv(output_dir + "model5.csv", index=False)

##################################### Parameters #####################################

output_dir  =  "/Users/qgf169/Documents/python/Usleep/results/noriaki_v2/"
groups      = ['Noriaki_cleaned_WT',"Noriaki_cleaned"]
splits      = ["TRAIN","TEST"]
#splits      = ["TEST"]
elec        = ["EEG1","EEG2"]

##################################### Load Data  #####################################
data_dir    = "/Users/qgf169/Documents/python/Usleep/usleepletsgo_nt_all/" 
data        = h5py.File(data_dir+"processed_data.h5")

##################################### Load Model #####################################

# EEG model 
project_dir =  "/Users/qgf169/Documents/python/Usleep/usleepletsgo_nt_EEG/" 
os.chdir(project_dir)
Defaults.PROJECT_DIRECTORY = project_dir 
hparams   =  YAMLHParams(Defaults.get_hparams_path(project_dir))
model_eeg,model_f = get_and_load_model(project_dir,
                    hparams,
                    weights_file_name="",
                    clear_previous=True)

# EMG model 
project_dir =  "/Users/qgf169/Documents/python/Usleep/usleepletsgo_nt_EMG/" 
os.chdir(project_dir)
Defaults.PROJECT_DIRECTORY = project_dir 
hparams   =  YAMLHParams(Defaults.get_hparams_path(project_dir))
model_emg,model_f = get_and_load_model(project_dir,
                    hparams,
                    weights_file_name="",
                    clear_previous=True)
# both 
project_dir =  "/Users/qgf169/Documents/python/Usleep/usleepletsgo_nt_all/" 
os.chdir(project_dir)
Defaults.PROJECT_DIRECTORY = project_dir 
hparams   =  YAMLHParams(Defaults.get_hparams_path(project_dir))
model_all,model_f = get_and_load_model(project_dir,
                    hparams,
                    weights_file_name="",
                    clear_previous=True)

##################################### Predictions #####################################

# predictions across animals 
pred_all_all = []
pred_EEG_all = []
pred_EMG_all = []
x_EEG_all    = []
x_EMG_all    = []
group_all    = []
aid_all      = []

for k in range(len(groups)): # across groups
    print(groups[k])

    for i in range(len(splits)): # across datasplits 
        print(splits[i])
        aid = list(data[groups[k]][splits[i]].keys())

        for j in range(len(aid)): # across animals 
            if aid[j]!='12':
                print(aid[j])
                batch_shape = [128,11,512,1]

                # load hyp, eeg2 and emg 
                sig  = data[groups[k]][splits[i]][aid[j]]["PSG"][elec[0]][:]
                emg  = data[groups[k]][splits[i]][aid[j]]["PSG"]['EMG'][:]

                mod_ = len(sig)%512
                if mod_>0:
                    print("here")
                    sig  = sig[0:-mod_]
                    emg  = emg[0:-mod_]
                
                # clean data 
                # iqr eeg 
                iqr_eeg  = calc_IQR(sig)
                iqr_emg  = calc_IQR(emg)
                null_array = np.zeros((len(sig)))
                null_array[iqr_eeg] = 1
                null_array[iqr_emg] = 1

                # truncate (n to epochs of 4, border to makes series of 11)
                n    = len(sig) // 512     
                hyp  = np.zeros(n)         
                reshaped_sig  = sig.reshape(n,512,1)
                reshaped_EMG  = emg.reshape(n,512,1)
                reshaped_art  = null_array.reshape(n,512)

                # only extract clean epochs 
                count_ones = np.sum(reshaped_art, axis=1)
                print(np.unique(count_ones))
                print(len(np.where(count_ones>=1)[0]))
                #reshaped_sig  = reshaped_sig[np.where(count_ones<1)[0],:,:]
                #reshaped_EMG  = reshaped_EMG[np.where(count_ones<1)[0],:,:]
                #hyp     = hyp[np.where(count_ones<1)[0]]
                shape   = [-1] + batch_shape[1:]
                border  = reshaped_sig.shape[0]%shape[1]

                if border:
                    reshaped_sig = reshaped_sig[:-border]
                    reshaped_EMG = reshaped_EMG[:-border]
                    reshaped_art = reshaped_art[:-border]
                    hyp = hyp[:-border]
                else:
                    hyp = hyp[:]

                x_eeg    = reshaped_sig.reshape(shape)
                x_emg    = reshaped_EMG.reshape(shape)
                y        = hyp.reshape(shape[:2] + [1])
                x_eeg,y  = process_batch(x_eeg, y,1,batch_shape)
                x_emg,y  = process_batch(x_emg, y,1,batch_shape)
                
                # pred eeg
                pred_EEG = model_eeg.predict_on_batch(x_eeg)
                pred_EEG_all.append(pred_EEG)
                x_EEG_all.append(x_eeg)

                # pred emg 
                pred_EMG = model_emg.predict_on_batch(x_emg) 
                pred_EMG_all.append(pred_EMG)
                x_EMG_all.append(x_emg)

                # pred all 
                reshaped_all = np.concatenate([x_eeg,x_emg],axis=3)
                pred_all = model_all.predict_on_batch(reshaped_all) 
                pred_all_all.append(pred_all)

                # save information 
                group_all.append(k)
                aid_all.append(aid[j])
                width_cm = 11.51
                height_cm = 4.47

                dpi = 600

                width_in_inches = width_cm / 2.54 
                height_in_inches = height_cm / 2.54 

                # plot 
                folder_path="/Users/qgf169/Documents/python/Usleep/results/noriaki/"+aid[j] + "/"
                if not os.path.exists(folder_path): 
                    os.makedirs(folder_path)
                fs = 128 
                fig, axs = plt.subplots(1, 2, figsize=(width_in_inches, height_in_inches))
                freqs, psd = signal.welch(sig, fs)
                axs[0].plot(freqs,psd,color=[21/255, 96/255, 130/255])
                axs[0].set_xlim([0, 35])
                axs[0].set_title("Power EEG",fontsize=10)
                axs[0].tick_params(axis='y', which='both', left=False, labelleft=False)
                freqs, psd = signal.welch(emg, fs)
                axs[1].plot(freqs,psd,color=[21/255, 96/255, 130/255])
                axs[1].set_title("Power EMG",fontsize=10)
                axs[1].set_xlim([0, 40])
                axs[1].tick_params(axis='y', which='both', left=False, labelleft=False)
                plt.tight_layout()
                plt.savefig(folder_path+"psd.png",dpi=300)
                plt.close()

                width_cm = 11.51
                height_cm = 6.47
                dpi = 600

                width_in_inches = width_cm / 2.54 
                height_in_inches = height_cm / 2.54 

                fig, axs = plt.subplots(2, 1, figsize=(width_in_inches, height_in_inches),sharex=True)
                t = np.arange(len(sig)) / fs
                axs[0].plot(t[0:128*60*60],sig[0:128*60*60],color=[21/255, 96/255, 130/255])
                axs[0].set_ylabel("EEG")
                axs[0].tick_params(axis='y', which='both', left=False, labelleft=False)
                axs[1].set_xlim([0,3600])
                axs[1].plot(t[0:128*60*60],emg[0:128*60*60],color=[21/255, 96/255, 130/255])
                axs[1].set_ylabel("EMG")
                axs[1].set_xlabel("Time [s]")
                axs[1].set_xlim([0,3600])
                axs[1].tick_params(axis='y', which='both', left=False, labelleft=False)

                plt.tight_layout()
                plt.savefig(folder_path+"signals.png",dpi=300)
                plt.close()

# experiment 0 GLOBAL MODEL 
n_classes        = 1
mixed            = 0
dis              = 0
agreement_exp    = 0
disagreement_exp = 0
run_exp(n_classes,mixed,dis,pred_EEG_all,pred_EMG_all,agreement_exp,disagreement_exp,output_dir,x_EEG_all,x_EMG_all,group_all,aid_all,ss_n=True,both=False)


# experiment 1 baseline model - ensemble model  
n_classes        = 3
mixed            = 0
dis              = 0
agreement_exp    = 0
disagreement_exp = 0
run_exp(n_classes,mixed,dis,pred_all_all,pred_EMG_all,agreement_exp,disagreement_exp,output_dir,x_EEG_all,x_EMG_all,group_all,aid_all,ss_n=False,both=True)
run_exp(n_classes,mixed,dis,pred_EEG_all,pred_EMG_all,agreement_exp,disagreement_exp,output_dir,x_EEG_all,x_EMG_all,group_all,aid_all,ss_n=False,both=False)

# experiment 2 mixing states 
n_classes        = 7
mixed            = 1
dis              = 0
agreement_exp    = 0
disagreement_exp = 0
run_exp(n_classes,mixed,dis,pred_EEG_all,pred_EMG_all,agreement_exp,disagreement_exp,output_dir,x_EEG_all,x_EMG_all,group_all,aid_all,ss_n=False,both=False)

# experiment 3 dissociation states 
n_classes        = 9
mixed            = 0
dis              = 1
agreement_exp    = 0
disagreement_exp = 0
run_exp(n_classes,mixed,dis,pred_EEG_all,pred_EMG_all,agreement_exp,disagreement_exp,output_dir,x_EEG_all,x_EMG_all,group_all,aid_all,ss_n=False,both=False)

# experiment 4 mixing and dissociation states 
n_classes        = 49
mixed            = 1
dis              = 1
agreement_exp    = 0
disagreement_exp = 0
run_exp(n_classes,mixed,dis,pred_EEG_all,pred_EMG_all,agreement_exp,disagreement_exp,output_dir,x_EEG_all,x_EMG_all,group_all,aid_all,ss_n=False,both=False)

# experiment 5  mixing and dissociation states (agreement)
n_classes        = 49
mixed            = 1
dis              = 1
agreement_exp    = 1
disagreement_exp = 0
run_exp(n_classes,mixed,dis,pred_EEG_all,pred_EMG_all,agreement_exp,disagreement_exp,output_dir,x_EEG_all,x_EMG_all,group_all,aid_all,ss_n=False,both=False)

# experiment 6 mixing and dissociation states (disagreement)
n_classes        = 49
mixed            = 1
dis              = 1
agreement_exp    = 0
disagreement_exp = 1
run_exp(n_classes,mixed,dis,pred_EEG_all,pred_EMG_all,agreement_exp,disagreement_exp,output_dir,x_EEG_all,x_EMG_all,group_all,aid_all,ss_n=False,both=False)
print("done")