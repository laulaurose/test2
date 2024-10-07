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


class ARGS2(object):   
    def __init__(self):
        self.folder_regex = ""
        self.data_per_prediction = 4*128
        self.channels = ""
        self.majority =  "store_true"
        self.datasets = ""
        self.data_split = "val_data"
        self.out_dir = "predictions"
        self.num_gpus = 0 
        self.strip_func = ""
        self.filter_settings = ""
        self.notch_filter_settings = ""
        self.num_test_time_augment = 0
        self.one_shot = "store_true"
        self.save_true = "store_true"
        self.force_gpus = ""
        self.no_argmax = "store_true"
        self.weights_file_name = ""
        self.continue_ = "store_true"
        self.overwrite = "store_true"
        self.log_file = "prediction_log"
        self.move_study_to_folder_on_error = ""


def count_df(df): 
        p_w   = []
        p_n   = []
        p_r   = []
        p_wr  = []
        p_nr  = []
        p_wn  = []
        p_wnr = []
        p_res = []
        p_ss  = []

        for k in range(len(df)):
            if df[k,0] > 0.8: # wake  
                p_w.append(df[k,:])
                p_ss.append(0)
            elif df[k,1] > 0.8: # nrem
                p_n.append(df[k,:])
                p_ss.append(1)
            elif df[k,2] > 0.8: # rem
                p_r.append(df[k,:])  
                p_ss.append(2)
            elif min(df[k,:]) > 0.1: #triplet  
                p_wnr.append(df[k,:])
                p_ss.append(6)
            elif df[k,:].argmin()==2: # duplet (p_wn)
                p_wn.append(df[k,:])
                p_ss.append(3)
            elif df[k,:].argmin()==0: # duplet (p_nr)
                p_nr.append(df[k,:])
                p_ss.append(5)
            elif df[k,:].argmin()==1: # duplet (p_wr)
                p_wr.append(df[k,:])
                p_ss.append(4)
            else: 
                p_res.append(df[k,:]) 
            
        p_ss = np.array(p_ss)
        
        df_wnr = pd.DataFrame(df)
        
        counts_mat = []
        for i in range(3): 
            df_wnr['cat'] = np.where(df_wnr.iloc[:,i].between(0,0.1), '0-0.1', 
                                        np.where(df_wnr.iloc[:,i].between(0.1,0.2), '0.1-0.2',
                                            np.where(df_wnr.iloc[:,i].between(0.2,0.3), '0.2-0.3',
                                                np.where(df_wnr.iloc[:,i].between(0.3,0.4), '0.3-0.4',
                                                    np.where(df_wnr.iloc[:,i].between(0.4,0.5), '0.4-0.5',
                                                        np.where(df_wnr.iloc[:,i].between(0.5,0.6), '0.5-0.6', 
                                                            np.where(df_wnr.iloc[:,i].between(0.6,0.7), '0.6-0.7',
                                                                np.where(df_wnr.iloc[:,i].between(0.7,0.8), '0.7-0.8',
                                                                    np.where(df_wnr.iloc[:,i].between(0.8,0.9), '0.8-0.9',
                                                                        np.where(df_wnr.iloc[:,i].between(0.9,1), '0.9-1',np.nan))))))))))

            counts_mat.append(np.array(df_wnr.groupby("cat").count()[0])/sum(np.array(df_wnr.groupby("cat").count()[0])))
            assert df_wnr.groupby("cat").count()[0].shape[0]==10
        cmat = np.array(counts_mat).T
        return p_ss, cmat 

def plot_hypnodensity(EEG,EMG,xEEG,xEMG,ss_joint,tname):
    
    # Argmax and CM elements
    xEEG = xEEG.reshape(xEEG.shape[0]*xEEG.shape[1],xEEG.shape[2],xEEG.shape[3])
    xEMG = xEMG.reshape(xEMG.shape[0]*xEMG.shape[1],xEMG.shape[2],xEMG.shape[3])

    df_EEG         = pd.DataFrame(EEG)
    df_EMG         = pd.DataFrame(EMG)

    df_EEG.columns = ["WAKE","NREM","REM"]
    df_EMG.columns = ["WAKE","NREM","REM"]

    ss = pd.DataFrame(np.where(ss_joint==4,1,0))
    ss.columns = ["ss"]
    ss['bout'] = (ss.ss!=ss.ss.shift(1)).cumsum() 
    ss = ss.merge(ss.groupby('bout').ss.count().reset_index().rename(columns={'ss':'len_bout'}), how='left')   
    bouts  = ss[ss.ss==1].drop_duplicates()
    bouts["index"] = bouts.index
    idx = bouts.index 
    #subset into dataframe for each idx 
    deltatime = 15 
    col_ = [(230/255, 240/255, 255/255), (60/255, 79/255, 79/255), (188/255, 208/255, 182/255)]
    directory = "/Users/qgf169/Documents/python/Usleep/results/eeg-emg/" + tname + "/"

    stft_size=256
    stft_stride=16
    lowcut  = 0 
    highcut = 20
    
    if not os.path.exists(directory):
        os.makedirs(directory)

    for j in range(len(idx)):
        st_id = idx[j]-15 
        sl_id = idx[j]+15 
        if (st_id>0) & (sl_id<len(EEG)-15): 
            plot_df_EEG     = df_EEG.iloc[st_id:sl_id].reset_index()
            plot_df_EMG     = df_EMG.iloc[st_id:sl_id].reset_index()
            repeated_df_EEG = pd.DataFrame(np.repeat(plot_df_EEG.values, 4, axis=0), columns=plot_df_EEG.columns)
            repeated_df_EMG = pd.DataFrame(np.repeat(plot_df_EMG.values, 4, axis=0), columns=plot_df_EMG.columns)

            t_hypnodensity  = np.linspace(0, 30*4, 30*4, endpoint=True)
            fig, ax= plt.subplots(4,1,figsize=(10,6),sharex=True, constrained_layout=True)#, sharex=True)

            df_hypnodensity = repeated_df_EEG.iloc[:,1:4]
            ys = np.zeros(t_hypnodensity.shape)
            for i, col in enumerate(df_hypnodensity.columns):
                ax[0].bar(t_hypnodensity, df_hypnodensity[col], label=col, width=t_hypnodensity[1]-t_hypnodensity[0], bottom=ys,color=col_[i])
                ys +=df_hypnodensity[col].astype('float64')
            ax[0].set_xlim(0,120)
            ax[0].set_ylim(0,1)
            ax[0].legend(bbox_to_anchor=(1, 0.5), loc="center left", ncol=1)
            ax[0].set_ylabel("EEG")

            df_hypnodensity = repeated_df_EMG.iloc[:,1:4]
            ys = np.zeros(t_hypnodensity.shape)
            for i, col in enumerate(df_hypnodensity.columns):
                ax[1].bar(t_hypnodensity, df_hypnodensity[col], label=col, width=t_hypnodensity[1]-t_hypnodensity[0], bottom=ys,color=col_[i])
                ys +=df_hypnodensity[col].astype('float64')
            ax[1].set_xlim(0,120)
            ax[1].set_ylim(0,1)
            ax[1].legend(bbox_to_anchor=(1, 0.5), loc="center left", ncol=1)
            ax[1].set_ylabel("EMG")

            nEEG = xEEG[np.arange(st_id,sl_id),:,0].flatten()
            nEMG = xEMG[np.arange(st_id,sl_id),:,0].flatten()
            time_sig = np.linspace(0, 120,len(nEEG), endpoint=False)

            f, t, Z = scipy.signal.stft(nEEG,
                                        fs=128,
                                        window='hamming',
                                        nperseg=128,
                                        noverlap=100
                                        )
            
            Z = Z[np.where(f == lowcut)[0][0]: np.where(f == highcut)[0][0] + 1, :]
            f = f[np.where(f == lowcut)[0][0]: np.where(f == highcut)[0][0] + 1]
            ax[2].pcolormesh(t, f,np.abs(Z), shading='gouraud')
            ax[2].set_ylabel('STFT EEG')

           
            ax[3].plot(time_sig,nEMG)
            ax[3].set_ylabel("EMG")
            ax[3].set_xlabel("Time [s]")

            plt.savefig(directory+ "hyp_EEG_4_" +str(j)+ ".png",dpi=300)
            plt.close()


def features(EEG, EMG,xEEG,xEMG,tname): 
    # Argmax and CM elements
    EEG = EEG.reshape(-1, 3)
    EMG = EMG.reshape(-1, 3)

    ss_EEG,cEEG = count_df(EEG)
    ss_EMG,cEMG = count_df(EMG)

    # make a common vec for EEG and EMG 
    ss_joint = np.empty(len(ss_EEG))
    ss_joint.fill(np.nan)

    bin_stage = np.zeros(49)

    # Loop through each combination of ss_EEG and ss_EMG
    for i in range(len(np.unique(ss_EEG))):
        for j in range(len(np.unique(ss_EMG))):
            # Calculate the index for the current combination
            index = i * 7 + j
            # Assign the corresponding value to ss_joint
            ss_joint[np.where((ss_EEG==i)&(ss_EMG==j))[0]] = index

            if len(np.where((ss_EEG==i)&(ss_EMG==j))[0])>0:
                bin_stage[index] = 1


    ss = pd.DataFrame([ss_EEG,ss_EMG,ss_joint]).T
    ss.columns = ["pEEG","pEMG","pEEG_EMG"]


    # if len(np.where(ss_joint==4)[0]>0):
    #     plot_hypnodensity(EEG,EMG,xEEG,xEMG,ss_joint,tname)
    rms_vec              = rms_fcn(xEEG,xEMG,ss)
    f1,f2,f3,f4,f5,f6,f7,f8 = bout_analysis(ss)
    slow_o, slow_d,fast_d,slow_t,fast_t,alpha,beta,gamma = power_analysis_v2(ss,xEEG,xEMG,tname)
 
    feature_vec = np.array([rms_vec,f1,f2,f3,f4,f5,f6,f7,f8,slow_o,slow_d,fast_d,slow_t,fast_t,alpha,beta,gamma]).flatten()

    rms_n = ["rms_" + str(i) for i in range(0, 49)]
    f1_n  = ["avr_bout_length_" + str(i) for i in range(0, 49)]
    f2_n  = ["total_time_" + str(i) for i in range(0, 49)]
    f3_n  = ["counts_h_" + str(i) for i in range(0, 49)]
    f4_n  = ["4sec_" + str(i) for i in range(0, 49)]
    f5_n  = ["4sec_32sec_" + str(i) for i in range(0, 49)]
    f6_n  = ["32sec_1min_" + str(i) for i in range(0, 49)]
    f7_n  = ["1min_5min_" + str(i) for i in range(0, 49)]
    f8_n  = [">5min_" + str(i) for i in range(0, 49)]
    f9_n  = ["slowoscillations_" + str(i) for i in range(0, 49)]
    f10_n  = ["slowdelta_" + str(i) for i in range(0, 49)]
    f11_n  = ["fastdelta_" + str(i) for i in range(0, 49)]
    f12_n  = ["slowtheta_" + str(i) for i in range(0, 49)]
    f13_n  = ["fasttheta_" + str(i) for i in range(0, 49)]
    f14_n  = ["alpha_" + str(i) for i in range(0, 49)]
    f15_n  = ["beta_" + str(i) for i in range(0, 49)]
    f16_n  = ["gamma_" + str(i) for i in range(0, 49)]

    combined_list = (rms_n + f1_n + f2_n + f3_n + f4_n + f5_n + f6_n + f7_n + f8_n +
                 f9_n + f10_n + f11_n + f12_n + f13_n+f14_n+f15_n+f16_n)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(ss_EEG, ss_EMG,labels=[0,1,2,3,4,5,6])

    return feature_vec,combined_list, rms_vec, f1, f2, f3, f4, f5, f6, f7, f8, slow_o, slow_d,fast_d,slow_t,fast_t,alpha,beta,gamma,cm, bin_stage



def power_(x,low,high):
    from scipy import signal
    from scipy.integrate import simps

    assert x.shape[0]>0 
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
    
    return delta_rel_power


def power_analysis_v2(ss,xEEG,xEMG,tname):
    bands = {'slowoscillations': [0.5, 1.5], 
         'slowdelta': [1, 2.25],
         'fastdelta': [2.5, 4],
         'slowtheta': [5, 8],
         'fasttheta': [8, 10],
         'alpha': [9, 14],
         'beta': [14, 30],
         'gamma': [30, 49]}
    
    
    #calculate rms from EMG in the different states: 
    xEEG      = xEEG.reshape(xEEG.shape[0]*xEEG.shape[1],xEEG.shape[2],xEEG.shape[3])
    fs        = 128 
    slowoscillations = []
    slowdelta = []
    fastdelta = []
    slowtheta = []
    fasttheta = []
    alpha     = []
    beta      = []
    gamma     = []
   
    for jj in range(49): 
        if len(np.unique(ss.pEEG_EMG==jj))==2:
            idx  = np.where(ss.pEEG_EMG==jj)[0]
            p    = power_(xEEG[idx,:,0],bands["slowoscillations"][0],bands["slowoscillations"][1])
            slowoscillations.append(np.mean(p))
    
            p    = power_(xEEG[idx,:,0],bands["slowdelta"][0],bands["slowdelta"][1])
            slowdelta.append(np.mean(p))

            p    = power_(xEEG[idx,:,0],bands["fastdelta"][0],bands["fastdelta"][1])
            fastdelta.append(np.mean(p))

            p    = power_(xEEG[idx,:,0],bands["slowtheta"][0],bands["slowtheta"][1])
            slowtheta.append(np.mean(p))
            
            p    = power_(xEEG[idx,:,0],bands["fasttheta"][0],bands["fasttheta"][1])
            fasttheta.append(np.mean(p))
        
            p    = power_(xEEG[idx,:,0],bands["alpha"][0],bands["alpha"][1])
            alpha.append(np.mean(p))

            p    = power_(xEEG[idx,:,0],bands["beta"][0],bands["beta"][1])
            beta.append(np.mean(p))

            p    = power_(xEEG[idx,:,0],bands["gamma"][0],bands["gamma"][1])
            gamma.append(np.mean(p))

        else:
            slowoscillations.append(np.nan)
            slowdelta.append(np.nan)
            fastdelta.append(np.nan)
            slowtheta.append(np.nan)
            fasttheta.append(np.nan)
            alpha.append(np.nan)
            beta.append(np.nan)
            gamma.append(np.nan)
   
    return np.array(slowoscillations), np.array(slowdelta),np.array(fastdelta), np.array(slowtheta), np.array(fasttheta),np.array(alpha), np.array(beta), np.array(gamma)


def power_analysis(ss,xEEG,xEMG,tname):
    bands = {'slowoscillations': [0.5, 1.5], 
         'slowdelta': [1, 2.25],
         'fastdelta': [2.5, 4],
         'slowtheta': [5, 8],
         'fasttheta': [8, 10],
         'alpha': [9, 14],
         'beta': [14, 30],
         'gamma': [30, 49]}
    
    
    #calculate rms from EMG in the different states: 
    xEEG      = xEEG.reshape(xEEG.shape[0]*xEEG.shape[1],xEEG.shape[2],xEEG.shape[3])
    fs        = 128 
    slowoscillations = []
    slowdelta = []
    fastdelta = []
    slowtheta = []
    fasttheta = []
    alpha     = []
    beta      = []
    gamma     = []
   
    for jj in range(49): 
        if len(np.unique(ss.pEEG_EMG==jj))==2:
            idx  = np.where(ss.pEEG_EMG==jj)[0]
            f, p = welch(xEEG[idx,:,0], fs, nperseg=4*fs, window='hanning', scaling='density')
            st_id = np.where(f == 0.5)[0][0]
            sl_id = np.where(f == 20)[0][0]
            norm = np.mean(np.mean(p[:,st_id:sl_id]))
            p_norm = p / norm * 100
            p_norm = p 
            # plt.figure()
            # plt.plot(f, np.mean(p_norm, axis=0))

            # directory = "/Users/qgf169/Documents/python/Usleep/results/eeg-emg/" + tname + "/"

            # if not os.path.exists(directory):
            #     os.makedirs(directory)

            # plt.savefig(directory+ str(jj)+ ".png")
            # plt.close()
            idxx = np.where((f >= bands["slowoscillations"][0]) & (f <= bands["slowoscillations"][1]))[0]
            bp   = np.mean(p_norm[:,idxx])
            slowoscillations.append(bp)

            idxx = np.where((f >= bands["slowdelta"][0]) & (f <= bands["slowdelta"][1]))[0]
            bp   = np.mean(p_norm[:,idxx])
            slowdelta.append(bp)

            idxx = np.where((f >= bands["fastdelta"][0]) & (f <= bands["fastdelta"][1]))[0]
            bp   = np.mean(p_norm[:,idxx])
            fastdelta.append(bp)

            idxx = np.where((f >= bands["slowtheta"][0]) & (f <= bands["slowtheta"][1]))[0]
            bp   = np.mean(p_norm[:,idxx])
            slowtheta.append(bp)
            
            idxx = np.where((f >= bands["fasttheta"][0]) & (f <= bands["fasttheta"][1]))[0]
            bp   = np.mean(p_norm[:,idxx])
            fasttheta.append(bp)

            idxx = np.where((f >= bands["alpha"][0]) & (f <= bands["alpha"][1]))[0]
            bp   = np.mean(p_norm[:,idxx])
            alpha.append(bp)

            idxx = np.where((f >= bands["beta"][0]) & (f <= bands["beta"][1]))[0]
            bp   = np.mean(p_norm[:,idxx])
            beta.append(bp)

            idxx = np.where((f >= bands["gamma"][0]) & (f <= bands["gamma"][1]))[0]
            bp   = np.mean(p_norm[:,idxx])
            gamma.append(bp)

        else:
            slowoscillations.append(np.nan)
            slowdelta.append(np.nan)
            fastdelta.append(np.nan)
            slowtheta.append(np.nan)
            fasttheta.append(np.nan)
            alpha.append(np.nan)
            beta.append(np.nan)
            gamma.append(np.nan)
   
    return np.array(slowoscillations), np.array(slowdelta),np.array(fastdelta), np.array(slowtheta), np.array(fasttheta),np.array(alpha), np.array(beta), np.array(gamma)


def bout_analysis(ss):
    df = ss.copy() 
    
    df['bout'] = (df.pEEG_EMG!=df.pEEG_EMG.shift(1)).cumsum() 
    df = df.merge(df.groupby('bout').pEEG_EMG.count().reset_index().rename(columns={'pEEG_EMG':'len_bout'}), how='left')   
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
    for j in range(49):
       
        bouts      = 0 
        x1         = 0
        bouts_bar  = 0
        ss["ss"]   =  np.where(ss.pEEG_EMG==j,1,0)
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
        
        ss = ss.iloc[:,0:3]
        
    return f1,f2,f3,np.array(bout_int1),np.array(bout_int2),np.array(bout_int3),np.array(bout_int4),np.array(bout_int5) 
    


def rms_fcn(xEEG,xEMG,ss):
    #calculate rms from EMG in the different states: 
    xEEG = xEEG.reshape(xEEG.shape[0]*xEEG.shape[1],xEEG.shape[2],xEEG.shape[3])
    xEMG = xEMG.reshape(xEMG.shape[0]*xEMG.shape[1],xEMG.shape[2],xEMG.shape[3])
    
    rms_vec = np.empty(49)
    rms_vec.fill(np.nan)
    for k in range(0,49): 
        idx = np.where(ss.pEEG_EMG==k)[0]
        if idx.size != 0:
            rms_vec[k] = np.sqrt(np.mean(xEMG[idx,:,0] **2, axis=1)).mean()
        
    return rms_vec

  
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

def plot_cm_triplets(cm,normalize=False,
            cmap="Blues",title=None,out_path=None,pr=True):
    import matplotlib.colors as colors

    normalized_cm = cm
    # Compute confusion matrix
    labels = ["W","N","R","W-N","W-R","N-R","W-N-R"] 
    fig, ax = plt.subplots(figsize=(10, 8))
    divnorm = colors.TwoSlopeNorm(vmin=normalized_cm.min(), vcenter=0, vmax=normalized_cm.max())

    #im = ax.imshow(normalized_cm, interpolation='nearest', cmap=plt.get_cmap(cmap))
    im = ax.imshow(normalized_cm, interpolation='nearest', cmap=plt.get_cmap(cmap),norm=divnorm)

    ax.figure.colorbar(im, ax=ax)
    #im.set_clim(vmin=0, vmax=100)
    ax.set_xlabel('EEG', fontsize=16)  # Adjust the font size as needed
    ax.set_ylabel('EMG', fontsize=16)  # Adjust the font size as needed

    # We want to show all ticks...
    ax.set(xticks=np.arange(normalized_cm.shape[1]),
           yticks=np.arange(normalized_cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=labels, yticklabels=labels,
           ylabel='EEG',
           xlabel='EMG')
   
    ax.tick_params(axis='both', labelsize=12)  # Adjust the font size as needed

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    if not os.path.exists(out_path):
         os.makedirs(out_path)
         
    # Loop over data dimensions and create text annotations.
    if pr == True: 
        fmt = '.1f' if normalize else 'd'
        thresh = normalized_cm.max() / 2.
        for i in range(normalized_cm.shape[0]):
            for j in range(normalized_cm.shape[1]):
                ax.text(j, i, f'{normalized_cm[i, j]:.1f}',
                    ha="center", va="center",
                    color="white" if normalized_cm[i, j] > thresh or normalized_cm[i, j] < -thresh else "black",fontsize=14)
        fig.tight_layout()
        fig.savefig(out_path+title+"n_cm.png", dpi=180)
        plt.close(fig)
        print("non-norm")
        print(title)
        print(normalized_cm)
    else:
        fmt = '.1f' if normalize else 'd'
        thresh = normalized_cm.max() / 2.
        for i in range(normalized_cm.shape[0]):
            for j in range(normalized_cm.shape[1]):
                ax.text(j, i, f'{normalized_cm[i, j]:.1f}',
                    ha="center", va="center",
                    color="white" if normalized_cm[i, j] > thresh or normalized_cm[i, j] < -thres else "black",fontsize=14)
        fig.tight_layout()
        fig.savefig(out_path+title+"cm.png", dpi=180)
        plt.close(fig)
        print("non-norm")
        print(title)
        print(normalized_cm) 



def cal_metric_from_cm(cm_all):
    print(cm_all)
    print(np.sum(cm_all))
    num_classes = cm_all.shape[0]

    # Calculate metrics for each class
    precision_all = []
    recall_all    = []
    accuracy_all  = []

    for class_idx in range(num_classes):
        true_positives  = cm_all[class_idx, class_idx]
        false_positives = np.sum(cm_all[:, class_idx]) - true_positives
        false_negatives = np.sum(cm_all[class_idx, :]) - true_positives
        true_negatives  = np.sum(cm_all) - (true_positives + false_positives + false_negatives)

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        accuracy = (true_positives + true_negatives) / np.sum(cm_all)
        
        precision_all.append(precision)
        recall_all.append(recall)
        accuracy_all.append(accuracy)
    
    f1_score = 2 * (np.array(precision_all) * np.array(recall_all)) / (np.array(precision_all) + np.array(recall_all))

    return precision_all,recall_all,accuracy_all,f1_score


########################### Load model and weights ###############################
from utime.bin.evaluate import get_and_load_model, get_sequencer
from utime.hyperparameters import YAMLHParams
from utime import Defaults

# Extract data from preprocess.h5 file that needs to be used for predictions 
from utime.bin.predict import get_datasets, run_pred
args        = ARGS2()
n_classes   = 3
E           = 1 #np.int(sys.argv[4])
splits      = ["TRAIN","TEST"]
elecs       = [["EEG1","EEG2"],
               ["EEG1","EEG2"]]
groups      = ['Kornum_cleaned','Kornum_cleaned_WT']
cols        = ["Greens","Blues","Reds"]

output_dir  =  "/Users/qgf169/Documents/python/Usleep/results/eeg-emg/"


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

#data_dir =  sys.argv[2]
data_dir        = "/Users/qgf169/Documents/python/Usleep/usleepletsgo_nt_all/" 

#data_dir = "/Users/qgf169/Desktop/Phd-project/Data-collaboration/SPINDLE-data/"
data        = h5py.File(data_dir+"processed_data.h5")
custom_palette = [[39/255, 94/255, 45/255],
                  [24/255, 56/255, 116/255]]
aid_all     = []
group_all   = []
feature_all = []
cm_both     = []
f0_all   = []
f1_all   = []
f2_all   = []
f3_all   = []
f4_all   = []
f5_all   = []
f6_all   = []
f7_all   = []
f8_all   = []
f9_all   = []
f10_all  = []
f11_all  = []
f12_all  = []
f13_all  = []
f14_all  = []
f15_all  = []
f16_all  = []
group_all_2 = []
state_all = []
#create dataframe 
data_h5 = h5py.File(output_dir+'_data.h5', 'w')
for k in range(len(groups)): # across groups
    print(groups[k])
    #cm_all =  np.zeros(shape=(n_classes,n_classes))
    cm_all =  np.zeros(shape=(7,7))
    bin_stage_all = np.zeros(shape=(1,49))

    print(cm_all)

    for i in range(len(splits)): # across datasplits 
        print(splits[i])
        aid = list(data[groups[k]][splits[i]].keys())
        print(aid)

        for j in range(len(aid)): # across animals 
            if aid[j]!='12':
                print(aid[j])
                batch_shape = [128,11,512,1]
                pred_EEG    = []
                #aid_all.append(aid[j])
                for e in range(len(elecs[k])):
                    e = 1 
                    print(elecs[k][e])
                    sig  = data[groups[k]][splits[i]][aid[j]]["PSG"][elecs[k][e]][:]
                    n = len(sig) // 512
                    hyp  = data[groups[k]][splits[i]][aid[j]]["hypnogram"][:]

                    reshaped_sig  = sig.reshape(n,512,1)
                    shape   = [-1] + batch_shape[1:]
                    border  = reshaped_sig.shape[0]%shape[1]

                    if border:
                        reshaped_sig = reshaped_sig[:-border]
                        hyp = hyp[:-border]
                    else:
                        hyp = hyp[:]

                    # EEG signal 
                    x_eeg   = reshaped_sig.reshape(shape)
                    y       = hyp.reshape(shape[:2] + [1])
                    x_eeg,y = process_batch(x_eeg, y,1,batch_shape)
                    pred    = model_eeg.predict_on_batch(x_eeg)
                    pred_EEG.append(pred) 

                #pred_EEG = np.array(pred_EEG).sum(axis=0)
                pred_EEG = pred # always channel 2
                # EMG signal 
                emg                = data[groups[k]][splits[i]][aid[j]]["PSG"]['EMG'][:]
                reshaped_EMG       = emg.reshape(n,512,1)
                reshaped_EMG       = reshaped_EMG[:-border]
                assert len(hyp)    == reshaped_sig.shape[0]
                x_emg              = reshaped_EMG.reshape(shape)
                y                  = hyp.reshape(shape[:2] + [1])
                x_emg,y            = process_batch(x_emg, y,1,batch_shape)
                pred_EMG           = model_emg.predict_on_batch(x_emg)      
                fvec,combined_list,rms_vec, f1_n, f2_n, f3_n, f4_n, f5_n, f6_n, f7_n, f8_n, f9_n, f10_n, f11_n, f12_n, f13_n, f14_n, f15_n, f16_n,cm, bin_stage = features(EEG=pred_EEG, EMG=pred_EMG,xEEG=x_eeg,xEMG=x_emg,tname=aid[j])
                f0_all.append(rms_vec)
                f1_all.append(f1_n)
                f2_all.append(f2_n)
                f3_all.append(f3_n)
                f4_all.append(f4_n)
                f5_all.append(f5_n)
                f6_all.append(f6_n)
                f7_all.append(f7_n)
                f8_all.append(f8_n)
                f9_all.append(f9_n)
                f10_all.append(f10_n)
                f11_all.append(f11_n)
                f12_all.append(f12_n)
                f13_all.append(f13_n)
                f14_all.append(f14_n)
                f15_all.append(f15_n)
                f16_all.append(f16_n)
                feature_all.append(fvec)
                aid_all.append(aid[j])
                group_all_2.append(1-k)
                state_all.append(np.arange(0,49))
                group_all.append(np.tile(k, 49))
                bin_stage_all +=bin_stage
                cm_all += cm
    cm_both.append(cm_all)
    bin_stage_all = bin_stage_all.flatten()
    plt.figure(figsize=(18, 3))  # Adjust the figure size if needed
    sns.barplot(x=np.arange(len(bin_stage_all)), y=bin_stage_all/14*100,color=custom_palette[k],linewidth=4)
    plt.ylim(0,100)
    plt.savefig(output_dir+"perc_mice_in_state"+str(k)+".png")
    plt.close()



import matplotlib.cm as cm

## odds ratio 
n_classes = 7 
odds_cm = np.zeros(shape=(n_classes,n_classes))
cm_NT   = cm_both[0]
cm_WT   = cm_both[1]

for r in range(7):
    for c in range(7):
        x1      =  cm_NT[r,c]
        x3      =  cm_WT[r,c]
        x2      =  cm_NT.sum(axis=1)[r] - x1
        x4      =  cm_WT.sum(axis=1)[r] - x3

        odds_cm[r,c] = round((x1*x4)/(x3*x2),1)

from matplotlib.colors import LinearSegmentedColormap
colors = [custom_palette[1],(1, 1, 1),custom_palette[0]]  # White, Red, Blue
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
plot_cm_triplets(np.log(odds_cm),normalize=False,cmap=cmap,title="odds_EEG2",out_path=output_dir,pr=True)

df = pd.DataFrame(feature_all)
df.columns = combined_list
df["aid"]    = aid_all
df["group"]  = group_all_2
df.to_csv(output_dir+'df_features_v2.csv', index=False)  # Set index=False to exclude index from the CSV file
print("done")

custom_palette = [[39/255, 94/255, 45/255],
                  [24/255, 56/255, 116/255]]

## dataframe 1
dfn = pd.DataFrame([np.concatenate(f0_all),np.concatenate(state_all),1-np.concatenate(group_all)])
dfn = dfn.T
dfn.columns = ["RMS_vec","state","group"]
dfn["GT"] = np.where(dfn.group==1,"NT","WT")
plt.figure(figsize=(30,4))
sns.barplot(x="state",y="RMS_vec",hue="GT",data=dfn,palette=custom_palette)
plt.title("RMS")
plt.savefig(output_dir+"RMS.png")

dfn = pd.DataFrame([np.concatenate(f1_all),np.concatenate(state_all),1-np.concatenate(group_all)])
dfn = dfn.T
dfn.columns = ["avr_bout_length","state","group"]
dfn["GT"] = np.where(dfn.group==1,"NT","WT")
plt.figure(figsize=(30,4))
sns.barplot(x="state",y="avr_bout_length",hue="GT",data=dfn,palette=custom_palette)
plt.title("avr_bout_length [s]")
plt.savefig(output_dir+"avr_bout_length.png")


dfn = pd.DataFrame([np.concatenate(f2_all),np.concatenate(state_all),1-np.concatenate(group_all)])
dfn = dfn.T
dfn.columns = ["total_time","state","group"]
dfn["GT"] = np.where(dfn.group==1,"NT","WT")
plt.figure(figsize=(30,4))
sns.barplot(x="state",y="total_time",hue="GT",data=dfn,palette=custom_palette)
plt.title("total_time [%]")
plt.savefig(output_dir+"total_time.png")


dfn = pd.DataFrame([np.concatenate(f3_all),np.concatenate(state_all),1-np.concatenate(group_all)])
dfn = dfn.T
dfn.columns = ["counts_h","state","group"]
dfn["GT"] = np.where(dfn.group==1,"NT","WT")
plt.figure(figsize=(30,4))
sns.barplot(x="state",y="counts_h",hue="GT",data=dfn,palette=custom_palette)
plt.title("counts_h [counts/h]")
plt.savefig(output_dir+"counts_h.png")


dfn = pd.DataFrame([np.concatenate(f4_all),np.concatenate(state_all),1-np.concatenate(group_all)])
dfn = dfn.T
dfn.columns = ["4sec","state","group"]
dfn["GT"] = np.where(dfn.group==1,"NT","WT")
plt.figure(figsize=(30,4))
sns.barplot(x="state",y="4sec",hue="GT",data=dfn,palette=custom_palette)
plt.title("4sec [%]")
plt.savefig(output_dir+"4sec.png")


dfn = pd.DataFrame([np.concatenate(f5_all),np.concatenate(state_all),1-np.concatenate(group_all)])
dfn = dfn.T
dfn.columns = ["4sec_32sec","state","group"]
dfn["GT"] = np.where(dfn.group==1,"NT","WT")
plt.figure(figsize=(30,4))
sns.barplot(x="state",y="4sec_32sec",hue="GT",data=dfn,palette=custom_palette)
plt.title("4sec_32sec [%]")
plt.savefig(output_dir+"4sec_32sec.png")

dfn = pd.DataFrame([np.concatenate(f6_all),np.concatenate(state_all),1-np.concatenate(group_all)])
dfn = dfn.T
dfn.columns = ["32sec_1min","state","group"]
dfn["GT"] = np.where(dfn.group==1,"NT","WT")
plt.figure(figsize=(30,4))
sns.barplot(x="state",y="32sec_1min",hue="GT",data=dfn,palette=custom_palette)
plt.title("32sec_1min [%]")
plt.savefig(output_dir+"32sec_1min.png")


dfn = pd.DataFrame([np.concatenate(f7_all),np.concatenate(state_all),1-np.concatenate(group_all)])
dfn = dfn.T
dfn.columns = ["1min_5min","state","group"]
dfn["GT"] = np.where(dfn.group==1,"NT","WT")
plt.figure(figsize=(30,4))
sns.barplot(x="state",y="1min_5min",hue="GT",data=dfn,palette=custom_palette)
plt.title("1min_5min [%]")
plt.savefig(output_dir+"1min_5min.png")


dfn = pd.DataFrame([np.concatenate(f8_all),np.concatenate(state_all),1-np.concatenate(group_all)])
dfn = dfn.T
dfn.columns = [">5min_","state","group"]
dfn["GT"] = np.where(dfn.group==1,"NT","WT")
plt.figure(figsize=(30,4))
sns.barplot(x="state",y=">5min_",hue="GT",data=dfn,palette=custom_palette)
plt.title(">5min_ [%]")
plt.savefig(output_dir+">5min_.png")

dfn = pd.DataFrame([np.concatenate(f9_all),np.concatenate(state_all),1-np.concatenate(group_all)])
dfn = dfn.T
dfn.columns = ["slowoscillations","state","group"]
dfn["GT"] = np.where(dfn.group==1,"NT","WT")
plt.figure(figsize=(30,4))
sns.barplot(x="state",y="slowoscillations",hue="GT",data=dfn,palette=custom_palette)
plt.title("slowoscillations")
plt.savefig(output_dir+"slowoscillations.png")


dfn = pd.DataFrame([np.concatenate(f10_all),np.concatenate(state_all),1-np.concatenate(group_all)])
dfn = dfn.T
dfn.columns = ["slowdelta","state","group"]
dfn["GT"] = np.where(dfn.group==1,"NT","WT")
plt.figure(figsize=(30,4))
sns.barplot(x="state",y="slowdelta",hue="GT",data=dfn,palette=custom_palette)
plt.title("slowdelta")
plt.savefig(output_dir+"slowdelta.png")

dfn = pd.DataFrame([np.concatenate(f11_all),np.concatenate(state_all),1-np.concatenate(group_all)])
dfn = dfn.T
dfn.columns = ["fastdelta","state","group"]
dfn["GT"] = np.where(dfn.group==1,"NT","WT")
plt.figure(figsize=(30,4))
sns.barplot(x="state",y="fastdelta",hue="GT",data=dfn,palette=custom_palette)
plt.title("fastdelta")
plt.savefig(output_dir+"fastdelta.png")

dfn = pd.DataFrame([np.concatenate(f12_all),np.concatenate(state_all),1-np.concatenate(group_all)])
dfn = dfn.T
dfn.columns = ["slowtheta","state","group"]
dfn["GT"] = np.where(dfn.group==1,"NT","WT")
plt.figure(figsize=(30,4))
sns.barplot(x="state",y="slowtheta",hue="GT",data=dfn,palette=custom_palette)
plt.title("slowtheta")
plt.savefig(output_dir+"slowtheta.png")

dfn = pd.DataFrame([np.concatenate(f13_all),np.concatenate(state_all),1-np.concatenate(group_all)])
dfn = dfn.T
dfn.columns = ["fasttheta","state","group"]
dfn["GT"] = np.where(dfn.group==1,"NT","WT")
plt.figure(figsize=(30,4))
sns.barplot(x="state",y="fasttheta",hue="GT",data=dfn,palette=custom_palette)
plt.title("fasttheta")
plt.savefig(output_dir+"fasttheta.png")

dfn = pd.DataFrame([np.concatenate(f14_all),np.concatenate(state_all),1-np.concatenate(group_all)])
dfn = dfn.T
dfn.columns = ["alpha","state","group"]
dfn["GT"] = np.where(dfn.group==1,"NT","WT")
plt.figure(figsize=(30,4))
sns.barplot(x="state",y="alpha",hue="GT",data=dfn,palette=custom_palette)
plt.title("alpha")
plt.savefig(output_dir+"alpha.png")

dfn = pd.DataFrame([np.concatenate(f15_all),np.concatenate(state_all),1-np.concatenate(group_all)])
dfn = dfn.T
dfn.columns = ["beta","state","group"]
dfn["GT"] = np.where(dfn.group==1,"NT","WT")
plt.figure(figsize=(30,4))
sns.barplot(x="state",y="beta",hue="GT",data=dfn,palette=custom_palette)
plt.title("beta")
plt.savefig(output_dir+"beta.png")

dfn = pd.DataFrame([np.concatenate(f16_all),np.concatenate(state_all),1-np.concatenate(group_all)])
dfn = dfn.T
dfn.columns = ["gamma","state","group"]
dfn["GT"] = np.where(dfn.group==1,"NT","WT")
plt.figure(figsize=(30,4))
sns.barplot(x="state",y="gamma",hue="GT",data=dfn,palette=custom_palette)
plt.title("gamma")
plt.savefig(output_dir+"gamma.png")