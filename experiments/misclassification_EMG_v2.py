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
from statsmodels.stats.contingency_tables import Table2x2
import seaborn as sns
from utime.bin.evaluate import get_and_load_model, get_sequencer
from utime.hyperparameters import YAMLHParams
from utime import Defaults
from utime.bin.predict import get_datasets, run_pred
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as colors
def compute_dice(tp, rel, sel):
        # Get data masks (to avoid div. by zero warnings)
        # We set precision, recall, dice to 0 in for those particular cls.
        print("summen af true - n_epochs input to compute dice")
        print(np.sum(rel))
        sel_mask = sel > 0
        rel_mask = rel > 0

        # prepare arrays
        precisions = np.zeros(shape=tp.shape, dtype=np.float32)
        recalls = np.zeros_like(precisions)
        dices = np.zeros_like(precisions)

        # Compute precisions, recalls
        precisions[sel_mask] = tp[sel_mask] / sel[sel_mask]
        recalls[rel_mask] = tp[rel_mask] / rel[rel_mask]

        # Compute dice
        intrs = (2 * precisions * recalls)
        union = (precisions + recalls)
        dice_mask = union > 0
        dices[dice_mask] = intrs[dice_mask] / union[dice_mask]
        return precisions, recalls, dices
    
    

def cat_char(pred, true,x):
        n_classes = 3
        # Argmax and CM elements
        pred = pred.argmax(-1).ravel()
        print(np.unique(pred))
        true = true.ravel()
        xnew = x.reshape(x.shape[0]*x.shape[1],x.shape[2])

        # True array may include negative or non-negative integers larger than n_classes, e.g. int class 5 "UNKNOWN"
        # Here we mask out any out-of-range values any evaluate only on in-range classes.
        
        idx_c    = np.where(true==3)
        counts_c = np.bincount(pred[idx_c[0]])

        c_w = xnew[np.where((true==3)&(pred==0))[0],:]
        c_n = xnew[np.where((true==3)&(pred==1))[0],:]
        c_r = xnew[np.where((true==3)&(pred==2))[0],:]

        xx = [c_w,c_n,c_r]
        
        return counts_c, xx 

def misclas(pred,true,x):
    pred = pred.argmax(-1).ravel()
    print(np.unique(pred))
    true = true.ravel()
    idx_w_n = np.where((true==0)&(pred==1))[0]
    idx_w_r = np.where((true==0)&(pred==2))[0]
    
    vec_mix          = true

    if  (len(idx_w_n)>0):
        vec_mix[idx_w_n] = 4 # w_n 
    
    if  (len(idx_w_r)>0):
        vec_mix[idx_w_r] = 5 # w_r 


    unique_states = [0,1,2,3,4,5]
    num_states = len(unique_states)

    transition_matrix = np.zeros((num_states, num_states), dtype=int)

    for i in range(len(vec_mix) - 1):
        current_state = np.where(unique_states == vec_mix[i])[0][0]
        next_state = np.where(unique_states == vec_mix[i + 1])[0][0]
        transition_matrix[current_state, next_state] += 1
        
    return  transition_matrix, vec_mix

def compute_counts(pred, true,x):
        n_classes = 3
        # Argmax and CM elements
        pred = pred.argmax(-1).ravel()
        print(np.unique(pred))
        true = true.ravel()
        xnew = x.reshape(x.shape[0]*x.shape[1],x.shape[2],x.shape[3])

        # True array may include negative or non-negative integers larger than n_classes, e.g. int class 5 "UNKNOWN"
        # Here we mask out any out-of-range values any evaluate only on in-range classes.
        mask = np.where(np.logical_and(
            np.greater_equal(true, 0),
            np.less(true, n_classes)
        ), np.ones_like(true), np.zeros_like(true)).astype(bool)
        pred = pred[mask]
        true = true[mask]
        xnew = xnew[mask,:,:]
       
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(true, pred,labels=[0,1,2])

        w_w = xnew[np.where((true==0)&(pred==0))[0],:] # alle de epoker der er w og bliver klassificeret som w 
        w_n = xnew[np.where((true==0)&(pred==1))[0],:] # alle de epoker der er w og bliver klassificeret som n 
        w_r = xnew[np.where((true==0)&(pred==2))[0],:] # alle de epoker der er w og bliver klassifiecret som r 
        
        n_w = xnew[np.where((true==1)&(pred==0))[0],:]
        n_n = xnew[np.where((true==1)&(pred==1))[0],:]
        n_r = xnew[np.where((true==1)&(pred==2))[0],:]
        
        r_w = xnew[np.where((true==2)&(pred==0))[0],:]
        r_n = xnew[np.where((true==2)&(pred==1))[0],:]
        r_r = xnew[np.where((true==2)&(pred==2))[0],:]
        xx = [w_w,w_n,w_r,n_w,n_n,n_r,r_w,r_n,r_r]

        print(cm)
     
        return cm, xx, xnew 


def plot_cm(cm,normalize=False,
            cmap="Blues",title=None,out_path=None,precision=False,odds=False):
    import matplotlib.colors as colors

    if precision & odds==False:
        total_samples = np.sum(cm, axis=0)
        normalized_cm = (cm / total_samples[np.newaxis,:]) * 100
        normalized_cm = np.round(normalized_cm, 2)

    elif precision & odds==True:
        normalized_cm = cm 
    else: 
        total_samples = np.sum(cm, axis=1)
        normalized_cm = (cm / total_samples[:, np.newaxis]) * 100
        normalized_cm = np.round(normalized_cm, 2)

    # Determine the colormap
    if isinstance(cmap, str) or isinstance(cmap, LinearSegmentedColormap):
        cmap2 = cmap
    else:
        n_shades = 100  # Adjust the number of shades as needed
        colors = [np.linspace(1, c, n_shades) for c in cmap]
        shades = np.column_stack(colors)
        cmap2 = LinearSegmentedColormap.from_list("custom_cmap", shades)

    # Create the plot
    fig, ax = plt.subplots(figsize=(4, 4))
    labels = ["Wake","NREM","REM"] 
    norm = None
    if odds:
        norm = colors.TwoSlopeNorm(vmin=normalized_cm.min(), vcenter=0, vmax=normalized_cm.max())
    im = ax.imshow(normalized_cm, interpolation='nearest', cmap=cmap2, norm=norm)

    # Adjust the colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=20)  # Shrink to 80% of the axes height, aspect ratio is thinner

    if not odds:
        im.set_clim(vmin=0, vmax=100)  # Normalize color bar for percentage values

    # Set labels and ticks

    ax.set_xlabel('Predicted label', fontsize=10)
    ax.set_ylabel('True label', fontsize=10)
    ax.set(xticks=np.arange(normalized_cm.shape[1]), yticks=np.arange(normalized_cm.shape[0]),
           xticklabels=labels, yticklabels=labels, ylabel='Manual label', xlabel='Predicted label')
    ax.tick_params(axis='both', labelsize=10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Text annotations
    if not odds:
        thresh = normalized_cm.max() / 2.
        for i in range(normalized_cm.shape[0]):
            for j in range(normalized_cm.shape[1]):
                ax.text(j, i, f'{normalized_cm[i, j]:.1f}%',
                        ha="center", va="center",
                        color="black" if normalized_cm[i, j] > thresh else "black", fontsize=10)
    else: 
        thresh = normalized_cm.max() / 2.
        for i in range(normalized_cm.shape[0]):
            for j in range(normalized_cm.shape[1]):
                ax.text(j, i, f'{normalized_cm[i, j]:.1f}',
                        ha="center", va="center",
                        color="black" if normalized_cm[i, j] > thresh else "black", fontsize=10)

    if precision:
        fig.savefig(out_path+title+"precision_EEG_cm.pdf", dpi=600)
        plt.close(fig)
    else: 
        fig.savefig(out_path+title+"EEG_cm.pdf", dpi=600)
        plt.close(fig)
 

def plot_trans(trans, labels, cmap="Blues",title=None,out_path=None):
    # Compute confusion matrix

    fig, ax = plt.subplots(figsize=(7,7))
    im = ax.imshow(trans, interpolation='nearest', cmap=plt.get_cmap(cmap))
    ax.figure.colorbar(im, ax=ax)
    im.set_clim(vmin=0, vmax=1)
    ax.set_xlabel('stage t+1', fontsize=14)  # Adjust the font size as needed
    ax.set_ylabel('stage t', fontsize=14)  # Adjust the font size as needed
    ax.set_title(title, fontsize=14)  # Adjust the font size as needed

    # We want to show all ticks...
    ax.set(xticks=np.arange(trans.shape[1]),
           yticks=np.arange(trans.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=labels, yticklabels=labels,
           title=title)
   
    ax.tick_params(axis='both', labelsize=14)  # Adjust the font size as needed


    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    if not os.path.exists(out_path):
         os.makedirs(out_path)
         
    # Loop over data dimensions and create text annotations.
    fmt = '.3f'
    thresh = trans.max() / 2.
    for i in range(trans.shape[0]):
        for j in range(trans.shape[1]):
            ax.text(j, i, f'{trans[i, j]:.2f}',
                 ha="center", va="center",
                 color="white" if trans[i, j] > thresh else "black",fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path+title+"transition_matrix.png", dpi=180)
    plt.close(fig)

def process_batch(X, y,n_channels,batch_shape):
        """
        Process a batch (X, y) of sampled data.

        The process_batch method should always be called in the end of any
        method that implements batch sampling.

        Processing includes:
          1) Casting of X to ndarray of dtype float32
          2) Ensures X has a channel dimension, even if self.n_channels == 1
          3) Ensures y has dtype uint8 and shape [-1, 1]
          4) Ensures both X and y has a 'batch dimension', even if batch_size
             is 1.
          5) If a 'batch_scaler' is set, scales the X data
          6) Performs augmentation on the batch if self.augmenters is set and
             self.augmentation_enabled is True

        Args:
            X:     A list of ndarrays corresponding to a batch of X data
            y:     A list of ndarrays corresponding to a batch of y labels

        Returns:
            Batch of (X, y) data
            OBS: Currently does not return the w (weights) array
        """
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


def cal_metric_from_cm(cm_all):
    print(cm_all)
    print(np.sum(cm_all))
    num_classes = cm_all.shape[0]

    # Calculate metrics for each class
    precision_all = []
    recall_all    = []
    accuracy_all  = []

    for class_idx in range(num_classes):
        true_positives = cm_all[class_idx, class_idx]
        false_positives = np.sum(cm_all[:, class_idx]) - true_positives
        false_negatives = np.sum(cm_all[class_idx, :]) - true_positives
        true_negatives = np.sum(cm_all) - (true_positives + false_positives + false_negatives)

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        accuracy = (true_positives + true_negatives) / np.sum(cm_all)
        
        precision_all.append(precision)
        recall_all.append(recall)
        accuracy_all.append(accuracy)
    
    f1_score = 2 * (np.array(precision_all) * np.array(recall_all)) / (np.array(precision_all) + np.array(recall_all))

    return precision_all,recall_all,accuracy_all,f1_score


########################### Load model and weights ###############################

n_classes   = 3
E           = 1 #np.int(sys.argv[4])
splits      = ["TRAIN","TEST"]
elecs       = [["EEG1","EEG2"],
               ["EEG1","EEG2"]]
groups      = ['Kornum_cleaned','Kornum_cleaned_WT']
modeltype   = "EMG"#sys.argv[3]
project_dir =  "/Users/qgf169/Documents/python/Usleep/usleepletsgo_nt_EMG/" #sys.argv[1]
output_dir  =  "/Users/qgf169/Desktop/papers/usleep/new/Figure2/" #sys.argv[5]

# load model 
os.chdir(project_dir)
Defaults.PROJECT_DIRECTORY = project_dir 
hparams       =  YAMLHParams(Defaults.get_hparams_path(project_dir))
model,model_f = get_and_load_model(project_dir,
                    hparams,
                    weights_file_name="",
                    clear_previous=True)
model.summary()
weights = h5py.File(project_dir+"model/"+model_f)


# load data 
data_dir  = "/Users/qgf169/Documents/python/Usleep/usleepletsgo_nt_all/" 
data      = h5py.File(data_dir+"processed_data.h5")
print(data.keys())

# preallocation of variables 
groups_all     = []
aid_all        = []
elec_all       = []
cm_both        = []
gt_df          = []


rms_ww = []
rms_wn = []
rms_wr = []
rms_nw = []
rms_nn = []
rms_nr = []
rms_rw = []
rms_rn = []
rms_rr = []

cols = [[218/255,229/255,195/255],
        [186/255,211/255,233/255]]

for k in range(len(groups)): # across groups
    cm_all =  np.zeros(shape=(n_classes,n_classes))
    print(groups[k])
    
    for i in range(len(splits)): # across datasplits 
        print(splits[i])
        aid = list(data[groups[k]][splits[i]].keys())
        print(aid)

        for j in range(len(aid)): # across animals 
            if aid[j]!='12':
                print(aid[j])
                batch_shape = [128,11,512,2]
                pred_all    = []

                for e in range(len(elecs[k])):
                    print(elecs[k][e])
                    emg  = data[groups[k]][splits[i]][aid[j]]["PSG"]['EMG'][:]
                    sig  = data[groups[k]][splits[i]][aid[j]]["PSG"][elecs[k][e]][:]
                    n    = len(sig) // 512
                    hyp  = data[groups[k]][splits[i]][aid[j]]["hypnogram"][:]

                    reshaped_sig  = sig.reshape(n,512,1)
                    reshaped_EMG  = emg.reshape(n,512,1)
                    reshaped_both = np.concatenate([reshaped_sig,reshaped_EMG],axis=2)
                    shape   = [-1] + batch_shape[1:]
                    border  = reshaped_sig.shape[0]%shape[1]

                    if border:
                        reshaped_both = reshaped_both[:-border]
                        hyp = hyp[:-border]
                    else:
                        hyp = hyp[:]
                    print(len(hyp)*4)
                    x = reshaped_both.reshape(shape)
                    y = hyp.reshape(shape[:2] + [1])
                    x,y = process_batch(x, y,2,batch_shape)
                    
                    if (E==2) & (modeltype=="all"):
                        print("usleep-all")
                        pred = model.predict_on_batch(x)
                        pred_all.append(pred) 
                    
                    elif (E==1) & (modeltype=="EEG"):
                        print("usleep-EEG")
                        pred = model.predict_on_batch(x[:,:,:,0])
                        pred_all.append(pred) 
                    
                    elif (E==1) & (modeltype=="EMG"):
                        print("usleep-EMG")
                        pred = model.predict_on_batch(x[:,:,:,1])
                
                    pred_all = pred
                    cm, xx, xnew = compute_counts(pred=pred_all, true=y,x=x)

                gt_df.append(groups[k])
                aid_all.append(aid[j])    
                
                rms_ww.append(np.mean(np.sqrt(np.mean(np.square(xx[0][:,:,1]),axis=1))))
                rms_wn.append(np.mean(np.sqrt(np.mean(np.square(xx[1][:,:,1]),axis=1))))
                rms_wr.append(np.mean(np.sqrt(np.mean(np.square(xx[2][:,:,1]),axis=1))))
                rms_nw.append(np.mean(np.sqrt(np.mean(np.square(xx[3][:,:,1]),axis=1))))
                rms_nn.append(np.mean(np.sqrt(np.mean(np.square(xx[4][:,:,1]),axis=1))))
                rms_nr.append(np.mean(np.sqrt(np.mean(np.square(xx[5][:,:,1]),axis=1))))
                rms_rw.append(np.mean(np.sqrt(np.mean(np.square(xx[6][:,:,1]),axis=1))))
                rms_rn.append(np.mean(np.sqrt(np.mean(np.square(xx[7][:,:,1]),axis=1))))
                rms_rr.append(np.mean(np.sqrt(np.mean(np.square(xx[8][:,:,1]),axis=1))))

                cm_all += cm 

    plot_cm(cm_all,normalize=False,cmap=cols[k],title=groups[k],out_path=output_dir,precision=True,odds=False)

    cm_both.append(cm_all)
    groups_all.append(groups[k])

## odds ratio 
n_classes = 3
odds_cm = np.zeros(shape=(n_classes,n_classes))
ci_low_cm = np.zeros(shape=(n_classes,n_classes))
ci_high_cm = np.zeros(shape=(n_classes,n_classes))

cm_NT   = cm_both[0]
cm_WT   = cm_both[1]

for r in range(3):
    for c in range(3):
        x1      =  cm_NT[r,c]
        x3      =  cm_WT[r,c]
        x2      =  cm_NT.sum(axis=1)[r] - x1
        x4      =  cm_WT.sum(axis=1)[r] - x3
        
        # Create a 2x2 table
        table = Table2x2(np.array([[x1, x2], [x3, x4]]))

        # Get log odds ratio and its confidence interval
        log_odds_ratio = table.log_oddsratio
        ci_low, ci_upp = table.log_oddsratio_confint()
        assert (x1*x4)/(x3*x2)==table.oddsratio
        odds_cm[r,c]    = table.log_oddsratio
        ci_low_cm[r,c]  = ci_low
        ci_high_cm[r,c] = ci_upp
print(odds_cm)
print(ci_low_cm)
print(ci_high_cm)

        
        #odds_cm[r,c] = round((x1*x4)/(x3*x2),1)
color_low  = [218/255, 229/255, 195/255]  # Low value color
color_high = [186/255, 211/255, 233/255]  # High value color
color_zero = [1, 1, 1]  # White color for zero

# Create the custom colormap
colors = [color_high, color_zero, color_low]  
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

plot_cm(odds_cm,normalize=False,cmap=cmap,title="odds",out_path=output_dir,precision=True,odds=True)


rms_df = pd.DataFrame(data={'aid':aid_all,
                        'gt_df':gt_df,
                        'ww': np.array(rms_ww),
                        'wn': np.array(rms_wn),
                        'wr': np.array(rms_wr),
                        'nw': np.array(rms_nw),
                        'nn': np.array(rms_nn),
                        'nr': np.array(rms_nr),
                        'rw': np.array(rms_rw),
                        'rn': np.array(rms_rn),
                        'rr':np.array(rms_rr)
                        })        

rms_df['GT'] = np.where(rms_df['gt_df'] == 'Kornum_cleaned', 'NT', 'WT')
melted_df = pd.melt(rms_df, id_vars=['aid', 'gt_df', 'GT'], var_name='sleep-stage', value_name='rms')
desired_order = ["ww", "nn", "rr", "nw", "rw", "wn", "rn", "wr", "nr"]
melted_df["ss"] = pd.Categorical(melted_df["sleep-stage"], categories=desired_order, ordered=True)

width_cm = 6.17
height_cm = 11.98
dpi = 600

width_in_inches = width_cm / 2.54 
height_in_inches = height_cm / 2.54 

plt.figure(figsize=(height_in_inches,width_in_inches))
sns.barplot(x="ss",y="rms",hue="GT",data=melted_df,palette=cols)
plt.ylabel("Root Mean Square", fontsize=10)
plt.xlabel("True label", fontsize=10)
plt.axvline(2.5,color="black")
plt.axvline(4.5,color="black")
plt.axvline(6.5,color="black")
plt.axvline(8.5,color="black")
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8], labels=["W", "N", "R", "N", "R", "W", "R", "W", "N"],fontsize=10)
plt.tight_layout()
plt.savefig(output_dir+"rms.pdf",dpi=600)