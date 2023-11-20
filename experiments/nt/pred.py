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
    

def compute_counts(pred, true):
        n_classes = 3
        # Argmax and CM elements
        pred = pred.argmax(-1).ravel()
        print(np.unique(pred))
        true = true.ravel()

        # True array may include negative or non-negative integers larger than n_classes, e.g. int class 5 "UNKNOWN"
        # Here we mask out any out-of-range values any evaluate only on in-range classes.
        mask = np.where(np.logical_and(
            np.greater_equal(true, 0),
            np.less(true, n_classes)
        ), np.ones_like(true), np.zeros_like(true)).astype(bool)
        pred = pred[mask]
        true = true[mask]
        print(len(pred))
        print(len(true))

        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(true, pred,labels=[0,1,2])
        print(cm)
        # Compute relevant CM elements
        # We select the number following the largest class integer when
        # y != pred, then bincount and remove the added dummy class
        
        return true, pred, cm 


def plot_cm(cm,normalize=False,
            cmap="Blues",title=None,out_path=None,precision=False):
    if precision==False:  
        total_samples = np.sum(cm,axis=1)
        normalized_cm = cm / total_samples[:, np.newaxis] * 100
        normalized_cm = np.round(normalized_cm, 2)
        print(cm)
    elif precision==True: 
        total_samples = np.sum(cm,axis=0)
        normalized_cm = cm / total_samples * 100
        normalized_cm = np.round(normalized_cm, 1) 
        out_path = out_path + "_prec"
    # Compute confusion matrix
    labels = ["W","NREM","REM"] 
    fig, ax = plt.subplots()
    im = ax.imshow(normalized_cm, interpolation='nearest', cmap=plt.get_cmap(cmap))
    ax.figure.colorbar(im, ax=ax)
    im.set_clim(vmin=0, vmax=100)
    ax.set_xlabel('Predicted label', fontsize=14)  # Adjust the font size as needed
    ax.set_ylabel('True label', fontsize=14)  # Adjust the font size as needed
    ax.set_title(title, fontsize=14)  # Adjust the font size as needed

    # We want to show all ticks...
    ax.set(xticks=np.arange(normalized_cm.shape[1]),
           yticks=np.arange(normalized_cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=labels, yticklabels=labels,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
   
    ax.tick_params(axis='both', labelsize=14)  # Adjust the font size as needed


    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    if not os.path.exists(out_path):
         os.makedirs(out_path)
         
    # Loop over data dimensions and create text annotations.
    fmt = '.3f' if normalize else 'd'
    thresh = normalized_cm.max() / 2.
    for i in range(normalized_cm.shape[0]):
        for j in range(normalized_cm.shape[1]):
            ax.text(j, i, f'{normalized_cm[i, j]:.1f}%',
                 ha="center", va="center",
                 color="white" if normalized_cm[i, j] > thresh else "black",fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path+title+"cm.png", dpi=180)
    plt.close(fig)
    print("non-norm")
    print(title)
    print(normalized_cm)
 
 

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
from utime.bin.evaluate import get_and_load_model, get_sequencer
from utime.hyperparameters import YAMLHParams
from utime import Defaults

# Extract data from preprocess.h5 file that needs to be used for predictions 
from utime.bin.predict import get_datasets, run_pred
args        = ARGS2()
n_classes   = 3
E           = np.int(sys.argv[4])
splits      = ["TRAIN","TEST"]
elecs       = [["EEG1","EEG2"],
               ["EEG1","EEG2"]]
groups      = ['Kornum_cleaned','Kornum_cleaned_WT']
cols        = ["Greens","Blues"]
modeltype   = sys.argv[3]
project_dir = sys.argv[1]
#project_dir = "/Users/qgf169/Documents/python/Usleep/usleep-spindle/"
output_dir  =  sys.argv[5]

os.chdir(project_dir)
Defaults.PROJECT_DIRECTORY = project_dir 
hparams       =  YAMLHParams(Defaults.get_hparams_path(project_dir))
model,model_f = get_and_load_model(project_dir,
                    hparams,
                    weights_file_name="",
                    clear_previous=True)
model.summary()
weights = h5py.File(project_dir+"model/"+model_f)
print("these weights should be the same")
print(weights['sequence_conv_out_2']['sequence_conv_out_2']["bias:0"][:])
print(model.non_trainable_weights[-1])
print(model_f)

data_dir =  sys.argv[2]
#data_dir = "/Users/qgf169/Desktop/Phd-project/Data-collaboration/SPINDLE-data/"
data = h5py.File(data_dir+"processed_data.h5")
print(data.keys())

groups_all     = []
aid_all        = []
elec_all       = []
f1_wake        = []
f1_nrem        = []
f1_rem         = []
recall_wake    = []
recall_nrem    = []
recall_rem     = []
precision_wake = []
precision_nrem = []
precision_rem  = []
accuracies_wake= []
accuracies_nrem= []
accuracies_rem = []
splits_all     = []
print(modeltype)

for k in range(len(groups)): # across groups
    cm_all =  np.zeros(shape=(n_classes,n_classes))
    dim_pred_all =  np.zeros(shape=(1,1))

    for i in range(len(splits)): # across datasplits 
        aid = list(data[groups[k]][splits[i]].keys())

        for j in range(len(aid)): # across animals 
            print(aid[j])
            batch_shape = [128,11,512,E]
            
            if (E==2) & (modeltype=="all"):
                pred_all    = []
                print("usleep-all")
                for e in range(len(elecs[k])):
                    print(elecs[k][e])
                    emg  = data[groups[k]][splits[i]][aid[j]]["PSG"]['EMG'][:]
                    sig  = data[groups[k]][splits[i]][aid[j]]["PSG"][elecs[k][e]][:]
                    n = len(sig) // 512
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

                    x = reshaped_both.reshape(shape)
                    y = hyp.reshape(shape[:2] + [1])
                    x,y = process_batch(x, y,E,batch_shape)
                    pred = model.predict_on_batch(x)
                    pred_all.append(pred) 
                pred_all = np.array(pred_all).sum(axis=0)
            
            elif (E==1) & (modeltype=="EEG"):
                pred_all    = []
                print("usleep-EEG")
                for e in range(len(elecs[k])):
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

                    x = reshaped_sig.reshape(shape)
                    y = hyp.reshape(shape[:2] + [1])
                    x,y = process_batch(x, y,E,batch_shape)
                    pred = model.predict_on_batch(x)
                    pred_all.append(pred) 
                pred_all = np.array(pred_all).sum(axis=0)
                
               
            elif (E==1) & (modeltype=="EMG"):
                print("usleep-EMG")
                
                sig  = data[groups[k]][splits[i]][aid[j]]["PSG"]["EMG"][:]
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

                x = reshaped_sig.reshape(shape)
                y = hyp.reshape(shape[:2] + [1])
                x,y = process_batch(x, y,E,batch_shape)
                pred_all = model.predict_on_batch(x)

        
            true, pred, cm  = compute_counts(pred=pred_all, true=y)
            assert np.sum(cm)==len(pred)
            cm_all += cm 
            dim_pred_all +=len(pred)
            assert dim_pred_all == np.sum(cm_all)
    p,r,a,f1_score  = cal_metric_from_cm(cm_all)

    plot_cm(cm_all,normalize=False,cmap=cols[k],title=groups[k],out_path=output_dir,precision=False)

    groups_all.append(groups[k])
    f1_wake.append(f1_score[0])
    f1_nrem.append(f1_score[1])
    f1_rem.append(f1_score[2])
    recall_wake.append(r[0])
    recall_nrem.append(r[1])
    recall_rem.append(r[2])
    precision_wake.append(p[0])
    precision_nrem.append(p[1])
    precision_rem.append(p[2])
    accuracies_wake.append(a[0])
    accuracies_nrem.append(a[1])
    accuracies_rem.append(a[2])

met = pd.DataFrame(data={'groups_all':groups_all,
                        'f1_wake': f1_wake,
                        'f1_nrem': f1_nrem,
                        'f1_rem': f1_rem,
                        'precision_wake': precision_wake,
                        'precision_nrem': precision_nrem,
                        'precision_rem': precision_rem,
                        'recall_wake': recall_wake,
                        'recall_nrem': recall_nrem,
                        'recall_rem': recall_rem,
                        'accuracies_wake': accuracies_wake,
                        'accuracies_nrem':accuracies_nrem,
                        'accuracies_rem':accuracies_rem
                        })        
#met.to_csv(sys.argv[3]+"metric.csv") 
met.to_csv(output_dir+"metric.csv") 
