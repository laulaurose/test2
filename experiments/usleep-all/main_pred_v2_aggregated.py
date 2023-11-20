import numpy as np
import os
import usleep
import typing
from psg_utils.dataset.sleep_study import SleepStudy
from utime.bin.predict_one import get_sleep_study
import h5py 
import pandas as pd 
import matplotlib.pyplot as plt 
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
        
        return cm

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
    ax.set_xlabel('Predicted label', fontsize=14)  # Adjust the font size as needed
    ax.set_ylabel('True label', fontsize=14)  # Adjust the font size as needed
    ax.set_title(title, fontsize=14)  # Adjust the font size as needed

    im.set_clim(vmin=0, vmax=100)
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

def calc_metrics(experiment,aid_vec,data,E,split,output_p):    
    n_classes = 3
    cm_all    = np.zeros(shape=(n_classes,n_classes))

    data_split_all      = []
    experiment_all      = []
    aid_all             = []
    recall_wake         = []
    recall_nrem         = []
    recall_rem          = []
    precision_wake      = []
    precision_nrem      = []
    precision_rem       = []
    accuracies_wake     = []
    accuracies_nrem     = []
    accuracies_rem      = []
    f1_score_wake       = []
    f1_score_nrem       = []
    f1_score_rem        = []
    length_all          = []
        
    for i in range(len(aid_vec)): # across animal id's 
        print(aid_vec[i])
        ne   = list(data[experiment]["TEST"][aid_vec[i]]['PSG'].keys())
        print(ne)
        sig  = data[experiment]["TEST"][aid_vec[i]]['PSG'][ne[0]][:] # prealloc
        # Assuming 'signal' is the 1D signal of length x
        signal_length = len(sig)
        # Calculate the number of instances 'n' required to form the target shape
        n = signal_length // 512
        batch_shape = [128,11,512,E]

        pred_all = [] # pred across electrodes 
        
        if E==2:
            print("here")
            for e in range(len(ne)-1):
                emg  = data[experiment]["TEST"][aid_vec[i]]['PSG']["EMG"][:]
                sig  = data[experiment]["TEST"][aid_vec[i]]['PSG'][ne[e]][:] # prealloc 
                hyp  = data[experiment]["TEST"][aid_vec[i]]['hypnogram']
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
        else: 
            for e in range(len(ne)):
                sig  = data[experiment]["TEST"][aid_vec[i]]['PSG'][ne[e]][:] # prealloc 
                hyp  = data[experiment]["TEST"][aid_vec[i]]['hypnogram']
                reshaped_sig = sig.reshape(n,512,1)
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
        cm = compute_counts(pred=pred_all, true=y)

        p,r,a,f1_score  = cal_metric_from_cm(cm)

        data_split_all.append(split)
        experiment_all.append(experiment)
        aid_all.append(aid_vec[i])
        recall_wake.append(r[0])
        recall_nrem.append(r[1])
        recall_rem.append(r[2])
        precision_wake.append(p[0])
        precision_nrem.append(p[1])
        precision_rem.append(p[2])
        accuracies_wake.append(a[0])
        accuracies_nrem.append(a[1])
        accuracies_rem.append(a[2])
        f1_score_wake.append(f1_score[0])
        f1_score_nrem.append(f1_score[1])
        f1_score_rem.append(f1_score[2])
        length_all.append(np.sum(cm))

        cm_all+=cm
        print(cm_all)
    
    met = pd.DataFrame(data={'data_split': data_split_all,
                             'experiment_all':experiment_all,
                             'animal':aid_all,
                             'precision_wake': precision_wake,
                             'precision_nrem': precision_nrem,
                            'precision_rem': precision_rem,
                            'recall_wake': recall_wake,
                            'recall_nrem': recall_nrem,
                            'recall_rem': recall_rem,
                            'accuracies_wake': accuracies_wake,
                            'accuracies_nrem':accuracies_nrem,
                            'accuracies_rem':accuracies_rem,
                            'f1_score_wake': f1_score_wake,
                            'f1_score_nrem': f1_score_nrem,
                            'f1_score_rem': f1_score_rem,
                            'length_all':length_all
                            })        
    
    met.to_csv(output_p+"metric_"+"exp_"+experiment+"split_"+str(split)+".csv")  

    return cm_all 

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


data_splits_cv = ["/scratch/users/laurose/usleep-all-split_0/",
                  "/scratch/users/laurose/usleep-all-split_1/",
                  "/scratch/users/laurose/usleep-all-split_2/",
                  "/scratch/users/laurose/usleep-all-split_3/",
                  "/scratch/users/laurose/usleep-all-split_4/"]


# Extract data from preprocess.h5 file that needs to be used for predictions 
from utime.bin.predict import get_datasets, run_pred
args      = ARGS2()
E         = np.int(sys.argv[1]) 
n_classes = 3
    

a_cm_all_al   = np.zeros(shape=(n_classes,n_classes))
a_cm_all_an   = np.zeros(shape=(n_classes,n_classes))
a_cm_all_ko   = np.zeros(shape=(n_classes,n_classes))
a_cm_all_ma   = np.zeros(shape=(n_classes,n_classes))
a_cm_all_se   = np.zeros(shape=(n_classes,n_classes))


groups = ["Alessandro-cleaned-data",
          "Antoine-cleaned-data",
          "Kornum-cleaned-data_v2",
          "Maiken-cleaned-data",
          "Sebastian-cleaned-data"]


for kk in range(len(data_splits_cv)): # across cross validation splits 
    project_dir = data_splits_cv[kk]
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
    data = h5py.File(project_dir+"processed_data.h5")

    # plot for each experiment and for each split 
    cm_all_al = calc_metrics(groups[0],list(data[groups[0]]["TEST"].keys()),data,E,kk,sys.argv[2])
    cm_all_an = calc_metrics(groups[1],list(data[groups[1]]["TEST"].keys()),data,E,kk,sys.argv[2])
    cm_all_ko = calc_metrics(groups[2],list(data[groups[2]]["TEST"].keys()),data,E,kk,sys.argv[2])
    cm_all_ma = calc_metrics(groups[3],list(data[groups[3]]["TEST"].keys()),data,E,kk,sys.argv[2])
    cm_all_se = calc_metrics(groups[4],list(data[groups[4]]["TEST"].keys()),data,E,kk,sys.argv[2])
    
    plot_cm(cm_all_al,normalize=False,cmap="Blues",title=groups[0]+"_split_"+str(kk),out_path=sys.argv[2])
    plot_cm(cm_all_an,normalize=False,cmap="Blues",title=groups[1]+"_split_"+str(kk),out_path=sys.argv[2])
    plot_cm(cm_all_ko,normalize=False,cmap="Blues",title=groups[2]+"_split_"+str(kk),out_path=sys.argv[2])
    plot_cm(cm_all_ma,normalize=False,cmap="Blues",title=groups[3]+"_split_"+str(kk),out_path=sys.argv[2])
    plot_cm(cm_all_se,normalize=False,cmap="Blues",title=groups[4]+"_split_"+str(kk),out_path=sys.argv[2])

    precision_al,recall_al,accuracy_al,f1_al  = cal_metric_from_cm(cm_all_al)
    precision_an,recall_an,accuracy_an,f1_an  = cal_metric_from_cm(cm_all_an)
    precision_ko,recall_ko,accuracy_ko,f1_ko  = cal_metric_from_cm(cm_all_ko)
    precision_ma,recall_ma,accuracy_ma,f1_ma  = cal_metric_from_cm(cm_all_ma)
    precision_se,recall_se,accuracy_se,f1_se  = cal_metric_from_cm(cm_all_se)

    met = pd.DataFrame(data={'alessandro_p_w': [precision_al[0]],
                        'alessandro_p_n': [precision_al[1]],
                        'alessandro_p_r': [precision_al[2]],
                        'alessandro_r_w': [recall_al[0]],
                        'alessandro_r_n': [recall_al[1]],
                        'alessandro_r_r': [recall_al[2]],
                        'alessandro_a_w': [accuracy_al[0]],
                        'alessandro_a_n': [accuracy_al[1]],
                        'alessandro_a_r': [accuracy_al[2]],
                        'alessandro_f1_w': [f1_al[0]],
                        'alessandro_f1_n': [f1_al[1]],
                        'alessandro_f1_r': [f1_al[2]],
                        'antoine_p_w': [precision_an[0]],
                        'antoine_p_n': [precision_an[1]],
                        'antoine_p_r': [precision_an[2]],
                        'antoine_r_w': [recall_an[0]],
                        'antoine_r_n': [recall_an[1]],
                        'antoine_r_r': [recall_an[2]],
                        'antoine_a_w': [accuracy_an[0]],
                        'antoine_a_n': [accuracy_an[1]],
                        'antoine_a_r': [accuracy_an[2]],
                        'antoine_f1_w': [f1_an[0]],
                        'antoine_f1_n': [f1_an[1]],
                        'antoine_f1_r': [f1_an[2]],  
                        'kornum_p_w': [precision_ko[0]],
                        'kornum_p_n': [precision_ko[1]],
                        'kornum_p_r': [precision_ko[2]],
                        'kornum_r_w': [recall_ko[0]],
                        'kornum_r_n': [recall_ko[1]],
                        'kornum_r_r': [recall_ko[2]],
                        'kornum_a_w': [accuracy_ko[0]],
                        'kornum_a_n': [accuracy_ko[1]],
                        'kornum_a_r': [accuracy_ko[2]],
                        'kornum_f1_w': [f1_ko[0]],
                        'kornum_f1_n': [f1_ko[1]],
                        'kornum_f1_r': [f1_ko[2]],
                        'maiken_p_w': [precision_ma[0]],
                        'maiken_p_n': [precision_ma[1]],
                        'maiken_p_r': [precision_ma[2]],
                        'maiken_r_w': [recall_ma[0]],
                        'maiken_r_n': [recall_ma[1]],
                        'maiken_r_r': [recall_ma[2]],
                        'maiken_a_w': [accuracy_ma[0]],
                        'maiken_a_n': [accuracy_ma[1]],
                        'maiken_a_r': [accuracy_ma[2]],
                        'maiken_f1_w': [f1_ma[0]],
                        'maiken_f1_n': [f1_ma[1]],
                        'maiken_f1_r': [f1_ma[2]],
                        'seb_p_w': [precision_se[0]],
                        'seb_p_n': [precision_se[1]],
                        'seb_p_r': [precision_se[2]],
                        'seb_r_w': [recall_se[0]],
                        'seb_r_n': [recall_se[1]],
                        'seb_r_r': [recall_se[2]],
                        'seb_a_w': [accuracy_se[0]],
                        'seb_a_n': [accuracy_se[1]],
                        'seb_a_r': [accuracy_se[2]],
                        'seb_f1_w': [f1_se[0]],
                        'seb_f1_n': [f1_se[1]],
                        'seb_f1_r': [f1_se[2]]
                        })        

    met.to_csv(sys.argv[2]+"split_"+str(kk)+"metric.csv") 

    # accumulate for each experiment across splits
    a_cm_all_al   += cm_all_al
    a_cm_all_an   += cm_all_an
    a_cm_all_ko   += cm_all_ko
    a_cm_all_ma   += cm_all_ma
    a_cm_all_se   += cm_all_se

# plot across cv splits 
plot_cm(a_cm_all_al,normalize=False,cmap="Blues",title=groups[0],out_path=sys.argv[2],precision=False)
plot_cm(a_cm_all_an,normalize=False,cmap="Blues",title=groups[1],out_path=sys.argv[2],precision=False)
plot_cm(a_cm_all_ko,normalize=False,cmap="Blues",title=groups[2],out_path=sys.argv[2],precision=False)
plot_cm(a_cm_all_ma,normalize=False,cmap="Blues",title=groups[3],out_path=sys.argv[2],precision=False)
plot_cm(a_cm_all_se,normalize=False,cmap="Blues",title=groups[4],out_path=sys.argv[2],precision=False)
    
# plot across cv splits and experiments 
plot_cm(a_cm_all_al+a_cm_all_an+a_cm_all_ko+a_cm_all_ma+a_cm_all_se,normalize=False,cmap="Blues",title="overall",out_path=sys.argv[2],precision=False)
ta_precision,ta_recall,ta_accuracy,ta_f1  = cal_metric_from_cm(a_cm_all_al+a_cm_all_an+a_cm_all_ko+a_cm_all_ma+a_cm_all_se)

met = pd.DataFrame(data={'p_w': [ta_precision[0]],
                        'p_n': [ta_precision[1]],
                        'p_r': [ta_precision[2]],
                        'r_w': [ta_recall[0]],
                        'r_n': [ta_recall[1]],
                        'r_r': [ta_recall[2]],
                        'a_w': [ta_accuracy[0]],
                        'a_n': [ta_accuracy[1]],
                        'a_r': [ta_accuracy[2]],
                        'f1_w': [ta_f1[0]],
                        'f1_n': [ta_f1[1]],
                        'f1_r': [ta_f1[2]]
                        })        

met.to_csv(sys.argv[2]+"metric.csv") 
