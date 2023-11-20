import ruamel.yaml
import sys 

# Load the hparams file
yaml        = ruamel.yaml.YAML()

project_dir = ["/scratch/users/laurose/usleep-all-split_0/",
               "/scratch/users/laurose/usleep-all-split_1/",
               "/scratch/users/laurose/usleep-all-split_2/",
               "/scratch/users/laurose/usleep-all-split_3/",
               "/scratch/users/laurose/usleep-all-split_4/",
               "/scratch/users/laurose/usleep-EEG-split_0/",
               "/scratch/users/laurose/usleep-EEG-split_1/",
               "/scratch/users/laurose/usleep-EEG-split_2/",
               "/scratch/users/laurose/usleep-EEG-split_3/",
               "/scratch/users/laurose/usleep-EEG-split_4/",
               "/scratch/users/laurose/usleep-EMG-split_0/",
               "/scratch/users/laurose/usleep-EMG-split_1/",
               "/scratch/users/laurose/usleep-EMG-split_2/",
               "/scratch/users/laurose/usleep-EMG-split_3/",
               "/scratch/users/laurose/usleep-EMG-split_4/"]

for k in range(len(project_dir)):
    # open and edit hparams 
    filename_hparams = project_dir[k] + "hyperparameters/hparams.yaml"

    with open(filename_hparams, 'r') as yaml_file:
        data = yaml.load(yaml_file)

    # Modify the learning rate
    data['fit']['optimizer_kwargs']['learning_rate'] = 1.0e-05# Change to your desired value
    data['fit']['n_epochs'] = 1200# Change to your desired value
    data['CB_es']['kwargs']['patience'] = 200

    # Save the modified YAML file while preserving formatting
    with open(filename_hparams, 'w') as modified_yaml_file:
        yaml.dump(data, modified_yaml_file)

    # open and change pre proc 
    filename_pre_proc = project_dir[k] + "hyperparameters/pre_proc_hparams.yaml"

    with open(filename_pre_proc, 'r') as yaml_file:
        data = yaml.load(yaml_file)

    # Modify the learning rate
    data['fit']['optimizer_kwargs']['learning_rate'] = 1.0e-05# Change to your desired value
    data['fit']['n_epochs'] = 1200# Change to your desired value
    data['CB_es']['kwargs']['patience'] = 200

    if 'init_epochs' in data.get('fit', {}):
        # If it exists, set it to 0
        data['fit']['init_epochs'] = 401

    # Save the modified YAML file while preserving formatting
    with open(filename_pre_proc, 'w') as modified_yaml_file:
        yaml.dump(data, modified_yaml_file)



