import ruamel.yaml
import sys 

# Load the hparams file
yaml        = ruamel.yaml.YAML()

project_dir = ["/scratch/users/laurose/models/usleep_all/",
               "/scratch/users/laurose/models/usleep_EEG/",
               "/scratch/users/laurose/models/usleep_EMG/"]

for k in range(len(project_dir)):
    # open and edit hparams 
    filename_hparams = project_dir[k] + "hyperparameters/hparams.yaml"

    with open(filename_hparams, 'r') as yaml_file:
        data = yaml.load(yaml_file)

    # Modify the learning rate
    data['fit']['optimizer_kwargs']['learning_rate'] = 1.0e-04# Change to your desired value
    data['fit']['n_epochs'] = 400# Change to your desired value
    data['CB_es']['kwargs']['patience'] = 400

    # Save the modified YAML file while preserving formatting
    with open(filename_hparams, 'w') as modified_yaml_file:
        yaml.dump(data, modified_yaml_file)

    # open and change pre proc 
    filename_pre_proc = project_dir[k] + "hyperparameters/pre_proc_hparams.yaml"

    with open(filename_pre_proc, 'r') as yaml_file:
        data = yaml.load(yaml_file)

    # Modify the learning rate
    data['fit']['optimizer_kwargs']['learning_rate'] = 1.0e-04# Change to your desired value
    data['fit']['n_epochs'] = 400# Change to your desired value
    data['CB_es']['kwargs']['patience'] = 400

    if 'init_epochs' in data.get('fit', {}):
        # If it exists, set it to 0
        data['fit']['init_epochs'] = 0

    # Save the modified YAML file while preserving formatting
    with open(filename_pre_proc, 'w') as modified_yaml_file:
        yaml.dump(data, modified_yaml_file)


