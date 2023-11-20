
import ruamel.yaml
import sys 

# Load the hparams file
yaml        = ruamel.yaml.YAML()
# open and change pre proc 
project_dirs_all = ["usleep-all-split_0/",
                    "usleep-all-split_1/",
                    "usleep-all-split_2/",
                    "usleep-all-split_3/",
                    "usleep-all-split_4/",
                    "usleep-EEG-split_0/",
                    "usleep-EEG-split_1/",
                    "usleep-EEG-split_2/",
                    "usleep-EEG-split_3/",
                    "usleep-EEG-split_4/",
                    "usleep-EMG-split_0/",
                    "usleep-EMG-split_1/",
                    "usleep-EMG-split_2/",
                    "usleep-EMG-split_3/",
                    "usleep-EMG-split_4/",
                    ]

files = ["Alessandro-cleaned-data.yaml","Antoine-cleaned-data.yaml","Kornum-cleaned-data_v2.yaml",
         "Maiken-cleaned-data.yaml","Sebastian-cleaned-data.yaml"]

main_p = ["/scratch/users/laurose/usleep_all/data/Alessandro-cleaned-data/processed/views/5_CV/",
          "/scratch/users/laurose/usleep_all/data/Antoine-cleaned-data/processed/views/5_CV/",
          "/scratch/users/laurose/usleep_all/data/Kornum-cleaned-data_v2/processed/kornum/views/5_CV/",
          "/scratch/users/laurose/usleep_all/data/Maiken-cleaned-data/processed/views/5_CV/",
          "/scratch/users/laurose/usleep_all/data/Sebastian-cleaned-data/processed/views/5_CV/"]

for k in range(len(project_dirs_all)): # outer CV split 
    for j in range(len(files)): # across experiments 
        filename = project_dirs_all[k] + "hyperparameters/dataset_configurations/"+files[j]
        print(filename)
        
        with open(filename, 'r') as yaml_file:
            data = yaml.load(yaml_file)

            # Modify pathways 
            data['train_data']['data_dir'] = main_p[j]+project_dirs_all[k].split("-")[-1]+"train/"
            data['val_data']['data_dir'] = main_p[j]+project_dirs_all[k].split("-")[-1]+"val/"
            data['test_data']['data_dir'] = main_p[j]+project_dirs_all[k].split("-")[-1]+"test/"

            

        # Save the modified YAML file while preserving formatting
        with open(filename, 'w') as modified_yaml_file:
            yaml.dump(data, modified_yaml_file)
