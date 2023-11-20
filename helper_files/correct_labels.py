from argparse import ArgumentParser
import glob
import os
import numpy as np 
import csv 
import pandas as pd 
import matplotlib.pyplot as plt


experts = ["/scratch/users/laurose/usleep_all/data/Alessandro-cleaned-data/processed/",
           "/scratch/users/laurose/usleep_all/data/Antoine-cleaned-data/processed/",
           "/scratch/users/laurose/usleep_all/data/Kornum-cleaned-data_v2/processed/kornum/",
           "/scratch/users/laurose/usleep_all/data/Maiken-cleaned-data/processed/",
           "/scratch/users/laurose/usleep_all/data/Sebastian-cleaned-data/processed/"
           ]

cv_splits = ["views/5_CV/split_0/",
          "views/5_CV/split_1/",
          "views/5_CV/split_2/",
          "views/5_CV/split_3/",
          "views/5_CV/split_4/"]

cv_splits_2  = ["test/","train/","val/"]

for j in range(len(experts)):
    for d in range(len(cv_splits)):
        for s in range(len(cv_splits_2)):
            file_paths = glob.glob(experts[j]+cv_splits[d]+cv_splits_2[s]+"*/labels_2.ids")
            for n in range(len(file_paths)): 
                df = pd.read_csv(file_paths[n], delimiter=',',header=None)

                if df.iloc[-1,2]==4:
                    df.iloc[-1,2] =  df.iloc[-2,2]
                elif df.iloc[0,2]==4:
                    df.iloc[0,2] = df.iloc[1,2]

                indices = df.index[df.iloc[:,2] == 4]

                for i in indices:
                    if (df.iloc[i,2]==4) &  (df.iloc[i+1,2]!=1):
                        df.iloc[i,2] = df.iloc[i-1,2]
                    else:
                        df.iloc[i,2] = 1
                output_file_path = file_paths[n][:-12]+'/labels_3.ids'
                df.to_csv(output_file_path, sep=',', index=False,header=False)
                assert len(np.where(df.iloc[:,2]==4)[0])==0
               


