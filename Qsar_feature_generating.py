#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os, time, copy
# import hyperopt
import pandas as pd
import numpy as np
from numpy.random import RandomState
from QsarUtils import CalMolDescriptors, Sec2Time
import docopt
import MiscUtil
import RDKitUtil


def feature_generation(in_file_path, saved_dir, 
         des_type=("all",), smi_column="smiles", label_column="high-accuracy result", low_accuracy_column="low-accuracy result",
         sep=",",):
    """
    :param in_file_path:
    :param saved_dir:
    :param des_type: str/list/tup, 'rdkit' / 'ecfp4'/ 'macss' / 'rdkfp' / 'ttfp' / 'all'
    :param smi_column:
    :param label_column:
    :param random_state:
    :return:
    """

    in_df = pd.read_csv(in_file_path,sep=sep)
    print("shape of the input data:",in_df.shape)
    print("the first 5 rows of the input data:\n", in_df.head())
    df_mol_des = CalMolDescriptors(in_file_path, saved_dir=saved_dir, smi_column=smi_column,
                                  label_column=label_column, low_accuracy_column=low_accuracy_column,
                                  des_type=des_type, sep=sep, out_file_prefix=None,)

    # df_mol_simple = df_mol_des[["Name","Smiles","low-accuracy results","high-accuracy results"]]
    print("the first 5 rows of the feature matrix: \n(including the data at different levels)\n", df_mol_des.head())

if __name__ == "__main__":

    t0 = time.time()
    data_dir = r"C:\Users\Jerry Wang\Desktop\delta_learning\deepchem\test"
    in_file_name = "HOF.csv"
    in_file_path = os.path.join(data_dir, in_file_name)
    smi_column = "SMILES"
    label_column = "Exp."            #high-accuracy result
    low_accuracy_column = "DFT"      #low-accuracy result

    # des_type = ( "FP (short for fingerprint, 6 types for now)":'ecfp4', 'maccs','rdkfp','ttfp', 'pubchemfp','circularfp',
                 # "CD (short for chemical descriptors, 2 types for now)": 'rdkit', 'mordred',
                 # "QM" (4 types for now): 'HOMO', 'LUMO', 'oxidation', 'reduction',   # not included for now
                 # "all"(short for all available representations, 12 types in total)
                 # )
    # des_type = ('rdkit','ecfp4', 'maccs','rdkfp','ttfp')
    # des_type = ('CD','LUMO','HOMO',)
    des_type = ("FP","CD") #"," is needed

    saved_dir = os.path.join(data_dir, "{}_{}".format(in_file_name[:-4],"_".join(des_type)))
    print("Data saved at:",saved_dir)

    
    feature_generation(in_file_path, saved_dir, 
                    des_type=des_type, smi_column=smi_column, label_column=label_column,low_accuracy_column=low_accuracy_column,)

    print("Feature generation done, time cost: {}".format(Sec2Time(time.time()-t0)))

