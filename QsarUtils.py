#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os,time,copy
from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem,MACCSkeys
from rdkit.Chem.SaltRemover import SaltRemover
import RDKitCalDes as rdcd
import warnings
import matplotlib.pyplot as plt
from matplotlib import colors
plt.switch_backend('agg')
import sys
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import deepchem as dc

def Sec2Time(seconds):  # convert seconds to time
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return ("{:02d}h:{:02d}m:{:02d}s".format(h, m, s))

def CalMolDescriptors(in_file_path, saved_dir=None, smi_column="Smiles", name_column="", label_column="",low_accuracy_column="",
                      des_type='all', sep=',', out_file_prefix=None, tmp_file_name="tmp",use_cache=False):
    """des_type: rdkit/ecfp4/macss/rdkfp/ttfp/..."""
    ### 输入文件处理
    # print(smi_column)
    t0 = time.time()
    init_df = pd.read_csv(in_file_path, sep=sep)
    # print(init_df.head())
    in_df = pd.DataFrame(init_df.loc[:, smi_column].values, columns=['Smiles'])

    if name_column not in ["",None,"None"]:
        in_df['Name'] = init_df.loc[:, name_column].values
    else:
        in_df['Name'] = in_df.index.astype(str) + '_' + in_df['Smiles']
        init_df['Name'] = in_df.index.astype(str) + '_' + in_df['Smiles']
        name_column = 'Name'

    parent_dir = os.path.dirname(in_file_path)
    if (saved_dir == None) or (saved_dir == "None"):
        saved_dir = parent_dir
    else:
        try:
            os.mkdir(saved_dir)
        except:
            pass

    if type(out_file_prefix) == str:
        pass
    else:
        in_file_base_name = os.path.basename(in_file_path)
        out_file_prefix = in_file_base_name[:in_file_base_name.rfind('.')]

    if type(des_type) == str:
        des_type = [des_type]
    if (type(des_type) == list) or (type(des_type) == tuple):
        for t in des_type:
            if t in ['rdkit', 'ecfp4', 'maccs', 'rdkfp', 'ttfp', 'circularfp','pubchemfp','mordred','HOMO', 'LUMO', 'oxidation','reduction',]:
                pass
            elif t == "all":
                des_type = ['rdkit', 'ecfp4', 'maccs','rdkfp','ttfp', 'circularfp','pubchemfp','mordred','HOMO', 'LUMO', 'oxidation','reduction',]
                break
            elif t == "FP":
                des_type = ['ecfp4', 'maccs','rdkfp','ttfp', 'circularfp','pubchemfp',]
                break
            elif t == "CD":
                des_type = ['rdkit', 'mordred',]
                break
            elif t == "QM":
                des_type = ['HOMO', 'LUMO', 'oxidation','reduction',]
                break

            else:
                print("The des_type '{}' is invalid".format(t))
                des_type.remove(t)

    else:
        print("The des_type '{}' is invalid".format(des_type))
        exit(0)
    
    des_file = os.path.join(saved_dir, out_file_prefix+"_{}.csv".format("_".join(des_type)))
    if os.path.exists(des_file) and use_cache:
        df_mol_des = pd.read_csv(des_file)
        print("Find existed des file: ".format(des_file))
        print('Molecule descriptors shape: {}'.format(df_mol_des.shape))
        return df_mol_des

    invalid_id = []
    mols = []
    smi_list = []
    temp_smi_path = os.path.join(saved_dir,'{}.smi'.format(tmp_file_name))
    wt = Chem.SmilesWriter(temp_smi_path, delimiter='\t', includeHeader=False)
    for i, smi in enumerate(in_df['Smiles']):
        # print(in_df.head())
        m = Chem.MolFromSmiles(smi)
        smi_list.append(smi)
        if m:
            remover = SaltRemover()
            m = remover.StripMol(m, dontRemoveEverything=True)
            m.SetProp('_Name', str(in_df.loc[i, 'Name']))
            wt.write(m)
            mols.append(m)
        else:
            invalid_id.append(i)
    wt.close()
    print('\nNumber of input molecules: {}\n'.format(len(in_df)))
    print('Number of invalid molecules: {}\n'.format(len(invalid_id)))
    in_df.drop(labels=invalid_id,axis=0,inplace=True)
    in_df.reset_index(drop=True,inplace=True)

    # mols = Chem.SmilesMolSupplier(temp_smi_path, smilesColumn=0, nameColumn=1, delimiter='\t', titleLine=False)
    t1 = time.time()
    print('Preprocessing done, time cost: {}'.format(Sec2Time(t1 - t0)))

    # 计算描述符
    temp_csv_path = temp_smi_path[:-4] + '.csv'
    des_dfs = []
    t2 = time.time()
    if "rdkit" in des_type:
        if os.path.exists(temp_csv_path):
            try:
                os.remove(temp_csv_path)
            except Exception as e:
                print(e)
                # shutil.rmtree(temp_csv_path)
        in_params = 'smilesColumn,1,smilesNameColumn,2,smilesDelimiter,tab,smilesTitleLine,False'
        options = {'--autocorr2DExclude': 'yes',
                   '--descriptorNames': '',
                   '--examples': False,
                   '--fragmentCount': 'yes',
                   '--help': False,
                   '--infile': temp_smi_path,
                   '--infileParams': in_params,
                   '--list': False,
                   '--mode': '2D',
                   '--mp': 'no',
                   '--mpParams': 'auto',
                   '--outfile': temp_csv_path,
                   '--outfileParams': 'auto',
                   '--overwrite': False,
                   '--precision': '3',
                   '--smilesOut': 'no',
                   '--workingdir': None}
        rdcd.main(options)
        rdkit2d_df = pd.read_csv(temp_csv_path).drop(labels='Ipc', axis=1)
        des_dfs.append(rdkit2d_df)
        # rdkit2d_df.insert(1,'Smiles',in_df['Smiles'].values)
        # rdkit2d_path = os.path.join(saved_dir, '{}_rdkit.csv'.format(out_file_prefix))
        # rdkit2d_df.to_csv(rdkit2d_path, index=False)
        # print("Calculated \033[36;1mrdkit2D descriptors\033[0m have been saved in '\033[45;1m{}\033[0m'".format(
        #     rdkit2d_path))
        print("Getting \033[36;1mrdkit2D Chemical descriptors\033[0m successfully")
#         print('Time cost: {}'.format(Sec2Time(time.time() - t2)))

    if 'ecfp4' in des_type:
        t2 = time.time()
        mgfps2 = np.array([list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048,
                                                                      useChirality=True)) for mol in mols])

        mg_df2 = pd.DataFrame(
            mgfps2, columns=['ECFP4_{}'.format(i) for i in range(mgfps2.shape[1])])
        mg_df2 = pd.concat([in_df['Name'], mg_df2], axis=1, sort=False)
        # mg_df2.to_csv(os.path.join(saved_dir, out_file_prefix + "_ecfp4.csv"), index=False)
        des_dfs.append(mg_df2)

        print("Getting \033[36;1mECFP4 fingerprints\033[0m successfully")
#         print('Time cost: {}'.format(Sec2Time(time.time() - t2)))

    if 'rdkfp' in des_type:
        t2 = time.time()
        rdkfps = np.array([list(Chem.RDKFingerprint(mol, fpSize=2048)) for mol in mols])

        rdk_df = pd.DataFrame(
            rdkfps, columns=['RDK_{}'.format(i) for i in range(rdkfps.shape[1])])
        rdk_df = pd.concat([in_df['Name'], rdk_df], axis=1, sort=False)
        # mg_df2.to_csv(os.path.join(saved_dir, out_file_prefix + "_rdkfp.csv"), index=False)
        des_dfs.append(rdk_df)
        print("Getting \033[36;1mRDKfingerprints\033[0m successfully")
        # print('Time cost: {}'.format(Sec2Time(time.time() - t2)))

    if 'ttfp' in des_type:
        t2 = time.time()
        ttfps = np.array([list(AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=2048)) for mol in mols])

        tt_df = pd.DataFrame(
            ttfps, columns=['TT_{}'.format(i) for i in range(ttfps.shape[1])])
        tt_df = pd.concat([in_df['Name'], tt_df], axis=1, sort=False)
        # mg_df2.to_csv(os.path.join(saved_dir, out_file_prefix + "_rdkfp.csv"), index=False)
        des_dfs.append(tt_df)

        print("Getting \033[36;1mTopologicalTorsionFingerprints\033[0m successfully")
        # print('Time cost: {}'.format(Sec2Time(time.time() - t2)))

    if 'maccs' in des_type:
        t2 = time.time()
        maccsfp = np.array([list(MACCSkeys.GenMACCSKeys(mol)) for mol in mols])
        maccs_df = pd.DataFrame(
            maccsfp, columns=['MACCS_{}'.format(i) for i in range(maccsfp.shape[1])])
        maccs_df = pd.concat([in_df['Name'], maccs_df], axis=1, sort=False)
        # maccs_df.to_csv(os.path.join(saved_dir, out_file_prefix + "_maccs.csv"), index=False)
        des_dfs.append(maccs_df)

        print("Getting \033[36;1mMACCS fingerprints\033[0m successfully")
        # print('Time cost: {}'.format(Sec2Time(time.time() - t2)))
    if 'circularfp' in des_type:
        t2 = time.time()
        # Example 1: (size = 2048, radius = 4)
        # featurizer = dc.feat.CircularFingerprint()
        featurizer_cc = dc.feat.CircularFingerprint(size=2048, radius=4)
        ccfp = featurizer_cc.featurize(smi_list)

        cc_df = pd.DataFrame(
            ccfp, columns=['circularfp_{}'.format(i) for i in range(ccfp.shape[1])])
        cc_df = pd.concat([in_df['Name'], cc_df], axis=1, sort=False)
        # mg_df2.to_csv(os.path.join(saved_dir, out_file_prefix + "_ecfp4.csv"), index=False)
        des_dfs.append(cc_df)

        print("Getting \033[36;1mCircular fingerprints\033[0m successfully")


    if 'pubchemfp' in des_type:
        t2 = time.time()
        featurizer_pc = dc.feat.PubChemFingerprint()
        Pbfp = featurizer_pc.featurize(smi_list)

        Pb_df = pd.DataFrame(
            Pbfp, columns=['pubchemfp_{}'.format(i) for i in range(Pbfp.shape[1])])
        Pb_df = pd.concat([in_df['Name'], Pb_df], axis=1, sort=False)
        # mg_df2.to_csv(os.path.join(saved_dir, out_file_prefix + "_ecfp4.csv"), index=False)
        des_dfs.append(Pb_df)

        print("Getting \033[36;1mPubChem fingerprints\033[0m successfully")

    if 'mordred' in des_type:
        t2 = time.time()
        featurizer_mordred = dc.feat.MordredDescriptors(ignore_3D=False)
        Mdfp = featurizer_mordred.featurize(smi_list)
        from mordred import Calculator, descriptors
        mordred_des_name_list = Calculator(descriptors, ignore_3D=False).descriptors
        print(mordred_des_name_list)
        print(type(mordred_des_name_list))
        print(len(mordred_des_name_list))
        Md_df = pd.DataFrame(
            Mdfp, columns=[str(mordred_des_name_list[i]) for i in range(len(mordred_des_name_list))])
        Md_df = pd.concat([in_df['Name'], Md_df], axis=1, sort=False)
        # mg_df2.to_csv(os.path.join(saved_dir, out_file_prefix + "_ecfp4.csv"), index=False)
        des_dfs.append(Md_df)

        print("Getting \033[36;1mMorderd Chemical\033[0m successfully")

    if 'HOMO' in des_type:
        t2 = time.time()
        sys.path.append('./qsroar_code/')
        from properties import properties
        HOMO_list = []
        for smi in smi_list:
            HOMO_list.append(properties(smi, homo=None, lumo=None)['homo'] )
        homo = np.array([HOMO_list]).reshape(-1,1)


        homo_df2 = pd.DataFrame(
            homo, columns=['HOMO_qsroar'])
        homo_df2 = pd.concat([in_df['Name'], homo_df2], axis=1, sort=False)
        des_dfs.append(homo_df2)

        print("Getting \033[36;1mHOMO by qsroar\033[0m successfully")

    if 'LUMO' in des_type:
        t2 = time.time()
        sys.path.append('./qsroar_code/')
        from properties import properties
        LUMO_list = []
        for smi in smi_list:
            LUMO_list.append(properties(smi, homo=None, lumo=None)['homo'] )
        lumo = np.array([LUMO_list]).reshape(-1,1)


        lumo_df2 = pd.DataFrame(
            lumo, columns=['LUMO_qsroar'])
        lumo_df2 = pd.concat([in_df['Name'], lumo_df2], axis=1, sort=False)

        des_dfs.append(lumo_df2)
        print("Getting \033[36;1mLUMO by qsroar\033[0m successfully")

    if 'oxidation' in des_type:
        t2 = time.time()
        sys.path.append('./qsroar_code/')
        from properties import properties
        oxidation_list = []
        for smi in smi_list:
            oxidation_list.append(properties(smi, homo=None, lumo=None)['oxidation'] )
        oxidation = np.array([oxidation_list]).reshape(-1,1)


        oxidation_df2 = pd.DataFrame(
            oxidation, columns=['oxidation_qsroar'])
        oxidation_df2 = pd.concat([in_df['Name'], oxidation_df2], axis=1, sort=False)

        des_dfs.append(oxidation_df2)
        print("Getting \033[36;1mOxidation potential by qsroar\033[0m successfully")

    if 'reduction' in des_type:
        t2 = time.time()
        sys.path.append('./qsroar_code/')
        from properties import properties
        reduction_list = []
        for smi in smi_list:
            reduction_list.append(properties(smi, homo=None, lumo=None)['reduction'] )
        reduction = np.array([reduction_list]).reshape(-1,1)


        reduction_df2 = pd.DataFrame(
            reduction, columns=['reduction_qsroar'])
        reduction_df2 = pd.concat([in_df['Name'], reduction_df2], axis=1, sort=False)

        des_dfs.append(reduction_df2)
        print("Getting \033[36;1mReduction potential by qsroar\033[0m successfully")




    if os.path.exists(temp_csv_path):
        try:
            os.remove(temp_csv_path)
        except Exception as e:
            print(e)
            # shutil.rmtree(temp_csv_path)
    if os.path.exists(temp_smi_path):
        try:
            os.remove(temp_smi_path)
        except Exception as e:
            print(e)
            # shutil.rmtree(temp_smi_path)

    df_mol_des = pd.concat(des_dfs, axis=1, sort=False).reset_index(drop=True).copy()
    cpd_idx = df_mol_des.iloc[:, 0]
    df_mol_des.drop(labels=['Name'], axis=1, inplace=True)
    print('Feature matrix shape: {}'.format(df_mol_des.shape))
    # print('Molecule descriptors shape: {}'.format(df_mol_des.shape))
    df_mol_des.insert(0, 'Name', cpd_idx)
    df_mol_des.insert(1, 'Smiles', init_df.loc[:,smi_column])
    df_mol_des.dropna(axis=0, inplace=True)

    if label_column not in ["", None, "None"]:
#         df_mol_des = pd.concat([df_mol_des,init_df.iloc[:,label_column-1]],axis=1)
        # assert len(df_mol_des) == len(init_df)
        label_df = init_df.loc[:,[name_column, label_column]].rename(columns={name_column: "Name", label_column: "labels"})
        des_columns = df_mol_des.columns.to_list()[2:]
        df_mol_des = df_mol_des.merge(label_df, on="Name", how="left")
        df_mol_des = df_mol_des.reindex(columns=["Name","Smiles","labels"]+des_columns)
        # df_mol_des.insert(2, 'labels', label_df)
#         df_mol_des = df_mol_des.rename(columns={init_df.columns[label_column-1]:"labels"})
    if low_accuracy_column not in ["", None, "None"]:
        low_df = init_df.loc[:,[name_column, low_accuracy_column]].rename(columns={name_column: "Name", low_accuracy_column: "low_accuracy_result"})
        # print("low_df",low_df)
        # print("df_mol_des",df_mol_des)        
        des_columns = df_mol_des.columns.to_list()[3:]
        # print("des_columns",des_columns)
        # sys.exit()
        df_mol_des = df_mol_des.merge(low_df, on="Name", how="left")
        df_mol_des = df_mol_des.reindex(columns=["Name","Smiles","labels","low_accuracy_result"]+des_columns)


    df_mol_des.to_csv(des_file,index=False)
    print("Des file saved: {}".format(des_file))

    return df_mol_des

if __name__ == "__main__":
    pass