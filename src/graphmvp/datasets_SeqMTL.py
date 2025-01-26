import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from src.graphmvp.datasets import mol_to_graph_data_obj_simple


from tqdm import tqdm
def shared_extractor(smiles_list, rdkit_mol_objs, labels):
    data_list = []
    data_smiles_list = []
    if labels.ndim == 1:
        labels = np.expand_dims(labels, axis=1)
    for i in tqdm(range(len(smiles_list)), desc='Processing data'):
        rdkit_mol = rdkit_mol_objs[i]
        if rdkit_mol is None:
            continue
        data = mol_to_graph_data_obj_simple(rdkit_mol)
        data.id = torch.tensor([i])
        data.y = torch.tensor(labels[i])
        data_list.append(data)
        data_smiles_list.append(smiles_list[i])

    return data_list, data_smiles_list

class MoleculeDataset_ADME(Dataset):
    def __init__(self, df_fname, transform=None,
                 pre_transform=None, pre_filter=None, empty=False, 
                 smiles_col='SMILES', label_col='Y', df = None):
        self.transform = transform
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform
        self.df_fname = df_fname
        self.processed_path = df_fname.replace('.csv', '_graphmvp.pt')
        self.smiles_col = smiles_col
        self.label_col = label_col
        
        import os
        if not os.path.exists(self.processed_path):
            self._process(df)
        self.data = torch.load(self.processed_path)
        print('Dataset: {}\nData #: {}'.format(self.df_fname, len(self.data)))

    def _process(self, df):
        smiles_list, rdkit_mol_objs, labels = \
                self.load_data_from_df(df)

        data_list, data_smiles_list = shared_extractor(
                smiles_list, rdkit_mol_objs, labels)
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(data_list, self.processed_path)
        return
    
    def load_data_from_df(self, df):
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        if df is None:
            df = pd.read_csv(self.df_fname)
        smiles_list = df[self.smiles_col]
        labels = df[self.label_col].values
        rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
        return smiles_list, rdkit_mol_objs_list, labels

    def __len__(self):
        return len(self.data)
    
    def get(self, idx):
        return self.data[idx]
    
    def __getitem__(self, idx):
        return self.get(idx)
    
    def _indices(self):
        return range(len(self.data))
    
class mvpDatasetWrapper_MTL(object):
    def __init__(self, train_data_pth, valid_data_path, test_data_path, target_cols, smiles_col='SMILES'):
        self.targets = target_cols

        train_df = pd.read_csv(train_data_pth)
        valid_df = pd.read_csv(valid_data_path)
        test_df = pd.read_csv(test_data_path)

        train_labels = train_df[self.targets].values
        valid_labels = valid_df[self.targets].values
        test_labels = test_df[self.targets].values

        self.tasks = []
        for i in range(len(self.targets)):
            train_label = train_labels[:, i]
            non_nan_train_label = train_label[~np.isnan(train_label)]
            if len(set(non_nan_train_label)) == 2:
                self.tasks.append('classification')
            else:
                self.tasks.append('regression')
        print('Task types:', self.tasks)

        self.scalers = []
        for i in range(len(self.targets)):
            if self.tasks[i] == 'regression':
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                train_labels[:, i] = scaler.fit_transform(train_labels[:, i].reshape(-1, 1)).flatten()
                valid_labels[:, i] = scaler.transform(valid_labels[:, i].reshape(-1, 1)).flatten()
                test_labels[:, i] = scaler.transform(test_labels[:, i].reshape(-1, 1)).flatten()
                self.scalers.append(scaler)
            else:
                self.scalers.append(None)

        train_df[self.targets] = train_labels
        valid_df[self.targets] = valid_labels
        test_df[self.targets] = test_labels

        self.train_dataset = MoleculeDataset_ADME(train_data_pth, smiles_col=smiles_col, label_col=self.targets, df = train_df)
        self.valid_dataset = MoleculeDataset_ADME(valid_data_path, smiles_col=smiles_col, label_col=self.targets, df = valid_df)
        self.test_dataset = MoleculeDataset_ADME(test_data_path, smiles_col=smiles_col, label_col=self.targets, df = test_df)
        
    def get_data_loaders(self, batch_size, shuffle=True):
        if shuffle:
            train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle, generator = torch.Generator().manual_seed(42))
        else:
            train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle)
        valid_loader = DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, valid_loader, test_loader
    

############################################################################################################
# Inference
############################################################################################################

class MoleculeDataset_inference_ADME(Dataset):
    def __init__(self, smiles_list, transform=None,
                 pre_transform=None, pre_filter=None, empty=False, labels=None):
        self.transform = transform
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform
        self.data, self.smiles_list = self._process_smiles(smiles_list, labels)
        print('Processed Data #: {}'.format(len(self.data)))

    def _process_smiles(self, smiles_list, labels):
        from tqdm import tqdm
        
        def shared_extractor(smiles_list, labels):
            from rdkit import Chem
            failed_list = []
            data_list = []
            success_list = []
            if labels is None:
                labels = np.zeros(len(smiles_list))
            if labels.ndim == 1:
                labels = np.expand_dims(labels, axis=1)
            
            for i in tqdm(range(len(smiles_list)), desc='Processing data'):
                rdkit_mol = Chem.MolFromSmiles(smiles_list[i])
                if rdkit_mol is None:
                    failed_list.append((i, smiles_list[i]))
                    continue
                try:
                    data = mol_to_graph_data_obj_simple(rdkit_mol)
                except:
                    failed_list.append((i, smiles_list[i]))
                    continue
                data.id = torch.tensor([i])
                data.y = torch.tensor(labels[i])
                data_list.append(data)
                success_list.append(smiles_list[i])

            return data_list, success_list, failed_list
        
        data_list, success_list, failed_list = shared_extractor(
                smiles_list, labels)
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # print failed list
        if len(failed_list) > 0:
            print(f'!!! Total {len(failed_list)} failed to process: {failed_list}')
        return data_list, success_list
    
    def __len__(self):
        return len(self.data)
    
    def get(self, idx):
        return self.data[idx]
    
    def __getitem__(self, idx):
        return self.get(idx)
    
    def _indices(self):
        return range(len(self.data))
    
class mvpDatasetWrapper_inference(object):
    def __init__(self, smiles_list):
        self.dataset = MoleculeDataset_inference_ADME(smiles_list)
        
    def get_data_loaders(self, batch_size, shuffle=False):
        if shuffle:
            loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, generator = torch.Generator().manual_seed(42))
        else:
            loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)
        return loader
    
    def get_smiles_list(self):
        return self.dataset.smiles_list