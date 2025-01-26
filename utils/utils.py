import numpy as np
import torch
from datetime import datetime

class Logger:
    def __init__(self, model_name):
        import os 
        if not os.path.exists('log'):
            os.makedirs('log')
        self.model_name=model_name
        self.date=str(datetime.now().date()).replace('-','')[2:]
        self.logger_file = f'log/{self.date}_{self.model_name}'
        
    def __call__(self, text, verbose=True, log=True):
        if log:
            with open(f'{self.logger_file}.log', 'a') as f:
                f.write(f'[{datetime.now().replace(microsecond=0)}] {text}\n')
        if verbose:
            print(f'[{datetime.now().time().replace(microsecond=0)}] {text}')

class EarlyStopper:
    def __init__(self, patience=7, printfunc=print,verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
            printfunc (func): 출력함수, 원하는 경우 personalized logger 사용 가능
                            Default: python print function
        """
        self.printfunc=printfunc
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.printfunc(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            self.printfunc(f'Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def set_seed(seed, printf=print):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'Random seed with {seed}')

def smiles2fp(smiles, fp_type='morgan', radius=2, n_bits=2048, numpy=True):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if fp_type == 'morgan':
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    elif fp_type == 'rdkit':
        fp = Chem.RDKFingerprint(mol)
    else:
        raise ValueError('Unsupported fingerprint type')
    
    if not numpy:
        return fp
    
    import numpy as np
    np_fp = np.zeros((1, n_bits), dtype=np.int32)
    Chem.DataStructs.ConvertToNumpyArray(fp, np_fp)
    return np_fp
