import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def encode_molecules(model, loader, device):
    model.eval()
    encodes = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            encode, _ = model.get_graph_representation(batch)
            encodes.append(encode.cpu())
    encodes = torch.cat(encodes, dim=0)
    return encodes

class MolDataset(Dataset):
    def __init__(self, embs, labels):
        self.embs = torch.tensor(embs, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embs[idx], self.labels[idx]

def get_loaders(train_embeddings, valid_embeddings, test_embeddings, train_labels, valid_labels, test_labels, batch_size):
    train_dataset = MolDataset(train_embeddings, train_labels)
    valid_dataset = MolDataset(valid_embeddings, valid_labels)
    test_dataset = MolDataset(test_embeddings, test_labels)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(42))
    validloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return trainloader, validloader, testloader

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, 512)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(256, out_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
    
    def inference(self, loader, device, sigmoid=True):
        self.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                embs = batch[0].to(device)
                outputs = self(embs).squeeze()
                preds.append(outputs.cpu())
        if sigmoid:
            return torch.sigmoid(torch.cat(preds, dim=0))
        else:
            return torch.cat(preds, dim=0)
        

def train(model, loader, optimizer, criterion, args):
    model.train()
    total_loss = 0
    for i, (embs, labels) in enumerate(loader):
        embs, labels = embs.to(args.device), labels.to(args.device)
        optimizer.zero_grad()
        outputs = model(embs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def valid(model, loader, criterion, args, return_preds = False):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for i, (embs, label) in enumerate(loader):
            embs, label = embs.to(args.device), label.to(args.device)
            outputs = model(embs).squeeze()
            preds.append(outputs.cpu())
            labels.append(label.cpu())
    preds = torch.cat(preds)#.numpy()
    labels = torch.cat(labels)#.numpy()
    loss = criterion(preds, labels).item()
    
    if return_preds:
        return loss, torch.sigmoid(preds), labels
    return loss


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef

def get_metrics(preds, labels):
    auroc = roc_auc_score(labels, preds)
    auprc = average_precision_score(labels, preds)

    preds = (preds > 0.5)#.astype(int)
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    return mcc, acc, precision, recall, f1, auroc, auprc


from torch.utils.data import DataLoader, TensorDataset
class DLPInferenceDataLoader(DataLoader):
    def __init__(self, embeddings, batch_size, shuffle=False):
        dataset = TensorDataset(embeddings)
        super(DLPInferenceDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, generator = torch.Generator().manual_seed(42))