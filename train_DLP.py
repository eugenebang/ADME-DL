import pandas as pd
import torch.nn as nn
from torch_geometric.data import DataLoader as GraphDataLoader

from src.graphmvp.config import args as mvp_args
from src.graphmvp.models import GNN, GNN_graphpred
from src.graphmvp.datasets_SeqMTL import MoleculeDataset_inference_ADME

from utils.utils import *
from utils.utils_DLP import encode_molecules, get_loaders, MLP, train, valid, get_metrics


def train_DLP(args):
    model_name = args.model_name
    logger = Logger(model_name)
    logger(f'Start training: {model_name}')
    set_seed(args.seed, logger)

    # ============================================================================
    # Step 1. Encode the DLP molecules with the pretrained model
    # ============================================================================
    num_tasks_adme = 21 # total task num
    molecule_model = GNN(num_layer=mvp_args.num_layer, emb_dim=mvp_args.emb_dim,
                            JK=mvp_args.JK, drop_ratio=mvp_args.dropout_ratio,
                            gnn_type=mvp_args.gnn_type)
    model = GNN_graphpred(args=mvp_args, num_tasks=num_tasks_adme,
                            molecule_model=molecule_model)

    dlp_df = pd.read_csv(args.data_path)
    dlp_labels = dlp_df['label'].values
    dlp_smiles = dlp_df['SMILES'].values

    dlp_enc_dataset = MoleculeDataset_inference_ADME(dlp_smiles)
    print(model.load_state_dict(torch.load(args.ADMEtrained_model, map_location=args.device), strict=False))
    model.to(args.device)

    dlp_enc_loader = GraphDataLoader(dlp_enc_dataset, batch_size=args.batch_size, shuffle=False)
    adme_embeddings = encode_molecules(model, dlp_enc_loader, args.device)

    # ============================================================================
    # Step 2. Train DLP classifier with the encoded features
    # ============================================================================

    # split dataset into train,valid,test with indices
    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(list(range(len(dlp_enc_dataset))), test_size=1/5, random_state=args.seed, stratify=dlp_labels)
    train_idx, valid_idx = train_test_split(train_idx, test_size=1/4, random_state=args.seed, stratify=dlp_labels[train_idx])

    train_labels = dlp_labels[train_idx]
    valid_labels = dlp_labels[valid_idx]
    test_labels = dlp_labels[test_idx]

    train_embeddings = adme_embeddings[train_idx]
    valid_embeddings = adme_embeddings[valid_idx]
    test_embeddings = adme_embeddings[test_idx]

    trainloader, validloader, testloader = get_loaders(train_embeddings, valid_embeddings, test_embeddings, 
                                                    train_labels, valid_labels, test_labels, args.batch_size)
    
    model = MLP(train_embeddings.shape[1], 1).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    modelf=f'ckpts/{model_name}.pt'
    early_stopper = EarlyStopper(patience=20,printfunc=print, 
                                verbose=True, path=modelf)
    
    epoch = 0
    while True:
        epoch+=1
        train_loss=train(model,trainloader,optimizer,criterion,args)
        valid_loss=valid(model,validloader,criterion,args)
        logger(f'[Epoch{epoch}] train_loss: {train_loss:.4f}, valid_loss: {valid_loss:.4f}')
        early_stopper(valid_loss,model)
        if early_stopper.early_stop:
            print('early stopping')
            break

    # Test with best model ckpt
    model.load_state_dict(torch.load(modelf, map_location = args.device))
    logger(f'Loaded best model with valid loss {early_stopper.val_loss_min:.4f}')

    test_loss, test_preds, test_labels = valid(model, testloader, criterion, args, return_preds=True)
    logger('Test loss: {:.4f}'.format(test_loss))

    mcc, acc, precision, recall, f1, auroc, auprc = get_metrics(test_preds, test_labels)
    logger(f"Test MCC: {mcc:.4f} | F1: {f1:.4f} | AUPRC: {auprc:.4f} | AUROC: {auroc:.4f} | ACC: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # run args
    parser.add_argument('--model_name', type=str, default='DLP_ADME_DL')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--data_path', type=str, default='data/DLP/drugmap_zinc.csv')
    parser.add_argument('--ADMEtrained_model', type=str, default='ckpts/SeqADME_ADME_DL.pt')
    # train args
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=int, default=1e-4)
    args = parser.parse_args()

    train_DLP(args)