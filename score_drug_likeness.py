import pandas as pd
import torch
from torch_geometric.data import DataLoader as GraphDataLoader

from src.graphmvp.config import args as mvp_args
from src.graphmvp.models import GNN, GNN_graphpred
from src.graphmvp.datasets_SeqMTL import MoleculeDataset_inference_ADME


from utils.utils_DLP import encode_molecules, get_loaders, MLP, DLPInferenceDataLoader


def validate_smiles(smiles_list):
    import rdkit
    from rdkit import Chem
    valid_smiles = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_smiles.append(smiles)
        except:
            print(f'Unable to parse SMILES: {smiles}')
    return valid_smiles
    
def score_molecules(args):
    smiles = pd.read_csv(args.smiles_file, header=None)[0].values
    smiles = validate_smiles(smiles)
    print(f'Valid SMILES: {len(smiles)}')

    # ============================================================================
    # Step 1. Encode the molecules with the pretrained ADME model
    # ============================================================================
    num_tasks_adme = 21 # total task num
    molecule_model = GNN(num_layer=mvp_args.num_layer, emb_dim=mvp_args.emb_dim,
                            JK=mvp_args.JK, drop_ratio=mvp_args.dropout_ratio,
                            gnn_type=mvp_args.gnn_type)
    model = GNN_graphpred(args=mvp_args, num_tasks=num_tasks_adme,
                            molecule_model=molecule_model)

    enc_dataset = MoleculeDataset_inference_ADME(smiles)
    model.load_state_dict(torch.load(args.ADMEtrained_model, map_location=args.device), strict=False)
    model.to(args.device)

    enc_loader = GraphDataLoader(enc_dataset, batch_size=args.batch_size, shuffle=False)
    adme_embeddings = encode_molecules(model, enc_loader, args.device)

    # ============================================================================
    # Step 2. Evaluate the molecules with trained DLP model
    # ============================================================================
    model = MLP(adme_embeddings.shape[1], 1)#.to(device)
    model.load_state_dict(torch.load(args.DLPtrained_model, map_location=args.device))
    model.to(args.device)

    dcc_loader = DLPInferenceDataLoader(adme_embeddings, batch_size=args.batch_size, shuffle=False)
    preds = model.inference(dcc_loader, args.device, sigmoid=True)

    res_df = pd.DataFrame({'SMILES': smiles, 'Score': preds})
    save_file = args.smiles_file.replace('.smi', '_scored.csv')
    res_df.to_csv(save_file, index=False)
    print(f'Saved the results to {save_file}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # run args
    parser.add_argument('--smiles_file', type=str, default='data/demo/demo_molecules.smi')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--DLPtrained_model', type=str, default='ckpts/DLP_ADME_DL.pt')
    parser.add_argument('--ADMEtrained_model', type=str, default='ckpts/SeqADME_ADME_DL.pt')
    # train args
    parser.add_argument('--batch_size', type=int, default=1024)
    args = parser.parse_args([])

    score_molecules(args)