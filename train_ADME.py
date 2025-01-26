import os 
import torch.nn as nn
import torch.optim as optim

from src.pcgrad import PCGrad
from src.graphmvp.models import GNN, GNN_graphpred
from src.graphmvp.datasets_SeqMTL import mvpDatasetWrapper_MTL
from src.graphmvp.config import args as mvp_args

from utils.utils import *
from utils.utils_ADME import *

def train_ADME(args):
    args.target_task = args.target_task.lower()
    model_name = f'ADME_DL_{args.target_task}MTL_seed{args.seed}'
    logger = Logger(model_name)
    logger('Start training {} model'.format(model_name))

    target_task_list = list(args.target_task)
    device = args.device

    train_loaders = []
    valid_loaders = []
    test_loaders = []

    for task in target_task_list:
        logger("="*50)
        logger(f'Training on {task.upper()} task')
        targets, tasks = target_task_dict[task]
        
        train_path = f'{args.data_dir}/{task}_train.csv'
        valid_path = f'{args.data_dir}/{task}_valid.csv'
        test_path = f'{args.data_dir}/{task}_test.csv'

        dataset = mvpDatasetWrapper_MTL(train_path, valid_path, test_path, targets)

        trainloader, validloader, testloader = dataset.get_data_loaders(batch_size=mvp_args.batch_size)
        train_loaders.append(trainloader)
        valid_loaders.append(validloader)
        test_loaders.append(testloader)

    # Define MTL tasks and indices
    targets, tasks = [], []
    for task in 'adme':
        targets_t, tasks_t = target_task_dict[task]
        targets += targets_t
        tasks += tasks_t

    criterion_list = []
    for _task in tasks:
        if _task == 'regression':
            criterion_list.append(nn.MSELoss())
        else:
            criterion_list.append(nn.BCEWithLogitsLoss())

    indices = {}
    indices['a'] = [i for i in range(len(a_names))]
    indices['d'] = [i + len(a_names) for i in range(len(d_names))]
    indices['m'] = [i + len(a_names) + len(d_names) for i in range(len(m_names))]
    indices['e'] = [i + len(a_names) + len(d_names) + len(m_names) for i in range(len(e_names))]

    task_indices = [indices[task] for task in target_task_list]

    # Define model
    molecule_model = GNN(num_layer=mvp_args.num_layer, emb_dim=mvp_args.emb_dim,
                        JK=mvp_args.JK, drop_ratio=mvp_args.dropout_ratio,
                        gnn_type=mvp_args.gnn_type)

    load_pretrained_weights_with_resize(pretrained_model_path=args.pretrained_file, 
                                        new_model=molecule_model, 
                                        device=device)

    model = GNN_graphpred(args=mvp_args, num_tasks=len(criterion_list),
                                molecule_model=molecule_model)
    model.to(device)

    model_param_group = [{'params': model.molecule_model.parameters()},
                        {'params': model.pred_layers.parameters(),
                        'lr': mvp_args.lr * mvp_args.lr_scale}]

    optimizer = PCGrad(optim.Adam(model_param_group, lr=mvp_args.lr,
                            weight_decay=mvp_args.decay)
    )

    modelf=f'ckpts/{model_name}.pt'
    
    if not os.path.exists('ckpts'):
        os.makedirs('ckpts', exist_ok=True)
    early_stopper = EarlyStopper(patience=20,printfunc=logger, 
                                verbose=True, path=modelf)
    epoch = 0
    while True:
        epoch+=1
        train_loss, valid_loss, valid_losses, test_loss, test_losses, test_preds, test_ys = train_valid_test(model, train_loaders, valid_loaders, test_loaders, device, optimizer, criterion_list, task_indices)
        logger(f"[Epoch{epoch}] train_loss: {train_loss:.4f}, valid_loss: {valid_loss:.4f}, test_loss: {test_loss:.4f}")
        early_stopper(valid_loss,model)
        if early_stopper.early_stop:
            logger('early stopping')
            break
        elif early_stopper.counter == 0:
            best_test_loss = f"Final test loss: {test_loss:.4f}, {' | '.join([f'{task}: {i:.4f}' for task,i in zip(targets, test_losses)])}"
            best_test_performance = evaluate_testset_return(test_preds, test_ys, tasks, targets)

    logger(best_test_loss)
    logger(best_test_performance)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_task', type=str, default='adme')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--data_dir', type=str, default='data/ADME/')
    parser.add_argument('--pretrained_file', type=str, default = 'src/graphmvp/pretraining_model.pth') ## GIN model provided by GraphMVP (Liu et al., 2022)
    args = parser.parse_args()

    train_ADME(args)