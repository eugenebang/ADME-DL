import torch
import numpy as np

# Task definition
a_names = ['Caco2_Wang', 'PAMPA_NCATS', 'HIA_Hou', 'Pgp_Broccatelli',
           'Bioavailability_Ma', 'Lipophilicity_AstraZeneca', 'Solubility_AqSolDB', 'HydrationFreeEnergy_FreeSolv']
d_names = ['BBB_Martins', 'PPBR_AZ', 'VDss_Lombardo']
m_names = ['CYP2C19_Veith', 'CYP2D6_Veith', 'CYP3A4_Veith', 'CYP1A2_Veith',
           'CYP2C9_Veith', 'CYP2C9_Substrate_CarbonMangels', 'CYP2D6_Substrate_CarbonMangels', 'CYP3A4_Substrate_CarbonMangels']
e_names = ['Half_Life_Obach', 'Clearance_Hepatocyte_AZ']

a_tasks = ['regression', 'classification', 'classification', 'classification',
           'classification', 'regression', 'regression', 'regression']
d_tasks = ['classification', 'regression', 'regression']
m_tasks = ['classification', 'classification', 'classification', 'classification',
           'classification', 'classification', 'classification', 'classification']
e_tasks = ['regression', 'regression']

target_task_dict = {
    'a': (a_names, a_tasks),
    'd': (d_names, d_tasks),
    'm': (m_names, m_tasks),
    'e': (e_names, e_tasks)
}

def train(model, trainloader, device, pcgrad_optimizer, criterion_list, task_indice):
    model.train()
    train_loss = 0
    for batch in trainloader:
        # reshape label to multi-task label
        label = batch.y.float().unsqueeze(1)
        label = label.reshape(-1, len(task_indice))
        mtl_label = torch.zeros(label.shape[0], model.num_tasks)
        mtl_label -= 666
        mtl_label[:, task_indice] = label
        mtl_label[mtl_label == -666] = float('nan')
        mtl_label = mtl_label.to(device)
    
        batch = batch.to(device)
        pred = model(batch)        

        total_loss = []
        for i in range(model.num_tasks):
            null_mask = torch.isnan(mtl_label[:,i])
            predi = pred[~null_mask, i]
            labeli = mtl_label[~null_mask, i]
            loss = criterion_list[i](predi, labeli)
            train_loss += loss.item()
            total_loss.append(loss)

        pcgrad_optimizer.pc_backward(total_loss)
        # total_loss.backward()
        pcgrad_optimizer.step()
    return train_loss/len(trainloader)

def get_preds(model, loader, device, task_indice):
    preds = []
    ys = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            # reshape label to multi-task label
            label = batch.y.float().unsqueeze(1)
            label = label.reshape(-1, len(task_indice))
            mtl_label = torch.zeros(label.shape[0], model.num_tasks)
            mtl_label -= 666
            mtl_label[:, task_indice] = label
            mtl_label[mtl_label == -666] = float('nan')
        
            batch = batch.to(device)
            pred = model(batch)        

            preds.append(pred.cpu())
            ys.append(mtl_label.cpu())
    preds = torch.cat(preds, dim=0)
    ys = torch.cat(ys, dim=0)
    return preds, ys

def eval(model, preds, ys, criterion_list, return_output=False):
    preds = torch.cat(preds, dim=0)
    ys = torch.cat(ys, dim=0)

    total_loss = 0
    losses = []
    for i in range(model.num_tasks):
        null_mask = torch.isnan(ys[:,i])
        predi = preds[~null_mask, i]
        labeli = ys[~null_mask, i]
        loss = criterion_list[i](predi, labeli)
        total_loss += loss
        losses.append(loss.item())

    if return_output:
        return total_loss.item(), losses, preds, ys
    else:
        return total_loss.item(), losses
    

def train_valid_test_per_task(model, train_loader, valid_loader, test_loader, device, optimizer, criterion_list, task_indices, task_index):
    task_indice = task_indices[task_index]
    train_loss = train(model, train_loader, device, optimizer, criterion_list, task_indice)
    valid_preds, valid_ys = get_preds(model, valid_loader, device, task_indice)
    test_preds, test_ys = get_preds(model, test_loader, device, task_indice)
    return train_loss, valid_preds, valid_ys, test_preds, test_ys

def train_valid_test(model, train_loaders, valid_loaders, test_loaders, device, optimizer, criterion_list, task_indices):
    train_losses = []
    valid_preds = []
    valid_ys = []
    test_preds = []
    test_ys = []
    for i in range(len(train_loaders)):
        train_loss, valid_pred, valid_y, test_pred, test_y = train_valid_test_per_task(model, train_loaders[i], valid_loaders[i], test_loaders[i], device, optimizer, criterion_list, task_indices, i)
        train_losses.append(train_loss)
        valid_preds.append(valid_pred)
        valid_ys.append(valid_y)
        test_preds.append(test_pred)
        test_ys.append(test_y)

    val_loss, val_losses = eval(model, valid_preds, valid_ys, criterion_list)
    test_loss, test_losses, test_preds, test_ys = eval(model, test_preds, test_ys, criterion_list, return_output=True)
    return sum(train_losses)/len(train_losses), val_loss, val_losses, test_loss, test_losses, test_preds, test_ys

import math
def load_pretrained_weights_with_resize(pretrained_model_path, new_model, device='cpu'):
    pretrained_state_dict = torch.load(pretrained_model_path, map_location=device)
    new_state_dict = new_model.state_dict()

    for name, param in new_state_dict.items():
        if name in pretrained_state_dict:
            pretrained_param = pretrained_state_dict[name]
            new_param = param
            
            # If the dimension of the embedding is different
            if pretrained_param.shape != new_param.shape:
                if 'weight' in name or 'bias' in name:
                    # Adjusting the weights by copying the available data
                    if len(pretrained_param.shape) == 2:  # For linear layers
                        if new_param.shape[0] > pretrained_param.shape[0]:
                            rep = math.ceil(new_param.shape[0] / pretrained_param.shape[0])
                            pretrained_param = pretrained_param.repeat(rep, 1)
                        if new_param.shape[1] > pretrained_param.shape[1]:
                            rep = math.ceil(new_param.shape[1] / pretrained_param.shape[1])
                            pretrained_param = pretrained_param.repeat(1, rep)
                        # new_param = pretrained_param[:new_param.shape[0],:new_param.shape[1]]
                        # print(new_param.shape)
                        # print(pretrained_param.shape)
                        new_param.copy_(pretrained_param[:new_param.shape[0],:new_param.shape[1]])
                    elif len(pretrained_param.shape) == 1:  # For bias and embedding layers
                        if new_param.shape[0] > pretrained_param.shape[0]:
                            rep = math.ceil(new_param.shape[0] / pretrained_param.shape[0])
                            pretrained_param = pretrained_param.repeat(rep)
                        # new_param = pretrained_param[:new_param.shape[0]]
                        # print(new_param.shape)
                        # print(pretrained_param.shape)
                        new_param.copy_(pretrained_param[:new_param.shape[0]])
                        
            else:
                new_param.copy_(pretrained_param)

        else: # new parameter
            name_split = name.split('.')
            num = int(name_split[1])
            num = num % 5 # new number
            name_split[1] = str(num)
            new_name ='.'.join(name_split)
            # print(f'Pasting from {new_name} to {name}.')

            pretrained_param = pretrained_state_dict[new_name]
            new_param = param

            if pretrained_param.shape != new_param.shape:
                if 'weight' in name or 'bias' in name:
                    # Adjusting the weights by copying the available data
                    if len(pretrained_param.shape) == 2:  # For linear layers
                        if new_param.shape[0] > pretrained_param.shape[0]:
                            rep = math.ceil(new_param.shape[0] / pretrained_param.shape[0])
                            pretrained_param = pretrained_param.repeat(rep, 1)
                        if new_param.shape[1] > pretrained_param.shape[1]:
                            rep = math.ceil(new_param.shape[1] / pretrained_param.shape[1])
                            pretrained_param = pretrained_param.repeat(1, rep)
                        # new_param = pretrained_param[:new_param.shape[0],:new_param.shape[1]]
                        new_param.copy_(pretrained_param[:new_param.shape[0],:new_param.shape[1]])
                    elif len(pretrained_param.shape) == 1:  # For bias and embedding layers
                        if new_param.shape[0] > pretrained_param.shape[0]:
                            rep = math.ceil(new_param.shape[0] / pretrained_param.shape[0])
                            pretrained_param = pretrained_param.repeat(rep)
                        # new_param = pretrained_param[:new_param.shape[0]]
                        new_param.copy_(pretrained_param[:new_param.shape[0]])

    # Load the modified state dict into the new model
    new_model.load_state_dict(new_state_dict)
    print("Loaded pretrained weights.")



### Evaluation
def evaluate_clf(preds, ys, threshold=0.5):
    isna = torch.isnan(ys)
    preds = preds[~isna]
    ys = ys[~isna]
    # acc, auroc, aupr, f1-score
    from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score

    acc = accuracy_score(ys, preds>threshold)
    auroc = roc_auc_score(ys, preds)
    aupr = average_precision_score(ys, preds)
    f1 = f1_score(ys, preds>threshold)
    return acc, auroc, aupr, f1

def evaluage_reg(preds, ys):
    isna = torch.isnan(ys)
    preds = preds[~isna]
    ys = ys[~isna]
    # rmse, r2
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    rmse = np.sqrt(mean_squared_error(ys, preds))
    mae = mean_absolute_error(ys, preds)
    r2 = r2_score(ys, preds)
    return rmse, mae, r2

def evaluate_testset(preds, ys, tasks, targets):
    '''
    Must input preds and ys as torch.Tensor
    '''
    for i, task in enumerate(tasks):
        if task == 'classification':
            acc, auroc, aupr, f1 = evaluate_clf(preds[:,i], ys[:,i])
            print(f'[{targets[i]}] ACC: {acc:.4f}, AUROC: {auroc:.4f}, AUPR: {aupr:.4f}, F1: {f1:.4f}')
        elif task == 'regression':
            rmse, mae, r2 = evaluage_reg(preds[:,i], ys[:,i])
            print(f'[{targets[i]}] RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}')

def evaluate_testset_return(preds, ys, tasks, targets):
    '''
    Must input preds and ys as torch.Tensor
    '''
    ret = ''
    for i, task in enumerate(tasks):
        if task == 'classification':
            acc, auroc, aupr, f1 = evaluate_clf(preds[:,i], ys[:,i])
            ret += f'[{targets[i]}] ACC: {acc:.4f}, AUROC: {auroc:.4f}, AUPR: {aupr:.4f}, F1: {f1:.4f}\n'
        elif task == 'regression':
            rmse, mae, r2 = evaluage_reg(preds[:,i], ys[:,i])
            ret += f'[{targets[i]}] RMSE: {rmse:.4f}, R2: {r2:.4f}\n'
    return ret