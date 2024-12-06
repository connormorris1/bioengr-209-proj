import torch
from sklearn.metrics import roc_auc_score
import torch.nn as nn 
import wandb

# This file contains the functions for the train and test loops of our neural network
# This code is adopted straight from the Pytorch tutorial with slight modifications
# https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

def train_loop(dataloader, model, loss_fn, device, batch_size, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):

        # Move X and y to GPU
        X = X.to(device)
        y = y.to(device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            wandb.log({'train_loss':loss})

def test_loop(dataloader, model, loss_fn, device):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    true_pos, true_neg, false_pos, false_neg = 0,0,0,0
    all_pred = []
    all_y = []

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for batchnum,(X, y) in enumerate(dataloader):

            # First move X and y to GPU
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            sigmoid = nn.Sigmoid()
            pred_sigmoid = sigmoid(pred) 
            pos_label = torch.where(y==1)[0]
            neg_label = torch.where(y==0)[0]
            pos_pred = torch.where(pred_sigmoid >= 0.5)[0]
            neg_pred = torch.where(pred_sigmoid < 0.5)[0]
            true_pos += torch.sum(torch.isin(pos_pred,pos_label)).item()
            true_neg += torch.sum(torch.isin(neg_pred,neg_label)).item()
            false_pos += torch.sum(torch.isin(pos_pred,neg_label)).item()
            false_neg += torch.sum(torch.isin(neg_pred,pos_label)).item()
            pred_np = pred_sigmoid.cpu().numpy().squeeze()
            y_np = y.cpu().numpy().squeeze()
            all_pred.extend(pred_np)
            all_y.extend(y_np)

    test_loss /= num_batches
    auc = roc_auc_score(all_y,all_pred)
    recall = true_pos/(true_pos + false_neg)
    precision = true_pos/(true_pos + false_pos)
    accuracy = (true_pos + true_neg)/(true_pos + true_neg + false_pos + false_neg)
    specificity = true_neg / (true_neg + false_pos)
    if true_pos + true_neg + false_pos + false_neg != size:
        print('Mismatch in # of examples used in evaluating model test performance.')
    print(f"Test Error: \n   Accuracy: {accuracy:>0.3f}\n   recall: {recall:>0.3f}\n   specificity: {specificity:>0.3f}\n   precision: {precision:>0.3f}\n   AUC: {auc:>0.3f}\n   Avg loss: {test_loss:>8f} \n") 
    wandb.log({'test_loss': test_loss,"test_acc":accuracy,"test_precision":precision,"test_recall":recall,"test_specificity":specificity,"test_auc":auc})
    return all_pred, all_y