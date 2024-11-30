import torch

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

def test_loop(dataloader, model, loss_fn, device):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    true_pos, true_neg, false_pos, false_neg = 0,0,0,0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for batchnum,(X, y) in enumerate(dataloader):

            # First move X and y to GPU
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            pos_label = torch.where(y==1)[0]
            neg_label = torch.where(y==0)[0]
            pos_pred = torch.where(pred >= 0.5)[0]
            neg_pred = torch.where(pred < 0.5)[0]
            true_pos += torch.sum(torch.isin(pos_pred,pos_label)).item()
            true_neg += torch.sum(torch.isin(neg_pred,neg_label)).item()
            false_pos += torch.sum(torch.isin(pos_pred,neg_label)).item()
            false_neg += torch.sum(torch.isin(neg_pred,pos_label)).item()
            if batchnum % 25 == 0 and batchnum != 0:
                print(batchnum)
                curr_loss = test_loss / (batchnum+1)
                try:
                    curr_recall = true_pos/(true_pos + false_neg)
                    curr_precision = true_pos/(true_pos + false_pos)
                    curr_accuracy = (true_pos + true_neg)/(true_pos + true_neg + false_pos + false_neg)
                    curr_specificity = true_neg / (true_neg + false_pos)
                except:
                    print(true_pos, true_neg, false_pos, false_neg)
                    print((batchnum + 1)*len(X))
                    continue
                print(f"Test Error: \n   Accuracy: {curr_accuracy:>0.3f}\n   recall: {curr_recall:>0.3f}\n   specificity: {curr_specificity:>0.3f}\n   precision: {curr_precision:>0.3f}\n   Avg loss: {curr_loss:>8f} \n") 


    test_loss /= num_batches
    recall = true_pos/(true_pos + false_neg)
    precision = true_pos/(true_pos + false_pos)
    accuracy = (true_pos + true_neg)/(true_pos + true_neg + false_pos + false_neg)
    specificity = true_neg / (true_neg + false_pos)
    if true_pos + true_neg + false_pos + false_neg != size:
        print('Mismatch in # of examples used in evaluating model test performance.')
    print(f"Test Error: \n   Accuracy: {accuracy:>0.3f}\n   recall: {recall:>0.3f}\n   specificity: {specificity:>0.3f}\n   precision: {precision:>0.3f}\n   Avg loss: {test_loss:>8f} \n") 