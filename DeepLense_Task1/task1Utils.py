import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, DatasetFolder
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import random
import os
import shutil
import pickle

import matplotlib.pyplot as plt
from matplotlib.pyplot import imread, cm



#############################################
#### Single Epoch Training Function
#############################################

def train_for_epoch(model, train_loader, criterion, optimizer, scheduler, device):

    # put model in train mode
    model.train()

    # keep track of the training losses during the epoch
    train_losses = []

    # Reconstruction
    for batch, targets in tqdm(train_loader):

        # Move the training data to the GPU
        batch = batch.to(device, dtype=torch.float)
        targets = targets.to(device)

        # clear previous gradient computation
        optimizer.zero_grad()

        # forward propagation
        predictions = model(batch)

        # calculate the loss
        loss = criterion(predictions, targets) # calculating loss

        # backpropagate to compute gradients
        loss.backward()

        # update model weights
        optimizer.step()

        # update running loss value
        train_losses.append(loss.item())
    
    # update scheduler
    scheduler.step()

    train_loss = np.mean(train_losses)

    return train_loss



#############################################
#### Test Function
#############################################

def test(model, test_loader, criterion, y_true, device):

    # put model in evaluation mode
    model.eval()

    # keep track of losses and predictions
    test_losses = []
    test_predictions = []

    # We don't need gradients for validation, so wrap in 
    # no_grad to save memory

    with torch.no_grad():

        for batch, targets in tqdm(test_loader):

            # Move the testing data to the GPU
            batch = batch.to(device, dtype=torch.float)
            targets = targets.to(device)

            # forward propagation
            predictions = model(batch)

            # calculate the loss
            loss = criterion(predictions, targets)

            # update running loss value
            test_losses.append(loss.item())

            # save predictions
            test_predictions.extend(predictions.argmax(dim=1).cpu().numpy())

    # compute the average TEST loss
    test_loss = np.mean(test_losses)

    # Collect predictions into y_pred
    y_pred = np.array(test_predictions, dtype=np.float32)

    # Calculate accuracy as the average number of times y_true == y_pred
    accuracy = np.mean([y_pred[i] == y_true[i] for i in range(len(y_true))])

    return test_loss, accuracy



#############################################
#### Final Train Function
#############################################

def train(first_epoch, num_epochs, name, model, train_loader, test_loader, initial_lr, step, gamma):
    
    # set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # collect true labels to calculate accuracy
    y_true = np.array([test_loader.dataset[i][1] for i in range(len(test_loader.dataset))], dtype=np.float32)
    
    
    # create log dictionary
    log_dict = {
        'current_loss'      : 999999,
        'current_accuracy'  : 0,
        'current_epoch'     : 0,
        'best_loss'         : 999999,
        'best_accuracy'     : 0,
        'best_epoch'        : 0,
        'train_losses'      : [],
        'test_losses'       : []
    }
    
    # Loss, Scheduler, Optimizer
    criterion = get_loss()
    optimizer, scheduler = get_optimizer_scheduler(model, initial_lr, step, gamma)
    
    # checkpoint directory
    checkpoint_dir = './checkpoints_'+name

    for epoch in range(first_epoch, first_epoch + num_epochs):

        # training phase
        train_loss = train_for_epoch(model, train_loader, criterion, optimizer, scheduler, device)

        # test phase
        test_loss, test_accuracy = test(model, test_loader, criterion, y_true, device)
        
        # print console log
        print(f'[{epoch:03d}] train loss: {train_loss:04f}',
                f'test loss: {test_loss:04f}',
                f'test accuracy: {test_accuracy:04f}\n')
        
        log_dict['train_losses'].append(train_loss)
        log_dict['test_losses'].append(test_loss)
        
        log_dict['current_loss'] = test_loss
        log_dict['current_accuracy'] = test_accuracy
        log_dict['current_epoch'] = epoch
        
        # update best accuracy
        if log_dict['current_accuracy'] > log_dict['best_accuracy']:
            log_dict['best_accuracy'] = log_dict['current_accuracy']
        
        # Save current checkpoint
        checkpoint_name = name + '-CURRENT.pkl'
        checkpoint_filepath = os.path.join(checkpoint_dir, checkpoint_name)
        save_checkpoint(optimizer, scheduler, model, epoch, checkpoint_filepath)
        
        # update best checkpoint
        if log_dict['current_loss'] < log_dict['best_loss']:
            log_dict['best_loss'] = log_dict['current_loss']
            log_dict['best_epoch'] = epoch

            checkpoint_name = name + '-BEST.pkl'
            checkpoint_filepath = os.path.join(checkpoint_dir, checkpoint_name)
            save_checkpoint(optimizer, scheduler, model, epoch, checkpoint_filepath)

    # save log dictionary
    log_name = name + '-LOG.pkl'
    log_filepath = os.path.join(checkpoint_dir, log_name)
    save_dict(log_filepath, log_dict)
    
    return log_dict



#############################################
#### Evaluation Function
#############################################

def evaluate(name, test_loader, checkpoint_path):

    # get model name
    model_name = name.split('-')[1]

    # get device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Build Model
    model = get_model(model_name)

    # load checkpoint
    load_checkpoint(None, None, model, checkpoint_path)

    # Change to available device
    model.to(device);

    # put model in evaluation mode
    model.eval();
    
    y_true = []
    y_pred = []

    with torch.no_grad():

        for batch, targets in tqdm(test_loader):
            a = F.one_hot(targets, 3).numpy()
            y_true.append(a)

            # Move the testing data to the GPU
            batch = batch.to(device, dtype=torch.float)
            targets = targets.to(device)

            # forward propagation
            predictions = model(batch)
            predictions = F.softmax(predictions, dim=1)
            y_pred.append(predictions.cpu().numpy())

    y_true = np.concatenate(tuple(y_true), axis=0)
    y_pred = np.concatenate(tuple(y_pred), axis=0)
    
    return y_true, y_pred



#############################################
#### Save/Load Functions
#############################################

def save_checkpoint(optimizer, scheduler, model, epoch, filename):
    checkpoint_dict = {
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'model': model.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint_dict, filename)


def load_checkpoint(optimizer, scheduler, model, filename):
    checkpoint_dict = torch.load(filename)
    epoch = checkpoint_dict['epoch']
    model.load_state_dict(checkpoint_dict['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint_dict['scheduler'])
    return epoch

def save_dict(path, log_dict):
    with open(path, 'wb') as f:
        pickle.dump(log_dict, f)

def load_dict(path):
    with open(path, 'rb') as f:
        log_dict = pickle.load(f)
    return log_dict



#############################################
#### Loss Plot Function
#############################################

def plot_loss(train_losses, test_losses):
    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(10,6))
    plt.plot(epochs, train_losses, '-o', label='Training loss')
    plt.plot(epochs, test_losses, '-o', label='Test loss')
    plt.legend()
    plt.title('Learning curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(epochs)
    plt.show()



#############################################
#### Build Model Function
#############################################

def get_model(model_name):
    
    if model_name == 'EfNetB2':

        # download pretrained model
        model = torchvision.models.efficientnet_b2(weights='EfficientNet_B2_Weights.DEFAULT')

        # define input and output features size
        in_features = model.classifier[1].in_features
        out_features = 3

        # replace the last layer
        model.classifier[1] = nn.Linear(in_features, out_features, bias=True)
        
        return model
    
    elif model_name == 'ResNet18':
        
        # download pretrained model
        model = torchvision.models.resnet18(weights='ResNet18_Weights.DEFAULT')

        # define input and output features size
        in_features = model.fc.in_features
        out_features = 3

        # replace the last layer
        model.fc = nn.Linear(in_features, out_features, bias=True)
        
        return model


#############################################
#### Load Data Function
#############################################


def npy_loader(path):
    arr = np.load(path)
    sample = torch.from_numpy(np.stack((arr.squeeze(),)*3, axis=-1)).permute(2,0,1)

    return sample

def load_dataset(train_batch_size):

    # Load data
    train_set = DatasetFolder(root='./dataset/train/', loader=npy_loader, extensions='.npy')
    test_set = DatasetFolder(root='./dataset/val', loader=npy_loader, extensions='.npy')

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)

    # Define the classes
    CLASSES = train_set.classes
    
    return train_loader, test_loader, CLASSES



#############################################
#### Get Loss Function
#############################################

def get_loss():
    
    criterion = torch.nn.CrossEntropyLoss()
    return criterion



#############################################
#### Get Optimizer/Scheduler Function
#############################################

def get_optimizer_scheduler(model, initial_lr, step, gamma):
    
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)
    
    return optimizer, scheduler



#############################################
#### Plot Multiclass ROC Curve
#############################################

def plot_multiclass_roc(name_list, fpr_tpr_list, auc_scores_list, figsize, linewidth, fontsize):

    figure, axes = plt.subplots(1, 2, figsize = (figsize*2, figsize))

    for i in range(len(name_list)):
        model_name = name_list[i].split('-')[1]
        fpr_dict, tpr_dict = fpr_tpr_list[i]
        auc_scores_dict = auc_scores_list[i]
        
        ax = axes[i]
        color = iter(cm.Dark2(np.linspace(0, 1, len(fpr_dict))))

        for key in auc_scores_dict:

            fpr = fpr_dict[key]
            tpr = tpr_dict[key]
            auc = auc_scores_dict[key]
            c = next(color)

            ax.plot(fpr, tpr, lw=linewidth, label = 'Class: {} (auc = {})'.format(key, auc), c=c)

        ax.plot([0, 1], [0, 1], color="navy", lw=linewidth, linestyle="--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=fontsize)
        ax.set_ylabel("True Positive Rate", fontsize=fontsize)
        ax.set_title("ROC Curve for {}".format(model_name), fontsize=fontsize)
        ax.legend(loc="lower right", prop={'size':fontsize})

    plt.suptitle('Task 1 ROC Curves', fontsize=fontsize+10)
    plt.savefig('task1_ROC.png', bbox_inches='tight')
    plt.show()



#############################################
#### Plot Losses
#############################################

def plot_loss(name_list, losses_list, figw, figh, linewidth, fontsize):
    plt.figure(figsize=(figw, figh))
    color = iter(cm.Dark2(np.linspace(0, 1, len(name_list))))
    model_name_list = [x.split('-')[1] for x in name_list]
    
    for i in range(len(model_name_list)):

        model_name = model_name_list[i]
        loss = losses_list[i]
        c = next(color)
        plt.plot(loss, lw=linewidth, label=model_name, c=c)

    plt.xlabel("Epoch", fontsize=fontsize)
    plt.ylabel("Loss", fontsize=fontsize)
    plt.title("Task 1 Test Loss", fontsize=fontsize)
    plt.legend(loc="upper right", prop={'size':fontsize})
    plt.savefig('task1_loss.png', bbox_inches='tight')
    plt.show()