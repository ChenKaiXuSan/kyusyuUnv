# %%
import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--cls', default=6, type=int, help='number of cycles')
parser.add_argument('--model', default='PredNet', help= 'models to train')
parser.add_argument('--gpunum', default=1, type=int, help='number of gpu used to train the model')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
args = parser.parse_args([])

# %%
'''Train CIFAR with PyTorch.'''
from __future__ import print_function

import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

from prednet import *
from utils import tensor2var , progress_bar

# %%
use_cuda = torch.cuda.is_available() # choose to use gpu if possible
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
batchsize = 64 #batch size
# %%
# Path
root = './'
rep = 1 #intial repitetion is 1

models = {'PredNet': PredNet}
model = 'PredNet'
modelname = model+'_'+str(args.lr)+'LR_'+str(args.cls)+'CLS_'+str(rep)+'REP'

# clearn folder 
checkpointpath = root + 'checkpoint/'
logpath = root + 'log/'

if not os.path.isdir(checkpointpath):
    os.mkdir(checkpointpath)
if not os.path.isdir(logpath):
    os.mkdir(logpath)
while(os.path.isfile(checkpointpath + modelname + '_last_ckpt.t7')): 
    rep += 1
    modelname = model+'_'+str(args.lr)+'LR_'+str(args.cls)+'CLS_'+str(rep)+'REP'  
    
# %%
# Data 
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

trainset = torchvision.datasets.CIFAR10(root='../../data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

# Define objective function
criterion = nn.CrossEntropyLoss()

# %%
# Model
print('==> Building model..')
net = PredNet(num_classes=10,cls=args.cls)
    
#set up optimizer
convparas = [p for p in net.FFconv.parameters()]+\
            [p for p in net.FBconv.parameters()]+\
            [p for p in net.linear.parameters()]

rateparas = [p for p in net.a0.parameters()]+\
            [p for p in net.b0.parameters()]

optimizer = optim.SGD([
            {'params': convparas},
            {'params': rateparas, 'weight_decay': 0},
            ], lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
# Parallel computing using mutiple gpu
if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(args.gpunum))
    cudnn.benchmark = True

print(net)
# %%
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0 

    # write to log 
    training_setting = 'batchsize=%d | epoch=%d | lr=%.1e ' % (batchsize, epoch, optimizer.param_groups[0]['lr'])
    statfile.write('\nTraining Setting: '+training_setting+'\n')
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = tensor2var(inputs), tensor2var(targets)
        optimizer.zero_grad()
        
        outputs = net(inputs)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*(float)(correct)/(float)(total), correct, total))
    #writing training record 
    statstr = 'Training: Epoch=%d | Loss: %.3f |  Acc: %.3f%% (%d/%d) | best acc: %.3f' \
                % (epoch, train_loss/(batch_idx+1), 100.*(float)(correct)/(float)(total), correct, total, best_acc)  
    statfile.write(statstr+'\n')  


# %%
# Testing 
@torch.no_grad() # out of memory 
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = tensor2var(inputs), tensor2var(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*(float)(correct)/(float)(total), correct, total))
    # write to log 
    statstr = 'Testing: Epoch=%d | Loss: %.3f |  Acc: %.3f%% (%d/%d) | best_acc: %.3f' \
                % (epoch, test_loss/(batch_idx+1), 100.*(float)(correct)/(float)(total), correct, total, best_acc)
    statfile.write(statstr+'\n')

    # Save checkpoint.
    acc = 100.*correct/total
    state = {
        'state_dict': net.state_dict(),
        'acc': acc,
        'epoch': epoch,           
    }
    torch.save(state, checkpointpath + modelname + '_last_ckpt')

    #check if current accuarcy is the best
    if acc >= best_acc:  
        print('Saving..')
        torch.save(state, checkpointpath + modelname + '_best_ckpt')
        best_acc = acc

# %%
def decrease_learning_rate():
    """Decay the previous learning rate by 10"""
    for param_group in optimizer.param_groups:
        param_group['lr'] /= 10
# %%
if __name__=='__main__':
    for epoch in range(start_epoch, start_epoch+250):
        statfile = open(logpath+'training_stats_'+modelname+'.txt', 'a+') ##open file for writing
        if epoch==80 or epoch == 140 or epoch==200:
            decrease_learning_rate()
            
        train(epoch)
        test(epoch)
