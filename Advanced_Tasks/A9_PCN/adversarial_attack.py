# %% 
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import torchvision
import matplotlib.pyplot as plt

from prednet import *
from utils import tensor2var
import torch.backends.cudnn as cudnn

# %%
epsilons = [0, .05, .1, .15, .2, .25, .3, .5, 1.]
pretrained_model = "checkpoint/PredNet_0.01LR_6CLS_1REP_best_ckpt"
use_cuda=True

# %% 
net = PredNet(num_classes=10,cls=6)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    
checkpoint = torch.load(pretrained_model, map_location='cpu')
net.load_state_dict(checkpoint['state_dict'])
epoch = checkpoint['epoch']
acc = checkpoint['acc']

net.eval()

# %%
# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
# %%
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

testset = torchvision.datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=50, shuffle=False)
# %%
# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

criterion = nn.CrossEntropyLoss()

# %%
def test_attack( net, device, test_loader, epsilon ):
    test_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if use_cuda:
            inputs, targets = tensor2var(inputs, grad=True), tensor2var(targets)

        outputs = net(inputs)

        loss = criterion(outputs, targets)

        net.zero_grad()
        loss.backward()

        del loss
        # collect datagrad 
        data_grad = inputs.grad.data

        # call FGSM Attack 
        perturbed_data = fgsm_attack(inputs, epsilon, data_grad)

        # re classify the perturbed image 
        output = net(perturbed_data)

        loss = criterion(output, targets)

        test_loss += loss.data
        _, predicted = torch.max(output.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
        # if batch_idx % 100 == 0:
        #     print('eps:', epsilon)
        #     print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #         % (test_loss/(batch_idx+1), 100.*(float)(correct)/(float)(total), correct, total))
    
    final_acc = 100.*(float)(correct)/(float)(total)
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}%".format(epsilon, correct, total, final_acc))
    
# %% 
# Testing
@torch.no_grad() # out of memory
def test(net, test_loader):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if use_cuda:
            inputs, targets = tensor2var(inputs), tensor2var(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        # if batch_idx % 100 == 0:
        #     print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #         % (test_loss/(batch_idx+1), 100.*(float)(correct)/(float)(total), correct, total))
    
    final_acc = 100.*(float)(correct)/(float)(total)
    print("without attack\tTest Accuracy = {} / {} = {}%".format(correct, total, final_acc))
    
# %%
# Run test for each epsilon
test(net, testloader)
for eps in epsilons:
    test_attack(net, device, testloader, eps)
    