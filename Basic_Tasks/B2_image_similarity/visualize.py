# -*- coding: utf-8 -*-
# %% 
import argparse

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from torchvision import datasets
from torchvision import transforms

import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import offsetbox
import seaborn as sns

from train import Siamese
from dataset import get_loaders

# %% 
def main(args):
    number_of_items = 15
    sns.set(style="whitegrid", font_scale=1.5)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("../../data", train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor()
        ])), batch_size=100, shuffle=True)
    model = torch.load(args.ck_path)

    model.eval()

    inputs, embs, targets = [], [], []
    for x1, t in tqdm(test_loader, total=len(test_loader)):
        x1 = Variable(x1.cuda())
        o1 = model(x1)

        inputs.append(x1.cpu().data.numpy())
        embs.append(o1.cpu().data.numpy())
        targets.append(t.numpy())

    inputs = np.array(inputs).reshape(-1, 28, 28)
    embs = np.array(embs).reshape((-1, 2))
    targets = np.array(targets).reshape((-1,))

    n_plots = args.n_plots

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111)
    ax.set_title("MNIST 2D embeddigs")
    for x, e, t in zip(inputs[:n_plots], embs[:n_plots], targets[:n_plots]):
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(x, zoom=0.5, cmap=plt.cm.gray_r),
            xy=e, frameon=False)
        ax.add_artist(imagebox)
    
    ax.set_xlim(embs[:, 0].min(), embs[:, 0].max())
    ax.set_ylim(embs[:, 1].min(), embs[:, 1].max())
    plt.tight_layout()
    plt.savefig("./vis.png")

# %%
def test(args):
    number_of_items = 10
    sns.set(style="whitegrid", font_scale=1.5)

    _, test_loader = get_loaders(100)
    model = torch.load(args.ck_path)

    model.eval()
    test_loss, test_n = 0, 0
    input1, input2 = [], []
    err= []
    for x1, x2, y in tqdm(test_loader, total=len(test_loader), leave=False):
        x1, x2 = Variable(x1.cuda()), Variable(x2.cuda())
        y = Variable(y.float().cuda()).view(y.size(0), 1)

        o1, o2 = model(x1), model(x2)

        input1.append(x1.cpu().data.numpy())
        input2.append(x2.cpu().data.numpy())
        
        loss = contractive_loss(o1, o2, y)
        test_loss = loss.item() * y.size(0)
        test_n += y.size(0)

        err.append(loss.item())

    print("\t{:.6f}".format(test_loss / test_n))

    input1 = np.array(input1).reshape(-1, 28, 28)
    input2 = np.array(input2).reshape(-1, 28, 28)

    plt.figure(figsize=(20, 10))
    for item in range(number_of_items):
        display = plt.subplot(1, number_of_items,item+1)
        plt.imshow(input1[item], cmap="gray")
        display.get_xaxis().set_visible(False)
        display.get_yaxis().set_visible(False)
    plt.savefig('./1.png')
    # plt.show()
    plt.close()
    
    plt.figure(figsize=(20, 10))
    for item in range(number_of_items):
        display = plt.subplot(1, number_of_items,item+1)
        plt.imshow(input2[item], cmap="gray")
        display.get_xaxis().set_visible(False)
        display.get_yaxis().set_visible(False)
    # plt.show()
    plt.savefig('./2.png')
    plt.close()

    for item in range(number_of_items):
        print("err:", err[item])
# %%
def contractive_loss(o1, o2, y):
    g, margin = F.pairwise_distance(o1, o2), 5.0
    loss = (1 - y) * (g ** 2) + y * (torch.clamp(margin - g, min=0) ** 2)
    return torch.mean(loss)

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ck_path", type=str, default=r"H:\kyusyuUnv\Basic_Tasks\B2_image_similarity\checkpoint\50.tar")
    parser.add_argument("--n_plots", type=int, default=500)
    args = parser.parse_args([])
    main(args)
    test(args=args)
