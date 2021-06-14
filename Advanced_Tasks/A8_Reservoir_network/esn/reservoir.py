# %%
import re
import torch
import torch.nn as nn 
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence
import torch.sparse
# %%
def apply_permutation(tensor, permutation, dim=1):
    return tensor.index_select(dim, permutation)
# %%
class Reservoir(nn.Module):

    def __init__(self):
        super(Reservoir, self).__init__()