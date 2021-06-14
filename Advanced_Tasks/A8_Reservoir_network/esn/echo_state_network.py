# %% 
import torch
import torch.nn as nn 
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from .reservoir import Reservoir