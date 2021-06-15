# %% 
import torch.nn 
import numpy as np 
import sys 

from esn import ESN
import time

import utils
from matplotlib import pyplot as plt 

# %%
device = torch.device('cuda')
dtype = torch.double
torch.set_default_dtype(dtype)

# %%
# Dataloader
if dtype == torch.double:
    data = np.loadtxt('data/mg17.csv', delimiter=',', dtype=np.float64)
elif dtype == torch.float:
    data = np.loadtxt('data/mg17.csv', delimiter=',', dtype=np.float32)

X_data = np.expand_dims(data[:, [0]], axis=1)
Y_data = np.expand_dims(data[:, [1]], axis=1)
X_data = torch.from_numpy(X_data).to(device)
Y_data = torch.from_numpy(Y_data).to(device)

trX = X_data[:5000]
trY = Y_data[:5000]
tsX = X_data[5000:]
tsY = Y_data[5000:]
# %%
washout = [500]
input_size = output_size = 1
hidden_size = 500

loss_fcn = torch.nn.MSELoss()
# %%
if __name__ == "__main__":
    start = time.time()

    # Training 
    trY_flat = utils.prepare_target(trY.clone(), [trX.size(0)], washout=washout)

    model = ESN(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    model.to(device)

    print(model)

    model(trX, washout, None, trY_flat)
    model.fit()
    output, hidden = model(trX, washout)
    loss = loss_fcn(output, trY[washout[0]:]).item()

    print("Training error:", loss)

    # Test 
    output, hidden = model(tsX, washout)
    loss = loss_fcn(output, tsY[washout[0]:]).item()
    print("Test error:", loss)
    print("Ended in:", time.time() - start, "seconds.")

# %%
plt.figure(figsize=(11,1.5))
plt.plot(range(0,len(trX)), data[:5000], 'k', label="target system")
plt.plot(range(len(trX),len(trX) + output.size(0)), output.detach().cpu().numpy().squeeze(), 'r', label="free running ESN")

lo,hi = plt.ylim()
plt.plot([len(trX),len(trX)],[lo+np.spacing(1),hi-np.spacing(1)],'k:')
plt.legend(loc=(0.61,1.1),fontsize='x-small')
