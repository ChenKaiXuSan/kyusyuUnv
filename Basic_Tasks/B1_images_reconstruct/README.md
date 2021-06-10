# Reconstruct images from CIFAR, MNIST or both from noisy images. 
This task requires reconstruct images from noise images.
So the first thing I think of is GANs.

The GAN is that put two neural networks contest with each other in a game (in the form of a zero-sum game, where one agent's gain is another agent's loss).
And the input of Generator is a noize, so i think the GAN can do well result in this task. 
But the noise input to Generator is randomly obtained data from the standard normal distribution, and has nothing relation with noisy images or orignal images.

Then I thought of Variational autoencoder(VAE).  
The aim of an autoencoder is to learn a representation (encoding) for a set of data, typically for dimensionality reduction, by training the network to ignore signal “noise”.

so, from images get noisy, and from noisy get generated images. 
i think VAE can have a good result in this task.

and in this task, i just use the mlp to reconstruct the images from MNIST.But if want to have a better performance, it should use cnn instead of mlp.

## Reference 
1. Kingma D P, Welling M. Auto-encoding variational bayes[J]. arXiv preprint arXiv:1312.6114, 2013.