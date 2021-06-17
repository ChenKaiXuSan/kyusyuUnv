# Create an algorithm for Blind Signal Separation (BSS) of 2 and 3 soundwaves.
This task need to do a Blind Signal Separation (BSS) of soundwaves. 

Source separation, blind signal separation (BSS) or blind source separation, is the separation of a set of source signals from a set of mixed signals, without the aid of information (or with very little information) about the source signals or the mixing process. 
It is most commonly applied in digital signal processing and involves the analysis of mixtures of signals; the objective is to recover the original component signals from a mixture signal. The classical example of a source separation problem is the cocktail party problem, where a number of people are talking simultaneously in a room (for example, at a cocktail party), and a listener is trying to follow one of the discussions. The human brain can handle this sort of auditory source separation problem, but it is a difficult problem in digital signal processing.

And then, i know that Independent component analysis (ICA) is a special case of blind source separation method. 

Independent component analysis (ICA) is a statistical and computational technique for revealing hidden factors that underlie sets of random variables, measurements, or signals.

ICA defines a generative model for the observed multivariate data, which is typically given as a large database of samples. In the model, the data variables are assumed to be linear mixtures of some unknown latent variables, and the mixing system is also unknown. The latent variables are assumed nongaussian and mutually independent, and they are called the independent components of the observed data. These independent components, also called sources or factors, can be found by ICA.
One of the famous problem “Cocktail Party Problem” — Listening particular One person’s voice in a noisy room, is a common example which is known as an application of ICA algorithm.

I have a look at [4]. It use the FastICA to estimating sources from noisy data.

## Reference 
1. [Signal separation(wikipad)](https://en.wikipedia.org/wiki/Signal_separation)
2. [What is Independent Component Analysis?](https://www.cs.helsinki.fi/u/ahyvarin/whatisica.shtml)
3. [Independent Component Analysis for Signal decomposition](https://medium.com/analytics-vidhya/independent-component-analysis-for-signal-decomposition-3db954ffe8aa)
4. [Blind source separation using FastICA](https://scikit-learn.org/stable/auto_examples/decomposition/plot_ica_blind_source_separation.html)