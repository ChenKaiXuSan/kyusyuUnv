# Create a new variation of an evolutionary algorithm (try to use Deep Learning) and compare its performance with a similar algorithm already published.

This task said to do a variation of an evolutionary algorithm. 

## Basic EC
So first i search the Evolution strategy(ES) and do a basic implement use the ES, in [Basic.py](Ecolution_Strategy/Basic.py).

$$
(\mu/\rho + or , \lambda)
$$
Here $\mu$ is the number of population, and $\rho$ is the number selected from the population to generate babies. $\lambda$ is the number of babies generated. 
If the form of $+$ is used, it is to use a mixture of $ρ + λ$ to make the fittest Survival, if it is$,$ form, then just use $λ$ for survival of the fittest.

The target is that to find the highest point in this picture.
![target](img/target.png)

## (1+1) EC
And then, i do a variation of the ES, use one parent and one kid instead of basic ES, in [(1+1)_ES.py](Ecolution_Strategy/(1+1)_ES.py).

What we consider is just a father, generate a baby, and then play the survival of the fittest game between the father and the baby, and choose the better of the father and the baby as the next generation father. 

(1+1) ES summarizes as follows:
1. Have one dad;
2. A baby was mutated according to the father;
3. The one chosen among fathers and babies becomes the next generation of fathers.

# NEAT
And the, i have a look at [3]. 


## Reference
1. [Evolution strategy](https://en.wikipedia.org/wiki/Evolution_strategy)
2. Salimans T, Ho J, Chen X, et al. Evolution strategies as a scalable alternative to reinforcement learning[J]. arXiv preprint arXiv:1703.03864, 2017.
3. K. O. Stanley and R. Miikkulainen, "Evolving Neural Networks through Augmenting Topologies," in Evolutionary Computation, vol. 10, no. 2, pp. 99-127, June 2002, doi: 10.1162/106365602320169811.
4. [Overview of the basic XOR example ](https://neat-python.readthedocs.io/en/latest/xor_example.html#)