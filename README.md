# Tree Variational Autoencoders
This is the PyTorch repository for the NeurIPS 2023 Spotlight Publication (https://neurips.cc/virtual/2023/poster/71188).

TreeVAE is a new generative method that learns the optimal tree-based posterior distribution of latent variables to capture the hierarchical structures present in the data. It adapts the architecture to discover the optimal tree for encoding dependencies between latent variables. TreeVAE optimizes the balance between shared and specialized architecture, enhancing the learning and adaptation capabilities of generative models. 
An example of a tree learned by TreeVAE is depicted in the figure below. Each edge and each split are encoded by neural networks, while the circles depict latent variables. Each sample is associated with a probability distribution over different paths of the discovered tree. The resulting tree thus organizes the data into an interpretable hierarchical structure in an unsupervised fashion, optimizing the amount of shared information between samples. In CIFAR-10, for example, the method divides the vehicles and animals into two different subtrees and similar groups (such as planes and ships) share common ancestors.

![Alt text](https://github.com/lauramanduchi/treevae/blob/main/treevae.png?raw=true)
For running TreeVAE:

1. Create a new environment with the ```treevae.yml``` or ```minimal_requirements.txt``` file.
2. Select the dataset you wish to use by changing the default config_name in the main.py parser. 
3. Potentially adapt default configuration in the config of the selected dataset (config/data_name.yml), the full set of config parameters with their explanations can be found in ```config/mnist.yml```.
4. For Weights & Biases support, set project & entity in ```train/train.py``` and change the value of ```wandb_logging``` to ```online``` in the config file.
5. Run ```main.py```.

For exploring TreeVAE results (including the discovered tree, the generation of new images, the clustering performances and much more) we created a jupyter notebook (```tree_exploration.ipynb```):
1. Run the steps above by setting ```save_model=True```.
2. Copy the experiment path where the model is saved (it will be printed out).
3. Open ```tree_exploration.ipynb```, replace the experiment path with yours, and have fun exploring the model!

DISCLAIMER: This PyTorch repository was thoroughly debugged and tested, however, please note that the experiments of the submission were performed using the repository with the Tensorflow code (https://github.com/lauramanduchi/treevae-tensorflow).

## Citing
To cite TreeVAE please use the following BibTEX entries:

```
@inproceedings{
manduchi2023tree,
title={Tree Variational Autoencoders},
author={Laura Manduchi and Moritz Vandenhirtz and Alain Ryser and Julia E Vogt},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=adq0oXb9KM}
}
```
