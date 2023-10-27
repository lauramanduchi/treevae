"""
SmallTreeVAE model (used for the growing procedure of TreeVAE).
"""
import torch
import torch.nn as nn
import torch.distributions as td
from models.networks import get_decoder, MLP, Router, Dense
from utils.model_utils import compute_posterior
from models.losses import loss_reconstruction_binary, loss_reconstruction_mse
from utils.training_utils import calc_aug_loss

class SmallTreeVAE(nn.Module):
    """
        A class used to represent a sub-tree VAE with one root and two children.

        SmallTreeVAE specifies a sub-tree of TreeVAE with one root and two children. It is used in the
        growing procedure of TreeVAE. At each growing step a new SmallTreeVAE is attached to a leaf of TreeVAE and
        trained separately to reduce computational time.

        Attributes
        ----------
        activation : str
            The name of the activation function for the reconstruction loss [sigmoid, mse]
        loss : models.losses
            The loss function used by the decoder to reconstruct the input
        alpha : float
            KL-annealing weight initialization
        depth : int
            The depth at which the sub-tree will be attached (root has depth 0 and a root with two leaves has depth 1)
        inp_shape : int
            The total dimensions of the input data (if images of 32x32 then 32x32x3)
        augment : bool
             Whether to use contrastive learning through augmentation, if False no augmentation is used
        augmentation_method : str
            The type of augmentation method used
        aug_decisions_weight : str
            The weight of the contrastive loss used in the decisions
        denses : nn.ModuleList
            List of dense layers for the sharing of top-down and bottom-up (MLPs) associated with each of the two leaf
             node of the tree from left to right.
        transformations : nn.ModuleList
            List of transformations (MLPs) associated with each of the two leaf node of the sub-tree from left to right
        decision : Router
            The decision associated with the root of the sub-tree.
        decoders : nn.ModuleList
            List of two decoders one for each leaf of the sub-tree
        decision_q : str
            The decision of the bottom-up associated with the root of the sub-tree

        Methods
        -------
        forward(x)
            Compute the forward pass of the SmallTreeVAE model and return a dictionary of losses.
        """
    def __init__(self, depth, **kwargs):
        """
        Parameters
        ----------
        depth: int
            The depth at which the sub-tree will be attached to TreeVAE
        kwargs : dict
            A dictionary of attributes (see config file).
        """
        super(SmallTreeVAE, self).__init__()
        self.kwargs = kwargs
        
        self.activation = self.kwargs['activation']
        if self.activation == "sigmoid":
            self.loss = loss_reconstruction_binary
        elif self.activation == "mse":
            self.loss = loss_reconstruction_mse
        else:
            raise NotImplementedError
        # KL-annealing weight initialization
        self.alpha=self.kwargs['kl_start'] 

        encoded_sizes = self.kwargs['latent_dim']
        hidden_layers = self.kwargs['mlp_layers']
        self.depth = depth
        encoded_size_gen = encoded_sizes[-(self.depth+1):-(self.depth-1)]  # e.g. encoded_size_gen = 32,16, depth 2
        self.encoded_size = encoded_size_gen[::-1]  # self.encoded_size = 32,16 => 16,32
        layers_gen = hidden_layers[-(self.depth+1):-(self.depth-1)]  # e.g. encoded_sizes 256,128,64, depth 2
        self.hidden_layer = layers_gen[::-1]  # encoded_size_gen = 256,128 => 128,256

        self.inp_shape = self.kwargs['inp_shape']
        self.augment = self.kwargs['augment']
        self.augmentation_method = self.kwargs['augmentation_method']
        self.aug_decisions_weight = self.kwargs['aug_decisions_weight']

        self.denses = nn.ModuleList([Dense(self.hidden_layer[1], self.encoded_size[1]) for _ in range(2)])
        self.transformations = nn.ModuleList([MLP(self.encoded_size[0], self.encoded_size[1], self.hidden_layer[0]) for _ in range(2)])
        self.decision = Router(self.encoded_size[0], hidden_units=self.hidden_layer[0])
        self.decision_q = Router(self.hidden_layer[0], hidden_units=self.hidden_layer[0])
        self.decoders = nn.ModuleList([get_decoder(architecture=self.kwargs['encoder'], input_shape=self.encoded_size[1],
                                                  output_shape=self.inp_shape, activation=self.activation) for _ in range(2)])

    def forward(self, x, z_parent, p, bottom_up):
        """
        Forward pass of the SmallTreeVAE model.

        Parameters
        ----------
        x : tensor
            Input data (batch-size, input-size)
        z_parent: tensor
            The embeddings of the parent of the two children of SmallTreeVAE (which are the embeddings of the TreeVAE
            leaf where the SmallTreeVAE will be attached)
        p: list
            Probabilities of falling into the selected TreeVAE leaf where the SmallTreeVAE will be attached
        bottom_up: list
            The list of bottom-up transformations [encoder, MLP, MLP, ...] up to the root

        Returns
        -------
        dict
            a dictionary
            {'rec_loss': reconstruction loss,
            'kl_decisions': the KL loss of the decisions,
            'kl_nodes': the KL loss of the nodes,
            'aug_decisions': the weighted contrastive loss,
            'p_c_z': the probability of each sample to be assigned to each leaf with size: #samples x #leaves,
            }
        """
        epsilon = 1e-7  # Small constant to prevent numerical instability
        device = x.device
        
        # Extract relevant bottom-up
        d_q = bottom_up[-self.depth]
        d = bottom_up[-self.depth - 1]
        
        prob_child_left = self.decision(z_parent).squeeze()
        prob_child_left_q = self.decision_q(d_q).squeeze()
        leaves_prob = [p * prob_child_left_q, p * (1 - prob_child_left_q)]

        kl_decisions = prob_child_left_q * torch.log(epsilon + prob_child_left_q / (prob_child_left + epsilon)) +\
                        (1 - prob_child_left_q) * torch.log(epsilon + (1 - prob_child_left_q) /
                                                                (1 - prob_child_left + epsilon))
        kl_decisions = torch.mean(p * kl_decisions)
        
        # Contrastive loss
        aug_decisions_loss = torch.zeros(1, device=device)
        if self.training is True and self.augment is True and 'simple' not in self.augmentation_method:
            aug_decisions_loss += calc_aug_loss(prob_parent=p, prob_router=prob_child_left_q,
                                                augmentation_methods=self.augmentation_method)

        reconstructions = []
        kl_nodes = torch.zeros(1, device=device)
        for i in range(2):
            # Compute posterior parameters
            z_mu_q_hat, z_sigma_q_hat = self.denses[i](d)
            _, z_mu_p, z_sigma_p = self.transformations[i](z_parent)
            z_p = td.Independent(td.Normal(z_mu_p, torch.sqrt(z_sigma_p+epsilon)), 1)
            z_mu_q, z_sigma_q = compute_posterior(z_mu_q_hat, z_mu_p, z_sigma_q_hat, z_sigma_p)

            # Compute sample z using mu_q and sigma_q
            z_q = td.Independent(td.Normal(z_mu_q, torch.sqrt(z_sigma_q + epsilon)), 1)
            z_sample = z_q.rsample()

            # Compute KL node
            kl_node = torch.mean(leaves_prob[i] * td.kl_divergence(z_q, z_p))
            kl_nodes += kl_node

            reconstructions.append(self.decoders[i](z_sample))

        kl_nodes_loss = torch.clamp(kl_nodes, min=-10, max=1e10)

        # Probability of falling in each leaf
        p_c_z = torch.cat([prob.unsqueeze(-1) for prob in leaves_prob], dim=-1)

        rec_losses = self.loss(x, reconstructions, leaves_prob)
        rec_loss = torch.mean(rec_losses, dim=0)    

        return {
            'rec_loss': rec_loss,
            'kl_decisions': kl_decisions,
            'kl_nodes': kl_nodes_loss,
            'aug_decisions': self.aug_decisions_weight * aug_decisions_loss,
            'p_c_z': p_c_z,
        }
