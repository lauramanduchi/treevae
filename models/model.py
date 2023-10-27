"""
TreeVAE model.
"""
import torch
import torch.nn as nn
import torch.distributions as td
from utils.model_utils import construct_tree, compute_posterior
from models.networks import get_encoder, get_decoder, MLP, Router, Dense
from models.losses import loss_reconstruction_binary, loss_reconstruction_mse
from utils.model_utils import return_list_tree
from utils.training_utils import calc_aug_loss

class TreeVAE(nn.Module):
    """
        A class used to represent a tree-based VAE.

        TreeVAE specifies a Variational Autoencoder with a tree structure posterior distribution of latent variables.
        It is defined by a bottom-up chain of deterministic transformations that from the input x compute the root
        representation of the data, and a probabilistic top-down architecture which takes the form of a tree. The
        top down tree is described by the probability distribution of its node (which depends on their parents) and the
        probability distribution of the decisions (what is the probability of following a certain path in the tree).
        Each node of the tree is described by the class Node in utils.model_utils.

        Attributes
        ----------
        activation : str
            The name of the activation function for the reconstruction loss [sigmoid, mse]
        loss : models.losses
            The loss function used by the decoder to reconstruct the input
        alpha : float
            KL-annealing weight initialization
        encoded_sizes : list
            A list of latent dimensions for each depth of the tree from the bottom to the root
        hidden_layers : float
            A list of hidden units number for the MLP transformations for each depth of the tree from bottom to root
        depth : int
            The depth of the tree (root has depth 0 and a root with two leaves has depth 1)
        inp_shape : int
            The total dimensions of the input data (if images of 32x32 then 32x32x3)
        augment : bool
             Whether to use contrastive learning through augmentation, if False no augmentation is used
        augmentation_method : str
            The type of augmentation method used
        aug_decisions_weight : str
            The weight of the contrastive loss used in the decisions
        return_x : float
            Whether to return the input in the return dictionary of the forward method
        return_elbo : float
            Whether to return the sample-specific elbo in the return dictionary of the forward method
        return_bottomup : float
            Whether to return the list of bottom-up transformations (including encoder)
        bottom_up : str
            The list of bottom-up transformations [encoder, MLP, MLP, ...] up to the root
        contrastive_mlp : list
            The list of transformations from the bottom-up embeddings to the latent spaces
            in which the contrastive losses are applied
        transformations : list
            List of transformations (MLPs) associated with each node of the tree from root to bottom (left to right)
        denses : list
            List of dense layers for the sharing of top-down and bottom-up (MLPs) associated with each node of the tree
            from root to bottom (left to right).
        decisions : list
            List of decisions associated with each node of the tree from root to bottom (left to right)
        decoders : list
            List of decoders one for each leaf
        decisions_q : str
            List of decisions of the bottom-up associated with each node of the tree from root to bottom (left to right)
        tree : utils.model_utils.Node
            The root node of the tree

        Methods
        -------
        forward(x)
            Compute the forward pass of the treeVAE model and return a dictionary of losses and optional outputs
            (like input, bottom-up and sample-specific elbo) when needed.
        compute_leaves()
            Return a list of leaf-nodes from left to right of the current tree (self.tree).
        compute_depth()
            Calculate the depth of the given tree (self.tree).
        attach_smalltree(node, small_model)
            Attach a sub tree (small_model) to the given node of the current tree.
        compute_reconstruction(x)
            Given the input x, it computes the reconstructions.
        generate_images(n_samples, device)
            Generate n_samples new images by sampling from the root and propagating through the entire tree.
        """

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        kwargs : dict
            A dictionary of attributes (see config file).
        """
        super(TreeVAE, self).__init__()
        self.kwargs = kwargs
        
        self.activation = self.kwargs['activation']
        if self.activation == "sigmoid":
            self.loss = loss_reconstruction_binary
        elif self.activation == "mse":
            self.loss = loss_reconstruction_mse
        else:
            raise NotImplementedError
        # KL-annealing weight initialization
        self.alpha = torch.tensor(self.kwargs['kl_start'])

        # saving important variables to initialize the tree
        self.encoded_sizes = self.kwargs['latent_dim']
        self.hidden_layers = self.kwargs['mlp_layers']
        # check that the number of layers for bottom up is equal to top down
        if len(self.encoded_sizes) != len(self.hidden_layers):
            raise ValueError('Model is mispecified!!')
        self.depth = self.kwargs['initial_depth']
        self.inp_shape = self.kwargs['inp_shape']
        self.augment = self.kwargs['augment']
        self.augmentation_method = self.kwargs['augmentation_method']
        self.aug_decisions_weight = self.kwargs['aug_decisions_weight']
        self.return_x = torch.tensor([False])
        self.return_bottomup = torch.tensor([False])
        self.return_elbo = torch.tensor([False])

        # bottom up: the inference chain that from input computes the d units till the root
        if self.activation == "mse":
            size = int((self.inp_shape / 3)**0.5)
            encoder = get_encoder(architecture=self.kwargs['encoder'], encoded_size=self.hidden_layers[0],
                                x_shape=self.inp_shape, size=size)
        else:
            encoder = get_encoder(architecture=self.kwargs['encoder'], encoded_size=self.hidden_layers[0],
                                x_shape=self.inp_shape)   

        self.bottom_up = nn.ModuleList([encoder])
        for i in range(1, len(self.hidden_layers)):
            self.bottom_up.append(MLP(self.hidden_layers[i-1], self.encoded_sizes[i], self.hidden_layers[i]))

        # MLP's if we use contrastive loss on d's
        if len([i for i in self.augmentation_method if i in ['instancewise_first', 'instancewise_full']]) > 0:
            self.contrastive_mlp = nn.ModuleList([])
            for i in range(0, len(self.hidden_layers)):
                self.contrastive_mlp.append(MLP(input_size=self.hidden_layers[i], encoded_size=self.encoded_sizes[i], hidden_unit=min(self.hidden_layers)))

        # top down: the generative model that from x computes the prior prob of all nodes from root till leaves
        # it has a tree structure which is constructed by passing a list of transformations and routers from root to
        # leaves visiting nodes layer-wise from left to right
        # N.B. root has None as transformation and leaves have None as routers
        # the encoded sizes and layers are reversed from bottom up
        # e.g. for bottom up [MLP(256, 32), MLP(128, 16), MLP(64, 8)] the list of top-down transformations are
        # [None, MLP(16, 64), MLP(16, 64), MLP(32, 128), MLP(32, 128), MLP(32, 128), MLP(32, 128)]

        # select the top down generative networks
        encoded_size_gen = self.encoded_sizes[-(self.depth+1):]  # e.g. encoded_sizes 32,16,8, depth 1
        encoded_size_gen = encoded_size_gen[::-1]  # encoded_size_gen = 16,8 => 8,16
        layers_gen = self.hidden_layers[-(self.depth+1):]  # e.g. encoded_sizes 256,128,64, depth 1
        layers_gen = layers_gen[::-1]  # encoded_size_gen = 128,64 => 64,128

        # add root transformation and dense layer, the dense layer is layer that connects the bottom-up with the nodes
        self.transformations = nn.ModuleList([None])
        self.denses = nn.ModuleList([Dense(layers_gen[0], encoded_size_gen[0])])
        # attach the rest of transformations and dense layers for each node
        for i in range(self.depth):
            for j in range(2 ** (i + 1)):
                self.transformations.append(MLP(encoded_size_gen[i], encoded_size_gen[i+1], layers_gen[i])) # MLP from depth i to i+1
                self.denses.append(Dense(layers_gen[i+1], encoded_size_gen[i+1])) # Dense at depth i+1 from bottom-up to top-down

        # compute the list of decisions for both bottom-up (decisions_q) and top-down (decisions)
        # for each node of the tree
        self.decisions = nn.ModuleList([])
        self.decisions_q = nn.ModuleList([])
        for i in range(self.depth):
            for _ in range(2 ** i):
                self.decisions.append(Router(encoded_size_gen[i], hidden_units=layers_gen[i])) # Router at node of depth i
                self.decisions_q.append(Router(layers_gen[i], hidden_units=layers_gen[i]))
        # the leaves do not have decisions (we set it to None)
        for _ in range(2 ** (self.depth)):
            self.decisions.append(None)
            self.decisions_q.append(None)

        # compute the list of decoders to attach to each node, note that internal nodes do not have a decoder
        # e.g. for a tree with depth 2: decoders = [None, None, None, Dec, Dec, Dec, Dec]
        self.decoders = nn.ModuleList([None for i in range(self.depth) for j in range(2 ** i)])
        for _ in range(2 ** (self.depth)):
            self.decoders.append(get_decoder(architecture=self.kwargs['encoder'], input_shape=encoded_size_gen[-1], 
                                            output_shape=self.inp_shape, activation=self.activation))

        # construct the tree
        self.tree = construct_tree(transformations=self.transformations, routers=self.decisions,
                                        routers_q=self.decisions_q, denses=self.denses, decoders=self.decoders)

    def forward(self, x):
        """
        Forward pass of the treeVAE model.

        Parameters
        ----------
        x : tensor
            Input data (batch-size, input-size)

        Returns
        -------
        dict
            a dictionary
            {'rec_loss': reconstruction loss,
            'kl_root': the KL loss of the root,
            'kl_decisions': the KL loss of the decisions,
            'kl_nodes': the KL loss of the nodes,
            'aug_decisions': the weighted contrastive loss,
            'p_c_z': the probability of each sample to be assigned to each leaf with size: #samples x #leaves,
            'node_leaves': a list of leaf nodes, each one described by a dictionary
                            {'prob': sample-wise probability of reaching the node, 'z_sample': sampled leaf embedding}
            }
        """
        # Small constant to prevent numerical instability
        epsilon = 1e-7  
        device = x.device
        
        # compute deterministic bottom up
        d = x
        encoders = []
        emb_contr = []

        for i in range(0, len(self.hidden_layers)):
            d, _, _ = self.bottom_up[i](d)
            # store bottom-up embeddings for top-down
            encoders.append(d)

            # pass through contrastive MLP's if contrastive learning is selected
            if 'instancewise_full' in self.augmentation_method:
                _, emb_c, _ = self.contrastive_mlp[i](d)
                emb_contr.append(emb_c)
            elif 'instancewise_first' in self.augmentation_method:
                if i == 0:
                    _, emb_c, _ = self.contrastive_mlp[i](d)
                    emb_contr.append(emb_c)

        # create a list of nodes of the tree that need to be processed, self.tree is the root of the tree
        list_nodes = [{'node': self.tree, 'depth': 0, 'prob': torch.ones(x.size(0), device=device), 'z_parent_sample': None}]
        # initializate KL losses
        kl_nodes_tot = torch.zeros(len(x), device=device)
        kl_decisions_tot = torch.zeros(len(x), device=device)
        aug_decisions_loss = torch.zeros(1, device=device)
        leaves_prob = []
        reconstructions = []
        node_leaves = []

        # iterates over all nodes in the tree
        while len(list_nodes) != 0:
            # store info regarding the current node
            current_node = list_nodes.pop(0)
            node, depth_level, prob = current_node['node'], current_node['depth'], current_node['prob']
            z_parent_sample = current_node['z_parent_sample']
            # access deterministic bottom up mu and sigma hat (computed above)
            d = encoders[-(1+depth_level)]
            z_mu_q_hat, z_sigma_q_hat = node.dense(d)

            # here we are in the root
            if depth_level == 0:
                # the root has a standard gaussian prior
                z_mu_p, z_sigma_p = torch.zeros_like(z_mu_q_hat, device=device), torch.ones_like(z_sigma_q_hat, device=device)
                z_p = td.Independent(td.Normal(z_mu_p, torch.sqrt(z_sigma_p + epsilon)), 1)
                # the samples z (from q(z|x)) is the top layer of deterministic bottom-up
                z_mu_q, z_sigma_q = z_mu_q_hat, z_sigma_q_hat

            # otherwise we are in the rest of the nodes of the tree
            else:
                # the generative probability distribution of internal nodes is a gaussian with mu and sigma that are
                # the outputs of the top-down network conditioned on the sampled parent
                _, z_mu_p, z_sigma_p = node.transformation(z_parent_sample)
                z_p = td.Independent(td.Normal(z_mu_p, torch.sqrt(z_sigma_p + epsilon)), 1)
                # to avoid posterior collapse there is a share of information between the bottom-up and top-down
                z_mu_q, z_sigma_q = compute_posterior(z_mu_q_hat, z_mu_p, z_sigma_q_hat, z_sigma_p)

            # compute sample z using mu_q and sigma_q
            z = td.Independent(td.Normal(z_mu_q, torch.sqrt(z_sigma_q + epsilon)), 1)
            z_sample = z.rsample()

            # compute KL node
            kl_node = prob * td.kl_divergence(z, z_p)
            kl_node = torch.clamp(kl_node, min=-1, max=1000)
        
            if depth_level == 0:
                kl_root = kl_node
            else:
                kl_nodes_tot += kl_node

            # if there is a router (i.e. decision probability) then we are in the internal nodes (not leaves)
            if node.router is not None:
                # compute the probability of the sample to go to the left child
                prob_child_left = node.router(z_sample).squeeze()
                prob_child_left_q = node.routers_q(d).squeeze()

                # compute the KL of the decisions
                kl_decisions = prob_child_left_q * (epsilon + prob_child_left_q / (prob_child_left + epsilon)).log() + \
                                (1 - prob_child_left_q) * (epsilon + (1 - prob_child_left_q) / (1 - prob_child_left + epsilon)).log()
                kl_decisions = prob * kl_decisions
                kl_decisions_tot += kl_decisions

                # compute the contrastive loss of the embeddings and the decisions
                if self.training is True and self.augment is True and 'simple' not in self.augmentation_method:
                    if depth_level == 0:
                        # compute the contrastive loss for all the bottom-up representations
                        aug_decisions_loss += calc_aug_loss(prob_parent=prob, prob_router=prob_child_left_q, augmentation_methods=self.augmentation_method, emb_contr=emb_contr)
                    else:
                        # compute the contrastive loss for the decisions
                        aug_decisions_loss += calc_aug_loss(prob_parent=prob, prob_router=prob_child_left_q, augmentation_methods=self.augmentation_method, emb_contr=[])

                # we are not in a leaf, so we have to add the left and right child to the list
                prob_node_left, prob_node_right = prob * prob_child_left_q, prob * (1 - prob_child_left_q)
                node_left, node_right = node.left, node.right
                list_nodes.append(
                    {'node': node_left, 'depth': depth_level + 1, 'prob': prob_node_left, 'z_parent_sample': z_sample})
                list_nodes.append({'node': node_right, 'depth': depth_level + 1, 'prob': prob_node_right,
                                'z_parent_sample': z_sample})

            # if there is a decoder then we are in one of the leaf
            elif node.decoder is not None:
                # if we are in a leaf we need to store the prob of reaching that leaf and compute reconstructions
                # as the nodes are explored left to right, these probabilities will be also ordered left to right
                leaves_prob.append(prob)
                dec = node.decoder
                reconstructions.append(dec(z_sample))
                node_leaves.append({'prob': prob, 'z_sample': z_sample})

            # here we are in an internal node with pruned leaves and thus only have one child
            elif node.router is None and node.decoder is None:
                node_left, node_right = node.left, node.right
                child = node_left if node_left is not None else node_right
                list_nodes.append(
                    {'node': child, 'depth': depth_level + 1, 'prob': prob, 'z_parent_sample': z_sample})

        kl_nodes_loss = torch.clamp(torch.mean(kl_nodes_tot), min=-10, max=1e10)
        kl_decisions_loss = torch.mean(kl_decisions_tot)
        kl_root_loss = torch.mean(kl_root)

        # p_c_z is the probability of reaching a leaf and is of shape [batch_size, num_clusters]
        p_c_z = torch.cat([prob.unsqueeze(-1) for prob in leaves_prob], dim=-1)
        
        rec_losses = self.loss(x, reconstructions, leaves_prob)
        rec_loss = torch.mean(rec_losses, dim=0)

        return_dict = {
            'rec_loss': rec_loss,
            'kl_root': kl_root_loss,
            'kl_decisions': kl_decisions_loss,
            'kl_nodes': kl_nodes_loss,
            'aug_decisions': self.aug_decisions_weight * aug_decisions_loss,
            'p_c_z': p_c_z,
            'node_leaves': node_leaves,
        }

        if self.return_elbo:
            return_dict['elbo_samples'] = kl_nodes_tot + kl_decisions_tot + kl_root + rec_losses

        if self.return_bottomup: 
            return_dict['bottom_up'] = encoders

        if self.return_x:
            return_dict['input'] = x

        return return_dict


    def compute_leaves(self):
        """
        Computes the leaves of the tree

        Returns
        -------
        list
            A list of the leaves from left to right.
            A leaf is defined by a dictionary: {'node': leaf node, 'depth': depth of the leaf node}.
            A leaf node is defined by the class Node in utils.model_utils.
        """
        # iterate over all nodes in the tree to find the leaves
        list_nodes = [{'node': self.tree, 'depth': 0}]
        nodes_leaves = []
        while len(list_nodes) != 0:
            current_node = list_nodes.pop(0)
            node, depth_level = current_node['node'], current_node['depth']
            if node.router is not None:
                node_left, node_right = node.left, node.right
                list_nodes.append(
                    {'node': node_left, 'depth': depth_level + 1})
                list_nodes.append({'node': node_right, 'depth': depth_level + 1})
            elif node.router is None and node.decoder is None:
                # we are in an internal node with pruned leaves and thus only have one child
                node_left, node_right = node.left, node.right
                child = node_left if node_left is not None else node_right
                list_nodes.append({'node': child, 'depth': depth_level + 1})                
            else:
                nodes_leaves.append(current_node)
        return nodes_leaves


    def compute_depth(self):
        """
        Computes the depth of the tree

        Returns
        -------
        int
            The depth of the tree (the root has depth 0 and a root with two leaves had depth 1).
        """
        # computes depth of the tree
        nodes_leaves = self.compute_leaves()
        d = []
        for i in range(len(nodes_leaves)):
            d.append(nodes_leaves[i]['depth'])
        return max(d)

    def attach_smalltree(self, node, small_model):
        """
        Attach a trained small tree of the class SmallTreeVAE (models.model_smalltree) to the given node of the full
        TreeVAE. The small tree has one root and two leaves. It does not return anything but changes self.tree

        Parameters
        ----------
        node : utils.model_utils.Node
            The selected node of TreeVAE where to attach the sub-tree, which was trained separately.
        small_model: models.model_smalltree.SmallTreeVAE
            The sub-tree with one root and two leaves that needs to be attached to TreeVAE.
        """
        assert node.left is None and node.right is None
        node.router = small_model.decision
        node.routers_q = small_model.decision_q
        node.decoder = None
        for j in range(2):
            dense = small_model.denses[j]
            transformation = small_model.transformations[j]
            decoder = small_model.decoders[j]
            # insert each leaf of the small tree as child of the node of TreeVAE
            node.insert(transformation, None, None, dense, decoder)

        # once the small tree is attached we re-compute the list of transformations, routers etc
        transformations, routers, denses, decoders, routers_q = return_list_tree(self.tree)

        # we then need to re-initialize the parameters of TreeVAE
        self.decisions_q = routers_q
        self.transformations = transformations
        self.decisions = routers
        self.denses = denses
        self.decoders = decoders
        self.depth = self.compute_depth()
        return


    def compute_reconstruction(self, x):
        """
        Given the input x, it computes the reconstructions.

        Parameters
        ----------
        x: Tensor
            Input data.

         Returns
        -------
        Tensor
            The reconstructions of the input data by computing a forward pass of the model.
        List
            A list of leaf nodes, each one described by a dictionary
            {'prob': sample-wise probability of reaching the node, 'z_sample': sampled leaf embedding}
        """
        assert self.training is False
        epsilon = 1e-7
        device = x.device
        
        # compute deterministic bottom up
        d = x
        encoders = []

        for i in range(0, len(self.hidden_layers)):
            d, _, _ = self.bottom_up[i](d)
            # store the bottom-up layers for the top down computation
            encoders.append(d)

        # create a list of nodes of the tree that need to be processed
        list_nodes = [{'node': self.tree, 'depth': 0, 'prob': torch.ones(x.size(0), device=device), 'z_parent_sample': None}]

        # initializate KL losses
        leaves_prob = []
        reconstructions = []
        node_leaves = []

        # iterate over the nodes
        while len(list_nodes) != 0:

            # store info regarding the current node
            current_node = list_nodes.pop(0)
            node, depth_level, prob = current_node['node'], current_node['depth'], current_node['prob']
            z_parent_sample = current_node['z_parent_sample']
            # access deterministic bottom up mu and sigma hat (computed above)
            d = encoders[-(1+depth_level)]
            z_mu_q_hat, z_sigma_q_hat = node.dense(d)

            if depth_level == 0:
                z_mu_q, z_sigma_q = z_mu_q_hat, z_sigma_q_hat
            else:
                # the generative mu and sigma is the output of the top-down network given the sampled parent
                _, z_mu_p, z_sigma_p = node.transformation(z_parent_sample)
                z_mu_q, z_sigma_q = compute_posterior(z_mu_q_hat, z_mu_p, z_sigma_q_hat, z_sigma_p)

            # compute sample z using mu_q and sigma_q
            z = td.Independent(td.Normal(z_mu_q, torch.sqrt(z_sigma_q + epsilon)), 1)
            z_sample = z.rsample()

            # if we are in the internal nodes (not leaves)
            if node.router is not None:

                prob_child_left_q = node.routers_q(d).squeeze()

                # we are not in a leaf, so we have to add the left and right child to the list
                prob_node_left, prob_node_right = prob * prob_child_left_q, prob * (1 - prob_child_left_q)

                node_left, node_right = node.left, node.right
                list_nodes.append(
                    {'node': node_left, 'depth': depth_level + 1, 'prob': prob_node_left, 'z_parent_sample': z_sample})
                list_nodes.append({'node': node_right, 'depth': depth_level + 1, 'prob': prob_node_right,
                                'z_parent_sample': z_sample})

            elif node.decoder is not None:
                # if we are in a leaf we need to store the prob of reaching that leaf and compute reconstructions
                # as the nodes are explored left to right, these probabilities will be also ordered left to right
                leaves_prob.append(prob)
                dec = node.decoder
                reconstructions.append(dec(z_sample))
                node_leaves.append({'prob': prob, 'z_sample': z_sample})

            elif node.router is None and node.decoder is None:
                # We are in an internal node with pruned leaves and thus only have one child
                node_left, node_right = node.left, node.right
                child = node_left if node_left is not None else node_right
                list_nodes.append(
                    {'node': child, 'depth': depth_level + 1, 'prob': prob, 'z_parent_sample': z_sample})

        return reconstructions, node_leaves

    def generate_images(self, n_samples, device):
        """
        Generate K x n_samples new images by sampling from the root and propagating through the entire tree.
        For each sample the method generates K images, where K is the number of leaves.

        Parameters
        ----------
        n_samples: int
            Number of generated samples the function should output.
        device: torch.device
            Either cpu or gpu

         Returns
        -------
        list
            A list of K tensors containing the leaf-specific generations obtained by sampling from the root and
            propagating through the entire tree, where K is the number of leaves.
        Tensor
            The probability of each generated sample to be assigned to each leaf with size: #samples x #leaves,
        """
        assert self.training is False
        epsilon = 1e-7
        sizes = self.encoded_sizes
        list_nodes = [{'node': self.tree, 'depth': 0, 'prob': torch.ones(n_samples, device=device), 'z_parent_sample': None}]
        leaves_prob = []
        reconstructions = []
        while len(list_nodes) != 0:
            current_node = list_nodes.pop(0)
            node, depth_level, prob = current_node['node'], current_node['depth'], current_node['prob']
            z_parent_sample = current_node['z_parent_sample']

            if depth_level == 0:
                z_mu_p, z_sigma_p = torch.zeros([n_samples, sizes[-1]], device=device), torch.ones([n_samples, sizes[-1]], device=device)
                z_p = td.Independent(td.Normal(z_mu_p, torch.sqrt(z_sigma_p+epsilon)), 1)
                z_sample = z_p.rsample()

            else:
                _, z_mu_p, z_sigma_p = node.transformation(z_parent_sample)
                z_p = td.Independent(td.Normal(z_mu_p, torch.sqrt(z_sigma_p+epsilon)), 1)
                z_sample = z_p.rsample()

            if node.router is not None:
                prob_child_left = node.router(z_sample).squeeze()
                prob_node_left, prob_node_right = prob * prob_child_left, prob * (
                        1 - prob_child_left)
                node_left, node_right = node.left, node.right
                list_nodes.append(
                    {'node': node_left, 'depth': depth_level + 1, 'prob': prob_node_left, 'z_parent_sample': z_sample})
                list_nodes.append({'node': node_right, 'depth': depth_level + 1, 'prob': prob_node_right,
                                'z_parent_sample': z_sample})

            elif node.decoder is not None:
                # here we are in a leaf node and we attach the corresponding generations
                leaves_prob.append(prob)
                dec = node.decoder
                reconstructions.append(dec(z_sample))

            elif node.router is None and node.decoder is None:
                # We are in an internal node with pruned leaves and thus only have one child
                node_left, node_right = node.left, node.right
                child = node_left if node_left is not None else node_right
                list_nodes.append(
                    {'node': child, 'depth': depth_level + 1, 'prob': prob, 'z_parent_sample': z_sample})
        p_c_z = torch.cat([prob.unsqueeze(-1) for prob in leaves_prob], dim=-1)
        
        return reconstructions, p_c_z
