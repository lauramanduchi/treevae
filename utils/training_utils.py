"""
Utility functions for training.
"""
import torch
import math
import numpy as np
import gc
import wandb
from tqdm import tqdm
import torch.optim as optim
from torchmetrics import Metric
from sklearn.metrics.cluster import normalized_mutual_info_score
from utils.utils import cluster_acc
from torch.utils.data import TensorDataset


def train_one_epoch(train_loader, model, optimizer, metrics_calc, epoch_idx, device, train_small_tree=False,
                    small_model=None, ind_leaf=None):
    """
    Train TreeVAE or SmallTreeVAE model for one epoch.

    Parameters
    ----------
    train_loader: DataLoader
        The train data loader
    model: models.model.TreeVAE
        The TreeVAE model
    optimizer: optim
        The optimizer for training the model
    metrics_calc: Metric
        The metrics to keep track while training
    epoch_idx: int
        The current epoch
    device: torch.device
        The device in which to validate the model
    train_small_tree: bool
        If set to True, then the subtree (small_model) will be trained (and afterwords attached to model)
    small_model: models.model.SmallTreeVAE
        The SmallTreeVAE model (which is then attached to a selected leaf of TreeVAE)
    ind_leaf: int
        The index of the TreeVAE leaf where the small_model will be attached
    """
    if train_small_tree:
        # if we train the small tree, then the full tree is frozen
        model.eval()
        small_model.train()
        model.return_bottomup[0] = True
        model.return_x[0] = True
        alpha = small_model.alpha
    else:
        # otherwise we are training the full tree
        model.train()
        alpha = model.alpha

    metrics_calc.reset()

    for batch_idx, batch in enumerate(tqdm(train_loader)):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        # Zero your gradients for every batch
        optimizer.zero_grad()

        # Make predictions for this batch
        if train_small_tree:
            # Gradient-free pass of full tree
            with torch.no_grad():
                outputs_full = model(inputs)
            x, node_leaves, bottom_up = outputs_full['input'], outputs_full['node_leaves'], outputs_full['bottom_up']
            # Passing through subtree for updating its parameters
            outputs = small_model(x, node_leaves[ind_leaf]['z_sample'], node_leaves[ind_leaf]['prob'], bottom_up)
            outputs['kl_root'] = torch.tensor(0., device=device)
        else:
            outputs = model(inputs)

        # Compute the loss and its gradients
        rec_loss = outputs['rec_loss']
        kl_losses = outputs['kl_root'] + outputs['kl_decisions'] + outputs['kl_nodes']
        loss_value = rec_loss + alpha * kl_losses + outputs['aug_decisions']
        loss_value.backward()

        # Adjust learning weights
        optimizer.step()

        # Store metrics
        # Note that y_pred is used for computing nmi.
        # During subtree training, the nmi is calculated relative to only the subtree.
        y_pred = outputs['p_c_z'].argmax(dim=-1)
        metrics_calc.update(loss_value, outputs['rec_loss'], outputs['kl_decisions'], outputs['kl_nodes'],
                            outputs['kl_root'], outputs['aug_decisions'],
                            (1 - torch.mean(y_pred.float()) if outputs['p_c_z'].shape[1] <= 2 else torch.tensor(0.,
                                                                                                                device=device)),
                            labels, y_pred)

    if train_small_tree:
        model.return_bottomup[0] = False
        model.return_x[0] = False

    # Calculate and log metrics
    metrics = metrics_calc.compute()
    metrics['alpha'] = alpha
    wandb.log({'train': metrics})
    prints = f"Epoch {epoch_idx}, Train     : "
    for key, value in metrics.items():
        prints += f"{key}: {value:.3f} "
    print(prints)
    metrics_calc.reset()
    _ = gc.collect()
    return


def validate_one_epoch(test_loader, model, metrics_calc, epoch_idx, device, test=False, train_small_tree=False,
                       small_model=None, ind_leaf=None):
    model.eval()
    if train_small_tree:
        small_model.eval()
        model.return_bottomup[0] = True
        model.return_x[0] = True
        alpha = small_model.alpha
    else:
        alpha = model.alpha

    metrics_calc.reset()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            # Make predictions for this batch
            if train_small_tree:
                # Sass of full tree
                outputs_full = model(inputs)
                x, node_leaves, bottom_up = outputs_full['input'], outputs_full['node_leaves'], outputs_full[
                    'bottom_up']
                # Passing through subtree
                outputs = small_model(x, node_leaves[ind_leaf]['z_sample'], node_leaves[ind_leaf]['prob'], bottom_up)
                outputs['kl_root'] = torch.tensor(0., device=device)
            else:
                outputs = model(inputs)

            # Compute the loss and its gradients
            rec_loss = outputs['rec_loss']
            kl_losses = outputs['kl_root'] + outputs['kl_decisions'] + outputs['kl_nodes']
            loss_value = rec_loss + alpha * kl_losses + outputs['aug_decisions']

            # Store metrics
            y_pred = outputs['p_c_z'].argmax(dim=-1)
            metrics_calc.update(loss_value, outputs['rec_loss'], outputs['kl_decisions'], outputs['kl_nodes'],
                                outputs['kl_root'],
                                outputs['aug_decisions'], (
                                    1 - torch.mean(outputs['p_c_z'].argmax(dim=-1).float()) if outputs['p_c_z'].shape[
                                                                                                   1] <= 2 else torch.tensor(
                                        0., device=device)), labels, y_pred)

    if train_small_tree:
        model.return_bottomup[0] = False
        model.return_x[0] = False

    # Calculate and log metrics
    metrics = metrics_calc.compute()
    if not test:
        wandb.log({'validation': metrics})
        prints = f"Epoch {epoch_idx}, Validation: "
    else:
        wandb.log({'test': metrics})
        prints = f"Test: "
    for key, value in metrics.items():
        prints += f"{key}: {value:.3f} "
    print(prints)
    metrics_calc.reset()
    _ = gc.collect()
    return


def predict(loader, model, device, *return_flags):
    model.eval()

    if 'bottom_up' in return_flags:
        model.return_bottomup[0] = True
    if 'X_aug' in return_flags:
        model.return_x[0] = True
    if 'elbo' in return_flags:
        model.return_elbo[0] = True

    results = {name: [] for name in return_flags}
    # Create a dictionary to map return flags to corresponding functions
    return_functions = {
        'node_leaves': lambda: move_to(outputs['node_leaves'], 'cpu'),
        'bottom_up': lambda: move_to(outputs['bottom_up'], 'cpu'),
        'prob_leaves': lambda: move_to(outputs['p_c_z'], 'cpu'),
        'X_aug': lambda: move_to(outputs['input'], 'cpu'),
        'y': lambda: labels,
        'elbo': lambda: move_to(outputs['elbo_samples'], 'cpu'),
        'rec_loss': lambda: move_to(outputs['rec_loss'], 'cpu')
    }

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(loader)):
            inputs = inputs.to(device)
            # Make predictions for this batch
            outputs = model(inputs)

            for return_flag in return_flags:
                results[return_flag].append(return_functions[return_flag]())

    for return_flag in return_flags:
        if return_flag == 'bottom_up':
            bottom_up = results[return_flag]
            results[return_flag] = [torch.cat([sublist[i] for sublist in bottom_up], dim=0) for i in
                                    range(len(bottom_up[0]))]
        elif return_flag == 'node_leaves':
            node_leaves_combined = []
            node_leaves = results[return_flag]
            for i in range(len(node_leaves[0])):
                node_leaves_combined.append(dict())
                for key in node_leaves[0][i].keys():
                    node_leaves_combined[i][key] = torch.cat([sublist[i][key] for sublist in node_leaves], dim=0)
            results[return_flag] = node_leaves_combined
        elif return_flag == 'rec_loss':
            results[return_flag] = torch.stack(results[return_flag], dim=0)
        else:
            results[return_flag] = torch.cat(results[return_flag], dim=0)

    if 'bottom_up' in return_flags:
        model.return_bottomup[0] = False
    if 'X_aug' in return_flags:
        model.return_x[0] = False
    if 'elbo' in return_flags:
        model.return_elbo[0] = False

    if len(return_flags) == 1:
        return list(results.values())[0]
    else:
        return tuple(results.values())


def move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    elif isinstance(obj, tuple):
        res = tuple(tensor.to(device) for tensor in obj)
        return res
    else:
        raise TypeError("Invalid type for move_to")


class AnnealKLCallback:
    def __init__(self, model, decay=0.01, start=0.):
        self.decay = decay
        self.start = start
        self.model = model
        self.model.alpha = torch.tensor(min(1, start))

    def on_epoch_end(self, epoch, logs=None):
        value = self.start + (epoch + 1) * self.decay
        self.model.alpha = torch.tensor(min(1, value))


class Decay():
    def __init__(self, lr=0.001, drop=0.1, epochs_drop=50):
        self.lr = lr
        self.drop = drop
        self.epochs_drop = epochs_drop

    def learning_rate_scheduler(self, epoch):
        initial_lrate = self.lr
        drop = self.drop
        epochs_drop = self.epochs_drop
        lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        return lrate


def calc_aug_loss(prob_parent, prob_router, augmentation_methods, emb_contr=[]):
    aug_decisions_loss = torch.zeros(1, device=prob_parent.device)
    prob_parent = prob_parent.detach()

    # Get router probabilities of X' and X''
    p1, p2 = prob_router[:len(prob_router) // 2], prob_router[len(prob_router) // 2:]
    # Perform invariance regularization
    for aug_method in augmentation_methods:
        # Perform invariance regularization in the decisions
        if aug_method == 'InfoNCE':
            p1_normed = torch.nn.functional.normalize(torch.stack([p1, 1 - p1], 1), dim=1)
            p2_normed = torch.nn.functional.normalize(torch.stack([p2, 1 - p2], 1), dim=1)
            pair_sim = torch.exp(torch.sum(p1_normed * p2_normed, dim=1))
            p_normed = torch.cat([p1_normed, p2_normed], dim=0)
            matrix_sim = torch.exp(torch.matmul(p_normed, p_normed.t()))
            norm_factor = torch.sum(matrix_sim, dim=1) - torch.diag(matrix_sim)
            pair_sim = pair_sim.repeat(2)  # storing sim for X' and X''
            info_nce_sample = -torch.log(pair_sim / norm_factor)
            info_nce = torch.sum(prob_parent * info_nce_sample) / torch.sum(prob_parent)
            aug_decisions_loss += info_nce
        # Perform invariance regularization in the bottom-up embeddings
        elif aug_method == 'instancewise_full':
            looplen = len(emb_contr)
            for i in range(looplen):
                temp_instance = 0.5
                emb = emb_contr[i]
                emb1, emb2 = emb[:len(emb) // 2], emb[len(emb) // 2:]
                emb1_normed = torch.nn.functional.normalize(emb1, dim=1)
                emb2_normed = torch.nn.functional.normalize(emb2, dim=1)
                pair_sim = torch.exp(torch.sum(emb1_normed * emb2_normed, dim=1) / temp_instance)
                emb_normed = torch.cat([emb1_normed, emb2_normed], dim=0)
                matrix_sim = torch.exp(torch.matmul(emb_normed, emb_normed.t()) / temp_instance)
                norm_factor = torch.sum(matrix_sim, dim=1) - torch.diag(matrix_sim)
                pair_sim = pair_sim.repeat(2)  # storing sim for X' and X''
                info_nce_sample = -torch.log(pair_sim / norm_factor)
                info_nce = torch.mean(info_nce_sample)
                info_nce = info_nce / looplen
                aug_decisions_loss += info_nce
        else:
            raise NotImplementedError

    return aug_decisions_loss


def get_ind_small_tree(node_leaves, n_effective_leaves):
    prob = node_leaves['prob']
    ind = np.where(prob >= min(1 / n_effective_leaves, 0.5))[0]  # To circumvent problems with n_effective_leaves==1
    return ind


def compute_leaves(tree):
    list_nodes = [{'node': tree, 'depth': 0}]
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
            # We are in an internal node with pruned leaves and thus only have one child
            node_left, node_right = node.left, node.right
            child = node_left if node_left is not None else node_right
            list_nodes.append(
                {'node': child, 'depth': depth_level + 1})
        else:
            nodes_leaves.append(current_node)
    return nodes_leaves


def compute_growing_leaf(loader, model, node_leaves, max_depth, batch_size, max_leaves, check_max=False):
    """
    Compute the leaf of the TreeVAE model that should be further split.

    Parameters
    ----------
    loader: DataLoader
        The data loader used to compute the leaf
    model: models.model.TreeVAE
        The TreeVAE model
    node_leaves: list
        A list of leaf nodes, each one described by a dictionary
        {'prob': sample-wise probability of reaching the node, 'z_sample': sampled leaf embedding}
    max_depth: int
        The maximum depth of the tree
    batch_size: int
        The batch size
    max_leaves: int
        The maximum number of leaves of the tree
    check_max: bool
        Whether to check that we reached the maximum number of leaves
    Returns
    ------
    list: List containing:
          ind_leaf: index of the selected leaf
          leaf: the selected leaf
          n_effective_leaves: number of leaves that are not empty
    """

    # count effective number of leaves (non empty leaves)
    weights = [node_leaves[i]['prob'] for i in range(len(node_leaves))]
    weights_summed = [weights[i].sum() for i in range(len(weights))]
    n_effective_leaves = len(np.where(weights_summed / np.sum(weights_summed) >= 0.01)[0])
    print("\nNumber of effective leaves: ", n_effective_leaves)

    # grow until reaching required n_effective_leaves
    if n_effective_leaves >= max_leaves:
        print('\nReached maximum number of leaves\n')
        return None, None, True

    elif check_max:
        return None, None, False

    else:
        leaves = compute_leaves(model.tree)
        n_samples = []
        if loader.dataset.dataset.__class__ is TensorDataset:
            y_train = loader.dataset.dataset.tensors[1][loader.dataset.indices]
        else:
            y_train = loader.dataset.dataset.targets[loader.dataset.indices]
        # Calculating ground-truth nodes-to-split for logging and model development
        # NOTE: labels are used to evaluate leaf metrics, they are not used to select the leaf
        for i in range(len(node_leaves)):
            depth, node = leaves[i]['depth'], leaves[i]['node']
            if not node.expand:
                continue
            ind = get_ind_small_tree(node_leaves[i], n_effective_leaves)
            y_train_small = y_train[ind]
            # printing distribution of ground-truth classes in leaves
            print(f"Leaf {i}: ", np.unique(y_train_small, return_counts=True))
            n_samples.append(len(y_train_small))

        # Highest number of samples indicates splitting
        split_values = n_samples
        ind_leaves = np.argsort(np.array(split_values))
        ind_leaves = ind_leaves[::-1]

        print("Ranking of leaves to split: ", ind_leaves)
        for i in ind_leaves:
            if n_samples[i] < batch_size:
                wandb.log({'Skipped Split': 1})
                print("We don't split leaves with fewer samples than batch size")
                continue
            elif leaves[i]['depth'] == max_depth or not leaves[i]['node'].expand:
                leaves[i]['node'].expand = False
                print('\nReached maximum architecture\n')
                print('\n!!ATTENTION!! architecture is not deep enough\n')
                break
            else:
                ind_leaf = i
                leaf = leaves[ind_leaf]
                print(f'\nSplitting leaf {ind_leaf}\n')
                return ind_leaf, leaf, n_effective_leaves

        return None, None, n_effective_leaves


def compute_pruning_leaf(model, node_leaves_train):
    leaves = compute_leaves(model.tree)
    n_leaves = len(node_leaves_train)
    weights = [node_leaves_train[i]['prob'] for i in range(n_leaves)]

    # Assign each sample to a leaf by argmax(weights)
    max_indeces = np.array([np.argmax(col) for col in zip(*weights)])

    n_samples = []
    for i in range(n_leaves):
        print(f"Leaf {i}: ", sum(max_indeces == i), "samples")
        n_samples.append(sum(max_indeces == i))

    # Prune leaves with less than 1% of all samples
    ind_leaf = np.argmin(n_samples)
    if n_samples[ind_leaf] < 0.01 * sum(n_samples):
        leaf = leaves[ind_leaf]
        return ind_leaf, leaf
    else:
        return None, None


def get_optimizer(model, configs):
    optimizer = optim.Adam(params=model.parameters(), lr=configs['training']['lr'],
                           weight_decay=configs['training']['weight_decay'])
    return optimizer


class Custom_Metrics(Metric):
    def __init__(self, device):
        super().__init__()
        self.add_state("loss_value", default=torch.tensor(0., device=device))
        self.add_state("rec_loss", default=torch.tensor(0., device=device))
        self.add_state("kl_root", default=torch.tensor(0., device=device))
        self.add_state("kl_decisions", default=torch.tensor(0., device=device))
        self.add_state("kl_nodes", default=torch.tensor(0., device=device))
        self.add_state("aug_decisions", default=torch.tensor(0., device=device))
        self.add_state("perc_samples", default=torch.tensor(0., device=device))
        self.add_state("y_true", default=[])
        self.add_state("y_pred", default=[])
        self.add_state("n_samples", default=torch.tensor(0, dtype=torch.int, device=device))

    def update(self, loss_value: torch.Tensor, rec_loss: torch.Tensor, kl_decisions: torch.Tensor,
               kl_nodes: torch.Tensor, kl_root: torch.Tensor, aug_decisions: torch.Tensor, perc_samples: torch.Tensor,
               y_true: torch.Tensor, y_pred: torch.Tensor):
        assert y_true.shape == y_pred.shape

        n_samples = y_true.numel()
        self.n_samples += n_samples
        self.loss_value += loss_value.item() * n_samples
        self.rec_loss += rec_loss.item() * n_samples
        self.kl_root += kl_root.item() * n_samples
        self.kl_decisions += kl_decisions.item() * n_samples
        self.kl_nodes += kl_nodes.item() * n_samples
        self.aug_decisions += aug_decisions.item() * n_samples
        self.perc_samples += perc_samples.item() * n_samples
        self.y_true.append(y_true)
        self.y_pred.append(y_pred)

    def compute(self):
        self.y_true = torch.cat(self.y_true, dim=0)
        self.y_pred = torch.cat(self.y_pred, dim=0)
        nmi = normalized_mutual_info_score(self.y_true.cpu().numpy(), self.y_pred.cpu().numpy())
        acc = cluster_acc(self.y_true.cpu().numpy(), self.y_pred.cpu().numpy(), return_index=False)

        metrics = dict({'loss_value': self.loss_value / self.n_samples, 'rec_loss': self.rec_loss / self.n_samples,
                        'kl_decisions': self.kl_decisions / self.n_samples, 'kl_root': self.kl_root / self.n_samples,
                        'kl_nodes': self.kl_nodes / self.n_samples,
                        'aug_decisions': self.aug_decisions / self.n_samples,
                        'perc_samples': self.perc_samples / self.n_samples, 'nmi': nmi, 'accuracy': acc})

        return metrics
