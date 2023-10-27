"""
Training function of TreeVAE and SmallTreeVAE.
"""
import wandb
import numpy as np
import gc
import torch
import torch.optim as optim

from utils.training_utils import train_one_epoch, validate_one_epoch, AnnealKLCallback, Custom_Metrics, \
	get_ind_small_tree, compute_growing_leaf, compute_pruning_leaf, get_optimizer, predict
from utils.data_utils import get_gen
from utils.model_utils import return_list_tree, construct_data_tree
from models.model import TreeVAE
from models.model_smalltree import SmallTreeVAE


def run_tree(trainset, trainset_eval, testset, device, configs):
	"""
	Run the TreeVAE model as defined in the config setting. The method will first train a TreeVAE model with initial
	depth defined in config (initial_depth). After training TreeVAE for epochs=num_epochs, if grow=True then it will
	start the iterative growing schedule. At each step, a SmallTreeVAE will be trained for num_epochs_smalltree and
	attached to the selected leaf of TreeVAE. The resulting TreeVAE will then grow at each step and will be finetuned
	throughout the growing procedure for num_epochs_intermediate_fulltrain and at the end of the growing procedure for
	num_epochs_finetuning.

	Parameters
	----------
	trainset: torch.utils.data.Dataset
		The train dataset
	trainset_eval: torch.utils.data.Dataset
		The validation dataset
	testset: torch.utils.data.Dataset
		The test dataset
	device: torch.device
		The device in which to validate the model
	configs: dict
		The config setting for training and validating TreeVAE defined in configs or in the command line

	Returns
	------
	models.model.TreeVAE
		The trained TreeVAE model
	"""

	graph_mode = not configs['globals']['eager_mode']
	gen_train = get_gen(trainset, configs, validation=False, shuffle=True)
	gen_train_eval = get_gen(trainset_eval, configs, validation=True, shuffle=False)
	gen_test = get_gen(testset, configs, validation=True, shuffle=False)
	_ = gc.collect()

	# Define model & optimizer
	model = TreeVAE(**configs['training'])
	model.to(device)

	if graph_mode:
		model = torch.compile(model)

	optimizer = get_optimizer(model, configs)

	# Initialize schedulers
	lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=configs['training']['decay_stepsize'],
											 gamma=configs['training']['decay_lr'])
	alpha_scheduler = AnnealKLCallback(model, decay=configs['training']['decay_kl'],
									   start=configs['training']['kl_start'])

	# Initialize Metrics
	metrics_calc_train = Custom_Metrics(device).to(device)
	metrics_calc_val = Custom_Metrics(device).to(device)

	################################# TRAINING TREEVAE with depth defined in config #################################
	
	# Training the initial tree
	for epoch in range(configs['training']['num_epochs']):  # loop over the dataset multiple times
		train_one_epoch(gen_train, model, optimizer, metrics_calc_train, epoch, device)
		validate_one_epoch(gen_test, model, metrics_calc_val, epoch, device)
		lr_scheduler.step()
		alpha_scheduler.on_epoch_end(epoch)
		_ = gc.collect()

	################################# GROWING THE TREE #################################

	# Start the growing loop of the tree
	# Compute metrics and set node.expand False for the nodes that should not grow
	# This loop goes layer-wise
	grow = configs['training']['grow']
	initial_depth = configs['training']['initial_depth']
	max_depth = len(configs['training']['mlp_layers']) - 1
	if initial_depth >= max_depth:
		grow = False
	growing_iterations = 0
	while grow and growing_iterations < 150:

		# full model finetuning during growing after every 3 splits
		if configs['training']['num_epochs_intermediate_fulltrain']>0:
			if growing_iterations != 0 and growing_iterations % 3 == 0:
				# Initialize optimizer and schedulers
				optimizer = get_optimizer(model, configs)
				lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=configs['training']['decay_stepsize'],
														 gamma=configs['training']['decay_lr'])
				alpha_scheduler = AnnealKLCallback(model, decay=configs['training']['decay_kl'],
												   start=configs['training']['kl_start'])

				# Training the initial split
				print('\nTree intermediate finetuning\n')
				for epoch in range(configs['training']['num_epochs_intermediate_fulltrain']):
					train_one_epoch(gen_train, model, optimizer, metrics_calc_train, epoch, device)
					validate_one_epoch(gen_test, model, metrics_calc_val, epoch, device)
					lr_scheduler.step()
					alpha_scheduler.on_epoch_end(epoch)
					_ = gc.collect()

		# extract information of leaves
		node_leaves_train = predict(gen_train_eval, model, device, 'node_leaves')
		node_leaves_test = predict(gen_test, model, device, 'node_leaves')

		# compute which leaf to grow and split
		ind_leaf, leaf, n_effective_leaves = compute_growing_leaf(gen_train_eval, model, node_leaves_train, max_depth,
																  configs['training']['batch_size'],
																  max_leaves=configs['training']['num_clusters_tree'])
		if ind_leaf == None:
			break
		else:
			print('\nGrowing tree: Leaf %d at depth %d\n' % (ind_leaf, leaf['depth']))
			depth, node = leaf['depth'], leaf['node']

		# get subset of data that has high prob. of falling in subtree
		ind_train = get_ind_small_tree(node_leaves_train[ind_leaf], n_effective_leaves)
		ind_test = get_ind_small_tree(node_leaves_test[ind_leaf], n_effective_leaves)
		gen_train_small = get_gen(trainset, configs, shuffle=True, smalltree=True, smalltree_ind=ind_train)
		gen_test_small = get_gen(testset, configs, shuffle=False, validation=True, smalltree=True,
								 smalltree_ind=ind_test)

		# preparation for the smalltree training
		# initialize the smalltree
		small_model = SmallTreeVAE(depth=depth+1, **configs['training'])
		small_model.to(device)
		if graph_mode:
			small_model = torch.compile(small_model)

		# Optimizer for smalltree
		optimizer = get_optimizer(small_model, configs)

		# Initialize schedulers
		lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=configs['training']['decay_stepsize'],
												 gamma=configs['training']['decay_lr'])
		alpha_scheduler = AnnealKLCallback(small_model, decay=configs['training']['decay_kl'],
										   start=configs['training']['kl_start'])

		# Training the smalltree subsplit
		for epoch in range(configs['training']['num_epochs_smalltree']):
			train_one_epoch(gen_train_small, model, optimizer, metrics_calc_train, epoch, device, train_small_tree=True,
							small_model=small_model, ind_leaf=ind_leaf)
			validate_one_epoch(gen_test_small, model, metrics_calc_val, epoch, device, train_small_tree=True,
							   small_model=small_model, ind_leaf=ind_leaf)
			lr_scheduler.step()
			alpha_scheduler.on_epoch_end(epoch)
			_ = gc.collect()

		# attach smalltree to full tree by assigning decisions and adding new children nodes to full tree
		model.attach_smalltree(node, small_model)

		# Check if reached the max number of effective leaves before finetuning unnecessarily
		if n_effective_leaves + 1 == configs['training']['num_clusters_tree']:
			node_leaves_train = predict(gen_train_eval, model, device, 'node_leaves')
			_, _, max_growth = compute_growing_leaf(gen_train_eval, model, node_leaves_train, max_depth,
													configs['training']['batch_size'],
													max_leaves=configs['training']['num_clusters_tree'], check_max=True)
			if max_growth is True:
				break

		growing_iterations += 1

	# The growing loop of the tree is concluded!
	# check whether we need to prune the final tree and log pre-pruning dendrogram
	prune = configs['training']['prune']
	if prune:
		node_leaves_test, prob_leaves_test = predict(gen_test, model, device, 'node_leaves', 'prob_leaves')
		if len(node_leaves_test)<2:
			prune = False
		else:
			print('\nStarting pruning!\n')
			yy = np.squeeze(np.argmax(prob_leaves_test, axis=-1))
			y_test = testset.dataset.targets[testset.indices]
			data_tree = construct_data_tree(model, y_predicted=yy, y_true=y_test, n_leaves=len(node_leaves_test),
											data_name=configs['data']['data_name'])

			table = wandb.Table(columns=["node_id", "node_name", "parent", "size"], data=data_tree)
			fields = {"node_name": "node_name", "node_id": "node_id", "parent": "parent", "size": "size"}
			dendro = wandb.plot_table(vega_spec_name="stacey/flat_tree", data_table=table, fields=fields)
			wandb.log({"dendogram_pre_pruned": dendro})

	# prune the tree
	while prune:
		# check pruning conditions
		node_leaves_train = predict(gen_train_eval, model, device, 'node_leaves')
		ind_leaf, leaf = compute_pruning_leaf(model, node_leaves_train)

		if ind_leaf == None:
			print('\nPruning finished!\n')
			break
		else:
			# prune leaves and internal nodes without children
			print(f'\nPruning leaf {ind_leaf}!\n')
			current_node = leaf['node']
			while all(child is None for child in [current_node.left, current_node.right]):
				if current_node.parent is not None:
					parent = current_node.parent
				# root does not get pruned
				else:
					break
				parent.prune_child(current_node)
				current_node = parent


			# reinitialize model
			transformations, routers, denses, decoders, routers_q = return_list_tree(model.tree)
			model.decisions_q = routers_q
			model.transformations = transformations
			model.decisions = routers
			model.denses = denses
			model.decoders = decoders
			model.depth = model.compute_depth()
	_ = gc.collect()

	################################# FULL MODEL FINETUNING #################################


	print('\n*****************model depth %d******************\n' % (model.depth))
	print('\n*****************model finetuning******************\n')

	# Initialize optimizer and schedulers
	optimizer = get_optimizer(model, configs)
	lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=configs['training']['decay_stepsize'], gamma=configs['training']['decay_lr'])
	alpha_scheduler = AnnealKLCallback(model, decay=max(0.01,1/max(1,configs['training']['num_epochs_finetuning']-1)), start=configs['training']['kl_start'])
	# finetune the full tree
	print('\nTree final finetuning\n')
	for epoch in range(configs['training']['num_epochs_finetuning']):  # loop over the dataset multiple times
		train_one_epoch(gen_train, model, optimizer, metrics_calc_train, epoch, device)
		validate_one_epoch(gen_test, model, metrics_calc_val, epoch, device)
		lr_scheduler.step()
		alpha_scheduler.on_epoch_end(epoch)
		_ = gc.collect()

	return model


