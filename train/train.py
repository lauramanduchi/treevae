"""
Run training and validation functions of TreeVAE.
"""
import time
from pathlib import Path
import wandb
import uuid
import os
import torch

from utils.data_utils import get_data
from utils.utils import reset_random_seeds
from train.train_tree import run_tree
from train.validate_tree import val_tree


def run_experiment(configs):
	"""
	Run the experiments for TreeVAE as defined in the config setting. This method will set up the device, the correct
	experimental paths, initialize Wandb for tracking, generate the dataset, train and grow the TreeVAE model, and
	finally it will validate the result. All final results and validations will be stored in Wandb, while the most
	important ones will be also printed out in the terminal. If specified, the model will also be saved for further
	exploration using the Jupyter Notebook: tree_exploration.ipynb.

	Parameters
	----------
	configs: dict
		The config setting for training and validating TreeVAE defined in configs or in the command line.
	"""
	# Setting device on GPU if available, else CPU
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Additional info when using cuda
	if device.type == 'cuda':
		print("Using", torch.cuda.get_device_name(0))
	else:
		print("No GPU available")

	# Set paths
	project_dir = Path(__file__).absolute().parent
	timestr = time.strftime("%Y%m%d-%H%M%S")
	ex_name = "{}_{}".format(str(timestr), uuid.uuid4().hex[:5])
	experiment_path = configs['globals']['results_dir'] / configs['data']['data_name'] / ex_name
	experiment_path.mkdir(parents=True)
	os.makedirs(os.path.join(project_dir, '../models/logs', ex_name))
	print("Experiment path: ", experiment_path)

	# Wandb
	os.environ['WANDB_CACHE_DIR'] = os.path.join(project_dir, '../wandb', '.cache', 'wandb')
	os.environ["WANDB_SILENT"] = "true"

	# ADD YOUR WANDB ENTITY
	wandb.init(
		project="treevae",
		entity="test",
		config=configs, 
		mode=configs['globals']['wandb_logging']
	)

	if configs['globals']['wandb_logging'] in ['online', 'disabled']:
		wandb.run.name = wandb.run.name.split("-")[-1] + "-"+ configs['run_name']
	elif configs['globals']['wandb_logging'] == 'offline':
		wandb.run.name = configs['run_name']
	else:
		raise ValueError('wandb needs to be set to online, offline or disabled.')

	# Reproducibility
	reset_random_seeds(configs['globals']['seed'])

	# Generate a new dataset each run
	trainset, trainset_eval, testset = get_data(configs)

	# Run the full training of treeVAE model, including the growing of the tree
	model = run_tree(trainset, trainset_eval, testset, device, configs)

	# Save model
	if configs['globals']['save_model']:
		print("\nSaving weights at ", experiment_path)
		torch.save(model.state_dict(), experiment_path / 'model_weights.pt')

	# Evaluation of TreeVAE
	print("\n" * 2)
	print("Evaluation")
	print("\n" * 2)
	val_tree(trainset_eval, testset, model, device, experiment_path, configs)
	wandb.finish(quiet=True)
	return
