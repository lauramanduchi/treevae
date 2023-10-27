"""
Utility functions for data loading.
"""
import os
import torch
import torchvision
import torchvision.transforms as T
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Subset, ConcatDataset
from PIL import Image
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from utils.utils import reset_random_seeds

def get_data(configs):
	"""Compute and process the data specified in the configs file.

	Parameters
	----------
	configs : dict
		A dictionary of config settings, where the data_name, the number of clusters in the data and augmentation
		details are specified.

	Returns
	------
	list
		A list of three tensor datasets: trainset, trainset_eval, testset
	"""
	data_name = configs['data']['data_name']
	augment = configs['training']['augment']
	augmentation_method = configs['training']['augmentation_method']
	n_classes = configs['data']['num_clusters_data']

	data_path = './data/'

	if data_name == 'mnist':
		reset_random_seeds(configs['globals']['seed'])
		full_trainset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=T.ToTensor())
		full_testset = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=T.ToTensor())

		# get only num_clusters digits
		indx_train, indx_test = select_subset(full_trainset.targets, full_testset.targets, n_classes)
		trainset = Subset(full_trainset, indx_train)
		trainset_eval = Subset(full_trainset, indx_train)
		testset = Subset(full_testset, indx_test)


	elif data_name == 'fmnist':
		reset_random_seeds(configs['globals']['seed'])
		full_trainset = torchvision.datasets.FashionMNIST(root=data_path, train=True, download=True, transform=T.ToTensor())
		full_testset = torchvision.datasets.FashionMNIST(root=data_path, train=False, download=True, transform=T.ToTensor())

		# get only num_clusters digits
		indx_train, indx_test = select_subset(full_trainset.targets, full_testset.targets, n_classes)
		trainset = Subset(full_trainset, indx_train)
		trainset_eval = Subset(full_trainset, indx_train)
		testset = Subset(full_testset, indx_test)


	elif data_name == 'news20':
		reset_random_seeds(configs['globals']['seed'])
		newsgroups_train = fetch_20newsgroups(subset='train')
		newsgroups_test = fetch_20newsgroups(subset='test')
		vectorizer = TfidfVectorizer(max_features=2000, dtype=np.float32)
		x_train = torch.from_numpy(vectorizer.fit_transform(newsgroups_train.data).toarray())
		x_test = torch.from_numpy(vectorizer.transform(newsgroups_test.data).toarray())
		y_train = torch.from_numpy(newsgroups_train.target)
		y_test = torch.from_numpy(newsgroups_test.target)

		# get only num_clusters digits
		indx_train, indx_test = select_subset(y_train, y_test, n_classes)
		trainset = Subset(TensorDataset(x_train, y_train), indx_train)
		trainset_eval = Subset(TensorDataset(x_train, y_train), indx_train)
		testset = Subset(TensorDataset(x_test, y_test), indx_test)
		trainset.dataset.targets = torch.tensor(trainset.dataset.tensors[1])
		trainset_eval.dataset.targets = torch.tensor(trainset_eval.dataset.tensors[1])
		testset.dataset.targets = torch.tensor(testset.dataset.tensors[1])


	elif data_name == 'omniglot':
		reset_random_seeds(configs['globals']['seed'])
		
		transform_eval = T.Compose([
				T.ToTensor(),
				T.Resize([28,28], antialias=True),
			])

		if augment and augmentation_method == ['simple']:
			transform = T.Compose([
				T.ToTensor(),
				T.Resize([28,28], antialias=True),
				T.RandomAffine(degrees=10, translate=(1/28, 1/28), scale=(0.9, 1.1), shear=0.01, fill=1),
			])
		elif augment is False:
			transform = transform_eval 
		else:
			raise NotImplementedError 
		
		# Download the datasets and apply transformations
		trainset_premerge = torchvision.datasets.Omniglot(root=data_path, background=True, download=True, transform=transform) 
		testset_premerge = torchvision.datasets.Omniglot(root=data_path, background=False, download=True, transform=transform)
		trainset_premerge_eval = torchvision.datasets.Omniglot(root=data_path, background=True, download=True, transform=transform_eval)
		testset_premerge_eval = torchvision.datasets.Omniglot(root=data_path, background=False, download=True, transform=transform_eval)

		# Get the corresponding labels y_train and y_test
		y_train_ind = torch.tensor([sample[1] for sample in trainset_premerge])
		y_test_ind = torch.tensor([sample[1] for sample in testset_premerge])

		# Create a list of all alphabet labels from both datasets
		alphabets = trainset_premerge._alphabets + testset_premerge._alphabets
		
		# Replace character labels by alphabet labels
		y_train_pre = []
		y_test_pre = []
		for value in y_train_ind:
			alphabet = trainset_premerge._characters[value].split("/")[0]
			alphabet_ind = alphabets.index(alphabet)
			y_train_pre.append(alphabet_ind)
		for value in y_test_ind:
			alphabet = testset_premerge._characters[value].split("/")[0]
			alphabet_ind = alphabets.index(alphabet) 
			y_test_pre.append(alphabet_ind)

		y = np.array(y_train_pre + y_test_pre)

		# Select alphabets
		num_clusters = n_classes
		if num_clusters !=50:
			alphabets_selected = get_selected_omniglot_alphabets()[:num_clusters]
			alphabets_ind = []
			for i in alphabets_selected:
				alphabets_ind.append(alphabets.index(i))
		else:
			alphabets_ind = np.arange(50)

		indx = np.array([], dtype=int)
		for i in range(num_clusters):
			indx = np.append(indx, np.where(y == alphabets_ind[i])[0])
		indx = np.sort(indx)

		# Split and stratify by digits
		digits_label = torch.concatenate([y_train_ind, y_test_ind+len(torch.unique(y_train_ind))])
		indx_train, indx_test = train_test_split(indx, test_size=0.2, random_state=configs['globals']['seed'], stratify=digits_label[indx])
		indx_train = np.sort(indx_train)
		indx_test = np.sort(indx_test)

		# Define alphabets as labels
		y = y+50
		for idx, alphabet in enumerate(alphabets_ind):
			y[y==alphabet+50] = idx

		# Define mapping from digit to label
		mapping_train = []
		for value in torch.unique(y_train_ind):
			alphabet = trainset_premerge._characters[value].split("/")[0]
			alphabet_ind = alphabets.index(alphabet)
			mapping_train.append(alphabet_ind)
		mapping_test = []
		for value in torch.unique(y_test_ind):
			alphabet = testset_premerge._characters[value].split("/")[0]
			alphabet_ind = alphabets.index(alphabet)
			mapping_test.append(alphabet_ind)

		custom_target_transform_train = T.Lambda(lambda y: mapping_train[y])
		custom_target_transform_test = T.Lambda(lambda y: mapping_test[y])
		
		trainset_premerge.target_transform = custom_target_transform_train
		trainset_premerge_eval.target_transform = custom_target_transform_train
		testset_premerge.target_transform = custom_target_transform_test
		testset_premerge_eval.target_transform = custom_target_transform_test

		# Define datasets
		fullset = ConcatDataset([trainset_premerge, testset_premerge])
		fullset_eval = ConcatDataset([trainset_premerge_eval, testset_premerge_eval])
		fullset.targets = torch.from_numpy(y)
		fullset_eval.targets = torch.from_numpy(y)
		trainset = Subset(fullset, indx_train)
		trainset_eval = Subset(fullset_eval, indx_train)
		testset = Subset(fullset_eval, indx_test)



	elif data_name in ['cifar10', 'cifar100', 'cifar10_vehicles', 'cifar10_animals']:
		reset_random_seeds(configs['globals']['seed'])
		aug_strength = 0.5


		transform_eval = T.Compose([
						T.ToTensor(),
			])

		if augment is True:
			aug_transforms = T.Compose([
						T.RandomResizedCrop(32, interpolation=Image.BICUBIC, scale=(0.2, 1.0)),
						T.RandomHorizontalFlip(),
						T.RandomApply([T.ColorJitter(0.8 * aug_strength, 0.8 * aug_strength, 0.8 * aug_strength, 0.2 * aug_strength)], p=0.8),
						T.RandomGrayscale(p=0.2),
						T.ToTensor(),
			])
			if augmentation_method == ['simple']:
				transform = aug_transforms
			else:
				transform = ContrastiveTransformations(aug_transforms, n_views=2)
		else:
			transform = transform_eval 

		if data_name == 'cifar100':
			if n_classes==20:
				dataset = CIFAR100Coarse
			else:
				dataset = torchvision.datasets.CIFAR100
		else:
			dataset = torchvision.datasets.CIFAR10

		full_trainset = dataset(root=data_path, train=True, download=True, transform=transform)
		full_trainset_eval = dataset(root=data_path, train=True, download=True, transform=transform_eval)
		full_testset = dataset(root=data_path, train=False, download=True, transform=transform_eval)

		if data_name == 'cifar10_vehicles':
			indx_train = [index for index, value in enumerate(full_trainset.targets) if value in (0, 1, 8, 9)]
			indx_test = [index for index, value in enumerate(full_testset.targets) if value in (0, 1, 8, 9)]
		elif data_name == 'cifar10_animals':
			indx_train = [index for index, value in enumerate(full_trainset.targets) if value not in (0, 1, 8, 9)]
			indx_test = [index for index, value in enumerate(full_testset.targets) if value not in (0, 1, 8, 9)]
		else:
			indx_train, indx_test = select_subset(full_trainset.targets, full_testset.targets, n_classes)

		trainset = Subset(full_trainset, indx_train)
		trainset_eval = Subset(full_trainset_eval, indx_train)
		testset = Subset(full_testset, indx_test)
		
		trainset.dataset.targets = torch.tensor(trainset.dataset.targets)
		trainset_eval.dataset.targets = torch.tensor(trainset_eval.dataset.targets)
		testset.dataset.targets = torch.tensor(testset.dataset.targets)

	elif data_name == 'celeba':
		reset_random_seeds(configs['globals']['seed'])
		aug_strength = 0.25

		# Slightly different reshaping from TF implementation to be inline with WAE
		transform_eval = T.Compose([
						T.Lambda(lambda x: T.functional.crop(x, left=15, top=40, width=148, height=148)),
						T.Resize([64,64], antialias=True),
						T.ToTensor(),
			])
		if augment is True:
			aug_transforms = T.Compose([
						T.Lambda(lambda x: T.functional.crop(x, left=15, top=40, width=148, height=148)),
						T.Resize([64,64], antialias=True),
						T.RandomResizedCrop(64, interpolation=Image.BICUBIC, scale = (3/4,1), ratio=(4/5,5/4)),
						T.RandomHorizontalFlip(),
						T.RandomApply([T.ColorJitter(0.8 * aug_strength, 0.8 * aug_strength, 0.8 * aug_strength)], p=0.8),
						T.ToTensor(),
			])
			if augmentation_method == ['simple']:
				transform = aug_transforms
			else:
				transform = ContrastiveTransformations(aug_transforms, n_views=2)
		else:
			transform = transform_eval 


		full_trainset = torchvision.datasets.CelebA(root=data_path, split='train', target_type='attr', target_transform=lambda y: 0, download=True, transform=transform)
		full_trainset_eval = torchvision.datasets.CelebA(root=data_path, split='train', target_type='attr', target_transform=lambda y: 0, download=True, transform=transform_eval)
		full_testset = torchvision.datasets.CelebA(root=data_path, split='test', target_type='attr', target_transform=lambda y: 0, download=True, transform=transform_eval)

		indx_train = np.arange(len(full_trainset))
		indx_test = np.arange(len(full_testset))

		trainset = Subset(full_trainset, indx_train)
		trainset_eval = Subset(full_trainset_eval, indx_train)
		testset = Subset(full_testset, indx_test)
		trainset.dataset.targets = torch.zeros(trainset.dataset.attr.shape[0], dtype=torch.int8)
		trainset_eval.dataset.targets = torch.zeros(trainset.dataset.attr.shape[0], dtype=torch.int8)
		testset.dataset.targets = torch.zeros(trainset.dataset.attr.shape[0], dtype=torch.int8)

	else:
		raise NotImplementedError('This dataset is not supported!')
	
	assert trainset.__class__ == testset.__class__ == trainset_eval.__class__ == Subset
	return trainset, trainset_eval, testset


def get_gen(dataset, configs, validation=False, shuffle=True, smalltree=False, smalltree_ind=None):
	"""Given the dataset and a config file, it will output the DataLoader for training.

	Parameters
	----------
	dataset : torch.dataset
		A tensor dataset.
	configs : dict
		A dictionary of config settings.
	validation : bool, optional
		If set to True it will not drop the last batch, during training it is preferrable to drop the last batch if it
		has a different shape to avoid changing the batch normalization statistics.
	shuffle : bool, optional
		Whether to shuffle the dataset at every epoch.
	smalltree : bool, optional
		Whether the method should output the DataLoader for the small tree training, where a subset of training inputs
		are used.
	smalltree_ind : list
		For training the small tree during the growing strategy of TreeVAE, only a subset of training inputs will be
		used for efficiency.

	Returns
	------
	DataLoader
		The dataloader of the provided dataset.
	"""
	batch_size = configs['training']['batch_size']
	drop_last = not validation
	try:
		num_workers = configs['parser']['num_workers']
	except:
		num_workers = 6

	if smalltree:
		dataset = Subset(dataset, smalltree_ind)

	# Call the DataLoader when contrastive learning is used
	if configs['training']['augment'] and configs['training']['augmentation_method'] != ['simple'] and not validation:
		# As one datapoint leads to two samples, we have to half the batch size to retain same number of samples per batch
		assert batch_size % 2 == 0
		batch_size = batch_size // 2 
		data_gen = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True,
							  persistent_workers=True, collate_fn=custom_collate_fn, drop_last=drop_last)

	# Call the DataLoader without contrastive learning
	else:
		data_gen = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True,
							  persistent_workers=True, drop_last=drop_last)
	return data_gen


def select_subset(y_train, y_test, num_classes):
	# Select a random subset of labels where the number of different labels equal num_classes.
	digits = np.random.choice([i for i in range(len(np.unique(y_train)))], size=num_classes, replace=False)
	indx_train = np.array([], dtype=int)
	indx_test = np.array([], dtype=int)
	for i in range(num_classes):
		indx_train = np.append(indx_train, np.where(y_train == digits[i])[0])
		indx_test = np.append(indx_test, np.where(y_test == digits[i])[0])
	return np.sort(indx_train), np.sort(indx_test)


def custom_collate_fn(batch):
	# Concatenate the augmented versions
	batch = torch.utils.data.default_collate(batch)
	batch[0] = batch[0].transpose(1, 0).reshape(-1,*batch[0].shape[2:])
	batch[1] = batch[1].repeat(2)
	return batch


class ContrastiveTransformations(object):

	def __init__(self, base_transforms, n_views=2):
		self.base_transforms = base_transforms
		self.n_views = n_views

	def __call__(self, x):
		return torch.stack([self.base_transforms(x) for i in range(self.n_views)],dim=0)


def get_selected_omniglot_alphabets():
	return ['Braille', 'Glagolitic', 'Old_Church_Slavonic_(Cyrillic)', 'Oriya', 'Bengali']


class CIFAR100Coarse(torchvision.datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR100Coarse, self).__init__(root, train, transform, target_transform, download)

        # update labels
        coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                                   3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                                   6, 11,  5, 10,  7,  6, 13, 15,  3, 15, 
                                   0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                                   5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                                   16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                                   10, 3,  2, 12, 12, 16, 12,  1,  9, 19, 
                                   2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                                  16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                                  18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
        self.targets = coarse_labels[self.targets]

        # update classes
        self.classes = [['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                        ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                        ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                        ['bottle', 'bowl', 'can', 'cup', 'plate'],
                        ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                        ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                        ['bed', 'chair', 'couch', 'table', 'wardrobe'],
                        ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                        ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                        ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                        ['cloud', 'forest', 'mountain', 'plain', 'sea'],
                        ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                        ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                        ['crab', 'lobster', 'snail', 'spider', 'worm'],
                        ['baby', 'boy', 'girl', 'man', 'woman'],
                        ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                        ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                        ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                        ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                        ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]