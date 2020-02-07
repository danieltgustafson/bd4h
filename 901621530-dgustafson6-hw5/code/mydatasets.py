import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.utils.data import TensorDataset, Dataset

##### DO NOT MODIFY OR REMOVE THIS VALUE #####
checksum = '169a9820bbc999009327026c9d76bcf1'
##### DO NOT MODIFY OR REMOVE THIS VALUE #####

def load_seizure_dataset(path, model_type):
	"""
	:param path: a path to the seizure data CSV file
	:return dataset: a TensorDataset consists of a data Tensor and a target Tensor
	"""
	# TODO: Read a csv file from path.
	# TODO: Please refer to the header of the file to locate X and y.
	# TODO: y in the raw data is ranging from 1 to 5. Change it to be from 0 to 4.
	# TODO: Remove the header of CSV file of course.
	# TODO: Do Not change the order of rows.
	# TODO: You can use Pandas if you want to.
	df=pd.read_csv(path)
	df['y']=df['y']-1
	tensor_data = torch.from_numpy(df.drop('y',axis=1).to_numpy().astype(float))
	tensor_target = torch.from_numpy(df['y'].to_numpy().astype('long'))
	# loader = torch.utils.data.DataLoader(dataset,
 #                                   shuffle=False, num_workers=2)
	if model_type == 'MLP':
		#data = torch.zeros((2, 2))
		#target = torch.zeros(2)
		dataset = TensorDataset(tensor_data, tensor_target)
	elif model_type == 'CNN':
		# data = torch.zeros((2, 2))
		# target = torch.zeros(2)
		dataset = TensorDataset(tensor_data.unsqueeze(1), tensor_target)
	elif model_type == 'RNN':
		# data = torch.zeros((2, 2))
		# target = torch.zeros(2)
		dataset = TensorDataset(tensor_data.unsqueeze(2), tensor_target)
	else:
		raise AssertionError("Wrong Model Type!")

	return dataset


def calculate_num_features(seqs):
	"""
	:param seqs:
	:return: the calculated number of features
	"""
	# TODO: Calculate the number of features (diagnoses codes in the train set)
	return max( [max(item) for sublist in seqs for item in sublist])+1


class VisitSequenceWithLabelDataset(Dataset):
	def __init__(self, seqs, labels, num_features):
		"""
		Args:
			seqs (list): list of patients (list) of visits (list) of codes (int) that contains visit sequences
			labels (list): list of labels (int)
			num_features (int): number of total features available
		"""

		if len(seqs) != len(labels):
			raise ValueError("Seqs and Labels have different lengths")

		self.labels = labels

		# TODO: Complete this constructor to make self.seqs as a List of which each element represent visits of a patient
		# TODO: by Numpy matrix where i-th row represents i-th visit and j-th column represent the feature ID j.
		# TODO: You can use Sparse matrix type for memory efficiency if you want.
		self.seqs=[]
		for patient in seqs:
			visits = len(patient)
			temp = np.zeros([int(visits),int(num_features)])
			for i in range(visits):
				for j in patient[i]:
					temp[i,int(j)]=1
			self.seqs.append(temp)
				
		#self.seqs = [i for i in range(len(labels))]  # replace this with your implementation.

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		# returns will be wrapped as List of Tensor(s) by DataLoader
		return self.seqs[index], self.labels[index]


def visit_collate_fn(batch):
	"""
	DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
	Thus, 'batch' is a list [(seq1, label1), (seq2, label2), ... , (seqN, labelN)]
	where N is minibatch size, seq is a (Sparse)FloatTensor, and label is a LongTensor

	:returns
		seqs (FloatTensor) - 3D of batch_size X max_length X num_features
		lengths (LongTensor) - 1D of batch_size
		labels (LongTensor) - 1D of batch_size
	"""

	# TODO: Return the following two things
	# TODO: 1. a tuple of (Tensor contains the sequence data , Tensor contains the length of each sequence),
	# TODO: 2. Tensor contains the label of each sequence
	labels = []
	lengths = []
	seqs = []
	for i in batch:
		labels.append(i[1])
		lengths.append(len(i[0]))
		seqs.append(i[0])

	max_len = np.max(lengths)
	orders = np.argsort(np.array(lengths)*-1)

	for i in range(len(seqs)):
		len_diff = max_len - len(seqs[i])
		temp = np.zeros([len_diff,len(seqs[i][0])])
		temp_main = np.copy(seqs[i])
		seqs[i] = np.concatenate([temp_main,temp])
	seqs,labels,lengths= np.array(seqs)[orders],np.array(labels)[orders],np.array(lengths)[orders]

	seqs_tensor = torch.FloatTensor(seqs)
	lengths_tensor = torch.LongTensor(lengths)
	labels_tensor = torch.LongTensor(labels)

	return (seqs_tensor, lengths_tensor), labels_tensor
