import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class MyMLP(nn.Module):
	def __init__(self):
		super(MyMLP, self).__init__()
		self.hidden1 = nn.Linear(178,45)
		self.hidden2 = nn.Linear(45,20)
		self.out = nn.Linear(20,5)
	def scale(self,tensor):
		scale = 1.0 / (tensor.max(dim=1, keepdim=True)[0] - tensor.min(dim=1, keepdim=True)[0]) 
		tensor.mul_(scale).sub_(tensor.min(dim=1, keepdim=True)[0])
		return tensor
	def forward(self, x):
		x = self.scale(x.float())
		x = torch.relu(self.hidden1(x))
		x = torch.relu(self.hidden2(x))
		x = self.out(x)
		return x


class MyCNN(nn.Module):
	def __init__(self):
		super(MyCNN, self).__init__()
		self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5,stride=1)
		self.pool = nn.MaxPool1d(kernel_size=2,stride=2)
		self.conv2 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=5,stride=1)
		self.fc1 = nn.Linear(in_features=16 * 41, out_features=200)
		self.fc2 = nn.Linear(200,50)
		self.out = nn.Linear(50, 5)
	def scale(self,tensor):
		scale = 1.0 / (tensor.max(dim=1, keepdim=True)[0] - tensor.min(dim=1, keepdim=True)[0]) 
		tensor.mul_(scale).sub_(tensor.min(dim=1, keepdim=True)[0])
		return tensor
	def forward(self, x):
		x = self.scale(x.float().squeeze()).unsqueeze(1)
		x = self.pool(torch.relu(self.conv1(x)))
		x = self.pool(torch.relu(self.conv2(x)))
		x = torch.relu(self.fc1(x.view(-1, 16 * 41)))
		x = torch.relu(self.fc2(x))
		x = self.out(x)
		
		return x


class MyRNN(nn.Module):
	def __init__(self):
		super(MyRNN, self).__init__()
		self.rnn = nn.GRU(input_size=1, hidden_size=32, num_layers=2, batch_first=True, dropout=0.5)
		self.fc = nn.Linear(in_features=32, out_features=5)
		#self.fc2 = nn.Linear(in_features=16, out_features=5)

	def forward(self, x):
		#x = self.scale(x.squeeze(2)).unsqueeze(2)
		x, _ = self.rnn(x.float())
		x = self.fc(x[:, -1, :])
		#x = self.fc2(x)
		
		return x



class MyVariableRNN(nn.Module):
	def __init__(self, dim_input):
		super(MyVariableRNN, self).__init__()
		# You may use the input argument 'dim_input', which is basically the number of features
		self.fc = nn.Linear(in_features=dim_input,out_features=100)
		self.rnn = nn.GRU(input_size = 100, hidden_size = 25,batch_first=True)
		#self.rnn = nn.LSTM(300,50)

		self.fc2 = nn.Linear(in_features = 25, out_features = 5)
		self.fc3 = nn.Linear(in_features = 5, out_features = 2)
	def forward(self, input_tuple):
		# HINT: Following two methods might be useful
		# 'pack_padded_sequence' and 'pad_packed_sequence' from torch.nn.utils.rnn
		seqs, lengths = input_tuple
		x = torch.tanh(self.fc(seqs))
		x,_ = self.rnn(x)
		#x,_ = self.rnn(pack_padded_sequence(x,lengths,batch_first=True))
		x = torch.relu(self.fc2(x[:, -1, :]))
		x = self.fc3(x)
		return x
