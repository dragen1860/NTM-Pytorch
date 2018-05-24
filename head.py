import  torch
from    torch import nn
import  torch.nn.functional as F
import  numpy as np


def _split_cols(mat, lengths):
	"""Split a 2D matrix to variable length columns."""
	assert mat.size()[1] == sum(lengths), "Lengths must be summed to num columns"
	l = np.cumsum([0] + lengths)
	results = []
	for s, e in zip(l[:-1], l[1:]):
		results += [mat[:, s:e]]
	return results




class NTMReadHead(nn.Module):

	def __init__(self, memory, ctrlr_sz):
		"""

		:param memory:
		:param ctrlr_sz:
		"""
		super(NTMReadHead, self).__init__()
		self.memory = memory
		self.N, self.M = memory.size()
		self.ctrlr_sz = ctrlr_sz

		# Corresponding to k, beta, g, s, gamma sizes from the paper
		# k is of size of M, and s unit of size 3
		self.read_len = [self.M, 1, 1, 3, 1]
		self.fc_read = nn.Linear(ctrlr_sz, sum(self.read_len))
		self.reset_parameters()

	def new_w(self, batchsz):
		# The state holds the previous time step address weightings
		return torch.zeros(batchsz, self.N).to('cuda')

	def reset_parameters(self):
		# Initialize the linear layers
		nn.init.xavier_uniform_(self.fc_read.weight, gain=1.4)
		nn.init.normal_(self.fc_read.bias, std=0.01)

	def is_read_head(self):
		return True

	def forward(self, h, w_prev):
		"""
		NTMReadHead forward function.

		:param h: controller hidden variable
		:param w_prev: previous step state
		"""
		o = self.fc_read(h)
		k, beta, g, s, gamma = _split_cols(o, self.read_len)

		# obtain address w
		w = self.memory.address(k, beta, g, s, gamma, w_prev)
		# read
		r = self.memory.read(w)

		return r, w




class NTMWriteHead(nn.Module):

	def __init__(self, memory, ctrlrsz):
		"""

		:param memory:
		:param ctrlrsz:
		"""
		super(NTMWriteHead, self).__init__()

		self.memory = memory
		self.N, self.M = memory.size()
		self.ctrlrsz = ctrlrsz

		# Corresponding to k, beta, g, s, gamma, e, a sizes from the paper
		self.write_len = [self.M, 1, 1, 3, 1, self.M, self.M]
		self.fc_write = nn.Linear(ctrlrsz, sum(self.write_len))
		self.reset_parameters()

	def new_w(self, batch_size):
		return torch.zeros(batch_size, self.N).to('cuda')

	def reset_parameters(self):
		# Initialize the linear layers
		nn.init.xavier_uniform_(self.fc_write.weight, gain=1.4)
		nn.init.normal_(self.fc_write.bias, std=0.01)

	def is_read_head(self):
		return False

	def forward(self, h, w_prev):
		"""
		NTMWriteHead forward function.

		:param h: controller hidden variable
		:param w_prev: previous step state
		"""
		o = self.fc_write(h)
		k, beta, g, s, gamma, e, a = _split_cols(o, self.write_len)

		# e should be in [0, 1]
		e = F.sigmoid(e)

		# retain address w
		w = self.memory.address(k, beta, g, s, gamma, w_prev)
		# write into memory
		self.memory.write(w, e, a)

		return w


