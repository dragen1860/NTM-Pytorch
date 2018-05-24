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


class NTMHeadBase(nn.Module):
	"""
	An NTM Read/Write Head Basic Class.
	"""

	def __init__(self, memory, ctrlrsz):
		"""
		Initilize the read/write head.

		:param memory: The :class:`NTMMemory` to be addressed by the head.
		:param controller_size: The size of the internal representation.
		"""
		super(NTMHeadBase, self).__init__()

		self.memory = memory
		self.N, self.M = memory.size()
		self.ctrlrsz = ctrlrsz

	def new_w(self, batchsz):
		raise NotImplementedError

	def register_parameters(self):
		raise NotImplementedError

	def is_read_head(self):
		return NotImplementedError

	def _address_memory(self, k, beta, g, s, gamma, w_prev):
		# Handle Activations
		k = k.clone()
		beta = F.softplus(beta)
		g = F.sigmoid(g)
		s = F.softmax(s, dim=1)
		gamma = 1 + F.softplus(gamma)

		w = self.memory.address(k, beta, g, s, gamma, w_prev)

		return w


class NTMReadHead(NTMHeadBase):

	def __init__(self, memory, ctrlrsz):
		super(NTMReadHead, self).__init__(memory, ctrlrsz)

		# Corresponding to k, β, g, s, γ sizes from the paper
		# k is of size of M, and s unit of size 3
		self.read_len = [self.M, 1, 1, 3, 1]
		self.fc_read = nn.Linear(ctrlrsz, sum(self.read_len))
		self.reset_parameters()

	def new_w(self, batchsz):
		# The state holds the previous time step address weightings
		return torch.zeros(batchsz, self.N)

	def reset_parameters(self):
		# Initialize the linear layers
		nn.init.xavier_uniform_(self.fc_read.weight, gain=1.4)
		nn.init.normal_(self.fc_read.bias, std=0.01)

	def is_read_head(self):
		return True

	def forward(self, embeddings, w_prev):
		"""
		NTMReadHead forward function.

		:param embeddings: input representation of the controller.
		:param w_prev: previous step state
		"""
		o = self.fc_read(embeddings)
		k, beta, g, s, gamma = _split_cols(o, self.read_len)

		# obtain address w
		w = self._address_memory(k, beta, g, s, gamma, w_prev)
		# read
		r = self.memory.read(w)

		return r, w


class NTMWriteHead(NTMHeadBase):

	def __init__(self, memory, ctrlrsz):
		super(NTMWriteHead, self).__init__(memory, ctrlrsz)

		# Corresponding to k, β, g, s, γ, e, a sizes from the paper
		self.write_len = [self.M, 1, 1, 3, 1, self.M, self.M]
		self.fc_write = nn.Linear(ctrlrsz, sum(self.write_len))
		self.reset_parameters()

	def new_w(self, batch_size):
		return torch.zeros(batch_size, self.N)

	def reset_parameters(self):
		# Initialize the linear layers
		nn.init.xavier_uniform_(self.fc_write.weight, gain=1.4)
		nn.init.normal_(self.fc_write.bias, std=0.01)

	def is_read_head(self):
		return False

	def forward(self, embeddings, w_prev):
		"""
		NTMWriteHead forward function.

		:param embeddings: input representation of the controller.
		:param w_prev: previous step state
		"""
		o = self.fc_write(embeddings)
		k, beta, g, s, gamma, e, a = _split_cols(o, self.write_len)

		# e should be in [0, 1]
		e = F.sigmoid(e)

		# retain address w
		w = self._address_memory(k, beta, g, s, gamma, w_prev)
		# write into memory
		self.memory.write(w, e, a)

		return w
