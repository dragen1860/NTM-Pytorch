import  torch
import  torch.nn.functional as F
from    torch import nn
import  numpy as np





class NTMMemory(nn.Module):
	"""
	Memory bank for NTM.
	"""

	def __init__(self, N, M):
		"""
		Initialize the NTM Memory matrix.

		The memory's dimensions are (batch size x N x M).
		Each batch has it's own memory matrix.

		:param N: Number of rows in the memory.
		:param M: Number of columns/features in the memory.
		"""
		super(NTMMemory, self).__init__()

		self.N = N
		self.M = M

		# The memory bias allows the heads to learn how to initially address
		# memory locations by content
		self.register_buffer('mem_bias', torch.Tensor(N, M))

		# Initialize memory bias
		stdev = 1 / (np.sqrt(N + M))
		nn.init.uniform_(self.mem_bias, -stdev, stdev)

	def reset(self, batchsz):
		"""
		Initialize memory from bias, for start-of-sequence.
		"""
		self.batchsz = batchsz
		# [b, N, M]
		self.memory = self.mem_bias.clone().repeat(batchsz, 1, 1)

	def size(self):
		return self.N, self.M

	def read(self, w):
		"""
		Read from memory (according to section 3.1).
		"""
		return torch.matmul(w.unsqueeze(1), self.memory).squeeze(1)

	def write(self, w, e, a):
		"""
		write to memory (according to section 3.2).
		"""
		self.prev_mem = self.memory
		self.memory = torch.Tensor(self.batchsz, self.N, self.M)
		erase = torch.matmul(w.unsqueeze(-1), e.unsqueeze(1))
		add = torch.matmul(w.unsqueeze(-1), a.unsqueeze(1))
		self.memory = self.prev_mem * (1 - erase) + add

	def address(self, k, beta, g, s, gamma, w_prev):
		"""
		NTM Addressing (according to section 3.3).

		Returns a softmax weighting over the rows of the memory matrix.

		:param k:
		:param beta:
		:param g:
		:param s:
		:param gamma:
		:param w_prev:
		"""
		# activation for each
		k = k.clone()
		beta = F.softplus(beta) # softplus is sort of activation function analogous to ReLU
		g = F.sigmoid(g)
		s = F.softmax(s, dim=1)
		gamma = 1 + F.softplus(gamma)

		# Content addressing
		wc = self._similarity(k, beta)

		# Location addressing
		wg = self._interpolate(w_prev, wc, g)
		w_wave = self._shift(wg, s)
		w = self._sharpen(w_wave, gamma)

		return w

	def _convolve(self, w, s):
		"""
		Circular convolution implementation.
		"""
		assert s.size(0) == 3  # w: [128], s[3]
		t = torch.cat([w[-1:], w, w[:1]])  # [128+2]
		c = F.conv1d(t.view(1, 1, -1), s.view(1, 1, -1)).view(-1)  # [128]
		return c

	def _similarity(self, k, beta):
		k = k.view(self.batchsz, 1, -1)
		w = F.softmax(beta * F.cosine_similarity(self.memory + 1e-16, k + 1e-16, dim=-1), dim=1)
		return w

	def _interpolate(self, w_prev, wc, g):
		return g * wc + (1 - g) * w_prev

	def _shift(self, wg, s):
		result = torch.zeros(wg.size()).to('cuda')
		for b in range(self.batchsz):
			result[b] = self._convolve(wg[b], s[b])
		return result

	def _sharpen(self, w_wave, gamma):
		w = w_wave ** gamma
		w = torch.div(w, torch.sum(w, dim=1).view(-1, 1) + 1e-16)
		return w

	def __repr__(self):
		return 'NTMMemory(N:%d, M:%d, mem_bias:(%d,%d, required_grad=False))'%(self.N, self.M, self.N, self.M)