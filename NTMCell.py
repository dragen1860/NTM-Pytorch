import  torch
from    torch import nn

from    ctrlr import Ctrlr
from    head import NTMReadHead, NTMWriteHead
from    memory import NTMMemory


class NTMCell(nn.Module):

	def __init__(self, input_sz, output_sz, ctrlr_sz, ctrlr_layers, num_heads, N, M):
		"""
		This is a wrapper which can be used as LSTMCell().

		:param input_sz:  input of NTM, 9, with end delimiter
		:param output_sz: outpt of NTM, 8, without end delimiter
		:param ctrlr_sz:  hidden units of controller, 128
		:param ctrlr_layers:    layers of hidden of controller, 3
		:param num_heads: heads number, 1 = 1 read and 1 write head
		:param N: memory rows, 128
		:param M: memory columns, 20
		"""
		super(NTMCell, self).__init__()

		self.input_sz = input_sz
		self.output_sz = output_sz
		self.ctrlr_sz = ctrlr_sz
		self.ctrlr_layers = ctrlr_layers
		self.num_heads = num_heads
		self.N = N
		self.M = M

		memory = NTMMemory(N, M)

		# the module added into nn.ModuleList will be included automatically as part of current module.
		heads = nn.ModuleList([])
		for i in range(num_heads):
			heads.extend([
				NTMReadHead(memory, ctrlr_sz),
				NTMWriteHead(memory, ctrlr_sz)
			])

		self.ctrlr = Ctrlr(input_sz, output_sz, N, M, heads, ctrlr_sz, ctrlr_layers)
		self.memory = memory

	def zero_state(self, batchsz):
		"""
		Initializing the state.
		"""
		self.batchsz = batchsz
		self.memory.reset(batchsz)
		self.prev_state = self.ctrlr.new_state(batchsz)

	def forward(self, x=None):
		"""

		:param x: [b, 9]
		:return:
		"""
		if x is None: # no input
			# [b, 9]
			x = torch.zeros(self.batchsz, self.input_sz)

		# x: [b, 9]
		# o: [b, 8]
		# state: (init_r, ctrlr_state, heads_state)
		o, self.prev_state = self.ctrlr(x, self.prev_state)

		return o, self.prev_state

	def calculate_num_params(self):
		"""
		Returns the total number of parameters.
		"""
		num_params = 0
		for p in self.parameters():
			num_params += p.data.view(-1).size(0)
		return num_params
