import  torch
from    torch import nn

from    ntm import NTM
from    ctrlr import LSTMCtrlr
from    head import NTMReadHead, NTMWriteHead
from    memory import NTMMemory


class NTMCell(nn.Module):

	def __init__(self, input_sz, output_sz, ctrlr_sz, ctrlr_layers, num_heads, N, M):
		"""
		This is a wrapper which can be used as LSTMCell().

		:param input_sz:
		:param output_sz:
		:param ctrlr_sz:
		:param ctrlr_layers:
		:param num_heads:
		:param N:
		:param M:
		"""
		super(NTMCell, self).__init__()

		self.input_sz = input_sz        # input sequence vector size, not sequences length
		self.output_sz = output_sz      # output sequence vector size, not sequences length
		self.ctrlr_sz = ctrlr_sz        # rnn cell hidden size
		self.ctrlr_layers = ctrlr_layers # rnn cell hidden layers number
		self.num_heads = num_heads      # number of headers of write&read head
		self.N = N                      # memory rows
		self.M = M                      # memory columns


		memory = NTMMemory(N, M)
		# the input of controller is formed by concatenating x and read vectors.
		ctrlr = LSTMCtrlr(input_sz + M * num_heads, ctrlr_sz, ctrlr_layers)
		heads = nn.ModuleList([])
		for i in range(num_heads):
			heads.extend([
				NTMReadHead(memory, ctrlr_sz),
				NTMWriteHead(memory, ctrlr_sz)
			])

		self.ntm = NTM(input_sz, output_sz, ctrlr, memory, heads)
		self.memory = memory

	def init_sequence(self, batchsz):
		"""
		Initializing the state.
		"""
		self.batchsz = batchsz
		self.memory.reset(batchsz)
		self.previous_state = self.ntm.create_new_state(batchsz)

	def forward(self, x=None):
		if x is None:
			x = torch.zeros(self.batchsz, self.input_sz)

		o, self.previous_state = self.ntm(x, self.previous_state)
		return o, self.previous_state

	def calculate_num_params(self):
		"""
		Returns the total number of parameters.
		"""
		num_params = 0
		for p in self.parameters():
			num_params += p.data.view(-1).size(0)
		return num_params
