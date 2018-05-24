import  torch
from    torch import nn

from    ntm import NTM
from    ctrlr import LSTMCtrlr
from    head import NTMReadHead, NTMWriteHead
from    memory import NTMMemory


class NTMCell(nn.Module):

	def __init__(self, input_sz, output_sz, ctrlr_sz, ctrlr_layers, num_heads, N, M):
		"""Initialize an EncapsulatedNTM.

		:param input_sz: External number of inputs.
		:param output_sz: External number of outputs.
		:param ctrlr_sz: The size of the internal representation.
		:param ctrlr_layers: Controller number of layers.
		:param num_heads: Number of heads.
		:param N: Number of rows in the memory bank.
		:param M: Number of cols/features in the memory bank.
		"""
		super(NTMCell, self).__init__()

		self.input_sz = input_sz
		self.output_sz = output_sz
		self.ctrlr_sz = ctrlr_sz
		self.ctrlr_layers = ctrlr_layers
		self.num_heads = num_heads
		self.N = N
		self.M = M

		# Create the NTM components
		memory = NTMMemory(N, M)
		controller = LSTMCtrlr(input_sz + M * num_heads, ctrlr_sz, ctrlr_layers)
		heads = nn.ModuleList([])
		for i in range(num_heads):
			heads += [
				NTMReadHead(memory, ctrlr_sz),
				NTMWriteHead(memory, ctrlr_sz)
			]

		self.ntm = NTM(input_sz, output_sz, controller, memory, heads)
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
