import  torch
from    torch import nn
import  torch.nn.functional as F


class NTM(nn.Module):
	"""
	A Neural Turing Machine.
	"""

	def __init__(self, input_sz, output_sz, ctrlr, N, M, heads):
		"""
		Initialize the NTM.

		:param input_sz:   9
		:param output_sz:  8
		:param ctrlr: :class:`LSTMCtrlr`
		:param memory: :class:`NTMMemory`
		:param heads: list of :class:`NTMReadHead` or :class:`NTMWriteHead`

		Note: This design allows the flexibility of using any number of read and
			  write heads independently, also, the order by which the heads are
			  called in controlled by the user (order in list)
		"""
		super(NTM, self).__init__()

		self.input_sz = input_sz
		self.output_sz = output_sz
		self.ctrlr = ctrlr
		self.heads = heads

		# 128    20
		self.N, self.M = N, M
		# controller output size:128 is distinct from NTM output sz:8
		# 29,    128
		_, self.ctrlr_output_sz = ctrlr.size()

		# Initialize the initial previous read values to random biases
		self.num_read_heads = 0
		self.init_r = []
		for head in heads:
			if head.is_read_head():
				init_r_bias = torch.randn(1, self.M) * 0.01
				self.register_buffer("read{}_bias".format(self.num_read_heads), init_r_bias)
				self.init_r.append(init_r_bias)
				self.num_read_heads += 1
		assert self.num_read_heads > 0


		# output of controller to output of NTM
		# [128 + heads*20] => [8]
		self.o2o = nn.Linear(self.ctrlr_output_sz + self.num_read_heads * self.M, output_sz)
		self.reset_parameters()

	def new_state(self, batchsz):
		# [1, M] => [b, M]
		init_r = [r.clone().repeat(batchsz, 1) for r in self.init_r]
		ctrlr_state = self.ctrlr.new_state(batchsz)
		w_list = [head.new_w(batchsz) for head in self.heads]

		return init_r, ctrlr_state, w_list

	def reset_parameters(self):
		nn.init.xavier_uniform_(self.o2o.weight, gain=1)
		nn.init.normal_(self.o2o.bias, std=0.01)

	def forward(self, x, prev_state):
		"""
		NTM forward function.

		:param x: input vector (batch_size x num_inputs)
		:param prev_state: The previous state of the NTM
		"""
		# Unpack the previous state
		prev_r_list, prev_ctrlr_state, prev_w_list = prev_state

		# concat x with read vector to fed into controller
		# [b, 9] concat [b, 20] on dim=1 => [b, 29]
		inp = torch.cat([x] + prev_r_list, dim=1)
		# [b, 29] + ([3, b, 128], [3, b,128]) => [b, 128] + ([3, b, 128], [3, b,128])
		ctrlr_outp, ctrlr_state = self.ctrlr(inp, prev_ctrlr_state)

		# Read/Write from the list of heads
		r_list = []
		w_list = []
		for head, prev_w in zip(self.heads, prev_w_list):
			if head.is_read_head():
				r, w = head(ctrlr_outp, prev_w)
				r_list.append(r)
			else:
				w = head(ctrlr_outp, prev_w)
			w_list.append(w)

		# concat output of controller and current read vector to yield output of NTM
		# ctrlr_outp: [b, 128]
		# reads[0]:   [b, 20]
		inp2 = torch.cat([ctrlr_outp] + r_list, dim=1)
		o = F.sigmoid(self.o2o(inp2))

		return o, (r_list, ctrlr_state, w_list)
