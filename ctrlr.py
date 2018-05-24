import  torch
from    torch import nn
import  numpy as np


class LSTMCtrlr(nn.Module):
	"""
	An NTM controller based on LSTM.
	"""

	def __init__(self, input_sz, output_sz, num_layer):
		"""

		:param input_sz:
		:param output_sz:
		:param num_layer:
		"""
		super(LSTMCtrlr, self).__init__()

		self.input_sz = input_sz
		self.output_sz = output_sz
		self.num_layer = num_layer

		self.lstm = nn.LSTM(input_size=input_sz, hidden_size=output_sz, num_layers=num_layer)

		# The hidden state is a learned parameter
		self.lstm_h_bias = nn.Parameter(torch.randn(self.num_layer, 1, self.output_sz) * 0.05)
		self.lstm_c_bias = nn.Parameter(torch.randn(self.num_layer, 1, self.output_sz) * 0.05)

		self.reset_parameters()

	def new_state(self, batchsz):
		# [num_layers * num_directions, batch, hidden_size]
		# [3, b, 128]
		lstm_h = self.lstm_h_bias.clone().repeat(1, batchsz, 1)
		lstm_c = self.lstm_c_bias.clone().repeat(1, batchsz, 1)
		return lstm_h, lstm_c

	def reset_parameters(self):
		for p in self.lstm.parameters():
			if p.dim() == 1:
				nn.init.constant_(p, 0)
			else:
				stdev = 5 / (np.sqrt(self.input_sz + self.output_sz))
				nn.init.uniform_(p, -stdev, stdev)

	def size(self):
		return self.input_sz, self.output_sz

	def forward(self, x, prev_state):
		"""

		:param x:   [b, 29] = [b, 8] extend with [b, ]
		:param prev_state:  [[lstm_layers, b, 100], [lstm_layers, b, 100]]
		:return:
		"""
		# print(x.size(), prev_state[0].size(), prev_state[1].size())
		# [b, input_sz] => [1, b, input_sz]
		x = x.unsqueeze(0)
		# LSTM: [T, b, input_sz] with [layers, b, hidden_sz] => [T, b, hidden_sz] with [layers, b, hidden_sz]
		outp, state = self.lstm(x, prev_state)

		return outp.squeeze(0), state
