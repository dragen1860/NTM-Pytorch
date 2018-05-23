import torch
from torch import nn



class NTMCell(nn.Module):


	def __init__(self, hidden_sz, layer_num, seq_len, batchsz, seq_sz):
		super(NTMCell, self).__init__()

		# addressing controller
		self.ctrlr = nn.LSTM(input_size=seq_sz, hidden_size=hidden_sz, num_layers=layer_num)

		# controller output to network output
		self.o2o = nn.Sequential(
								nn.Linear(128, 9)
								)

		# controller output to addressing variable: k, beta, g, gamma
		self.o2c = nn.Sequential(
								nn.Linear(128, 92)
								)


	def forward(self, x):
		"""

		:param x: [seq_len, batchsz, seq_sz], [1~20, 10, 8]
		:return:
		"""

		preds, ()



