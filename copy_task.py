import  random
import  torch
from    torch import nn
from    torch import optim
import  numpy as np
from    NTMCell import NTMCell




def DataLoader(num_batches, batchsz, seq_sz, min_len, max_len):
	"""
	Generator of random sequences for the copy task.

	Creates random batches of "bits" sequences.

	All the sequences within each batch have the same length.
	The length is [`min_len`, `max_len`]

	:param num_batches: Total number of batches to generate.
	:param seq_sz: squence dimension=[seq_len, seq_sz].
	:param batchsz: Batch size.
	:param min_len: Sequence minimum length.
	:param max_len: Sequence maximum length.

	NOTE: The input size is `seq_sz + 1`, the additional input contains the delimiter.
	"""
	for batch_idx in range(num_batches):
		# All batches have the same sequence length
		seq_len = random.randint(min_len, max_len)

		seq = np.random.binomial(1, 0.5, (seq_len, batchsz, seq_sz))
		seq = torch.tensor(seq).float()

		# The input includes an additional channel used for the delimiter
		input = torch.zeros(seq_len + 1, batchsz, seq_sz + 1)
		input[:seq_len, :, :seq_sz] = seq
		input[seq_len, :, seq_sz] = 1.0  # delimiter in our control channel
		output = seq.clone()

		yield batch_idx + 1, input, output





def train():

	ctrlr_sz = 128
	ctrlr_layers = 1
	num_heads = 1
	seq_sz = 8
	seq_min_len = 1
	seq_max_len = 20
	memory_N = 128
	memory_M = 20

	num_batches = 50000
	batchsz = 10

	device = torch.device('cuda')

	db = DataLoader(num_batches, batchsz, seq_sz, seq_min_len, seq_max_len)
	cell = NTMCell(seq_sz + 1, seq_sz, ctrlr_sz, ctrlr_layers, num_heads, memory_N, memory_M).to(device)
	print(cell)
	criteon = nn.BCELoss().to(device)
	optimizer = optim.RMSprop(cell.parameters(), momentum=0.9, alpha=0.95, lr=1e-4)
	for p in cell.parameters():
		print(p.size())

	losses = []
	costs = []
	seq_lengths = []

	for epoch, x, y in db:
		x = x.to(device)
		y = y.to(device)
		# train
		inp_seq_len = x.size(0)
		outp_seq_len, batchsz, _ = y.size()

		# new sequence
		cell.zero_state(batchsz)

		# feed the sequence + delimiter
		for i in range(inp_seq_len):
			cell(x[i])

		# read the output (no input given)
		pred = torch.zeros(y.size()).to('cuda')
		for i in range(outp_seq_len):
			pred[i], _ = cell(None)

		# pred: [seq_len, b, seq_sz]
		loss = criteon(pred, y)
		optimizer.zero_grad()
		loss.backward()
		nn.utils.clip_grad_norm_(cell.parameters(), 10)
		optimizer.step()

		# we just take use of its vallia data, so stop gradients here.
		pred_binarized = pred.clone().detach().cpu()
		pred_binarized.apply_(lambda x: 0 if x < 0.5 else 1)

		# the cost is the number of error bits per sequence
		cost = torch.sum(torch.abs(pred_binarized - y.cpu()))

		# convert to numpy data
		loss, cost =  loss.item(), cost.item() / batchsz

		losses += [loss]
		costs += [cost]
		seq_lengths += [outp_seq_len]



		# report
		if epoch % 100 == 0:
			mean_loss = np.array(losses[-100:]).mean()
			mean_cost = np.array(costs[-100:]).mean()

			print("epoch %d loss: %.6f cost: %.2f"%(epoch, mean_loss, mean_cost))
			print(x[:,0,:].cpu().numpy()[:2])
			print(pred_binarized[:,0,:].numpy()[:2])


if __name__ == '__main__':
	train()
