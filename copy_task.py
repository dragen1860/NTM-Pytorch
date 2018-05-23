import argparse, random
import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import numpy as np
from aio import EncapsulatedNTM

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123, help="seed value for rngs")
parser.add_argument('--task', default='copy',
                    help="choose the task to train (default: copy)")
parser.add_argument('-p', '--param', action='append', default=[],
                    help='override model  example: "-pbatch_size=4 -pnum_heads=2"')
parser.add_argument('--checkpoint-interval', type=int, default='1000',
                    help="checkpoint interval (default: {1000}). "
                         "use 0 to disable checkpointing")
parser.add_argument('--checkpoint-path', action='store', default='./',
                    help="path for saving checkpoint data (default: './')")
parser.add_argument('--report-interval', type=int, default=100,
                    help="reporting interval")

args = parser.parse_args()
args.checkpoint_path = args.checkpoint_path.rstrip('/')





# Generator of randomized test sequences
def DataLoader(num_batches, batch_size, seq_width, min_len, max_len):
	"""Generator of random sequences for the copy task.

	Creates random batches of "bits" sequences.

	All the sequences within each batch have the same length.
	The length is [`min_len`, `max_len`]

	:param num_batches: Total number of batches to generate.
	:param seq_width: The width of each item in the sequence.
	:param batch_size: Batch size.
	:param min_len: Sequence minimum length.
	:param max_len: Sequence maximum length.

	NOTE: The input width is `seq_width + 1`, the additional input
	contain the delimiter.
	"""
	for batch_num in range(num_batches):
		# All batches have the same sequence length
		seq_len = random.randint(min_len, max_len)
		seq = np.random.binomial(1, 0.5, (seq_len, batch_size, seq_width))
		seq = Variable(torch.from_numpy(seq))

		# The input includes an additional channel used for the delimiter
		inp = Variable(torch.zeros(seq_len + 1, batch_size, seq_width + 1))
		inp[:seq_len, :, :seq_width] = seq
		inp[seq_len, :, seq_width] = 1.0  # delimiter in our control channel
		outp = seq.clone()

		yield batch_num + 1, inp.float(), outp.float()


def clip_grads(net):
	"""gradient clipping to the range [10, 10]."""
	parameters = list(filter(lambda p: p.grad is not None, net.parameters()))
	for p in parameters:
		p.grad.data.clamp_(-10, 10)

def train():
	name = "copy-task"
	controller_size = 100
	controller_layers = 1
	num_heads = 1
	sequence_width = 8
	sequence_min_len = 1
	sequence_max_len = 20
	memory_n = 128
	memory_m = 20
	num_batches = 50000
	batch_size = 1
	rmsprop_lr = 1e-4
	rmsprop_momentum = 0.9
	rmsprop_alpha = 0.95

	net = EncapsulatedNTM(sequence_width + 1, sequence_width,
	                           controller_size, controller_layers,
	                           num_heads, memory_n, memory_m)

	db = DataLoader(num_batches, batch_size, sequence_width, sequence_min_len, sequence_max_len)

	criteon = nn.BCELoss()

	optimizer = optim.RMSprop(net.parameters(),
	                               momentum=rmsprop_momentum,
	                               alpha=rmsprop_alpha,
	                               lr=rmsprop_lr)



	losses = []
	costs = []
	seq_lengths = []

	for batch_num, x, y in db:
		# train
		optimizer.zero_grad()
		inp_seq_len = x.size(0)
		outp_seq_len, batch_size, _ = y.size()

		# new sequence
		net.init_sequence(batch_size)

		# feed the sequence + delimiter
		for i in range(inp_seq_len):
			net(x[i])

		# read the output (no input given)
		y_out = Variable(torch.zeros(y.size()))
		for i in range(outp_seq_len):
			y_out[i], _ = net()

		loss = criteon(y_out, y)
		loss.backward()
		clip_grads(net)
		optimizer.step()

		y_out_binarized = y_out.clone().data
		y_out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)

		# the cost is the number of error bits per sequence
		cost = torch.sum(torch.abs(y_out_binarized - y.data))

		loss, cost =  loss.data[0], cost / batch_size

		losses += [loss]
		costs += [cost]
		seq_lengths += [y.size(0)]



		# report
		if batch_num % args.report_interval == 0:
			mean_loss = np.array(losses[-args.report_interval:]).mean()
			mean_cost = np.array(costs[-args.report_interval:]).mean()

			print("batch %d loss: %.6f cost: %.2f"%(batch_num, mean_loss, mean_cost))


if __name__ == '__main__':
	train()
