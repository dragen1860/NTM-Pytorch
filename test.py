import torch
from    torch import nn



class You(nn.Module):

	def __init__(self):
		super(You, self).__init__()

		for i in range(3):
			self.register_parameter('a', torch.Tensor(1,2,3))


class My(nn.Module):

	def __init__(self, n):
		super(My, self).__init__()

		self.n = n



def main():
	u = You()
	my = My(u)

	print(u)
	print(my)

	for i in u.parameters():
		print(i.shape)

if __name__ == '__main__':
    main()

