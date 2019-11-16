import torch


w = torch.tensor([0.1])
b = torch.tensor([0.1])
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

X = torch.tensor([1, 2, 3]).view((1, 3))
y_hat = torch.tensor([2, 8, 18]).view((1, 3))	# y = 2 * x^2


def squared_loss(y_hat, y_pre):
	return (y_hat - y_pre)**2


def sgd(params, lr):
	for param in params:
		print("grad:", param.grad)
		param.data -= lr * param.grad


for i in range(X.shape[0]+1):
	x = X[0, i].clone()
	y_pre = w*x + b
	loss = squared_loss(y_hat[0, i], y_pre)/2
	loss.backward()
	sgd([w, b], 0.1)

	w.grad.data.zero_()
	b.grad.data.zero_()

	print(loss)