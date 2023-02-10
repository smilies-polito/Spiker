import numpy as np

def print_batch_accuracy(net, data, batch_size, num_steps, targets, train=False):

	output, _ = net(data.view(batch_size, -1), num_steps)

	_, idx = output.sum(dim=0).max(1)

	acc = np.mean((targets == idx).detach().cpu().numpy())

	if train:
		print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
	else:
		print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")



def train_printer(net, batch_size, num_steps, epoch, iter_counter, loss_hist,
		test_loss_hist, counter, data, targets, test_data,
		test_targets):

	print(f"Epoch {epoch}, Iteration {iter_counter}")

	print(f"Train Set Loss: {loss_hist[counter]:.2f}")

	print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")

	print_batch_accuracy(net, data, batch_size, num_steps, targets, train=True)

	print_batch_accuracy(net, test_data, batch_size, num_steps, test_targets, 
			train=False)

	print("\n")


def test_printer(net, batch_size, num_steps, iter_counter, test_data, test_targets):

	print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")

	print_batch_accuracy(net, test_data, batch_size, num_steps, test_targets, 
			train=False)

	print("\n")
