from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def loadDataset(data_path, batch_size):

	# Define a transform
	transform = transforms.Compose([
		transforms.Resize((28, 28)),
		transforms.Grayscale(),
		transforms.ToTensor(),
		transforms.Normalize((0,), (1,))]
	)

	mnist_train = datasets.MNIST(
		data_path, 
		train=True, 
		download=True, 
		transform=transform
	)

	mnist_test = datasets.MNIST(
		data_path,
		train=False,
		download=True,
		transform=transform
	)


	# Create DataLoaders
	train_loader = DataLoader(
		mnist_train,
		batch_size=batch_size,
		shuffle=True,
		drop_last=True
	)

	test_loader = DataLoader(
		mnist_test,
		batch_size=batch_size,
		shuffle=True,
		drop_last=True
	)

	return train_loader, test_loader
