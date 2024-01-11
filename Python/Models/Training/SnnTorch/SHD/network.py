import snntorch as snn
from snntorch import spikegen

import torch
import torch.nn as nn

# Define Network
class Net(nn.Module):
	def __init__(self, num_inputs, num_hidden, num_outputs, alpha, beta):
		super().__init__()

		self.num_inputs = num_inputs
		self.num_hidden = num_hidden
		self.num_outputs = num_outputs

		# Initialize layers
		self.fc1 = nn.Linear(num_inputs, num_hidden)
		self.fb1 = nn.Linear(num_hidden, num_hidden)
		self.lif1 = snn.Synaptic(alpha = alpha, beta = beta)
		self.fc2 = nn.Linear(num_hidden, num_outputs)
		self.lif2 = snn.Synaptic(alpha = alpha, beta = beta)

	def forward(self, input_spikes):

		# Initialize hidden states at t=0
		syn1, mem1 = self.lif1.init_synaptic()
		syn2, mem2 = self.lif2.init_synaptic()

		# Record the final layer
		spk2_rec = []
		mem2_rec = []

		spk1 = torch.zeros(self.num_hidden)

		for step in range(input_spikes.shape[0]):
			cur1 = self.fc1(input_spikes[step]) + self.fb1(spk1)
			spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
			cur2 = self.fc2(spk1)
			spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)

			spk2_rec.append(spk2)
			mem2_rec.append(mem2)

		return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

class TonicTransform:

	def __init__(self, min_time : float, max_time : float, n_samples :
			int = 100, n_inputs : int = 700):

		self.min_time	= min_time
		self.max_time	= max_time
		self.n_samples	= n_samples
		self.n_inputs	= n_inputs

	def __call__(self, sparse_tensor : torch.Tensor) -> torch.Tensor:
		return self.events_to_sparse(sparse_tensor)


	def events_to_sparse(self, sparse_tensor : torch.Tensor) -> \
		torch.Tensor:

		assert "t" and "x" 
		times = self.resample(sparse_tensor["t"])

		units = sparse_tensor["x"]

		indexes = np.stack((times, units), axis = 0)
		values = np.ones(times.shape[0])

		return torch.sparse_coo_tensor(indexes, values, (self.n_samples,
			self.n_inputs), dtype = torch.float)

	def resample(self, np_array : np.array) -> np.array:

		sampling_index = np.linspace(
			self.min_time,
			self.max_time,
			self.n_samples
		)

		return np.digitize(np_array, sampling_index)

def train(x_data, y_data, lr=1e-3, nb_epochs=10):
    
    optimizer = torch.optim.Adamax(net.parameters(), lr=lr, betas=(0.9,0.999))

    log_softmax_fn = nn.LogSoftmax(dim=1)
    loss_fn = nn.NLLLoss()
    
    loss_hist = []
    for e in range(nb_epochs):
        local_loss = []
        for x_local, y_local in sparse_data_generator_from_hdf5_spikes(x_data,
			y_data, batch_size, nb_steps, nb_inputs, max_time):
            output,recs = run_snn(x_local.to_dense())
            _,spks=recs
            m,_=torch.max(output,1)
            log_p_y = log_softmax_fn(m)
            
            # Here we set up our regularizer loss
            # The strength paramters here are merely a guess and there should be ample room for improvement by
            # tuning these paramters.
            # reg_loss = 2e-6*torch.sum(spks) # L1 loss on total number of spikes
            # reg_loss += 2e-6*torch.mean(torch.sum(torch.sum(spks,dim=0),dim=0)**2) # L2 loss on spikes per neuron
            
            # Here we combine supervised loss and the regularizer
            loss_val = loss_fn(log_p_y, y_local) # + reg_loss

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            local_loss.append(loss_val.item())
        mean_loss = np.mean(local_loss)
        loss_hist.append(mean_loss)
        live_plot(loss_hist)
        print("Epoch %i: loss=%.5f"%(e+1,mean_loss))
        
    return loss_hist

def compute_classification_accuracy(dataset, transform):

	""" 
	Computes classification accuracy on supplied data in batches.
	"""

	accs = []

	for data in dataset:

		dense_data = transform(data[0]).to_dense()
		label = data[1]

		output = snn(dense_data)

		# Max over time
		m,_= torch.max(output, 1)

		# Argmax over output units
		_, am=torch.max(m, 0)

		accs.append(label == am)

	return np.mean(accs)

nb_epochs = 300


n_samples = 100
n_inputs = tonic.datasets.hsd.SHD.sensor_size[0]
nb_hidden  = 200
nb_outputs = 20

time_step = 1e-3
batch_size = 256

tau_mem = 10e-3
tau_syn = 5e-3

alpha   = float(np.exp(-time_step/tau_syn))
beta    = float(np.exp(-time_step/tau_mem))

min_time = 0
max_time = 1.4 * 10**6

dataset = tonic.datasets.hsd.SHD(save_to='./data', train=False)

transform = TonicTransform(
	min_time	= min_time,
	max_time	= max_time,
	n_samples	= n_samples,
	n_inputs	= n_inputs
)

# Check whether a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")     
else:
    device = torch.device("cpu")

net = Net(num_inputs, num_hidden, num_outputs, alpha, beta)

log_softmax_fn = nn.LogSoftmax(dim=1)
loss_fn = nn.NLLLoss()

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999))

# Outer training loop
for epoch in range(num_epochs):

	iter_counter = 0
	train_batch = iter(train_loader)

	# Minibatch training loop
	for data in dataset:

		# Forward pass
		net.train()

		dense_data = transform(data[0]).to_dense()
		label = data[1]

		spk_rec, mem_rec = net(dense_data)

		# Initialize the loss & sum over time
		loss_val = torch.zeros((1), dtype=dtype)
		for step in range(num_steps):
			loss_val += loss(mem_rec[step], targets)

		# Gradient calculation + weight update
		optimizer.zero_grad()
		loss_val.backward()
		optimizer.step()

		# Store loss history for future plotting
		loss_hist.append(loss_val.item())

		# Test set
		with torch.no_grad():
			net.eval()
			test_data, test_targets = next(iter(test_loader))

			# Test set forward pass
			test_spk, test_mem = net(test_data.view(batch_size, -1),
					num_steps)

			# Test set loss
			test_loss = torch.zeros((1), dtype=dtype)
			for step in range(num_steps):
				test_loss += loss(test_mem[step], test_targets)
			test_loss_hist.append(test_loss.item())

			# Print train/test loss/accuracy
			if counter % 50 == 0:
				train_printer(net, batch_size, num_steps, epoch,
						iter_counter, loss_hist,
						test_loss_hist, counter, data,
						targets, test_data,
						test_targets)
			counter += 1
			iter_counter +=1
