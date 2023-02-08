import torch

# dataloader arguments
batch_size = 128

dtype = torch.float

# Network Architecture
num_inputs = 28*28
num_hidden1 = 400
num_hidden2 = 100
num_outputs = 10

# Temporal Dynamics
num_steps = 25
beta = 0.95

num_epochs = 10
loss_hist = []
test_loss_hist = []
counter = 0
