import torch

# dataloader arguments
batch_size = 256

dtype = torch.float

# Network Architecture
num_inputs = 28*28
num_hidden = 128
num_outputs = 10

# Temporal Dynamics
num_steps = 100
beta = 0.9375

num_epochs = 30
loss_hist = []
test_loss_hist = []
counter = 0
