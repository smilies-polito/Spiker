# spiker package

spiker is a python package which aims to reduce at a minimum the effort required
to build, train, optimize and generage the VHDL description of hardware
accelerators for Spiking Neural Networks, targeting in particular FPGA
implementations. Figure 1 shows spiker building blocks.

![Figure 1: spiker framework building blocks](../Doc/framework.png)
Figure 1: spiker framework building blocks

# Usage

Once installed the package can be imported with

```python
    import spikerplus
```

At this point it becomes possible to instantiate the various blocks. Next
sections will go more in detail on the framework's components.

The tool is based on the *logging* python built-in module. To enable progress printing at the various steps run

```python
logging.basicConfig(level=logging.INFO)
```


## Net builder

The **NetBuilder** is in charge of translating a high-level description of the network in an [snnTorch](https://snntorch.readthedocs.io/en/latest/)-based trainable object.

The network can be described through a simple Python dictionary:

```python
net_dict = {

		"n_cycles"			: 73,
		"n_inputs"			: 40,

		"layer_0"	: {
			
			"neuron_model"		: "lif",
			"n_neurons"		: 128,
			"alpha"			: None,
			"learn_alpha"		: False,
			"beta"			: 0.9375,
			"learn_beta"		: False,
			"threshold"		: 1.,
			"learn_threshold"	: False,
			"reset_mechanism"	: "subtract"
		},

		"layer_1"	: {
			
			"neuron_model"		: "lif",
			"n_neurons"		: 10,
			"alpha"			: None,
			"learn_alpha"		: False,
			"beta"			: 0.9375,
			"learn_beta"		: False,
			"threshold"		: 1.,
			"learn_threshold"	: False,
			"reset_mechanism"	: "none"
		}
}
```

At this point the network can be built with two lines of code:

```python
net_builder = NetBuilder(net_dict)
snn = net_builder.build()
```

## Trainer

Once the network is built it can be trained on the desired dataset. The
**Trainer** component is the block in charge of training the network

```python
trainer = Trainer(snn)
trainer.train(train_loader, test_loader)
```

Notice that the trainer expects two [torch](https://pytorch.org/) dataloaders.  They are not covered here, refer to the tutorials for more information.

## Optimizer

The **Optimizer** role is to convert internal parameters in a format that can be implemented on the target hardware architecture, namely fixed-point. To do this a quantization step is required. For each parameter, the **Optimizer** performs a grid search between the specified values and returns a log with all the explored configurations and the corresponding loss and accuracy. The range of search for the different bit-widths can be again specified using a python dictionary.

```python
optim_config = {

	"weights_bw"	: {
		"min"	: 5,
		"max"	: 6
	},

	"neurons_bw"	: {
		"min"	: 5,
		"max"	: 6
	},

	"fp_dec"	: {
		"min"	: 2,
		"max"	: 3
	}
}
```

The **Optimizer** expects to receive a trained network object, the network configuration dictionary, as the one used for the **NetBuilder**, and the dictionary containing the search ranges.

```
opt = Optimizer(snn, net_dict, optim_config)
opt.optimize(test_loader)
```

## VHDL generator

Finally, the network is ready to be translated into the corresponding accelerator, described using VHDL language. Between the different parameters explored by the optimizer the user is asked to pick the trade-off which best satisfies its requirements. Again, this can be done using a python dictionary.

```python
optim_params = {

	"weights_bw"	: 6,
	"neurons_bw"	: 8,
	"fp_dec"	: 4

}
```

Starting from the trained network, and using the specified set of bitwidths,
**VhdlGenerator** automatically translates the network into the corresponding
VHDL description.

```python
vhdl_generator = VhdlGenerator(snn, optim_params)
vhdl_snn = vhdl_generator.generate()
```

To print the code of the generated network you can run

```python
print(vhdl_snn.code())
```

The **vhdl** directory contains all the code necessary to translate a *snnTorch* description of the network into a synthesizable VHDL description of the corresponding accelerator.

Figure 2 shows a block diagram of the generated architecture. For more details about the various components refer to [Spiker: a framework for the generation of efficient Spiking Neural Networks FPGA accelerators for inference at the edge](https://arxiv.org/abs/2401.01141).

![Figure 2: spiker hardware architecture, building blocks](../Doc/spiker.png)
Figure 2: spiker hardware architecture, building blocks
