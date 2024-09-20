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
    import spiker
```

At this point it becomes possible to instantiate the various blocks. Next
sections will go more in detail on the framework's components.

## Net builder

The **NetBuilder** is in charge of translating a high-level description of the
network in an [snnTorch](https://snntorch.readthedocs.io/en/latest/)based
trainable object.

The network can be described through a simple Python dictionary

```python
net_dict = {

		"n_cycles"				: 73,
		"n_inputs"				: 40,

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


![Figure 2: spiker hardware architecture, building blocks](../Doc/spiker.png)
Figure 2: spiker hardware architecture, building blocks
