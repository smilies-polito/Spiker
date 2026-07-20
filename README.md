# Spiker: a framework for the generation of efficient Spiking Neural Networks FPGA accelerators for inference at the edge
This is the official repo of spiker, a comprehensive framework for generating efficient, low-power, and low-area customized Spiking Neural Networks (SNN) accelerators on FPGA for inference at the edge. spiker presents a library of highly efficient neuron architectures and a design framework, enabling the development of complex neural network accelerators with few lines of Python code. In addition to inference-only accelerators, spiker can now generate accelerators with **on-chip local learning** (Spiker-LL), which keep training the network directly in hardware using the STSF local learning rule (see [On-chip local learning](#on-chip-local-learning-spiker-ll)).

# Video tutorial
Spiker comes together with a series of [video tutorials](https://www.youtube.com/watch?v=y3OvFHBXrDE&list=PLkIAXI4vJ8EgfZki2WRh2Da_h-w6gKbsd) which guides you through all the design steps, from the textual description of the Spiking Neural Network etwork to the generation of the hardware accelerator, described using VHDL.  Everything using python language. 


# Project structure
|	Component		|															Description																|
|:-----------------:|:---------------------------------------------------------------------------------------------------------------------------------:|
|	**spiker**		|	Python package to build, train, quantize and generate the VHDL description of hardware accelerators for Spiking Neural Networks, for inference only or with on-chip local learning	|
|	**Tutorials**	|									Examples on how to use the different components of spiker										|
|	**Doc**			|				Project documentation. It will be gradually filled with schematics, timing diagrams and similar						|


# Requirements

- numpy >= 1.20
- torch >= 1.12
- snntorch >= 0.9.1
- tabulate >= 0.9.0

# Installation

From pip repositories

    pip install spikerplus

Or to install the last version from the repo

    git clone https://github.com/smilies-polito/Spiker.git
    cd Spiker/spiker
	pip install .

or using the conda environment 

    git clone https://github.com/smilies-polito/Spiker.git
    cd Spiker/
    conda env create -f environment.yaml
    conda activate spiker

# On-chip local learning (Spiker-LL)

Besides inference-only accelerators, spiker can generate accelerators that
learn **on-chip**: the generated hardware embeds per-layer trainers that keep
updating the synaptic weights while the accelerator runs, using the **STSF
local learning rule**. Training in hardware is online, sample by sample;
neuron thresholds are trained off-chip and remain fixed on-chip.

To enable it, add a `learning` block to the network dictionary, train the
network in software with the `STSFTrainer` (this produces the initial
quantized weights baked into the accelerator), and generate the VHDL as
usual:

```python
from spikerplus import NetBuilder, STSFTrainer, VhdlGenerator
from spikerplus.vhdl import write_vhdl

net_dict = {
	"n_cycles": 10,
	"n_inputs": 64,
	"layer_0": {
		"neuron_model":    "lif",
		"n_neurons":       60,
		"beta":            0.9375,
		"threshold":       1.0,
		"reset_mechanism": "subtract",
	},
	"layer_1": {
		"neuron_model":    "lif",
		"n_neurons":       10,
		"beta":            0.9375,
		"threshold":       1.0,
		"reset_mechanism": "subtract",
	},
	"learning": {
		"rule":               "stsf",
		"update_every":       5,	# hardware weight-update period, in windows
		"lr":                 0.01,
		"loss_value":         0.2,
		"output_const_value": 5,	# output-layer update constant
		"bw":                 16,	# uniform bitwidth of the accelerator
		"fp_dec":             8,	# fixed-point fractional bits
	},
}

snn = NetBuilder(net_dict).build()

trainer = STSFTrainer(snn, lr=0.01, loss_value=0.2, update_every=5)
trainer.train(train_loader, val_loader, n_epochs=1)

vhdl_net = VhdlGenerator(
	snn, {"weights_bw": 16, "neurons_bw": 16, "fp_dec": 8}).generate()
write_vhdl(vhdl_net, rm=True, output_dir="output")
```

## Supported configurations

When the `learning` block is present, the generator supports a specific
hardware configuration. Any violation is rejected with a clear `ValueError`
at build/generation time, never by emitting broken VHDL:

- **Topology**: exactly two trainable layers, input → hidden → output.
- **LIF neurons only**, with `reset_mechanism: "subtract"` on both layers
  (the learning hardware implements the subtractive-reset neuron).
- **No lateral/inhibitory connections** on the trainable layers.
- **Uniform bitwidth**: `weights_bw == neurons_bw == learning["bw"]`
  (any width; typical values are 16 and 8), with matching `fp_dec`.
- **Single generation flavor**: `generate()` with learning enabled supports
  the default `functional=True, interface=False, debug=False` configuration
  (behavioral memories that synthesis tools infer as BRAM).
- `update_every` must fit the hardware update counter
  (`1 <= update_every <= 2**cycles_cnt_bitwidth`).

Inference-only generation is unaffected: without a `learning` block, all the
network formats supported so far (IF/LIF/Synaptic neurons, recurrent
networks, any number of layers, all reset mechanisms, mixed bitwidths,
debug and interface variants) are generated exactly as before.

# Citations

[Spiker+: a framework for the generation of efficient Spiking Neural Networks FPGA accelerators for inference at the edge](https://doi.org/10.1109/TETC.2024.3511676)

    @article{carpegna\_spiker\_2024,
        title = {Spiker+: a framework for the generation of efficient Spiking Neural Networks FPGA accelerators for inference at the edge},
        issn = {2168-6750},
        shorttitle = {Spiker+},
        url = {https://ieeexplore.ieee.org/document/10794606},
        doi = {10.1109/TETC.2024.3511676},
        urldate = {2025-02-05},
        journal = {IEEE Transactions on Emerging Topics in Computing},
        author = {Carpegna, Alessio and Savino, Alessandro and Carlo, Stefano Di},
        year = {2024},
        pages = {1--15},
    }

[Spiker-LL: An Energy-Efficient FPGA Accelerator Enabling Adaptive Local Learning in Spiking Neural Networks](https://arxiv.org/abs/2605.18003)

    @misc{caviglia2026spikerllenergyefficientfpgaaccelerator,
        title={Spiker-LL: An Energy-Efficient FPGA Accelerator Enabling Adaptive Local Learning in Spiking Neural Networks}, 
        author={Alessio Caviglia and Filippo Marostica and Alessandro Savino and Stefano Di Carlo},
        year={2026},
        eprint={2605.18003},
        archivePrefix={arXiv},
        primaryClass={cs.NE},
        url={https://arxiv.org/abs/2605.18003}, 
    }

You can find the very first version of spiker at:

[Spiker: an FPGA-optimized Hardware accelerator for Spiking Neural Networks](https://doi.org/10.1109/ISVLSI54635.2022.00016)

    @inproceedings{carpegna\_spiker\_2022,
        title = {Spiker: an FPGA-optimized Hardware accelerator for Spiking Neural Networks},
        shorttitle = {Spiker},
        url = {https://ieeexplore.ieee.org/document/9911998},
        doi = {10.1109/ISVLSI54635.2022.00016},
        urldate = {2025-02-05},
        booktitle = {2022 {IEEE} {Computer} {Society} {Annual} {Symposium} on {VLSI} ({ISVLSI})},
        author = {Carpegna, Alessio and Savino, Alessandro and Di Carlo, Stefano},
        month = jul,
        year = {2022},
        pages = {14--19},
    }
    

# Acknowledgements

[Neuropuls](https://neuropuls.eu/)

This project has received funding from the European Union’s Horizon Europe research and innovation programme under grant agreement No. 101070238. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union. Neither the European Union nor the granting authority can be held responsible for them.

The code in spiker/vhdl/vhdltools was modified starting from [rftafas/hdltools](https://github.com/rftafas/hdltools).

I would like to thank Domenico Elia Sabella for their valuable assistance in revising and cleaning the final version of the code published on the open repository.
