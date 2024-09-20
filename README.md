# Spiker: a framework for the generation of efficient Spiking Neural Networks FPGA accelerators for inference at the edge
This is the official repo of spiker, a comprehensive framework for generating efficient, low-power, and low-area customized Spiking Neural Networks (SNN) accelerators on FPGA for inference at the edge. spiker presents a library of highly efficient neuron architectures and a design framework, enabling the development of complex neural network accelerators with few lines of Python code. 


# Project structure
|	Component		|															Description																|
|:-----------------:|:---------------------------------------------------------------------------------------------------------------------------------:|
|	**spiker**		|	Python package to build, train, quantize and generate the VHDL description of hardware accelerators for Spiking Neural Networks	|
|	**Tutorials**	|									Examples on how to use the different components of spiker										|
|	**Doc**			|				Project documentation. It will be gradually filled with schematics, timing diagrams and similar						|


# Requirements

- numpy >= 1.20
- torch >= 1.12
- snntorch >= 0.9.1
- tabulate >= 0.9.0

# Installation

    git clone https://github.com/smilies-polito/Spiker.git
    cd Spiker
	pip install .

Or alternatively

	python setup.py install

# Citation
[Spiker: a framework for the generation of efficient Spiking Neural Networks FPGA accelerators for inference at the edge](https://arxiv.org/abs/2401.01141)

    @misc{carpegna\_spiker\_2024,
    	title = {Spiker+: a framework for the generation of efficient {Spiking} {Neural} {Networks} {FPGA} accelerators for inference at the edge},  
    	shorttitle = {Spiker+},  
    	url = {http://arxiv.org/abs/2401.01141},  
    	doi = {10.48550/arXiv.2401.01141},  
    	urldate = {2024-01-26},  
    	publisher = {arXiv},  
    	author = {Carpegna, Alessio and Savino, Alessandro and Di Carlo, Stefano},  
    	month = jan,  
    	year = {2024},  
    	keywords = {Computer Science - Neural and Evolutionary Computing, Computer Science - Artificial Intelligence, Computer Science - Hardware Architecture}   
    }

# Acknowledgements

[Neuropuls](https://neuropuls.eu/)

This project has received funding from the European Union’s Horizon Europe research and innovation programme under grant agreement No. 101070238. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union. Neither the European Union nor the granting authority can be held responsible for them.

The code in spiker/vhdl/vhdltools was modified starting from [rftafas/hdltools](https://github.com/rftafas/hdltools).

I would like to thank Domenico Elia Sabella for their valuable assistance in revising and cleaning the final version of the code published on the open repository.
