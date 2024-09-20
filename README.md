# Spiker: a framework for the generation of efficient Spiking Neural Networks FPGA accelerators for inference at the edge


# Project structure
|	Component		|															Description																|
|:-----------------:|:---------------------------------------------------------------------------------------------------------------------------------:|
|	**spiker**		|	Python package to build, train, quantize and generate the VHDL description of hardware accelerators for Spiking Neural Networks	|
|	**Tutorials**	|									Examples on how to use the different components of spiker										|
|	**Doc**			|				Project documentation. It will be gradually filled with schematics, timing diagrams and similar						|

You can access the documentation for spiker directly in the associated directory.

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

# Installation

    git clone https://github.com/smilies-polito/Spiker.git
    cd Spiker
	pip install .

Or alternatively

	python setup.py install
