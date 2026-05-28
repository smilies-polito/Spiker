# Spiker: a framework for the generation of efficient Spiking Neural Networks FPGA accelerators for inference at the edge
This is the official repo of spiker, a comprehensive framework for generating efficient, low-power, and low-area customized Spiking Neural Networks (SNN) accelerators on FPGA for inference at the edge. spiker presents a library of highly efficient neuron architectures and a design framework, enabling the development of complex neural network accelerators with few lines of Python code. 

# Video tutorial
Spiker comes together with a series of [video tutorials](https://www.youtube.com/watch?v=y3OvFHBXrDE&list=PLkIAXI4vJ8EgfZki2WRh2Da_h-w6gKbsd) which guides you through all the design steps, from the textual description of the Spiking Neural Network etwork to the generation of the hardware accelerator, described using VHDL.  Everything using python language. 


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

# Citation

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
