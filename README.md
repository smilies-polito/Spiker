# Spiker: FPGA-oriented hardware accelerator for Spiking Neural Networks (SNN)
Spiker is a neuromorphic processor to accelerate inference in edge applications, targeting area- and performance-constrained applications.

Here you can find all the code developed to obtain a VHDL description of Spiker. Next image shows the step followed for the design.

![DesignFlow](Doc/Figures/designFlow.png){fig:spikerDesignFlow}
Spiker design flow

## CHARACTERISTICS
* __Neuron model__: Leaky Integrate and Fire
* __Network architecture__: single layer fully-connected
* __Target dataset__: MNIST

## PROJECT ORGANIZATION
* __MNIST__
	* MNIST dataset files in IDX format
	* mnist.py: script to import the dataset in form of numpy array

* __Python__
	* Models: python models of the Spiking Neural Network.
		* Brian2: models developed using [Brian 2 simulator](https://brian2.readthedocs.io/en/stable/)
			* PeterDiehl: brian2 translation of the original work from Peter Diel et al. (see related documents).
			* Simplified: simplified LIF neuron model with respect to the conductance-based model used by Peter Diehl et al.
	* Simulations: scripts to atomate simulations.

* __Vhdl__


## HOW TO CITE
1. A. Carpegna, A. Savino and S. Di Carlo, "Spiker: an FPGA-optimized Hardware accelerator for Spiking Neural Networks," 2022 IEEE Computer Society Annual Symposium on VLSI (ISVLSI), Nicosia, Cyprus, 2022, pp. 14-19, doi: [10.1109/ISVLSI54635.2022.00016](https://doi.org/10.1109/ISVLSI54635.2022.00016).

2. Alessio Carpegna: [Design of an hardware accelerator for a Spiking Neural Network](https://webthesis.biblio.polito.it/20606/).  Rel. Stefano Di Carlo, Alessandro Savino. Politecnico di Torino, Corso di laurea magistrale in Ingegneria Elettronica (Electronic Engineering), 2021 



## RELATED DOCUMENTS
1. S. Li, Z. Zhang, R. Mao, J. Xiao, L. Chang and J. Zhou, "A Fast and Energy-Efficient SNN Processor With Adaptive Clock/Event-Driven Computation Scheme and Online Learning," in IEEE Transactions on Circuits and Systems I: Regular Papers, vol. 68, no. 4, pp. 1543-1552, April 2021, doi: [10.1109/TCSI.2021.3052885](https://doi.org/10.1109/TCSI.2021.3052885).

2. Diehl, P. U., & Cook, M. (2015). Unsupervised learning of digit recognition using spike-timing-dependent plasticity. Frontiers in Computational Neuroscience, doi: [10.3389/fncom.2015.00099](https://doi.org/10.3389/fncom.2015.00099) 

3. Heeger, David. "Poisson model of spike generation." Handout, University of Standford 5.1-13 (2000): 76.



## TO DO
1. Fix Peter Diehl Parameters
2. Fix automatic VHDL simulation in Python/Simulations/VhdlSim
3. Upload python models' flowcharts
4. Upload architecture schematics
5. Merge Linux driver to interface the accelerator
6. Upload complete Xilinx project with compiled accelerator + linux driver
7. Singularity container
