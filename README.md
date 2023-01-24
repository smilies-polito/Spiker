# HOW TO
Explain here how to work with this project (compile, build, execute, install etc. etc.)



# HOW TO CITE THIS WORK

1. A. Carpegna, A. Savino and S. Di Carlo, "Spiker: an FPGA-optimized Hardware accelerator for Spiking Neural Networks," 2022 IEEE Computer Society Annual Symposium on VLSI (ISVLSI), Nicosia, Cyprus, 2022, pp. 14-19, doi: [10.1109/ISVLSI54635.2022.00016](https://doi.org/10.1109/ISVLSI54635.2022.00016).

2. Alessio Carpegna: [Design of an hardware accelerator for a Spiking Neural Network](https://webthesis.biblio.polito.it/20606/).  Rel. Stefano Di Carlo, Alessandro Savino. Politecnico di Torino, Corso di laurea magistrale in Ingegneria Elettronica (Electronic Engineering), 2021 



# RELATED DOCUMENTS
1. S. Li, Z. Zhang, R. Mao, J. Xiao, L. Chang and J. Zhou, "A Fast and Energy-Efficient SNN Processor With Adaptive Clock/Event-Driven Computation Scheme and Online Learning," in IEEE Transactions on Circuits and Systems I: Regular Papers, vol. 68, no. 4, pp. 1543-1552, April 2021, doi: [10.1109/TCSI.2021.3052885](https://doi.org/10.1109/TCSI.2021.3052885).

2. Diehl, P. U., & Cook, M. (2015). Unsupervised learning of digit recognition using spike-timing-dependent plasticity. Frontiers in Computational Neuroscience, doi: [10.3389/fncom.2015.00099](https://doi.org/10.3389/fncom.2015.00099) 

3. Heeger, David. "Poisson model of spike generation." Handout, University of Standford 5.1-13 (2000): 76.

# TODO
1) Implement the developed model in VHDL. 
2) Test the developed neuron on the FPGA, evaluate the performances and
   the required resources in order to have an idea of the maximum reachable
   dimensions of the network.
3) Write the linux driver in order to be able to test, evaluate and use the
   developed structure.
4) Write a complete python program that, provided the characteristics of
   the network, creates the accelerator and interacts with it.
