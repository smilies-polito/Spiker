# HOW TO
Explain here how to work with this project (compile, build, execute, install etc. etc.)



# HOW TO CITE THIS WORK

1. A. Carpegna, A. Savino and S. Di Carlo, "Spiker: an FPGA-optimized Hardware accelerator for Spiking Neural Networks," 2022 IEEE Computer Society Annual Symposium on VLSI (ISVLSI), Nicosia, Cyprus, 2022, pp. 14-19, doi: [10.1109/ISVLSI54635.2022.00016](https://doi.org/10.1109/ISVLSI54635.2022.00016).

2. Alessio Carpegna: [Design of an hardware accelerator for a Spiking Neural Network](https://webthesis.biblio.polito.it/20606/).  Rel. Stefano Di Carlo, Alessandro Savino. Politecnico di Torino, Corso di laurea magistrale in Ingegneria Elettronica (Electronic Engineering), 2021 



# RELATED DOCUMENTS
1. Sixu Li, Zhaomin Zhang, Ruixin Mao, Jianbiao Xiao, Liang Chang and Jun Zhou "A
  Fast and Energy-Efficient SNN Processor With Adaptive Clock/Event-Driven
  Computation Scheme and Online Learning", IEEE transactions on circuits and
  systems—i: regular papers, vol. 68, no. 4, april 2021
2. Peter U. Diehl and Matthew Cook, "Unsupervised learning of digit recognition using 
  spike-timing-dependent plasticity", Institute of Neuroinformatics, ETH Zurich and 
  University Zurich, Zurich, Switzerland
3. Professor David Heeger, "Poisson Model of Spike Generation", September 5, 2000
  networks on FPGA", in Tsinghua Science and Technology, vol. 25, no. 4, pp. 479-486, 
  Aug. 2020, doi: 10.26599/TST.2019.9010019.



# TODO
1) Implement the developed model in VHDL. 
2) Test the developed neuron on the FPGA, evaluate the performances and
   the required resources in order to have an idea of the maximum reachable
   dimensions of the network.
3) Write the linux driver in order to be able to test, evaluate and use the
   developed structure.
4) Write a complete python program that, provided the characteristics of
   the network, creates the accelerator and interacts with it.
