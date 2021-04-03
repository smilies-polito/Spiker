# HOW TO
Explain here how to work with this project (compile, build, execute, install etc. etc.)


# RELATED DOCUMENTS
- J. Han, Z. Li, W. Zheng and Y. Zhang, "Hardware implementation of spiking neural 
  networks on FPGA", in Tsinghua Science and Technology, vol. 25, no. 4, pp. 479-486, 
  Aug. 2020, doi: 10.26599/TST.2019.9010019.
- B. Fang, Y. Zhang, R. Yan and H. Tang, "Spike Trains Encoding Optimization for 
  Spiking Neural Networks Implementation in FPGA," 2020 12th International 
  Conference on Advanced Computational Intelligence (ICACI), Dali, China, 2020, 
  pp. 412-418, doi: 10.1109/ICACI49185.2020.9177793.
- Edris Zaman Farsa, Arash Ahmadi, Senior Member, Mohammad Ali Maleki, Morteza 
  Gholami and Hima Nikafshan Rad, "A Low-Cost High-Speed Neuromorphic Hardware 
  Based on Spiking Neural Network", IEEE transactions on circuits and systems—II: 
  express briefs, Vol. 66, No. 9, September 2019
- Amirhossein Tavanaei, Masoud Ghodrati, Saeed Reza Kheradpisheh, Timothè e 
  Masquelier and Anthony Maida, "Deep Learning in Spiking Neural Networks"
- Daniel Neil and Shih-Chii Liu, "Minitaur, an Event-Driven FPGA-Based 
  Spiking Network Accelerator", IEEE TRANSACTIONS ON VERY LARGE SCALE 
  INTEGRATION (VLSI) SYSTEMS, VOL. 22, NO. 12, DECEMBER 2014


# TODO
1) Implement the spiking neuron. At the moment I've chosen the LIF model
   and I'm evaluating the differences between the time-stepped and the
   event driven implementations.
2) Test the developed model with MATLAB or python.
3) Implement the developed model in VHDL.
4) Test the developed neuron on the FPGA, evaluate the performances and
   the required resources in order to have an idea of the maximum reachable
   dimensions of the network.
5) Create the network with a modular organization.
6) Find a way to make the network learn. At the moment my idea is to use
   STDP but let's see. The goal would be to be able to perform online
   training.
7) Write the linux driver in order to be able to test, valuate and use the
   developed structure.
8) Write a complete python program that, provided the characteristics of
   the network, creates the VHDL structure.

