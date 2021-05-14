# HOW TO
Explain here how to work with this project (compile, build, execute, install etc. etc.)


# RELATED DOCUMENTS
- J. Han, Z. Li, W. Zheng and Y. Zhang, "Hardware implementation of spiking neural 
  networks on FPGA", in Tsinghua Science and Technology, vol. 25, no. 4, pp. 479-486, 
  Aug. 2020, doi: 10.26599/TST.2019.9010019.
- B. Fang, Y. Zhang, R. Yan and H. Tang, "Spike Trains Encoding Optimization for 
  Spiking Neural Networks Implementation in FPGA", 2020 12th International 
  Conference on Advanced Computational Intelligence (ICACI), Dali, China, 2020, 
  pp. 412-418, doi: 10.1109/ICACI49185.2020.9177793.
- Edris Zaman Farsa, Arash Ahmadi, Senior Member, Mohammad Ali Maleki, Morteza 
  Gholami and Hima Nikafshan Rad, "A Low-Cost High-Speed Neuromorphic Hardware 
  Based on Spiking Neural Network", IEEE transactions on circuits and systems—II: 
  express briefs, Vol. 66, No. 9, September 2019
- Amirhossein Tavanaei, Masoud Ghodrati, Saeed Reza Kheradpisheh, Timothè e Masquelier 
  and Anthony Maida, "Deep Learning in Spiking Neural Networks"
- Daniel Neil and Shih-Chii Liu, "Minitaur, an Event-Driven FPGA-Based 
  Spiking Network Accelerator", IEEE TRANSACTIONS ON VERY LARGE SCALE INTEGRATION 
  (VLSI) SYSTEMS, VOL. 22, NO. 12, DECEMBER 2014
- Hajar Asgari, Babak Mazloom-Nezhad Maybodi, Raphaela Kreiser and Yulia Sandamirskaya,
  "Digital Multiplier-Less Spiking Neural Network Architecture of Reinforcement 
  Learning in a Context-Dependent Task", IEEE JOURNAL ON EMERGING AND SELECTED TOPICS 
  IN CIRCUITS AND SYSTEMS, VOL. 10, NO. 4, DECEMBER 2020
- Rafael Medina and Morillas Pablo Ituero, "STDP Design Trade-offs for FPGA-Based 
  Spiking Neural Networks", ETSI Telecomunicacion, Universidad Politecnica de Madrid 
  Ciudad Universitaria s/n, 28040 Madrid, Spain
- Peter U. Diehl and Matthew Cook, "Unsupervised learning of digit recognition using 
  spike-timing-dependent plasticity", Institute of Neuroinformatics, ETH Zurich and 
  University Zurich, Zurich, Switzerland
- Professor David Heeger, "Poisson Model of Spike Generation", September 5, 2000



# TODO
1) Find a way to make the network learn. At the moment I want to use STDP but let's
   see. Once the network is able to learn it can be tested in python before starting
   to develop the VHDL structure.
2) Find an encoding/decoding method in order to be able to use real data in input
   and to use the net to classificate them for example.
3) If the obtained results are reasonable implement the developed model in VHDL.
   here I want to develop two types of circuit in parallel, one with full resources,
   simpler from a control point of view; the other with limited resources. These
   two models at the end will be used to create the circuit basing on the requests
   made by the user.
4) Test the developed neuron on the FPGA, evaluate the performances and
   the required resources in order to have an idea of the maximum reachable
   dimensions of the network.
5) Write the linux driver in order to be able to test, evaluate and use the
   developed structure.
6) Write a complete python program that, provided the characteristics of
   the network, creates the accelerator and interacts with it.

