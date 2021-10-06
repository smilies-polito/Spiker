# HOW TO
Explain here how to work with this project (compile, build, execute, install etc. etc.)


# RELATED DOCUMENTS
- Sixu Li, Zhaomin Zhang, Ruixin Mao, Jianbiao Xiao, Liang Chang and Jun Zhou "A
  Fast and Energy-Efficient SNN Processor With Adaptive Clock/Event-Driven
  Computation Scheme and Online Learning", IEEE transactions on circuits and
  systems—i: regular papers, vol. 68, no. 4, april 2021
- Peter U. Diehl and Matthew Cook, "Unsupervised learning of digit recognition using 
  spike-timing-dependent plasticity", Institute of Neuroinformatics, ETH Zurich and 
  University Zurich, Zurich, Switzerland
- Professor David Heeger, "Poisson Model of Spike Generation", September 5, 2000
  networks on FPGA", in Tsinghua Science and Technology, vol. 25, no. 4, pp. 479-486, 
  Aug. 2020, doi: 10.26599/TST.2019.9010019.
- Hajar Asgari, Babak Mazloom-Nezhad Maybodi, Raphaela Kreiser and Yulia Sandamirskaya,
  "Digital Multiplier-Less Spiking Neural Network Architecture of Reinforcement 
  Learning in a Context-Dependent Task", IEEE Journal on emerging and selected topics 
  in circuits and systems, vol. 10, no. 4, december 2020
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



# TODO
1) Implement the developed model in VHDL. 
2) Test the developed neuron on the FPGA, evaluate the performances and
   the required resources in order to have an idea of the maximum reachable
   dimensions of the network.
3) Write the linux driver in order to be able to test, evaluate and use the
   developed structure.
4) Write a complete python program that, provided the characteristics of
   the network, creates the accelerator and interacts with it.
