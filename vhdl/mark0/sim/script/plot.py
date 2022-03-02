import matplotlib.pyplot as plt
import numpy as np

inSpikes = "../testbench/testbench.sim/sim_1/behav/xsim/inSpikes.txt"
v = "../testbench/testbench.sim/sim_1/behav/xsim/v.txt"
v_th = "../testbench/testbench.sim/sim_1/behav/xsim/v_th.txt"
outSpikes = "../testbench/testbench.sim/sim_1/behav/xsim/outSpikes.txt"

with open(inSpikes) as inSpikes_fp:
	inSpikes = np.loadtxt(inSpikes_fp)



with open(v) as v_fp:
	v = np.loadtxt(v_fp)



with open(v_th) as v_th_fp:
	v_th = np.loadtxt(v_th_fp)



with open(outSpikes) as outSpikes_fp:
	outSpikes = np.loadtxt(outSpikes_fp)

print(inSpikes)
print(v)
print(v_th)
print(outSpikes)
