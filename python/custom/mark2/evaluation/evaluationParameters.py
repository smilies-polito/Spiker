import sys

import numpy as np
from parameters import *



N_sim = 3500

density = 0.001

networkList = [3, 2]

mode = "train"

excDictList = [excDict] * (len(networkList) - 1)
inhDictList = [inhDict] * (len(networkList) - 1)

exc2inhWeights = exc2inhWeight * np.ones(len(networkList) - 1)
inh2excWeights = inh2excWeight * np.ones(len(networkList) - 1)

scaleFactors = np.array([10])

dt = 1

dt_tauDict = {

	"exc" 	: dt/tauExc,
	"inh"	: dt/tauExc,
	"theta"	: dt/tauTheta

}

stdpDict["ltp_dt_tau"] = dt/stdpDict["ltp_tau"]
stdpDict["ltd_dt_tau"] = dt/stdpDict["ltd_tau"]
