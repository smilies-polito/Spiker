import sys

import numpy as np
from parameters import *



N_sim = 1500

density = 0.01

networkList = [2, 3]

mode = "train"

excDictList = [excDict] * (len(networkList) - 1)

inh2excWeights = inh2excWeight * np.ones(len(networkList) - 1)

scaleFactors = np.array([10])

dt = 1

dt_tauDict = {

	"exc" 		: dt/tauExc,
	"inh"		: dt/tauExc,
	"thresh"	: dt/tauThresh

}

stdpDict["ltp_dt_tau"] = dt/stdpDict["ltp_tau"]
stdpDict["ltd_dt_tau"] = dt/stdpDict["ltd_tau"]
