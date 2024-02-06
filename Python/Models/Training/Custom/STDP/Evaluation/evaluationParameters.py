import sys

import numpy as np
from parameters import *


N_sim = 1500

density = 0.005

networkList = [3, 2]

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
