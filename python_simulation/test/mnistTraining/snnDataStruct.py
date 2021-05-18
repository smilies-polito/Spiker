#!/Users/alessio/anaconda3/bin/python3

import numpy as np

def createNetworkDictList(v_th_list, v_reset, w_min_list, w_max_list, networkList):
	return [createLayerDict(v_th_list[i], v_reset, w_min_list[i], w_max_list[i], 
		networkList[i+1], networkList[i]) for i in range(len(networkList)-1)]



def createLayerDict(v_th, v_reset, w_min, w_max, currLayerDim, prevLayerDim):

	layerDict = {}

	layerDict["v_mem"] = v_reset*np.ones(currLayerDim)

	layerDict["v_th"] = v_th

	layerDict["outEvents"] = np.zeros(currLayerDim).astype(bool)

	layerDict["weights"] = (np.random.random((currLayerDim, prevLayerDim)).T*\
				(w_max-w_min) + w_min).T

	layerDict["t_in"] = np.zeros(prevLayerDim).astype(int)

	layerDict["t_out"] = np.zeros(currLayerDim).astype(int)

	return layerDict

