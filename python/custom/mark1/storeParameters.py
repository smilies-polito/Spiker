def storeParameters(networkList, network, assignments, weightFilename,
			thetaFilename, assignmentsFilename):

	storeArray(weightFilename + str(1) + ".npy", network["poisson2exc"].w)

	storeArray(thetaFilename + str(1) + ".npy", network["excLayer1"].theta)

	for layer in range(2, len(networkList)):

		storeArray(weightFilename + str(layer) + ".npy", 
			network["exc2exc" + str(layer)].w)

		storeArray(thetaFilename + str(layer) + ".npy", 
			network["excLayer" + str(layer)].theta)


	storeArray(assignmentsFilename + ".npy", assignments)




def storeArray(filename, numpyArray):

	with open(filename, 'wb') as fp:
		np.save(fp, numpyArray)




def storePerformace(startTimeTraining, accuracies, performanceFilename):

	timeString = "Total training time : " + \
		seconds2hhmmss(timeit.default_timer() - startTimeTraining)

	accuracyString = "Accuracy evolution:\n" + "\n".join(accuracies)

	with open(performanceFilename + ".txt", 'w') as fp:
		fp.write(timeString)
		fp.write("\n\n")
		fp.write(accuracyString)


def seconds2hhmmss(seconds):

	hours = int(seconds // 3600)
	minutes = int((seconds % 3600) // 60)
	seconds = int(seconds % 60)

	return str(hours) + "h " + str(minutes) + "min " + str(seconds) + "s"
