import subprocess as sp


def createParamDir(dirName):

	cmdString = "if [[ -d " + dirName + " ]]; then "
	cmdString += "rm -r " + dirName + "; "
	cmdString += "fi; "
	cmdString += "mkdir " + dirName + "; "
	
	sp.run(cmdString, shell=True, executable="/bin/bash")
