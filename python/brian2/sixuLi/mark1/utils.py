import subprocess as sp


def createParamDir(dirName):

	cmdString = "if [[ -d " + dirName + " ]]; then "
	cmdString += "rm -r " + dirName + "; "
	cmdString += "mkdir " + dirName + "; "
	cmdString += "else "
	cmdString += "mkdir " + dirName + "; "
	cmdString += "fi"
	
	sp.run(cmdString, shell=True)
