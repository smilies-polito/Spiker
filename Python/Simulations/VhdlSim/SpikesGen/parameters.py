import numpy as np

# Time step duration in milliseconds
dt = 0.1		# ms

# Training and resting periods in milliseconds
trainDuration = 350	# ms
trainingSteps = int(trainDuration/dt)

# NumPy default random generator.
rng = np.random.default_rng()

# Initial intensity of the input pixels
startInputIntensity = 2.
inputIntensity = startInputIntensity
