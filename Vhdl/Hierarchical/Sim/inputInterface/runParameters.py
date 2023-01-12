import numpy as np

# Training and resting periods in milliseconds
trainDuration = 350	# ms
restTime = 150		# ms


# Time step duration in milliseconds
dt = 0.1		# ms

trainingSteps = int(trainDuration/dt)

# Initial intensity of the input pixels
startInputIntensity = 1
inputIntensity = startInputIntensity


# NumPy default random generator.
rng = np.random.default_rng()

bitWidth = 16
taps = np.array([15, 14, 12, 3])
seed = 5
