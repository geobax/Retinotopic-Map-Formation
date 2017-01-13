'''
George Baxter
Project - Version 2.13
10/01/17
'''
import numpy as np
import scipy as sp
from scipy import ndimage

# Variables
# Retinal X and Y dimensions
XR = 6
YR = 6
# Tectal X and Y dimensions
XT = 6
YT = 6
# Initial synaptic weighting normal distribution variables
ND_mean = 2.5
ND_sd = 0.14
# Threshold
theta = 10
# dH/dt Coefficients
# Retinal neuron decay constant
alpha = 0.5
# Very short range excitation constant
beta = 0.05
# Short range excitation constant
gamma = 0.025
# Long range inhibition constant
delta = 0.06
# Integration time
dt = 1.0
# Update variables
# Organisation speed
h = 0.016
# Modification threshold
epsilon = 2.0



# Functions
# Threshold
def threshold(H, theta):
	for j in range(0, YT+6):
		for i in range(0, XT+6):
			if H[j, i] > theta:
				H[j, i] = H[j, i] - theta
			else:
				H[j, i] = 0
	return H

# Normalisation
def normalise(s):
	for y in range(XT*YT):
		mean_T = np.mean(s[y, :])
		for x in range(XR*YR):
			s[y, x] = (ND_mean / mean_T) * s[y, x]
	return s

# Convergence
def convergence(H2, H1):
	if np.abs((H2 - H1)) < (0.005 * H1):
		converged = True
	else:
		converged = False
	return converged

# Update synaptic modification
def update(s):
	# Caluclate 'new_H_star'
	new_H_star = new_H_grid.copy()
	new_H_star = threshold(new_H_star, theta)
	# Convert 'new_H_star' into a linear array, removing boundary zeros
	lin_new_H_star = np.reshape(new_H_star[3:-3, 3:-3], (XT*YT, 1))
	# Update 's'
	for y in range(XT*YT):
		if lin_new_H_star[y] > epsilon:
			s[y, RN1] = s[y, RN1] + h * lin_new_H_star[y]
			s[y, RN2] = s[y, RN2] + h * lin_new_H_star[y]
		else:
			s[y, RN1] = s[y, RN1]
			s[y, RN2] = s[y, RN2]
	return s

# Synaptic centre of mass (COM)
def calculate_COM(s):
	# Create zeros arrays for the COM X and Y values
	COM_Y = np.zeros((YT * XT))
	COM_X = np.zeros((YT * XT))
	for retinal_neuron in range(XR*YR):
		COM_grid = np.reshape(s[:, retinal_neuron], (YT, XT))
		# Calculate COM for 'retinal_neuron'
		COM = sp.ndimage.measurements.center_of_mass(COM_grid)
		# Store X and Y coordinates of COM in two seperate arrays
		# Populate 'COM_Y' and 'COM_X'
		COM_Y[retinal_neuron] = COM[0]
		COM_X[retinal_neuron] = COM[1]
	return COM_Y, COM_X



''' MODEL STARTS HERE '''

# Matrix of initial synaptic weightings - X axis is the neuron number in the retina, and Y in the tectum
s = np.random.normal(ND_mean, ND_sd, (XT * YT, XR * YR))


# Polarity Markers

# Define RETINAL polarity markers (square of 4 retinal neurons)
# Generate coordinates in the retinal grid: 'R_PM_a' is the Y coordinate and 'R_PM_b' the X coordinate
# Coordinates for retinal polarity marker 1 ('R_PM1')
R_PM_a1 = np.random.randint(0, YR-1)
R_PM_b1 = np.random.randint(0, XR-1)
# Generate coordinates for adjacent retinal neurons
# Generate coordinates ('R_PM_a2', 'R_PM_b2') for retinal polarity marker 2 ('R_PM2') which is in the same row as 'R_PM1'
# Generate 'R_PM_a2' -> same as 'R_PM_a1'
R_PM_a2 = R_PM_a1
# Generate 'R_PM_b2'
R_PM_b2 = R_PM_b1+1
# Generate coordinates for 'R_PM3' and 'R_PM4' which lie either directly below 'R_PM1' and 'R_PM2'
# Generate 'a' coordinates
R_PM_a3 = R_PM_a1-1
R_PM_a4 = R_PM_a2-1
# Generate 'b' coordinates
R_PM_b3 = R_PM_b1
R_PM_b4 = R_PM_b2

# Convert 'a' and 'b' coordinates for the R_PMs to correspond to specific retinal neurons in 's'
R_PM1 = R_PM_a1 * XT + R_PM_b1
R_PM2 = R_PM_a2 * XT + R_PM_b2
R_PM3 = R_PM_a3 * XT + R_PM_b3
R_PM4 = R_PM_a4 * XT + R_PM_b4

# Define TECTAL polarity markers (square of 4 tectal neurons)
# Generate coordinates in the tectal grid: 'T_PM_a' is the Y coordinate and 'T_PM_b' the X coordinate
# Coordinates for tectal polarity marker 1 ('T_PM1')
T_PM_a1 = np.random.randint(0, YT-1)
T_PM_b1 = np.random.randint(0, XT-1)
# Generate coordinates for adjacent tectal neurons
# Generate coordinates ('T_PM_a2', 'T_PM_b2') for tectal polarity marker 2 ('T_PM2') which is in the same row as 'T_PM1'
# Generate 'T_PM_a2' -> same as 'T_PM_a1'
T_PM_a2 = T_PM_a1
# Generate 'T_PM_b2'
T_PM_b2 = T_PM_b1+1
# Generate coordinates for 'T_PM3' and 'T_PM4' which lie either directly below 'T_PM1' and 'T_PM2'
# Generate 'a' coordinates
T_PM_a3 = T_PM_a1-1
T_PM_a4 = T_PM_a2-1
# Generate 'b' coordinates
T_PM_b3 = T_PM_b1
T_PM_b4 = T_PM_b2

# Convert 'a' and 'b' coordinates for the T_PMs to correspond to specific tectal neurons in 's'
T_PM1 = T_PM_a1 * XT + T_PM_b1
T_PM2 = T_PM_a2 * XT + T_PM_b2
T_PM3 = T_PM_a3 * XT + T_PM_b3
T_PM4 = T_PM_a4 * XT + T_PM_b4

# Increase the stregnth of connection between the corresponding retinal and tectal polarity markers
s[T_PM1, R_PM1] = 5 * s[T_PM1, R_PM1]
s[T_PM2, R_PM2] = 5 * s[T_PM2, R_PM2]
s[T_PM3, R_PM3] = 5 * s[T_PM3, R_PM3]
s[T_PM4, R_PM4] = 5 * s[T_PM4, R_PM4]

# Normalise the synaptic stregnths such that each tectal neuron has an average synaptic weighting of ND_mean
normalise(s)


''' LOOP 1 STARTS HERE '''
loop_count = 0
while loop_count <= 8000:

	# Select random pair of adjacent retinal neurons to activate
	# Generate coordinates in the retinal grid: 'a' is the Y coordinate and 'b' the X coordinate
	# Coordinates of retinal neuron 1 ('RN1')
	a1 = np.random.randint(0, YR)
	b1 = np.random.randint(0, XR)
	# Generate coordinates ('a2', 'b2') for an adjacent retinal neuron ('RN2')
	# Generate 'a2'
	if a1 == 0:
		a2 = np.random.choice([a1, a1+1])
	elif a1 == YR-1:
		a2 = np.random.choice([a1, a1-1])
	else:
		a2 = np.random.choice([a1-1, a1, a1+1])
	# Generate 'b2'
	if a2 == a1:
		if b1 == 0:
			b2 = b1+1
		elif b1 == XR-1:
			b2 = b1-1
		else:
			b2 = np.random.choice([b1-1, b1+1])
	else:
		b2 = b1
	# Convert 'a' and 'b' coordinates for 'RN1' and 'RN2' to correspond to specific retinal neurons in 's'
	RN1 = a1 * XT + b1
	RN2 = a2 * XT + b2

	# Linear zeros matrix of tectal activity
	H_linear = np.zeros((YT * XT))
	# Update H_linear
	for i in range(XT * YT):
		H_linear[i] = s[i, RN1] + s[i, RN2]

	# Change shape of H_linear to be a grid
	H_grid = np.reshape(H_linear, (YT, XT))
	# Create zeros array that has dimensions YT+6, XT+6
	zeros1 = np.zeros((YT+6, XT+6))
	# Populate the inside of 'zeros1' with H_grid such that there are 3 layers of zeros surrounding H_grid -> eliminates need for BCs
	for y in range(0, YT):
		for x in range(0, XT):
			zeros1[y+3, x+3] = H_grid[y, x]
	# Rename 'zeros1' matrix -> this will be the initial activity of the tectal neurons due only to the activity of the activated retinal neurons
	H_grid_initial = zeros1
	# Set H_grid_initial as the first H_grid
	H_grid = H_grid_initial.copy()

	''' LOOP 2 STARTS HERE '''

	# Declare 'converged = False'
	converged = False
	# Iterate whilst 'converged' remains false
	while converged == False:

		# Thresholding
		# Create a copy of H_grid
		H_star = H_grid.copy()
		# Apply thresholding
		H_star = threshold(H_star, theta)
		# Calculate mean of 'H_grid' excluding the boundary zeros
		mean_H_grid = np.mean(H_grid[3:-3, 3:-3])

		# Calculate dH/dt
		# Create zeros matrix to hold dH/dts
		dH_dt = np.zeros((YT, XT))
		# Calculate dH/dts
		for y in range(0, YT):
			for x in range(0, XT):
				dH_dt[y, x] = H_grid_initial[y+3, x+3] \
						+ beta * (H_star[y+3, x+2] + H_star[y+2, x+3] + H_star[y+3, x+4] + H_star[y+4, x+3]) \
						+ gamma * (H_star[y+2, x+2] + H_star[y+1, x+3] + H_star[y+2, x+4] + H_star[y+3, x+5] + H_star[y+4, x+4] + H_star[y+5, x+3] + H_star[y+4, x+2] + H_star[y+3, x+1]) \
						- delta * (H_star[y+1, x+2] + H_star[y, x+3] + H_star[y+1, x+4] + H_star[y+2, x+5] + H_star[y+3, x+6] + H_star[y+4, x+5] + H_star[y+5, x+4] + H_star[y+6, x+3] + H_star[y+5, x+2] + H_star[y+4, x+1] + H_star[y+3, x] +H_star[y+2, x+1]) \
						- alpha * H_grid[y+3, x+3]

		# Add boundary zeros to 'dH_dt'
		zeros2 = np.zeros((YT+6, XT+6))
		for y in range(0, YT):
			for x in range(0, XT):
				zeros2[y+3, x+3] = dH_dt[y, x]
		# Rename 'zeros2' as 'dH_dt'
		dH_dt = zeros2

		# Update H_grid based on dH_dt
		H_grid = H_grid + dH_dt * dt
		# Create a copy of 'H_grid'
		new_H_grid = H_grid.copy()
		# Calculate  mean of 'new_H_grid' excluding boundary zeros
		mean_new_H_grid = np.mean(new_H_grid[3:-3, 3:-3])
		# Has the tectal activity converged?
		converged = convergence(mean_new_H_grid, mean_H_grid)


	# Update 's' based on the tectal neuron firing rates
	update(s)
	# Normalise 's'
	normalise(s)
	# Update 'loop_count'
	loop_count = loop_count + 1


''' CHART PLOTTING '''
# Calculate the 'centre of mass' of the retinal-tectal connections for each of the tectal neurons
COM_Y, COM_X = calculate_COM(s)

import matplotlib.pyplot
import pylab

x = COM_X
y = COM_Y

matplotlib.pyplot.scatter(x,y)
matplotlib.pyplot.xlim([0, XT-1])
matplotlib.pyplot.ylim([0, YT-1])

matplotlib.pyplot.show()

# Plot using seaborn

import seaborn as sns
COM_X_grid = np.reshape(COM_X, (YR, XR))
COM_Y_grid = np.reshape(COM_Y, (YR, XR))



sns.plt.figure()
for i in range(XR):
    sns.plt.plot(COM_X_grid[i,:], COM_Y_grid[i,:], color='r')
    sns.plt.plot(COM_X_grid[:,i], COM_Y_grid[:,i], color='r')

sns.plt.savefig('fisheyeii.png')

''' CLEAN UP THE CODE '''