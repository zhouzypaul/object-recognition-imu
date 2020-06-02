import numpy as np
import random
import matplotlib.pyplot as plt

from kf_v1 import KF
from observation import ls_of_observations


plt.ion()
plt.figure()

# assume the pic is 300 in x-length, and 200 in y-height
real_state = np.zeros((400, 1))
real_state[0] = 1
real_state[1] = 150
real_state[2] = 100
real_state[3] = 30
real_state[4] = 20
real_state[5] = 1
real_state[6] = 50
real_state[7] = 60
real_state[8] = 10
real_state[9] = 10

kf = KF(state_var=1)

NUMSTEPS = 10  # number of loops to run the algorithm
delta_x = 1  # suppose the user turns his head at a constant speed
delta_y = -0.5
var = 1 ** 2

con1s = []
real_con1s = []

for step in range(NUMSTEPS):
    real_state[1] += delta_x
    real_state[2] += delta_y
    real_state[6] += delta_x
    real_state[7] += delta_y

    kf.predict(delta_x=delta_x,
               delta_y=delta_y,
               process_noise_var=np.random.normal(0, 1))
    kf.update(meas_value=ls_of_observations[step])

    con1s.append(kf.mean[0])
    real_con1s.append(real_state[0])

plt.subplot(2, 1, 1)
plt.title('Confidence 1')
plt.plot([con1 for con1 in con1s], 'r')  # red is the one recursively calculated, expected to converge to blue
plt.plot([real for real in real_con1s], 'b')  # blue is the real state

plt.show()
plt.ginput(1)
