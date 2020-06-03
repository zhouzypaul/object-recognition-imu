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
x1s = []
real_x1s = []
y1s = []
real_y1s = []
w1s = []
real_w1s = []
h1s = []
real_h1s = []


con2s = []
real_con2s = []

for step in range(NUMSTEPS):
    real_state[1] += delta_x
    real_state[2] += delta_y
    real_state[6] += delta_x
    real_state[7] += delta_y

    kf.predict(delta_x=delta_x,
               delta_y=delta_y,
               process_noise_var=0.5)
    kf.update(meas_value=ls_of_observations[step],
              meas_variance=1)

    con1s.append(kf.mean[0])
    real_con1s.append(real_state[0])
    x1s.append(kf.mean[1])
    real_x1s.append(real_state[1])
    y1s.append(kf.mean[2])
    real_y1s.append(real_state[2])
    w1s.append(kf.mean[3])
    real_w1s.append(real_state[3])
    h1s.append(kf.mean[4])
    real_h1s.append(real_state[4])

    con2s.append(kf.mean[5])
    real_con2s.append(real_state[5])

plt.subplot(5, 2, 1)
plt.title('Confidence 1')
plt.plot(con1s, 'r')  # red is the one recursively calculated, expected to converge to blue
plt.plot(real_con1s, 'b')  # blue is the real state

plt.subplot(5, 2, 3)
plt.title('x1')
plt.plot(x1s, 'r')
plt.plot(real_x1s, 'b')

plt.subplot(5, 2, 5)
plt.title('y1')
plt.plot(y1s, 'r')
plt.plot(real_y1s, 'b')

plt.subplot(5, 2, 7)
plt.title('width1')
plt.plot(w1s, 'r')
plt.plot(real_w1s, 'b')

plt.subplot(5, 2, 9)
plt.title('heights1')
plt.plot(h1s, 'r')
plt.plot(real_h1s, 'b')

# plt.subplot(5, 2, 2)
# plt.title('Confidence 2')
# plt.plot(con2s, 'r')
# plt.plot(real_con2s, 'b')

plt.show()
plt.ginput(1)
