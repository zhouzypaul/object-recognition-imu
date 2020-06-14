import numpy as np
import matplotlib.pyplot as plt

from kf_v4 import f
from simulated_observation import ls_of_observations_v4, real_state_v4


plt.ion()
plt.figure()

# assume the pic is 300 in x-length, and 200 in y-height
real_state = real_state_v4
f.x = ls_of_observations_v4[0]

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


for step in range(NUMSTEPS):
    real_state[1] += delta_x
    real_state[2] += delta_y
    real_state[6] += delta_x
    real_state[7] += delta_y
    real_state[21] += delta_x
    real_state[22] += delta_y

    f.predict(u=np.array([[delta_x], [delta_y]]))
    f.update(z=ls_of_observations_v4[step])

    con1s.append(f.x[20])
    print(f.x[20])
    real_con1s.append(real_state[20])
    x1s.append(f.x[21])
    real_x1s.append(real_state[21])
    y1s.append(f.x[22])
    real_y1s.append(real_state[22])
    w1s.append(f.x[23])
    real_w1s.append(real_state[23])
    h1s.append(f.x[24])
    real_h1s.append(real_state[24])


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


plt.show()
plt.ginput(timeout=300)
