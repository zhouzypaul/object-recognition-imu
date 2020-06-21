import pandas as pd
import h5py
import numpy as np


# the path to the imu data file
PATH = '/home/h2r/VP/input/imu/imu.h5'

# read in the hdf file
hf = h5py.File(PATH, 'r')

# get the datasets
time = hf.get('time')
time = np.array(time)

gyro = hf.get('gyro')
gyro = np.array(gyro)
gyro = np.transpose(gyro)  # in the shape of gyro[time index][axis index]

acc = hf.get('acc')
acc = np.array(acc)
acc = np.transpose(acc)  # in the shape of acc[time index][axis index]

hf.close()
