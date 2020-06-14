import numpy as np

# assume the pic is 300 in x-length, and 200 in y-height
delta_x = 1  # suppose the user turns his head at a constant speed
delta_y = -0.5


def make_state_v1(con1, x1, y1, w1, h1, con2, x2, y2, w2, h2) -> np.array:
    state = np.zeros((400, 1))
    state[0] = con1
    state[1] = x1
    state[2] = y1
    state[3] = w1
    state[4] = h1
    state[5] = con2
    state[6] = x2
    state[7] = y2
    state[8] = w2
    state[9] = h2
    return state


real_state_v1 = make_state_v1(1.00, 150, 100, 30, 20,
                              1.00, 50, 60, 10, 10)

first_state_v1 = make_state_v1(0.88, 150, 99, 29, 22,
                               0.95, 51, 59.5, 13, 10)
second_state_v1 = make_state_v1(0.97, 148, 99, 30, 21,
                                0.87, 51.5, 59.5, 10, 11)
third_state_v1 = make_state_v1(0.9, 153, 98, 30, 20,
                               0.9, 52, 58.5, 11, 11)
fourth_state_v1 = make_state_v1(0.93, 153.6, 98, 31, 20,
                                0.88, 54, 58, 10, 10)
fifth_state_v1 = make_state_v1(0.92, 154, 97, 28, 22,
                               0.99, 54, 57, 9, 9)
sixth_state_v1 = make_state_v1(0.93, 155, 97, 30, 23,
                               0.84, 57, 57, 10, 10.5)
seventh_state_1 = make_state_v1(0.98, 157, 96.5, 30, 20,
                                0.95, 57, 56.5, 10, 10)
eighth_state_v1 = make_state_v1(0.62, 159, 95, 31, 20,
                                0.93, 59, 56, 10, 9)
nineth_state_v1 = make_state_v1(0.80, 159, 95, 30, 19,
                                0.55, 59, 55.5, 10, 11)
tenth_state_v1 = make_state_v1(0.92, 160, 95, 30, 20,
                               0.80, 59, 55, 10, 10.5)

ls_of_observations_v1 = [first_state_v1, second_state_v1, third_state_v1, fourth_state_v1, fifth_state_v1, sixth_state_v1,
                         seventh_state_1, eighth_state_v1, nineth_state_v1, tenth_state_v1]


def make_state_v4(con1, x1, y1, w1, h1, con2, x2, y2, w2, h2, con3, x3, y3, w3, h3) -> np.array:
    state = np.zeros((800, 1))
    state[0] = con1
    state[1] = x1
    state[2] = y1
    state[3] = w1
    state[4] = h1
    state[5] = con2
    state[6] = x2
    state[7] = y2
    state[8] = w2
    state[9] = h2
    state[20] = con3
    state[21] = x3
    state[22] = y3
    state[23] = w3
    state[24] = h3
    return state


real_state_v4 = make_state_v4(1.00, 150, 100, 30, 20,
                              1.00, 50, 60, 10, 10,
                              1.00, 10, 190, 150, 100)

first_state_v4 = make_state_v4(0.88, 150, 99, 29, 22,
                               0.95, 51, 59.5, 13, 10,
                               0.98, 9.9, 189, 151, 99)
second_state_v4 = make_state_v4(0.97, 148, 99, 30, 21,
                                0.87, 51.5, 59.5, 10, 11,
                                0.95, 11.6, 188.5, 150, 101)
third_state_v4 = make_state_v4(0.9, 153, 98, 30, 20,
                               0.9, 52, 58.5, 11, 11,
                               0.70, 13, 188.5, 152, 98)
fourth_state_v4 = make_state_v4(0.93, 153.6, 98, 31, 20,
                                0.88, 54, 58, 10, 10,
                                0.55, 13.5, 188, 151, 101)
fifth_state_v4 = make_state_v4(0.92, 154, 97, 28, 22,
                               0.99, 54, 57, 9, 9,
                               0.78, 14.7, 187.5, 148, 100)
sixth_state_v4 = make_state_v4(0.93, 155, 97, 30, 23,
                               0.84, 57, 57, 10, 10.5,
                               0.88, 16.3, 187.5, 148, 100)
seventh_state_4 = make_state_v4(0.98, 157, 96.5, 30, 20,
                                0.95, 57, 56.5, 10, 10,
                                0.80, 17.1, 186.9, 151, 103)
eighth_state_v4 = make_state_v4(0.62, 159, 95, 31, 20,
                                0.93, 59, 56, 10, 9,
                                0.92, 18, 186, 154, 95)
nineth_state_v4 = make_state_v4(0.80, 159, 95, 30, 19,
                                0.55, 59, 55.5, 10, 11,
                                0.95, 19.3, 185, 142, 100)
tenth_state_v4 = make_state_v4(0.92, 160, 95, 30, 20,
                               0.80, 59, 55, 10, 10.5,
                               0.98, 20, 184, 154, 103)

ls_of_observations_v4 = [first_state_v4, second_state_v4, third_state_v4, fourth_state_v4, fifth_state_v4, sixth_state_v4,
                         seventh_state_4, eighth_state_v4, nineth_state_v4, tenth_state_v4]
