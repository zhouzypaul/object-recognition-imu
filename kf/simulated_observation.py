import numpy as np

# assume the pic is 300 in x-length, and 200 in y-height
delta_x = 1  # suppose the user turns his head at a constant speed
delta_y = -0.5


def make_state(con1, x1, y1, w1, h1, con2, x2, y2, w2, h2) -> np.array:
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


real_state = make_state(1.00, 150, 100, 30, 20,
                        1.00, 50, 60, 10, 10)

first_state = make_state(0.88, 150, 99, 29, 22,
                         0.95, 51, 59.5, 13, 10)
second_state = make_state(0.97, 148, 99, 30, 21,
                          0.87, 51.5, 59.5, 10, 11)
third_state = make_state(0.9, 153, 98, 30, 20,
                         0.9, 52, 58.5, 11, 11)
fourth_state = make_state(0.93, 153.6, 98, 31, 20,
                          0.88, 54, 58, 10, 10)
fifth_state = make_state(0.92, 154, 97, 28, 22,
                         0.99, 54, 57, 9, 9)
sixth_state = make_state(0.93, 155, 97, 30, 23,
                         0.84, 57, 57, 10, 10.5)
seventh_state = make_state(0.98, 157, 96.5, 30, 20,
                           0.95, 57, 56.5, 10, 10)
eighth_state = make_state(0.62, 159, 95, 31, 20,
                          0.93, 59, 56, 10, 9)
nineth_state = make_state(0.80, 159, 95, 30, 19,
                          0.55, 59, 55.5, 10, 11)
tenth_state = make_state(0.92, 160, 95, 30, 20,
                         0.80, 59, 55, 10, 10.5)

ls_of_observations = [first_state, second_state, third_state, fourth_state, fifth_state, sixth_state,
                      seventh_state, eighth_state, nineth_state, tenth_state]
