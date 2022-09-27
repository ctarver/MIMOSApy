import itertools as it
import math

import matplotlib.pyplot as plt
import numpy as np


def calculate_vue_angles(ue_azimuth):
    # Generating the iterator to calculate number of elements
    combinations = it.product(ue_azimuth, ue_azimuth, ue_azimuth)
    n_vue_angles = len(list(combinations))

    # Need to regenerate teh iterator so we can loop over it
    combinations = it.product(ue_azimuth, ue_azimuth, ue_azimuth)
    v_ue_angles = np.zeros(n_vue_angles)
    for i, combination in enumerate(combinations):
        combination = np.deg2rad(combination)
        theta1, theta2, theta3 = combination
        vue_angle_radians = np.arccos(np.cos(theta1) + np.cos(theta2) - np.cos(theta3))

        if math.isnan(vue_angle_radians):
            vue_angle_radians = np.arccos(np.cos(theta1) + np.cos(theta2) - np.cos(theta3) - 2)
        if math.isnan(vue_angle_radians):
            vue_angle_radians = np.arccos(np.cos(theta1) + np.cos(theta2) - np.cos(theta3) + 2)

        # vue_angle_radians = np.arccos(((np.cos(theta1) + np.cos(theta2) - np.cos(theta3)) % 2 ) - 1)
        v_ue_angles[i] = np.rad2deg(vue_angle_radians)

    unique_vue_angles = np.setdiff1d(v_ue_angles, ue_azimuth)
    n_vues = len(unique_vue_angles)
    ue_azimuth = np.append(ue_azimuth, unique_vue_angles)
    return ue_azimuth, n_vues


def plot_training(losses):
    plt.plot(losses)
    plt.yscale('log', base=10)
    plt.grid()
    plt.title('PyTorch Training Loss')
    plt.show()

def save_training(losses):
    backslash_char = '\\\\'
    nl = '\n'
    with open('train.txt', 'w') as f:
        for i, loss in enumerate(losses):
            f.write(f'{i} {loss} {backslash_char}{nl}')