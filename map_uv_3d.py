"""
Description:
Author: Xiongjun Guan
Date: 2024-05-28 16:40:30
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-07-18 19:37:39

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
"""

import os
import os.path as osp
import pickle
from glob import glob

import numpy as np
from scipy.optimize import curve_fit


def func_dim2_power4(xy, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o):
    x, y = xy
    return (
        a
        + b * x
        + c * y
        + d * x**2
        + e * y**2
        + f * x * y
        + g * x**3
        + h * y * x**2
        + i * x * y**2
        + j * y**3
        + k * x**4
        + l * y * x**3
        + m * y**2 * x**2
        + n * x * y**3
        + o * y**4
    )


def func_dim2_power3(xy, a, b, c, d, e, f, g, h, i, j):
    x, y = xy
    return (
        a
        + b * x
        + c * y
        + d * x**2
        + e * y**2
        + f * x * y
        + g * x**3
        + h * y * x**2
        + i * x * y**2
        + j * y**3
    )


def func_dim2_power2(xy, a, b, c, d, e, f):
    x, y = xy
    return a + b * x + c * y + d * x**2 + e * y**2 + f * x * y


def func_dim2_power1(xy, a, b, c):
    x, y = xy
    return a + b * x + c * y


def func_dim1_power1(x, a, b):
    return a + b * x


def func_dim1_power2(x, a, b, c):
    return a + b * x + c * x**2


def func_dim1_power3(x, a, b, c, d):
    return a + b * x + c * x**2 + d * x**3


def func_dim1_power4(x, a, b, c, d, e):
    return a + b * x + c * x**2 + d * x**3 + e * x**4


def test_model(func, popt, inp, y_true, zero_inp):
    y_pred = func(inp, *popt)
    err_mean = np.mean(np.absolute(y_pred - y_true))
    err_std = np.std(np.absolute(y_pred - y_true))
    r2score = (
        1 - ((y_pred - y_true) ** 2).sum() / ((y_true - np.mean(y_true)) ** 2).sum()
    )
    zero_value = func(zero_inp, *popt)
    return err_mean, err_std, r2score, zero_value


def fit_predict_model(inp, y_true, zero_inp, func_title, params_dict):
    if "func_dim2_power4" in func_title:
        func = func_dim2_power4
    elif "func_dim2_power3" in func_title:
        func = func_dim2_power3
    elif "func_dim2_power2" in func_title:
        func = func_dim2_power2
    elif "func_dim2_power1" in func_title:
        func = func_dim2_power1
    if "func_dim1_power4" in func_title:
        func = func_dim1_power4
    elif "func_dim1_power3" in func_title:
        func = func_dim1_power3
    elif "func_dim1_power2" in func_title:
        func = func_dim1_power2
    elif "func_dim1_power1" in func_title:
        func = func_dim1_power1

    popt, pcov = curve_fit(func, inp, y_true)
    err_mean, err_std, r2score, zero_value = test_model(
        func, popt, inp, y_true, zero_inp
    )

    if len(inp) == 2:
        c1 = np.mean(inp[0])
        c2 = np.mean(inp[1])
        center_inp = (c1, c2)
    else:
        center_inp = np.mean(inp)
    center_zero_value = func(center_inp, *popt)

    params_dict[func_title] = {
        "popt": popt,
        "err_mean": err_mean,
        "err_std": err_std,
        "r2score": r2score,
        "zero_value": zero_value,
        "center_inp": center_inp,
        "center_zero_value": center_zero_value,
    }
    return params_dict


if __name__ == "__main__":
    save_dir = "./saved/map/"

    center_pose_dir = " "  # The folder path for storing all user pose information

    person = " "  # user name
    finger = " "  # user finger

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    # --- load data
    with open(osp.join(center_pose_dir, f"{person}-{finger}.pkl"), "rb") as f:
        center_pose_dict = pickle.load(f)

        # (num, ) all data
        r_arr = center_pose_dict["r_arr"]
        c_arr = center_pose_dict["c_arr"]
        roll_arr = center_pose_dict["roll_arr"]
        pitch_arr = center_pose_dict["pitch_arr"]

    # fit function
    params_dict = {}
    for power in range(1, 5):
        params_dict = fit_predict_model(
            (c_arr, r_arr),
            roll_arr,
            (256, 256),
            f"roll_func_dim2_power{power}",
            params_dict,
        )
        params_dict = fit_predict_model(
            (c_arr, r_arr),
            pitch_arr,
            (256, 256),
            f"pitch_func_dim2_power{power}",
            params_dict,
        )
    for power in range(1, 5):
        params_dict = fit_predict_model(
            c_arr, roll_arr, 256, f"roll_func_dim1_power{power}", params_dict
        )
        params_dict = fit_predict_model(
            r_arr, pitch_arr, 256, f"pitch_func_dim1_power{power}", params_dict
        )

    # --- save
    with open(osp.join(save_dir, f"example.pkl"), "wb") as f:
        pickle.dump(params_dict, f)
