"""
Description:
Author: Xiongjun Guan
Date: 2025-04-25 16:32:20
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2025-04-25 16:48:56

Copyright (C) 2025 by Xiongjun Guan, Tsinghua University. All rights reserved.
"""

import logging

import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset


def prob_normalization(prob, eps=1e-6):
    prob = prob / np.clip(np.sum(prob, axis=0, keepdims=True), eps, np.inf)
    return prob


class load_dataset_test(Dataset):

    def __init__(
        self,
        fp_lst: None,
        patch_lst: None,
        mask_lst: None,
        cap_lst: None,
        inp_mode="fp",
    ):
        self.fp_lst = fp_lst
        self.patch_lst = patch_lst
        self.mask_lst = mask_lst
        self.cap_lst = cap_lst
        self.inp_mode = inp_mode

    def __len__(self):
        if self.inp_mode == "fp":
            return len(self.fp_lst)
        elif self.inp_mode == "cap":
            return len(self.cap_lst)
        elif self.inp_mode == "patch":
            return len(self.patch_lst)
        elif self.inp_mode == "patch_cap":
            return len(self.patch_lst)

    def __getitem__(self, idx):
        if self.inp_mode == "fp":
            fp_path = self.fp_lst[idx]
            fp = cv2.imread(fp_path, 0).astype(np.float32)
            load_path = fp_path
        else:
            fp = np.array([-1])

        if "mask" in self.inp_mode:
            mask_path = self.mask_lst[idx]
            mask = cv2.imread(mask_path, 0).astype(np.float32)
            load_path = mask_path
        else:
            mask = np.array([-1])

        if "cap" in self.inp_mode:
            cap_path = self.cap_lst[idx]
            cap = cv2.imread(cap_path, 0).astype(np.float32)
            load_path = cap_path
        else:
            cap = np.array([-1])

        if "patch" in self.inp_mode:
            patch_path = self.patch_lst[idx]
            patch = cv2.imread(patch_path, 0).astype(np.float32)
            load_path = patch_path
        else:
            patch = np.array([-1])

        if self.inp_mode == "fp":
            fp = (255.0 - fp) / 255.0  # 0 to background
            fp = fp[None, :, :]
        if "patch" in self.inp_mode:
            patch = (255.0 - patch) / 255.0  # 0 to background
            patch = patch[None, :, :]
        if "cap" in self.inp_mode:
            cap = cap / 255.0
            cap = cap[None, :, :]
        if "mask" in self.inp_mode:
            mask = mask / 255.0
            mask = mask[None, :, :]

        return fp, patch, cap, mask, load_path


def get_dataloader_test(
    fp_lst: None,
    patch_lst: None,
    mask_lst: None,
    cap_lst: None,
    inp_mode="fp",
    batch_size=1,
    shuffle=False,
):
    # Create dataset
    try:
        dataset = load_dataset_test(
            fp_lst=fp_lst,
            patch_lst=patch_lst,
            mask_lst=mask_lst,
            cap_lst=cap_lst,
            inp_mode=inp_mode,
        )
    except Exception as e:
        logging.error("Error in DataLoader: ", repr(e))
        return

    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
    )
    logging.info(f"n_test:{len(dataset)}")

    return test_loader
