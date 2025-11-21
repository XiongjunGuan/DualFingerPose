"""
Description:
Author: Xiongjun Guan
Date: 2024-06-13 16:10:11
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2025-04-25 16:43:10

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
"""

import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Train parameters")
    # train settings
    parser.add_argument(
        "--config-name",
        "-c",
        dest="config_name",
        type=str,
        default="DualFingerPose",
        help="suffix of the train parameter yaml",
    )
    parser.add_argument(
        "--cuda-ids",
        "-cids",
        dest="cuda_ids",
        default="2",
    )

    # model settings
    parser.add_argument(
        "--trans-out-form",
        "-tof",
        dest="trans_out_form",
        type=str,
        default="claSum",
    )
    parser.add_argument(
        "--trans-num-classes",
        "-tnc",
        dest="trans_num_classes",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--rot-out-form",
        "-rof",
        dest="rot_out_form",
        type=str,
        default="claSum",
    )
    parser.add_argument(
        "--rot-num-classes",
        "-rnc",
        dest="rot_num_classes",
        type=int,
        default=120,
    )

    # loss settings
    parser.add_argument(
        "--supervise-mode",
        "-sm",
        dest="supervise_mode",
        type=str,
        default="rot_trans",
    )
    parser.add_argument(
        "--trans-loss-form",
        "-tlf",
        dest="trans_loss_form",
        type=str,
        default="CE",
    )
    parser.add_argument(
        "--rot-loss-form",
        "-rlf",
        dest="rot_loss_form",
        type=str,
        default="CE",
    )
    parser.add_argument(
        "--trans-loss-weight",
        "-tlw",
        dest="trans_loss_weight",
        type=float,
        default=1.0,
    )

    return parser.parse_args()
