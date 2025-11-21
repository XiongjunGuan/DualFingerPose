"""
Description:
Author: Xiongjun Guan
Date: 2024-06-04 15:50:36
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2025-02-25 23:23:28

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

eps = 1e-6


class CELoss(nn.Module):

    def __init__(self, need_softmax=False, need_log=True):
        super().__init__()
        self.need_softmax = need_softmax
        self.need_log = need_log

    def forward(self, pred, target):
        if self.need_softmax:
            pred = F.softmax(pred, dim=1)
        if self.need_log:
            pred = pred.clamp_min(1e-6).log()
        loss = torch.mean(-torch.sum(target * pred, dim=1))
        return loss


class FinalLoss(torch.nn.Module):

    def __init__(
        self,
        supervise_mode="rot",
        trans_loss_form="mse",
        rot_out_form="claSum",
        rot_loss_form="mse",
        trans_loss_weight=0.2,
    ):
        super().__init__()
        self.supervise_mode = supervise_mode
        self.trans_loss_form = trans_loss_form
        self.rot_loss_form = rot_loss_form
        self.trans_loss_weight = trans_loss_weight

        if self.trans_loss_form == "mse":
            self.trans_func = nn.MSELoss()
        elif self.trans_loss_form == "SmoothL1":
            self.trans_func = nn.SmoothL1Loss()
        elif self.trans_loss_form == "L1":
            self.trans_func = nn.L1Loss()
        elif self.trans_loss_form == "CE":
            self.trans_func = CELoss()

        if self.rot_loss_form in ["mse_ang", "mse_tan"]:
            self.rot_func = nn.MSELoss()
        elif self.rot_loss_form in ["SmoothL1_ang", "SmoothL1_tan"]:
            self.rot_func = nn.SmoothL1Loss()
        elif self.rot_loss_form in ["L1_ang", "L1_tan"]:
            self.rot_func = nn.L1Loss()
        elif self.rot_loss_form == "CE":
            self.rot_func = CELoss()

    def forward(
        self,
        pred_xy,
        pred_theta,
        vec_xy,
        vec_theta,
        vec_target,
        target_prob_x,
        target_prob_y,
        target_prob_theta,
    ):

        loss_items = {}
        loss = 0

        if "rot" in self.supervise_mode:
            if self.rot_loss_form in [
                "mse_ang",
                "SmoothL1_ang",
                "L1_ang",
                "mse_tan",
                "SmoothL1_tan",
                "L1_tan",
            ]:
                if self.rot_loss_form.split("_")[1] == "tan":
                    loss_cos = self.rot_func(vec_theta[:, -3], vec_target[:, -3])
                    loss_sin = self.rot_func(vec_theta[:, -2], vec_target[:, -2])
                    loss_items["theta-cos"] = loss_cos.item()
                    loss_items["theta-sin"] = loss_sin.item()
                    loss += loss_cos + loss_sin
                elif self.rot_loss_form.split("_")[1] == "ang":
                    rot_const = 90  # [-180,180] -> [-2,2]
                    dtheta = torch.abs(vec_theta[:, -1] - vec_target[:, -1])
                    dtheta = torch.min(dtheta, 360 - dtheta) / rot_const

                    loss_theta = self.rot_func(
                        dtheta, torch.zeros_like(dtheta).to(dtheta.device)
                    )
                    loss_items["theta"] = loss_theta.item()
                    loss += loss_theta
            elif self.rot_loss_form in ["CE"]:
                loss_theta = self.rot_func(pred_theta, target_prob_theta)
                loss_items["theta"] = loss_theta.item()
                loss += loss_theta

        if "trans" in self.supervise_mode:
            if self.trans_loss_form in ["mse", "SmoothL1", "L1"]:
                trans_const = 64  # [-256, 256] -> [-4, 4]
                loss_x = self.trans_func(
                    vec_xy[:, 0] / trans_const, vec_target[:, 0] / trans_const
                )
                loss_y = self.trans_func(
                    vec_xy[:, 1] / trans_const, vec_target[:, 1] / trans_const
                )

                loss_items["pos-x"] = self.trans_loss_weight * loss_x.item()
                loss_items["pos-y"] = self.trans_loss_weight * loss_y.item()
                loss += loss_x + loss_y
            elif self.trans_loss_form in ["CE"]:
                _, c = pred_xy.shape[:2]
                loss_x = self.trans_func(pred_xy[:, : c // 2], target_prob_x)
                loss_y = self.trans_func(pred_xy[:, c // 2 :], target_prob_y)
                loss_items["pos-x"] = self.trans_loss_weight * loss_x.item()
                loss_items["pos-y"] = self.trans_loss_weight * loss_y.item()
                loss += loss_x + loss_y

        return loss, loss_items


class EvalLoss(torch.nn.Module):

    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, vec_xy, vec_theta, vec_target):
        loss_items = {}

        dx = vec_xy[:, 0] - vec_target[:, 0]
        dy = vec_xy[:, 1] - vec_target[:, 1]
        loss_x = torch.mean(torch.abs(dx))
        loss_y = torch.mean(torch.abs(dy))
        loss_dis = torch.mean(torch.sqrt(torch.square(dx) + torch.square(dy)))
        loss_items["valid-x"] = loss_x.item()
        loss_items["valid-y"] = loss_y.item()
        loss_items["valid-trans"] = loss_dis.item()

        dtheta = torch.abs(vec_theta[:, -1] - vec_target[:, -1])
        dtheta = torch.min(dtheta, 360 - dtheta)
        loss_theta = torch.mean(dtheta)
        loss_items["valid-rot"] = loss_theta.item()

        return loss_items
