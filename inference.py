"""
Description:
Author: Xiongjun Guan
Date: 2023-12-11 10:21:53
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2025-02-27 20:15:59

Copyright (C) 2023 by Xiongjun Guan, Tsinghua University. All rights reserved.
"""

import argparse
import os
import os.path as osp
import pickle
import random
from glob import glob

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from data_loader_test import get_dataloader_test
from models.DualFingerPose import DualFingerPose
from utils.affine_func import translate_vec
from utils.trans_est import classify2vector_rot, classify2vector_trans
from utils.visual import draw_img_with_pose


def set_seed(seed=7):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_uvpose(vec):
    xc, yc = 256, 256
    center_vec = np.array([256, 256, 0])
    img_vec = translate_vec(center_vec, xc, yc, xc - vec[0], yc - vec[1], -vec[2])
    return img_vec


def func_dim2_power1(xy, a, b, c):
    x, y = xy
    return a + b * x + c * y


if __name__ == "__main__":
    set_seed(7)

    data_basedir = "./example"  # data dir
    model_dir = "./ckpts/DualFingerPose/"  # model weights

    parser = argparse.ArgumentParser(description="Generate parameters")
    parser.add_argument(
        "-inp_mode",
        type=str,
        default="patch_cap",
        help="fp / patch / cap / mask / patch_cap / patch_mask",
    )
    parser.add_argument(
        "-cuda_ids",
        dest="cuda_ids",
        default="2",
    )
    parser.add_argument(
        "-batch_size",
        dest="batch_size",
        default=4,
    )
    parser.add_argument(
        "-show_mode",
        dest="show_mode",  # patch / fp
        default="patch",
    )
    args = parser.parse_args()

    cuda_ids = args.cuda_ids
    if "," in cuda_ids:
        cuda_ids = [int(x) for x in args.cuda_ids.split(",")]
    else:
        cuda_ids = [int(cuda_ids)]

    batch_size = args.batch_size

    # --- load model
    pth_path = osp.join(model_dir, "best.pth")
    config_path = osp.join(model_dir, "config.yaml")
    with open(config_path, "r") as config:
        cfg = yaml.safe_load(config)

    device = torch.device(
        "cuda:{}".format(str(cuda_ids[0])) if torch.cuda.is_available() else "cpu"
    )

    model = DualFingerPose(
        inp_mode=cfg["model_cfg"]["inp_mode"],
        trans_out_form=cfg["model_cfg"]["trans_out_form"],
        trans_num_classes=cfg["model_cfg"]["trans_num_classes"],
        rot_out_form=cfg["model_cfg"]["rot_out_form"],
        rot_num_classes=cfg["model_cfg"]["rot_num_classes"],
    )
    model.load_state_dict(torch.load(pth_path, map_location=f"cuda:{cuda_ids[0]}"))

    model = torch.nn.DataParallel(
        model,
        device_ids=cuda_ids,
        output_device=cuda_ids[0],
    ).to(device)
    model.eval()

    # --- load transformation param
    with open(osp.join(model_dir, "map_example.pkl"), "rb") as f:
        map_params = pickle.load(f)
        roll_popt = map_params["roll_func_dim2_power1"]["popt"]
        pitch_popt = map_params["pitch_func_dim2_power1"]["popt"]

    # --- load dataset
    test_patch_dir = osp.join(data_basedir, "patch")
    test_cap_dir = osp.join(data_basedir, "cap")
    ftitle_lst = os.listdir(osp.join(data_basedir, "patch"))
    ftitle_lst = [x.replace(".png", "") for x in ftitle_lst]
    patch_lst = []
    cap_lst = []
    for ftitle in ftitle_lst:
        patch_lst.append(osp.join(test_patch_dir, ftitle + ".png"))
        cap_lst.append(osp.join(test_cap_dir, ftitle + ".png"))

    # --- set save dir
    save_dir = osp.join(data_basedir, "result")
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    valid_loader = get_dataloader_test(
        fp_lst=None,
        patch_lst=patch_lst,
        mask_lst=None,
        cap_lst=cap_lst,
        inp_mode=cfg["model_cfg"]["inp_mode"],
        batch_size=batch_size,
        shuffle=False,
    )

    # --- inherit the settings
    trans_out_form = cfg["model_cfg"]["trans_out_form"]
    trans_num_classes = cfg["model_cfg"]["trans_num_classes"]
    rot_out_form = cfg["model_cfg"]["rot_out_form"]
    rot_num_classes = cfg["model_cfg"]["rot_num_classes"]

    # --- test phase
    with torch.no_grad():
        pbar = tqdm(valid_loader)
        for fp, patch, cap, mask, fp_path in pbar:
            assert args.inp_mode == "patch_cap"

            # --- estimation
            cap = cap.float().to(device)
            patch = patch.float().to(device)
            [pred_xy, pred_theta] = model([patch, cap])

            # --- pose probability distribution -> 2d pose
            vec_xy = classify2vector_trans(
                pred_xy, out_form=trans_out_form, trans_num_classes=trans_num_classes
            )
            vec_theta = classify2vector_rot(
                pred_theta, out_form=rot_out_form, rot_num_classes=rot_num_classes
            )

            # --- visualize & output
            imgs = patch.cpu().numpy()
            vec_xys = vec_xy.cpu().numpy()
            vec_thetas = vec_theta.cpu().numpy()

            img_size = 512
            for i in range(imgs.shape[0]):
                vec_xy_i = vec_xys[i, :] + img_size // 2
                vec_theta_i = vec_thetas[i, -1]

                # --- 2D pose
                vec_pred = [vec_xy_i[0], vec_xy_i[1], vec_theta_i]
                # --- 2D pose -> UV pose
                uvpose = get_uvpose(vec_pred)
                # --- UV pose -> 3D pose
                roll = func_dim2_power1(uvpose[:2], *roll_popt)
                pitch = func_dim2_power1(uvpose[:2], *pitch_popt)
                pose3d = np.array([roll, pitch, uvpose[-1]])

                # --- visualize
                fpath_i = fp_path[i]
                ftitle_i = osp.basename(fpath_i).replace(".png", "")

                if args.show_mode == "patch":
                    img_path_i = osp.join(test_patch_dir, ftitle_i + ".png")
                    img_i = cv2.imread(img_path_i, 0)
                    pw = (512 - img_i.shape[0]) // 2
                    img_i = np.pad(
                        img_i,
                        ((pw, pw), (pw, pw)),
                        mode="constant",
                        constant_values=255,
                    )
                elif args.show_mode == "fp":
                    img_path_i = osp.join(
                        "/disk3/guanxiongjun/backup_clean/HCI25_pose/example/fp",
                        ftitle_i + ".png",
                    )
                    img_i = cv2.imread(img_path_i, 0)
                    patch = img_i[196:-196, 196:-196]
                    img_i = np.uint8(255 - (255 - img_i * 1.0) / 4)
                    img_i[196:-196, 196:-196] = patch

                save_path_i = osp.join(save_dir, ftitle_i + f"_{args.show_mode}.png")
                if osp.exists(save_path_i):
                    os.remove(save_path_i)
                draw_img_with_pose(img_i, vec_pred, save_path=save_path_i)

                # --- output
                save_path_i = osp.join(save_dir, ftitle_i + ".txt")
                np.savetxt(save_path_i, vec_pred, fmt="%.2f")
                with open(save_path_i, "w", encoding="utf-8") as f:
                    f.write("# 2d pose (x, y, theta)" + "\n")
                    f.write(" ".join(map(str, vec_pred)) + "\n")

                    f.write("# uv pose (u, v, phi)" + "\n")
                    f.write(" ".join(map(str, uvpose)) + "\n")

                    f.write("# 3d pose (roll, pitch, yaw)" + "\n")
                    f.write(" ".join(map(str, pose3d)) + "\n")

                pass

        pbar.close()
