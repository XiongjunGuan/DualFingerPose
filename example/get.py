import os
import os.path as osp

import cv2

cap_dir = "/disk3/guanxiongjun/data/generate_FingerPose/valid_fixed/cap"
patch_dir = "/disk3/guanxiongjun/data/generate_FingerPose/valid_fixed/fp_patch_bimg/"
fp_dir = "/disk3/guanxiongjun/data/generate_FingerPose/valid_fixed/fp_bimg/"

flst = os.listdir(cap_dir)

cnt = 0
for fname in flst[0:10]:
    cap = cv2.imread(osp.join(cap_dir, fname), 0)
    patch = cv2.imread(osp.join(patch_dir, fname), 0)
    fp = cv2.imread(osp.join(fp_dir, fname), 0)

    save_name = str(cnt) + ".png"
    cv2.imwrite(
        osp.join("/disk3/guanxiongjun/backup_clean/HCI25_pose/example/cap/", save_name),
        cap,
    )
    cv2.imwrite(
        osp.join(
            "/disk3/guanxiongjun/backup_clean/HCI25_pose/example/patch/", save_name
        ),
        patch,
    )

    cv2.imwrite(
        osp.join("/disk3/guanxiongjun/backup_clean/HCI25_pose/example/fp/", save_name),
        fp,
    )

    cnt += 1
