import argparse
import json
import os

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image


def obtain_seq_index(index, num_frames, radius):
    seq = list(range(index - radius, index + radius + 1))
    seq = [min(max(item, 0), num_frames - 1) for item in seq]
    return seq


@torch.no_grad()
def get_netG(checkpoint_path, device):
    import yaml

    from generators.face_model import FaceGenerator

    with open("generators/renderer_conf.yaml", "r") as f:
        renderer_config = yaml.load(f, Loader=yaml.FullLoader)

    renderer = FaceGenerator(**renderer_config).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    renderer.load_state_dict(checkpoint["net_G_ema"], strict=False)

    renderer.eval()

    return renderer


@torch.no_grad()
def render_video(
    net_G,
    src_img_path,
    exp_path,
    wav_path,
    output_path,
    device,
    silent=False,
    semantic_radius=13,
    fps=30,
    split_size=16,
    no_move=False,
):
    """
    exp: (N, 73)
    """
    target_exp_seq = np.load(exp_path)
    if target_exp_seq.shape[1] == 257:
        exp_coeff = target_exp_seq[:, 80:144]
        angle_trans_crop = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9370641, 126.84911, 129.03864],
            dtype=np.float32,
        )
        target_exp_seq = np.concatenate(
            [exp_coeff, angle_trans_crop[None, ...].repeat(exp_coeff.shape[0], axis=0)],
            axis=1,
        )
        # (L, 73)
    elif target_exp_seq.shape[1] == 73:
        if no_move:
            target_exp_seq[:, 64:] = np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9370641, 126.84911, 129.03864],
                dtype=np.float32,
            )
    else:
        raise NotImplementedError

    frame = cv2.imread(src_img_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    src_img_raw = Image.fromarray(frame)
    image_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    src_img = image_transform(src_img_raw)

    target_win_exps = []
    for frame_idx in range(len(target_exp_seq)):
        win_indices = obtain_seq_index(
            frame_idx, target_exp_seq.shape[0], semantic_radius
        )
        win_exp = torch.tensor(target_exp_seq[win_indices]).permute(1, 0)
        # (73, 27)
        target_win_exps.append(win_exp)

    target_exp_concat = torch.stack(target_win_exps, dim=0)
    target_splited_exps = torch.split(target_exp_concat, split_size, dim=0)
    output_imgs = []
    for win_exp in target_splited_exps:
        win_exp = win_exp.to(device)
        cur_src_img = src_img.expand(win_exp.shape[0], -1, -1, -1).to(device)
        output_dict = net_G(cur_src_img, win_exp)
        output_imgs.append(output_dict["fake_image"].cpu().clamp_(-1, 1))

    output_imgs = torch.cat(output_imgs, 0)
    transformed_imgs = ((output_imgs + 1) / 2 * 255).to(torch.uint8).permute(0, 2, 3, 1)

    if silent:
        torchvision.io.write_video(output_path, transformed_imgs.cpu(), fps)
    else:
        silent_video_path = f"{output_path}-silent.mp4"
        torchvision.io.write_video(silent_video_path, transformed_imgs.cpu(), fps)
        os.system(
            f"ffmpeg -loglevel quiet -y -i {silent_video_path} -i {wav_path} -shortest {output_path}"
        )
        os.remove(silent_video_path)
