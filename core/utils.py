import os
import argparse
from collections import defaultdict
import logging
import pickle
import json

import numpy as np
import torch
from torch import nn
from scipy.io import loadmat

from configs.default import get_cfg_defaults
import dlib
import cv2


def _reset_parameters(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


def get_video_style(video_name, style_type):
    person_id, direction, emotion, level, *_ = video_name.split("_")
    if style_type == "id_dir_emo_level":
        style = "_".join([person_id, direction, emotion, level])
    elif style_type == "emotion":
        style = emotion
    elif style_type == "id":
        style = person_id
    else:
        raise ValueError("Unknown style type")

    return style


def get_style_video_lists(video_list, style_type):
    style2video_list = defaultdict(list)
    for video in video_list:
        style = get_video_style(video, style_type)
        style2video_list[style].append(video)

    return style2video_list


def get_face3d_clip(
    video_name, video_root_dir, num_frames, start_idx, dtype=torch.float32
):
    """_summary_

    Args:
        video_name (_type_): _description_
        video_root_dir (_type_): _description_
        num_frames (_type_): _description_
        start_idx (_type_): "random" , middle, int
        dtype (_type_, optional): _description_. Defaults to torch.float32.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    video_path = os.path.join(video_root_dir, video_name)
    if video_path[-3:] == "mat":
        face3d_all = loadmat(video_path)["coeff"]
        face3d_exp = face3d_all[:, 80:144]  # expression 3DMM range
    elif video_path[-3:] == "txt":
        face3d_exp = np.loadtxt(video_path)
    else:
        raise ValueError("Invalid 3DMM file extension")

    length = face3d_exp.shape[0]
    clip_num_frames = num_frames
    if start_idx == "random":
        clip_start_idx = np.random.randint(low=0, high=length - clip_num_frames + 1)
    elif start_idx == "middle":
        clip_start_idx = (length - clip_num_frames + 1) // 2
    elif isinstance(start_idx, int):
        clip_start_idx = start_idx
    else:
        raise ValueError(f"Invalid start_idx {start_idx}")

    face3d_clip = face3d_exp[clip_start_idx : clip_start_idx + clip_num_frames]
    face3d_clip = torch.tensor(face3d_clip, dtype=dtype)

    return face3d_clip


def get_video_style_clip(
    video_name,
    video_root_dir,
    style_max_len,
    start_idx="random",
    dtype=torch.float32,
    return_start_idx=False,
):
    video_path = os.path.join(video_root_dir, video_name)
    if video_path[-3:] == "mat":
        face3d_all = loadmat(video_path)["coeff"]
        face3d_exp = face3d_all[:, 80:144]  # expression 3DMM range
    elif video_path[-3:] == "txt":
        face3d_exp = np.loadtxt(video_path)
    else:
        raise ValueError("Invalid 3DMM file extension")

    face3d_exp = torch.tensor(face3d_exp, dtype=dtype)

    length = face3d_exp.shape[0]
    if length >= style_max_len:
        clip_num_frames = style_max_len
        if start_idx == "random":
            clip_start_idx = np.random.randint(low=0, high=length - clip_num_frames + 1)
        elif start_idx == "middle":
            clip_start_idx = (length - clip_num_frames + 1) // 2
        elif isinstance(start_idx, int):
            clip_start_idx = start_idx
        else:
            raise ValueError(f"Invalid start_idx {start_idx}")

        face3d_clip = face3d_exp[clip_start_idx : clip_start_idx + clip_num_frames]
        pad_mask = torch.tensor([False] * style_max_len)
    else:
        clip_start_idx = None
        padding = torch.zeros(style_max_len - length, face3d_exp.shape[1])
        face3d_clip = torch.cat((face3d_exp, padding), dim=0)
        pad_mask = torch.tensor([False] * length + [True] * (style_max_len - length))

    if return_start_idx:
        return face3d_clip, pad_mask, clip_start_idx
    else:
        return face3d_clip, pad_mask


def get_video_style_clip_from_np(
    face3d_exp,
    style_max_len,
    start_idx="random",
    dtype=torch.float32,
    return_start_idx=False,
):
    face3d_exp = torch.tensor(face3d_exp, dtype=dtype)

    length = face3d_exp.shape[0]
    if length >= style_max_len:
        clip_num_frames = style_max_len
        if start_idx == "random":
            clip_start_idx = np.random.randint(low=0, high=length - clip_num_frames + 1)
        elif start_idx == "middle":
            clip_start_idx = (length - clip_num_frames + 1) // 2
        elif isinstance(start_idx, int):
            clip_start_idx = start_idx
        else:
            raise ValueError(f"Invalid start_idx {start_idx}")

        face3d_clip = face3d_exp[clip_start_idx : clip_start_idx + clip_num_frames]
        pad_mask = torch.tensor([False] * style_max_len)
    else:
        clip_start_idx = None
        padding = torch.zeros(style_max_len - length, face3d_exp.shape[1])
        face3d_clip = torch.cat((face3d_exp, padding), dim=0)
        pad_mask = torch.tensor([False] * length + [True] * (style_max_len - length))

    if return_start_idx:
        return face3d_clip, pad_mask, clip_start_idx
    else:
        return face3d_clip, pad_mask


def get_wav2vec_audio_window(audio_feat, start_idx, num_frames, win_size):
    """

    Args:
        audio_feat (np.ndarray): (N, 1024)
        start_idx (_type_): _description_
        num_frames (_type_): _description_
    """
    center_idx_list = [2 * idx for idx in range(start_idx, start_idx + num_frames)]
    audio_window_list = []
    padding = np.zeros(audio_feat.shape[1], dtype=np.float32)
    for center_idx in center_idx_list:
        cur_audio_window = []
        for i in range(center_idx - win_size, center_idx + win_size + 1):
            if i < 0:
                cur_audio_window.append(padding)
            elif i >= len(audio_feat):
                cur_audio_window.append(padding)
            else:
                cur_audio_window.append(audio_feat[i])
        cur_audio_win_array = np.stack(cur_audio_window, axis=0)
        audio_window_list.append(cur_audio_win_array)

    audio_window_array = np.stack(audio_window_list, axis=0)
    return audio_window_array


def setup_config():
    parser = argparse.ArgumentParser(description="voice2pose main program")
    parser.add_argument(
        "--config_file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--resume_from", type=str, default=None, help="the checkpoint to resume from"
    )
    parser.add_argument(
        "--test_only", action="store_true", help="perform testing and evaluation only"
    )
    parser.add_argument(
        "--demo_input", type=str, default=None, help="path to input for demo"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="the checkpoint to test with"
    )
    parser.add_argument("--tag", type=str, default="", help="tag for the experiment")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        help="local rank for DistributedDataParallel",
    )
    parser.add_argument(
        "--master_port",
        type=str,
        default="12345",
    )
    parser.add_argument(
        "--max_audio_len",
        type=int,
        default=450,
        help="max_audio_len for inference",
    )
    parser.add_argument(
        "--ddim_num_step",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--inference_seed",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--inference_sample_method",
        type=str,
        default="ddim",
    )
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return args, cfg


def setup_logger(base_path, exp_name):
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)

    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-0.5s] %(message)s")

    log_path = "{0}/{1}.log".format(base_path, exp_name)
    fileHandler = logging.FileHandler(log_path)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    rootLogger.handlers[0].setLevel(logging.INFO)

    logging.info("log path: %s" % log_path)


def cosine_loss(a, v, y, logloss=nn.BCELoss()):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)
    return loss


def get_pose_params(mat_path):
    """Get pose parameters from mat file

    Args:
        mat_path (str): path of mat file

    Returns:
        pose_params (numpy.ndarray): shape (L_video, 9), angle, translation, crop paramters
    """
    mat_dict = loadmat(mat_path)

    np_3dmm = mat_dict["coeff"]
    angles = np_3dmm[:, 224:227]
    translations = np_3dmm[:, 254:257]

    np_trans_params = mat_dict["transform_params"]
    crop = np_trans_params[:, -3:]

    pose_params = np.concatenate((angles, translations, crop), axis=1)

    return pose_params


def sinusoidal_embedding(timesteps, dim):
    """

    Args:
        timesteps (_type_): (B,)
        dim (_type_): (C_embed)

    Returns:
        _type_: (B, C_embed)
    """
    # check input
    half = dim // 2
    timesteps = timesteps.float()

    # compute sinusoidal embedding
    sinusoid = torch.outer(
        timesteps, torch.pow(10000, -torch.arange(half).to(timesteps).div(half))
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    if dim % 2 != 0:
        x = torch.cat([x, torch.zeros_like(x[:, :1])], dim=1)
    return x


def get_wav2vec_audio_window(audio_feat, start_idx, num_frames, win_size):
    """

    Args:
        audio_feat (np.ndarray): (250, 1024)
        start_idx (_type_): _description_
        num_frames (_type_): _description_
    """
    center_idx_list = [2 * idx for idx in range(start_idx, start_idx + num_frames)]
    audio_window_list = []
    padding = np.zeros(audio_feat.shape[1], dtype=np.float32)
    for center_idx in center_idx_list:
        cur_audio_window = []
        for i in range(center_idx - win_size, center_idx + win_size + 1):
            if i < 0:
                cur_audio_window.append(padding)
            elif i >= len(audio_feat):
                cur_audio_window.append(padding)
            else:
                cur_audio_window.append(audio_feat[i])
        cur_audio_win_array = np.stack(cur_audio_window, axis=0)
        audio_window_list.append(cur_audio_win_array)

    audio_window_array = np.stack(audio_window_list, axis=0)
    return audio_window_array


def reshape_audio_feat(style_audio_all_raw, stride):
    """_summary_

    Args:
        style_audio_all_raw (_type_): (stride * L, C)
        stride (_type_): int

    Returns:
        _type_: (L, C * stride)
    """
    style_audio_all_raw = style_audio_all_raw[
        : style_audio_all_raw.shape[0] // stride * stride
    ]
    style_audio_all_raw = style_audio_all_raw.reshape(
        style_audio_all_raw.shape[0] // stride, stride, style_audio_all_raw.shape[1]
    )
    style_audio_all = style_audio_all_raw.reshape(style_audio_all_raw.shape[0], -1)
    return style_audio_all


import random


def get_derangement_tuple(n):
    while True:
        v = [i for i in range(n)]
        for j in range(n - 1, -1, -1):
            p = random.randint(0, j)
            if v[p] == j:
                break
            else:
                v[j], v[p] = v[p], v[j]
        else:
            if v[0] != 0:
                return tuple(v)


def compute_aspect_preserved_bbox(bbox, increase_area, h, w):
    left, top, right, bot = bbox
    width = right - left
    height = bot - top

    width_increase = max(
        increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width)
    )
    height_increase = max(
        increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height)
    )

    left_t = int(left - width_increase * width)
    top_t = int(top - height_increase * height)
    right_t = int(right + width_increase * width)
    bot_t = int(bot + height_increase * height)

    left_oob = -min(0, left_t)
    right_oob = right - min(right_t, w)
    top_oob = -min(0, top_t)
    bot_oob = bot - min(bot_t, h)

    if max(left_oob, right_oob, top_oob, bot_oob) > 0:
        max_w = max(left_oob, right_oob)
        max_h = max(top_oob, bot_oob)
        if max_w > max_h:
            return left_t + max_w, top_t + max_w, right_t - max_w, bot_t - max_w
        else:
            return left_t + max_h, top_t + max_h, right_t - max_h, bot_t - max_h

    else:
        return (left_t, top_t, right_t, bot_t)


def crop_src_image(src_img, save_img, increase_ratio, detector=None):
    if detector is None:
        detector = dlib.get_frontal_face_detector()

    img = cv2.imread(src_img)
    faces = detector(img, 0)
    h, width, _ = img.shape
    if len(faces) > 0:
        bbox = [faces[0].left(), faces[0].top(), faces[0].right(), faces[0].bottom()]
        l = bbox[3] - bbox[1]
        bbox[1] = bbox[1] - l * 0.1
        bbox[3] = bbox[3] - l * 0.1
        bbox[1] = max(0, bbox[1])
        bbox[3] = min(h, bbox[3])
        bbox = compute_aspect_preserved_bbox(
            tuple(bbox), increase_ratio, img.shape[0], img.shape[1]
        )
        img = img[bbox[1] : bbox[3], bbox[0] : bbox[2]]
        img = cv2.resize(img, (256, 256))
        cv2.imwrite(save_img, img)
    else:
        raise ValueError("No face detected in the input image")
        # img = cv2.resize(img, (256, 256))
        # cv2.imwrite(save_img, img)
