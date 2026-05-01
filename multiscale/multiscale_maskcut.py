#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import sys
import argparse
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from tqdm import tqdm
import datetime
import PIL
import PIL.Image as Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from pycocotools import mask
import pycocotools.mask as mask_util
from scipy import ndimage
from scipy.linalg import eigh
import json

REPO_ROOT = Path(__file__).resolve().parents[1]
CUTLER_ROOT = REPO_ROOT / "CutLER"
CUTLER_MASKCUT_DIR = CUTLER_ROOT / "maskcut"
CUTLER_THIRD_PARTY_DIR = CUTLER_ROOT / "third_party"

# Reuse upstream CutLER/TokenCut modules while keeping this entrypoint in the parent repo.
sys.path.insert(0, str(CUTLER_MASKCUT_DIR))
sys.path.insert(0, str(CUTLER_ROOT))
sys.path.insert(0, str(CUTLER_THIRD_PARTY_DIR))

import dino
# modfied by Xudong Wang based on third_party/TokenCut
from TokenCut.unsupervised_saliency_detection import utils, metric
from TokenCut.unsupervised_saliency_detection.object_discovery import detect_box
# bilateral_solver codes are modfied based on https://github.com/poolio/bilateral_solver/blob/master/notebooks/bilateral_solver.ipynb
# from TokenCut.unsupervised_saliency_detection.bilateral_solver import BilateralSolver, BilateralGrid
# crf codes are are modfied based on https://github.com/lucasb-eyer/pydensecrf/blob/master/pydensecrf/tests/test_dcrf.py
from crf import densecrf

# Image transformation applied to all images
ToTensor = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(
                                (0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225)),])


@dataclass
class MaskCandidate:
    mask: np.ndarray
    source: str
    crop_box: tuple = None
    crop_score: float = 0.0
    mask_score: float = 0.0
    meta: dict = field(default_factory=dict)


def parse_float_list(text):
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if len(values) == 0:
        values = [1.0]
    return values


def parse_size_list(text):
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if len(values) == 0:
        values = [0.35, 0.5, 0.75]
    return values


MULTISCALE_PRESETS = {
    "small": {
        "crop_mode": "heatmap",
        "heatmap_crop_sizes": "0.25,0.35,0.5",
        "heatmap_top_k": 12,
        "keep_topk": 12,
        "min_mask_area_ratio": 0.0001,
        "max_mask_area_ratio": 0.02,
        "containment_thresh": 0.85,
        "box_expand_ratio": 0.05,
        "merge_max_aspect_ratio": 3.0,
        "two_stage_crop": True,
        "primary_output": "multiscale",
    },
    "balanced": {
        "crop_mode": "heatmap",
        "heatmap_crop_sizes": "0.35,0.5,0.75",
        "heatmap_top_k": 16,
        "keep_topk": 20,
        "min_mask_area_ratio": 0.0001,
        "max_mask_area_ratio": 0.05,
        "containment_thresh": 0.8,
        "box_expand_ratio": 0.1,
        "merge_max_aspect_ratio": 4.0,
        "two_stage_crop": True,
        "primary_output": "multiscale",
    },
    "legacy": {
        "crop_mode": "grid",
        "heatmap_crop_sizes": "0.35,0.5,0.75",
        "heatmap_top_k": 12,
        "keep_topk": 20,
        "min_mask_area_ratio": 0.0001,
        "max_mask_area_ratio": 0.25,
        "containment_thresh": 0.7,
        "box_expand_ratio": 0.15,
        "merge_max_aspect_ratio": 5.0,
        "two_stage_crop": True,
        "primary_output": "combined",
    },
}


def cli_flag_was_set(flag):
    return any(arg == flag or arg.startswith(flag + "=") for arg in sys.argv[1:])


def apply_multiscale_preset(args):
    if not args.multi_crop:
        return
    preset = MULTISCALE_PRESETS.get(args.ms_preset, {})
    flag_names = {
        "crop_mode": "--crop-mode",
        "heatmap_crop_sizes": "--heatmap-crop-sizes",
        "heatmap_top_k": "--heatmap-top-k",
        "keep_topk": "--keep-topk",
        "min_mask_area_ratio": "--min-mask-area-ratio",
        "max_mask_area_ratio": "--max-mask-area-ratio",
        "containment_thresh": "--containment-thresh",
        "box_expand_ratio": "--box-expand-ratio",
        "merge_max_aspect_ratio": "--merge-max-aspect-ratio",
        "two_stage_crop": "--two-stage-crop",
        "primary_output": "--primary-output",
    }
    for attr, value in preset.items():
        flag = flag_names[attr]
        if not cli_flag_was_set(flag):
            setattr(args, attr, value)

def get_affinity_matrix(feats, tau, eps=1e-5):
    # get affinity matrix via measuring patch-wise cosine similarity
    feats = F.normalize(feats, p=2, dim=0) # normalizing the columns
    A = (feats.transpose(0,1) @ feats).cpu().numpy()
    # convert the affinity matrix to a binary one.
    A = A > tau
    A = np.where(A.astype(float) == 0, eps, A) # susbstituting 0 for eps
    d_i = np.sum(A, axis=1) # summing rows
    D = np.diag(d_i) # diagonal of that
    return A, D

def second_smallest_eigenvector(A, D):
    # get the second smallest eigenvector from affinity matrix
    _, eigenvectors = eigh(D-A, D, subset_by_index=[1,2])
    eigenvec = np.copy(eigenvectors[:, 0])
    second_smallest_vec = eigenvectors[:, 0]
    return eigenvec, second_smallest_vec

def get_salient_areas(second_smallest_vec):
    # get the area corresponding to salient objects.
    avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
    bipartition = second_smallest_vec > avg
    return bipartition

def check_num_fg_corners(bipartition, dims):
    # check number of corners belonging to the foreground
    bipartition_ = bipartition.reshape(dims)
    top_l, top_r, bottom_l, bottom_r = bipartition_[0][0], bipartition_[0][-1], bipartition_[-1][0], bipartition_[-1][-1]
    nc = int(top_l) + int(top_r) + int(bottom_l) + int(bottom_r)
    return nc

def get_masked_affinity_matrix(painting, feats, mask, ps):
    # mask out affinity matrix based on the painting matrix 
    dim, num_patch = feats.size()[0], feats.size()[1]
    painting = painting + mask.unsqueeze(0)
    painting[painting > 0] = 1
    painting[painting <= 0] = 0
    feats = feats.clone().view(dim, ps, ps)
    feats = ((1 - painting) * feats).view(dim, num_patch)
    return feats, painting

def maskcut_forward(feats, dims, scales, init_image_size, tau=0, N=3, cpu=False):
    """
    Implementation of MaskCut.
    Inputs
      feats: the pixel/patche features of an image
      dims: dimension of the map from which the features are used
      scales: from image to map scale
      init_image_size: size of the image
      tau: thresold for graph construction
      N: number of pseudo-masks per image.
    """
    bipartitions = []
    eigvecs = []

    for i in range(N):
        if i == 0:
            painting = torch.from_numpy(np.zeros(dims))
            if not cpu: painting = painting.cuda()
        else:
            feats, painting = get_masked_affinity_matrix(painting, feats, current_mask, ps)

        # construct the affinity matrix
        A, D = get_affinity_matrix(feats, tau)
        # get the second smallest eigenvector
        eigenvec, second_smallest_vec = second_smallest_eigenvector(A, D)
        # get salient area
        bipartition = get_salient_areas(second_smallest_vec)

        # check if we should reverse the partition based on:
        # 1) peak of the 2nd smallest eigvec 2) object centric bias
        seed = np.argmax(np.abs(second_smallest_vec))
        nc = check_num_fg_corners(bipartition, dims)
        if nc >= 3:
            reverse = True
        else:
            reverse = bipartition[seed] != 1

        if reverse:
            # reverse bipartition, eigenvector and get new seed
            eigenvec = eigenvec * -1
            bipartition = np.logical_not(bipartition)
            seed = np.argmax(eigenvec)
        else:
            seed = np.argmax(second_smallest_vec)

        # get pxiels corresponding to the seed
        bipartition = bipartition.reshape(dims).astype(float)
        _, _, _, cc = detect_box(bipartition, seed, dims, scales=scales, initial_im_size=init_image_size)
        pseudo_mask = np.zeros(dims)
        pseudo_mask[cc[0],cc[1]] = 1
        pseudo_mask = torch.from_numpy(pseudo_mask)
        if not cpu: pseudo_mask = pseudo_mask.to('cuda')
        ps = pseudo_mask.shape[0]

        # check if the extra mask is heavily overlapped with the previous one or is too small.
        if i >= 1:
            ratio = torch.sum(pseudo_mask) / pseudo_mask.size()[0] / pseudo_mask.size()[1]
            if metric.IoU(current_mask, pseudo_mask) > 0.5 or ratio <= 0.01:
                pseudo_mask = np.zeros(dims)
                pseudo_mask = torch.from_numpy(pseudo_mask)
                if not cpu: pseudo_mask = pseudo_mask.to('cuda')
        current_mask = pseudo_mask

        # mask out foreground areas in previous stages
        masked_out = 0 if len(bipartitions) == 0 else np.sum(bipartitions, axis=0)
        bipartition = F.interpolate(pseudo_mask.unsqueeze(0).unsqueeze(0), size=init_image_size, mode='nearest').squeeze()
        bipartition_masked = bipartition.cpu().numpy() - masked_out
        bipartition_masked[bipartition_masked <= 0] = 0
        bipartitions.append(bipartition_masked)

        # unsample the eigenvec
        eigvec = second_smallest_vec.reshape(dims)
        eigvec = torch.from_numpy(eigvec)
        if not cpu: eigvec = eigvec.to('cuda')
        eigvec = F.interpolate(eigvec.unsqueeze(0).unsqueeze(0), size=init_image_size, mode='nearest').squeeze()
        eigvecs.append(eigvec.cpu().numpy())

    return seed, bipartitions, eigvecs

def maskcut(img_path, backbone,patch_size, tau, N=1, fixed_size=480, cpu=False) :
    I = Image.open(img_path).convert('RGB')
    return maskcut_from_pil(I, backbone, patch_size, tau, N=N, fixed_size=fixed_size, cpu=cpu)


def maskcut_from_pil(I, backbone, patch_size, tau, N=1, fixed_size=480, cpu=False):
    return maskcut_from_pil_batch(
        [I],
        backbone,
        patch_size,
        tau,
        N=N,
        fixed_size=fixed_size,
        cpu=cpu,
        batch_size=1,
    )[0]


def prepare_maskcut_input(I, patch_size, fixed_size):
    I_new = I.resize((int(fixed_size), int(fixed_size)), PIL.Image.LANCZOS)
    I_resize, w, h, feat_w, feat_h = utils.resize_pil(I_new, patch_size)
    tensor = ToTensor(I_resize)
    return I_new, tensor, w, h, feat_w, feat_h


def maskcut_from_pil_batch(
    images,
    backbone,
    patch_size,
    tau,
    N=1,
    fixed_size=480,
    cpu=False,
    batch_size=8,
):
    if len(images) == 0:
        return []

    prepared = [prepare_maskcut_input(I, patch_size, fixed_size) for I in images]
    results = []
    batch_size = max(1, int(batch_size))

    for start in range(0, len(prepared), batch_size):
        chunk = prepared[start:start + batch_size]
        tensor = torch.stack([item[1] for item in chunk], dim=0)
        if not cpu:
            tensor = tensor.cuda()

        feats = backbone(tensor)
        for idx, (I_new, _, w, h, feat_w, feat_h) in enumerate(chunk):
            feat = feats[idx]
            _, bipartition, eigvec = maskcut_forward(
                feat,
                [feat_h, feat_w],
                [patch_size, patch_size],
                [h, w],
                tau,
                N=N,
                cpu=cpu,
            )
            results.append((bipartition, eigvec, I_new))

    return results


def generate_windows(image_size, crop_scales, crop_overlap=0.3, max_windows_per_scale=0):
    windows = []
    seen = set()

    for scale in crop_scales:
        crop_size = max(16, int(round(image_size * scale)))
        crop_size = min(crop_size, image_size)

        if crop_size == image_size:
            key = (0, 0, image_size, image_size)
            if key not in seen:
                windows.append(key)
                seen.add(key)
            continue

        stride = max(1, int(round(crop_size * (1.0 - crop_overlap))))

        xs = list(range(0, max(1, image_size - crop_size + 1), stride))
        ys = list(range(0, max(1, image_size - crop_size + 1), stride))

        if xs[-1] != image_size - crop_size:
            xs.append(image_size - crop_size)
        if ys[-1] != image_size - crop_size:
            ys.append(image_size - crop_size)

        scale_windows = []
        for y in ys:
            for x in xs:
                scale_windows.append((x, y, crop_size, crop_size))

        if max_windows_per_scale > 0 and len(scale_windows) > max_windows_per_scale:
            if max_windows_per_scale >= 5:
                center = (image_size - crop_size) // 2
                selected = [
                    (0, 0, crop_size, crop_size),
                    (image_size - crop_size, 0, crop_size, crop_size),
                    (0, image_size - crop_size, crop_size, crop_size),
                    (image_size - crop_size, image_size - crop_size, crop_size, crop_size),
                    (center, center, crop_size, crop_size),
                ]
                scale_windows = selected[:max_windows_per_scale]
            else:
                scale_windows = scale_windows[:max_windows_per_scale]

        for win in scale_windows:
            if win not in seen:
                windows.append(win)
                seen.add(win)

    return windows


def project_window_to_original(window, fixed_size, orig_w, orig_h):
    x, y, w, h = window
    left = int(round((x / fixed_size) * orig_w))
    top = int(round((y / fixed_size) * orig_h))
    right = int(round(((x + w) / fixed_size) * orig_w))
    bottom = int(round(((y + h) / fixed_size) * orig_h))
    left = max(0, min(left, orig_w - 1))
    top = max(0, min(top, orig_h - 1))
    right = max(left + 1, min(right, orig_w))
    bottom = max(top + 1, min(bottom, orig_h))
    return left, top, right, bottom


def add_refined_masks_to_candidates(
    candidates,
    bipartitions,
    crop_resized,
    target_box,
    output_shape,
    crf_iou_thresh=0.3,
    source="crop",
    crop_score=0.0,
    protected_masks=None,
):
    left, top, right, bottom = target_box
    orig_h, orig_w = output_shape
    added = 0

    for bipartition in bipartitions:
        refined, crf_iou = postprocess_crop_mask(crop_resized, bipartition, crf_iou_thresh)
        if refined is None:
            continue
        refined = resize_binary_mask(refined, (right - left, bottom - top))
        full_mask = np.zeros((orig_h, orig_w), dtype=np.bool_)
        full_mask[top:bottom, left:right] = np.logical_or(
            full_mask[top:bottom, left:right], refined
        )
        candidates.append(make_mask_candidate(
            full_mask,
            source=source,
            crop_box=target_box,
            crop_score=crop_score,
            protected_masks=protected_masks,
            crf_iou=crf_iou,
        ))
        added += 1
    return added


def binary_iou(mask_a, mask_b):
    a = mask_a.astype(np.bool_)
    b = mask_b.astype(np.bool_)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def intersection_over_smaller(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    smaller = min(mask_a.sum(), mask_b.sum())
    if smaller == 0:
        return 0.0
    return float(inter) / float(smaller)


def mask_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return int(rmin), int(cmin), int(rmax), int(cmax)


def unwrap_mask(item):
    return item.mask if isinstance(item, MaskCandidate) else item


def unwrap_masks(items):
    return [unwrap_mask(item) for item in (items or [])]


def mask_bbox_xywh(mask):
    bbox = mask_bbox(mask)
    if bbox is None:
        return None
    rmin, cmin, rmax, cmax = bbox
    return [int(cmin), int(rmin), int(cmax - cmin + 1), int(rmax - rmin + 1)]


def mask_area_ratio(mask):
    return float(mask.sum()) / float(mask.shape[0] * mask.shape[1])


def mask_compactness(mask):
    bbox = mask_bbox(mask)
    if bbox is None:
        return 0.0
    rmin, cmin, rmax, cmax = bbox
    bbox_area = max(1, (rmax - rmin + 1) * (cmax - cmin + 1))
    return float(mask.sum()) / float(bbox_area)


def mask_aspect_ratio(mask):
    bbox = mask_bbox(mask)
    if bbox is None:
        return 1.0
    rmin, cmin, rmax, cmax = bbox
    h = rmax - rmin + 1
    w = cmax - cmin + 1
    return float(max(h, w)) / float(max(1, min(h, w)))


def crop_border_metrics(mask, crop_box):
    if crop_box is None:
        return 0.0, 0
    left, top, right, bottom = crop_box
    top = max(0, min(top, mask.shape[0] - 1))
    bottom = max(top + 1, min(bottom, mask.shape[0]))
    left = max(0, min(left, mask.shape[1] - 1))
    right = max(left + 1, min(right, mask.shape[1]))

    crop_mask = mask[top:bottom, left:right]
    area = int(crop_mask.sum())
    if area == 0:
        return 0.0, 0
    sides = [
        bool(crop_mask[0, :].any()),
        bool(crop_mask[-1, :].any()),
        bool(crop_mask[:, 0].any()),
        bool(crop_mask[:, -1].any()),
    ]
    border_pixels = (
        int(crop_mask[0, :].sum()) +
        int(crop_mask[-1, :].sum()) +
        int(crop_mask[:, 0].sum()) +
        int(crop_mask[:, -1].sum())
    )
    # A one-pixel border is small compared with area, so scale by sqrt(area).
    border_touch = min(1.0, float(border_pixels) / float(max(1.0, np.sqrt(area))))
    return border_touch, int(sum(sides))


def max_binary_iou(mask, others):
    best = 0.0
    for other in unwrap_masks(others):
        best = max(best, binary_iou(mask, other))
    return best


def small_object_area_prior(area_ratio):
    if area_ratio <= 0:
        return 0.0
    target = 0.004
    sigma = 0.65
    distance = (np.log10(area_ratio) - np.log10(target)) / sigma
    return float(np.exp(-0.5 * distance * distance))


def make_mask_candidate(
    mask,
    source,
    crop_box=None,
    crop_score=0.0,
    protected_masks=None,
    crf_iou=1.0,
):
    bm = mask.astype(np.bool_)
    area = int(bm.sum())
    area_ratio = mask_area_ratio(bm)
    compactness = mask_compactness(bm)
    aspect_ratio = mask_aspect_ratio(bm)
    border_touch, border_sides = crop_border_metrics(bm, crop_box)
    normal_iou = max_binary_iou(bm, protected_masks) if protected_masks else 0.0
    crop_prior = float(np.clip(crop_score / 4.0, 0.0, 1.0))
    area_prior = small_object_area_prior(area_ratio)
    aspect_penalty = float(np.clip((aspect_ratio - 3.0) / 4.0, 0.0, 1.0))
    border_penalty = max(border_touch, border_sides / 4.0)

    score = (
        1.4 * area_prior +
        1.1 * compactness +
        0.7 * crop_prior +
        0.5 * float(np.clip(crf_iou, 0.0, 1.0)) -
        1.0 * aspect_penalty -
        0.9 * border_penalty -
        0.6 * normal_iou
    )

    meta = {
        "source": source,
        "crop_box": list(crop_box) if crop_box is not None else None,
        "crop_score": float(crop_score),
        "mask_score": float(score),
        "area": area,
        "area_ratio": float(area_ratio),
        "bbox": mask_bbox_xywh(bm),
        "compactness": float(compactness),
        "aspect_ratio": float(aspect_ratio),
        "area_prior": float(area_prior),
        "border_touch": float(border_touch),
        "border_sides": int(border_sides),
        "normal_iou": float(normal_iou),
        "crf_iou": float(crf_iou),
    }
    return MaskCandidate(
        mask=bm,
        source=source,
        crop_box=crop_box,
        crop_score=float(crop_score),
        mask_score=float(score),
        meta=meta,
    )


def candidate_to_record(candidate, rank=None):
    record = dict(candidate.meta)
    if rank is not None:
        record["rank"] = int(rank)
    return record


def boxes_overlap_expanded(box_a, box_b, expand_ratio=0.15):
    """True if boxes overlap after each is expanded by expand_ratio of its own size."""
    def expand(b, r):
        h = b[2] - b[0]; w = b[3] - b[1]
        return (b[0] - h * r, b[1] - w * r, b[2] + h * r, b[3] + w * r)
    ea = expand(box_a, expand_ratio)
    eb = expand(box_b, expand_ratio)
    return ea[0] < eb[2] and eb[0] < ea[2] and ea[1] < eb[3] and eb[1] < ea[3]


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def clip_square_box(cx, cy, size, orig_w, orig_h):
    size = max(2, min(int(round(size)), max(orig_w, orig_h)))
    left = int(round(cx - size / 2.0))
    top = int(round(cy - size / 2.0))
    right = left + size
    bottom = top + size

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > orig_w:
        left -= right - orig_w
        right = orig_w
    if bottom > orig_h:
        top -= bottom - orig_h
        bottom = orig_h

    left = max(0, left)
    top = max(0, top)
    right = max(left + 1, min(right, orig_w))
    bottom = max(top + 1, min(bottom, orig_h))
    return left, top, right, bottom


def crop_sizes_to_pixels(crop_sizes, orig_w, orig_h):
    base = min(orig_w, orig_h)
    sizes = []
    for value in crop_sizes:
        if value <= 1.0:
            size = value * base
        else:
            size = value
        sizes.append(max(2, min(int(round(size)), max(orig_w, orig_h))))
    return sorted(set(sizes))


def compute_edge_density(image_array, top, left, bottom, right):
    crop = image_array[top:bottom, left:right]
    if crop.size == 0:
        return 0.0
    gray = crop.mean(axis=2).astype(np.float32) if crop.ndim == 3 else crop.astype(np.float32)
    if gray.shape[0] < 2 or gray.shape[1] < 2:
        return 0.0
    gy = np.gradient(gray, axis=0)
    gx = np.gradient(gray, axis=1)
    return float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))


def compute_feature_contrast_heatmap(I, backbone, patch_size, fixed_size, cpu=False):
    _, tensor, _, _, feat_w, feat_h = prepare_maskcut_input(I, patch_size, fixed_size)
    tensor = tensor.unsqueeze(0)
    if not cpu:
        tensor = tensor.cuda()
    with torch.no_grad():
        feat = backbone(tensor)[0]
    feat = F.normalize(feat, p=2, dim=0).detach().cpu().numpy()
    feat = feat.reshape(feat.shape[0], feat_h, feat_w)

    heatmap = np.zeros((feat_h, feat_w), dtype=np.float32)
    counts = np.zeros((feat_h, feat_w), dtype=np.float32)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        y_src_start = max(0, -dy)
        y_src_end = min(feat_h, feat_h - dy)
        x_src_start = max(0, -dx)
        x_src_end = min(feat_w, feat_w - dx)
        y_dst_start = y_src_start + dy
        y_dst_end = y_src_end + dy
        x_dst_start = x_src_start + dx
        x_dst_end = x_src_end + dx

        src = feat[:, y_src_start:y_src_end, x_src_start:x_src_end]
        dst = feat[:, y_dst_start:y_dst_end, x_dst_start:x_dst_end]
        cosine = np.sum(src * dst, axis=0)
        distance = 1.0 - cosine
        heatmap[y_src_start:y_src_end, x_src_start:x_src_end] += distance
        counts[y_src_start:y_src_end, x_src_start:x_src_end] += 1.0

    heatmap = heatmap / np.maximum(counts, 1.0)
    heatmap -= heatmap.min()
    max_value = heatmap.max()
    if max_value > 0:
        heatmap /= max_value
    return heatmap


def score_heatmap_box(heatmap, box, orig_w, orig_h, covered_mask, image_array):
    left, top, right, bottom = box
    h, w = heatmap.shape
    x1 = max(0, min(w - 1, int(np.floor(left / orig_w * w))))
    x2 = max(x1 + 1, min(w, int(np.ceil(right / orig_w * w))))
    y1 = max(0, min(h - 1, int(np.floor(top / orig_h * h))))
    y2 = max(y1 + 1, min(h, int(np.ceil(bottom / orig_h * h))))
    patch = heatmap[y1:y2, x1:x2]
    object_mean = float(patch.mean()) if patch.size else 0.0
    object_max = float(patch.max()) if patch.size else 0.0

    crop_area = max(1, (right - left) * (bottom - top))
    coverage = 0.0
    if covered_mask is not None and covered_mask.any():
        coverage = float(covered_mask[top:bottom, left:right].sum()) / crop_area
    edge = compute_edge_density(image_array, top, left, bottom, right)

    touches = int(left == 0) + int(top == 0) + int(right == orig_w) + int(bottom == orig_h)
    border_penalty = 0.1 * touches
    return (1.5 * object_mean) + object_max + (2.0 * object_mean * (1.0 - coverage)) + (0.5 * edge / 128.0) - border_penalty


def generate_heatmap_windows(
    I,
    backbone,
    patch_size,
    fixed_size,
    crop_sizes,
    covered_mask,
    top_k,
    nms_iou,
    percentile,
    cpu=False,
):
    orig_w, orig_h = I.size
    heatmap = compute_feature_contrast_heatmap(I, backbone, patch_size, fixed_size, cpu=cpu)
    threshold = np.percentile(heatmap, percentile)
    image_array = np.array(I)
    sizes = crop_sizes_to_pixels(crop_sizes, orig_w, orig_h)

    peak_indices = np.argsort(heatmap.ravel())[::-1]
    raw_boxes = []
    h, w = heatmap.shape
    max_peaks = max(top_k * 10, 50) if top_k > 0 else 200
    for flat_idx in peak_indices:
        score = heatmap.ravel()[flat_idx]
        if score < threshold and len(raw_boxes) >= max_peaks:
            break
        y, x = np.unravel_index(flat_idx, heatmap.shape)
        cx = (x + 0.5) / w * orig_w
        cy = (y + 0.5) / h * orig_h
        for size in sizes:
            box = clip_square_box(cx, cy, size, orig_w, orig_h)
            raw_boxes.append((score_heatmap_box(heatmap, box, orig_w, orig_h, covered_mask, image_array), box))
        if len(raw_boxes) >= max_peaks * max(1, len(sizes)):
            break

    raw_boxes.sort(key=lambda item: item[0], reverse=True)
    selected = []
    selected_boxes = []
    for score, box in raw_boxes:
        if any(box_iou(box, kept) > nms_iou for kept in selected_boxes):
            continue
        selected.append({"box": box, "score": float(score)})
        selected_boxes.append(box)
        if top_k > 0 and len(selected) >= top_k:
            break
    return selected


def merge_mask_candidates(
    candidates,
    merge_iou_thresh,
    keep_topk,
    min_area_ratio,
    max_area_ratio,
    small_first=True,
    containment_thresh=0.7,
    box_expand_ratio=0.15,
    max_aspect_ratio=5.0,
    protected_candidates=None,
):
    """Merge candidates while ranking by mask quality score.

    Area filters define the intent of the crop branch; score ranking decides
    which surviving masks are most object-like.
    """
    def _filter_candidates(items, upper_area_ratio):
        filtered_items = []
        for item in items or []:
            candidate = item if isinstance(item, MaskCandidate) else make_mask_candidate(item, "legacy")
            area_ratio = mask_area_ratio(candidate.mask)
            if min_area_ratio <= area_ratio <= upper_area_ratio:
                filtered_items.append(candidate)
        return filtered_items

    protected = _filter_candidates(protected_candidates, 1.0)
    filtered = _filter_candidates(candidates, max_area_ratio)

    if not filtered:
        return protected

    n = len(filtered)
    boxes = [mask_bbox(c.mask) for c in filtered]
    total_px = filtered[0].mask.shape[0] * filtered[0].mask.shape[1]

    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            connected = binary_iou(filtered[i].mask, filtered[j].mask) > merge_iou_thresh
            if not connected:
                connected = intersection_over_smaller(filtered[i].mask, filtered[j].mask) > containment_thresh
            if not connected and boxes[i] is not None and boxes[j] is not None:
                connected = boxes_overlap_expanded(boxes[i], boxes[j], box_expand_ratio)
            if connected:
                adj[i].add(j)
                adj[j].add(i)

    visited = [False] * n
    components = []
    for start in range(n):
        if visited[start]:
            continue
        comp = []
        queue = [start]
        visited[start] = True
        while queue:
            node = queue.pop()
            comp.append(node)
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        components.append(comp)

    merged_list = []
    for comp in components:
        if len(comp) == 1:
            merged_list.append(filtered[comp[0]])
            continue

        members = [filtered[i] for i in comp]
        union = np.logical_or.reduce([c.mask for c in members])
        area_ratio = float(union.sum()) / total_px
        valid = area_ratio <= max_area_ratio
        if valid:
            bbox = mask_bbox(union)
            if bbox is not None:
                bh = bbox[2] - bbox[0] + 1
                bw = bbox[3] - bbox[1] + 1
                if max(bh, bw) / max(1, min(bh, bw)) > max_aspect_ratio:
                    valid = False

        if valid:
            best_member = max(members, key=lambda c: c.mask_score)
            merged = make_mask_candidate(
                union,
                source="merged_multiscale",
                crop_box=best_member.crop_box,
                crop_score=best_member.crop_score,
                protected_masks=protected,
                crf_iou=best_member.meta.get("crf_iou", 1.0),
            )
            merged.mask_score = max(c.mask_score for c in members) - 0.03 * (len(members) - 1)
            merged.meta["mask_score"] = float(merged.mask_score)
            merged.meta["component_size"] = int(len(members))
            merged.meta["component_sources"] = [c.source for c in members]
            merged_list.append(merged)
        else:
            merged_list.extend(members)

    if small_first:
        merged_list.sort(key=lambda c: (-c.mask_score, mask_area_ratio(c.mask)))
    else:
        merged_list.sort(key=lambda c: (-c.mask_score, -mask_area_ratio(c.mask)))

    kept = list(protected)
    protected_count = len(protected)
    crop_kept = 0
    for candidate in merged_list:
        keep = True
        for idx, kept_candidate in enumerate(kept):
            duplicate = binary_iou(candidate.mask, kept_candidate.mask) > merge_iou_thresh
            if idx >= protected_count:
                duplicate = duplicate or intersection_over_smaller(
                    candidate.mask, kept_candidate.mask
                ) > containment_thresh
            if duplicate:
                keep = False
                break
        if keep:
            kept.append(candidate)
            crop_kept += 1
        if keep_topk > 0 and crop_kept >= keep_topk:
            break

    return kept


def merge_masks(
    candidates,
    merge_iou_thresh,
    keep_topk,
    min_area_ratio,
    max_area_ratio,
    small_first=True,
    containment_thresh=0.7,
    box_expand_ratio=0.15,
    max_aspect_ratio=5.0,
    protected_masks=None,
):
    """Graph-based mask merging.

    Protected masks, usually full-image MaskCut proposals, are kept before crop
    proposals. Crop proposals are merged among themselves, then appended only if
    they are not near-duplicates of protected masks.

    Builds a graph where masks are nodes; edges connect masks that are
    near-duplicate (high IoU), partial overlaps (high IoS containment), or
    adjacent fragments (expanded bounding boxes overlap).  Connected components
    are merged by union, subject to a validity check (area and aspect ratio).
    A final dedup pass suppresses any residual near-duplicates across components.
    """
    def _filter_masks(masks, upper_area_ratio):
        filtered_masks = []
        for m in masks:
            bm = m.astype(np.bool_)
            area_ratio = float(bm.sum()) / float(bm.shape[0] * bm.shape[1])
            if min_area_ratio <= area_ratio <= upper_area_ratio:
                filtered_masks.append(bm)
        return filtered_masks

    protected = _filter_masks(protected_masks or [], 1.0)
    filtered = _filter_masks(candidates, max_area_ratio)

    if not filtered:
        return protected

    n = len(filtered)
    boxes = [mask_bbox(m) for m in filtered]
    total_px = filtered[0].shape[0] * filtered[0].shape[1]

    # Build adjacency graph for crop proposals only. Full-image proposals stay
    # protected and are used later as duplicate filters.
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            connected = binary_iou(filtered[i], filtered[j]) > merge_iou_thresh
            if not connected:
                connected = intersection_over_smaller(filtered[i], filtered[j]) > containment_thresh
            if not connected and boxes[i] is not None and boxes[j] is not None:
                connected = boxes_overlap_expanded(boxes[i], boxes[j], box_expand_ratio)
            if connected:
                adj[i].add(j)
                adj[j].add(i)

    # BFS to collect connected components.
    visited = [False] * n
    components = []
    for start in range(n):
        if visited[start]:
            continue
        comp = []
        queue = [start]
        visited[start] = True
        while queue:
            node = queue.pop()
            comp.append(node)
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        components.append(comp)

    # Merge each component; fall back to individual masks if union is invalid.
    merged_list = []
    for comp in components:
        if len(comp) == 1:
            m = filtered[comp[0]]
            merged_list.append((m, float(m.sum()) / total_px))
            continue
        union = np.logical_or.reduce([filtered[i] for i in comp])
        area_ratio = float(union.sum()) / total_px
        valid = area_ratio <= max_area_ratio
        if valid:
            bbox = mask_bbox(union)
            if bbox is not None:
                bh = bbox[2] - bbox[0] + 1
                bw = bbox[3] - bbox[1] + 1
                if max(bh, bw) / max(1, min(bh, bw)) > max_aspect_ratio:
                    valid = False
        if valid:
            merged_list.append((union, area_ratio))
        else:
            for i in comp:
                m = filtered[i]
                merged_list.append((m, float(m.sum()) / total_px))

    if small_first:
        merged_list.sort(key=lambda x: x[1])
    else:
        merged_list.sort(key=lambda x: x[1], reverse=True)

    # Final dedup. Protected full-image masks are never dropped by crop masks.
    kept = list(protected)
    protected_count = len(protected)
    crop_kept = 0
    for m, _ in merged_list:
        keep = True
        for idx, km in enumerate(kept):
            duplicate = binary_iou(m, km) > merge_iou_thresh
            if idx >= protected_count:
                duplicate = duplicate or intersection_over_smaller(m, km) > containment_thresh
            if duplicate:
                keep = False
                break
        if keep:
            kept.append(m)
            crop_kept += 1
        if keep_topk > 0 and crop_kept >= keep_topk:
            break

    return kept


def postprocess_crop_mask(crop_rgb, bipartition, crf_iou_thresh=0.3):
    pseudo_mask = densecrf(np.array(crop_rgb), bipartition)
    pseudo_mask = ndimage.binary_fill_holes(pseudo_mask >= 0.5)
    crf_iou = binary_iou(bipartition > 0, pseudo_mask)
    if crf_iou < crf_iou_thresh:
        return None, crf_iou
    return pseudo_mask.astype(np.bool_), crf_iou


def maskcut_multicrop(
    img_path,
    backbone,
    patch_size,
    tau,
    N=3,
    fixed_size=480,
    cpu=False,
    crop_scales=None,
    crop_overlap=0.3,
    max_windows_per_scale=0,
    merge_iou_thresh=0.5,
    keep_topk=12,
    min_area_ratio=0.0001,
    max_area_ratio=0.02,
    small_first=True,
    two_stage_crop=False,
    two_stage_max_covered_ratio=0.9,
    crop_batch_size=8,
    containment_thresh=0.85,
    box_expand_ratio=0.05,
    merge_max_aspect_ratio=3.0,
    crop_top_k=0,
    crop_mode="heatmap",
    heatmap_crop_sizes=None,
    heatmap_top_k=12,
    heatmap_nms_iou=0.4,
    heatmap_percentile=85.0,
    crf_iou_thresh=0.3,
    return_stats=False,
    return_splits=False,
    return_debug=False,
):
    if crop_scales is None:
        crop_scales = [1.0, 0.75, 0.5]
    if heatmap_crop_sizes is None:
        heatmap_crop_sizes = [0.25, 0.35, 0.5]

    I = Image.open(img_path).convert("RGB")
    orig_w, orig_h = I.size
    output_shape = (orig_h, orig_w)

    protected_masks = []
    crop_candidates = []
    covered_mask = np.zeros(output_shape, dtype=np.bool_)
    crop_items = []
    stats = {
        "full_masks": 0,
        "total_windows": 0,
        "skipped_covered": 0,
        "eligible_windows": 0,
        "ranked_windows": 0,
        "crop_windows": 0,
        "crop_candidates": 0,
        "crop_merged_masks": 0,
        "merged_masks": 0,
        "scored_candidates": 0,
    }

    if two_stage_crop:
        full_bipartitions, _, full_resized = maskcut_from_pil(
            I,
            backbone,
            patch_size,
            tau,
            N=N,
            fixed_size=fixed_size,
            cpu=cpu,
        )
        stats["full_masks"] = add_refined_masks_to_candidates(
            protected_masks,
            full_bipartitions,
            full_resized,
            (0, 0, orig_w, orig_h),
            output_shape,
            crf_iou_thresh=crf_iou_thresh,
            source="normal",
            crop_score=0.0,
            protected_masks=None,
        )
        if protected_masks:
            covered_mask = np.logical_or.reduce(unwrap_masks(protected_masks))

    eligible = []
    if crop_mode == "heatmap":
        top_k = crop_top_k if crop_top_k > 0 else heatmap_top_k
        eligible = generate_heatmap_windows(
            I,
            backbone,
            patch_size,
            fixed_size,
            heatmap_crop_sizes,
            covered_mask,
            top_k,
            heatmap_nms_iou,
            heatmap_percentile,
            cpu=cpu,
        )
        stats["total_windows"] = len(eligible)
        stats["eligible_windows"] = len(eligible)
        stats["ranked_windows"] = len(eligible)
    else:
        windows = generate_windows(
            image_size=fixed_size,
            crop_scales=crop_scales,
            crop_overlap=crop_overlap,
            max_windows_per_scale=max_windows_per_scale,
        )
        stats["total_windows"] = len(windows)
        # Collect eligible windows (apply two-stage coverage filter inline so we
        # have projected boxes available for ranking without a second projection pass).
        for window in windows:
            # In two-stage mode the full image has already been processed once.
            if two_stage_crop and window == (0, 0, fixed_size, fixed_size):
                continue

            # Windows are generated on a normalized fixed-size square canvas. Map
            # them back to the original image, crop there, then resize the crop for
            # inference so smaller objects gain effective resolution.
            left, top, right, bottom = project_window_to_original(
                window, fixed_size, orig_w, orig_h
            )

            if two_stage_crop and covered_mask.any():
                crop_area = float((right - left) * (bottom - top))
                covered_ratio = covered_mask[top:bottom, left:right].sum() / crop_area
                if covered_ratio >= two_stage_max_covered_ratio:
                    stats["skipped_covered"] += 1
                    continue

            eligible.append({"box": (left, top, right, bottom), "score": 0.0})
        stats["eligible_windows"] = len(eligible)

        # Rank remaining windows by unexplained coverage times edge detail and keep top-k.
        # Unexplained coverage: fraction of the window NOT yet covered by full-image masks.
        # Edge detail: mean gradient magnitude, high in textured/object-rich regions.
        img_array = np.array(I)
        cov_mask = covered_mask if two_stage_crop else None

        def _crop_score(box):
            l, t, r, b = box
            crop_area = max(1, (r - l) * (b - t))
            coverage = float(cov_mask[t:b, l:r].sum()) / crop_area if cov_mask is not None else 0.0
            edge = compute_edge_density(img_array, t, l, b, r)
            return (1.0 - coverage) * (1.0 + edge / 128.0)

        for item in eligible:
            item["score"] = _crop_score(item["box"])
        if crop_top_k > 0 and len(eligible) > crop_top_k:
            eligible.sort(key=lambda item: item["score"], reverse=True)
            eligible = eligible[:crop_top_k]
        stats["ranked_windows"] = len(eligible)

    for item in eligible:
        left, top, right, bottom = item["box"]
        crop_items.append({
            "crop": I.crop((left, top, right, bottom)),
            "box": (left, top, right, bottom),
            "crop_score": item.get("score", 0.0),
        })
    stats["crop_windows"] = len(crop_items)

    crop_results = maskcut_from_pil_batch(
        [item["crop"] for item in crop_items],
        backbone,
        patch_size,
        tau,
        N=N,
        fixed_size=fixed_size,
        cpu=cpu,
        batch_size=crop_batch_size,
    )
    for item, (bipartitions, _, crop_resized) in zip(crop_items, crop_results):
        stats["crop_candidates"] += add_refined_masks_to_candidates(
            crop_candidates,
            bipartitions,
            crop_resized,
            item["box"],
            output_shape,
            crf_iou_thresh=crf_iou_thresh,
            source="raw_multiscale",
            crop_score=item.get("crop_score", 0.0),
            protected_masks=protected_masks,
        )

    stats["scored_candidates"] = len(crop_candidates)
    crop_merged_candidates = merge_mask_candidates(
        candidates=crop_candidates,
        merge_iou_thresh=merge_iou_thresh,
        keep_topk=keep_topk,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        small_first=small_first,
        containment_thresh=containment_thresh,
        box_expand_ratio=box_expand_ratio,
        max_aspect_ratio=merge_max_aspect_ratio,
        protected_candidates=None,
    )
    merged_candidates = merge_mask_candidates(
        candidates=crop_candidates,
        merge_iou_thresh=merge_iou_thresh,
        keep_topk=keep_topk,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        small_first=small_first,
        containment_thresh=containment_thresh,
        box_expand_ratio=box_expand_ratio,
        max_aspect_ratio=merge_max_aspect_ratio,
        protected_candidates=protected_masks,
    )
    stats["merged_masks"] = len(merged_candidates)
    stats["crop_merged_masks"] = len(crop_merged_candidates)
    split_candidates = {
        "normal": protected_masks,
        "raw_multiscale": crop_candidates,
        "multiscale": crop_merged_candidates,
        "combined": merged_candidates,
    }
    splits = {
        name: [candidate.mask for candidate in candidates]
        for name, candidates in split_candidates.items()
    }
    debug_splits = {
        name: [candidate_to_record(candidate, rank=i + 1) for i, candidate in enumerate(candidates)]
        for name, candidates in split_candidates.items()
    }
    if return_splits:
        if return_debug:
            if return_stats:
                return splits["combined"], I, stats, splits, debug_splits
            return splits["combined"], I, splits, debug_splits
        if return_stats:
            return splits["combined"], I, stats, splits
        return splits["combined"], I, splits
    if return_stats:
        return splits["combined"], I, stats
    return splits["combined"], I

def resize_binary_mask(array, new_size):
    image = Image.fromarray(array.astype(np.uint8)*255)
    image = image.resize(new_size, PIL.Image.NEAREST)
    return np.asarray(image).astype(np.bool_)

def create_image_info(image_id, file_name, image_size, 
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):
    """Return image_info in COCO style
    Args:
        image_id: the image ID
        file_name: the file name of each image
        image_size: image size in the format of (width, height)
        date_captured: the date this image info is created
        license: license of this image
        coco_url: url to COCO images if there is any
        flickr_url: url to flickr if there is any
    """
    image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[0],
            "height": image_size[1],
            "date_captured": date_captured,
            "license": license_id,
            "coco_url": coco_url,
            "flickr_url": flickr_url
    }
    return image_info


def create_annotation_info(annotation_id, image_id, category_info, binary_mask, 
                           image_size=None, bounding_box=None):
    """Return annotation info in COCO style
    Args:
        annotation_id: the annotation ID
        image_id: the image ID
        category_info: the information on categories
        binary_mask: a 2D binary numpy array where '1's represent the object
        file_name: the file name of each image
        image_size: image size in the format of (width, height)
        bounding_box: the bounding box for detection task. If bounding_box is not provided, 
        we will generate one according to the binary mask.
    """
    upper = np.max(binary_mask)
    lower = np.min(binary_mask)
    thresh = upper / 2.0
    binary_mask[binary_mask > thresh] = upper
    binary_mask[binary_mask <= thresh] = lower
    if image_size is not None:
        binary_mask = resize_binary_mask(binary_mask.astype(np.uint8), image_size)

    binary_mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))

    area = mask.area(binary_mask_encoded)
    if area < 1:
        return None

    if bounding_box is None:
        bounding_box = mask.toBbox(binary_mask_encoded)

    rle = mask_util.encode(np.array(binary_mask[...,None], order="F", dtype="uint8"))[0]
    rle['counts'] = rle['counts'].decode('ascii')
    segmentation = rle

    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_info["id"],
        "iscrowd": 0,
        "area": area.tolist(),
        "bbox": bounding_box.tolist(),
        "segmentation": segmentation,
        "width": binary_mask.shape[1],
        "height": binary_mask.shape[0],
    } 

    return annotation_info

# necessay info used for coco style annotations
INFO = {
    "description": "ImageNet-1K: pseudo-masks with MaskCut",
    "url": "https://github.com/facebookresearch/CutLER",
    "version": "1.0",
    "year": 2023,
    "contributor": "Xudong Wang",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Apache License",
        "url": "https://github.com/facebookresearch/CutLER/blob/main/LICENSE"
    }
]

# only one class, i.e. foreground
CATEGORIES = [
    {
        'id': 1,
        'name': 'fg',
        'supercategory': 'fg',
    },
]

output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []}

category_info = {
    "is_crowd": 0,
    "id": 1
}


def new_coco_output():
    return {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }


def append_masks_to_output(
    coco_output,
    masks,
    image_info,
    image_name,
    seen_image_names,
    segmentation_id,
):
    if image_name not in seen_image_names:
        coco_output["images"].append(image_info)
        seen_image_names.add(image_name)

    for binary_mask in masks:
        annotation_info = create_annotation_info(
            segmentation_id,
            image_info["id"],
            category_info,
            binary_mask.astype(np.uint8),
            None,
        )
        if annotation_info is not None:
            coco_output["annotations"].append(annotation_info)
            segmentation_id += 1
    return segmentation_id

if __name__ == "__main__":

    parser = argparse.ArgumentParser('MaskCut script')
    # default arguments
    parser.add_argument('--out-dir', type=str, help='output directory')
    parser.add_argument('--vit-arch', type=str, default='small', choices=['base', 'small'], help='which architecture')
    parser.add_argument('--vit-feat', type=str, default='k', choices=['k', 'q', 'v', 'kqv'], help='which features')
    parser.add_argument('--patch-size', type=int, default=16, choices=[16, 8], help='patch size')
    parser.add_argument('--nb-vis', type=int, default=20, choices=[1, 200], help='nb of visualization')
    parser.add_argument('--img-path', type=str, default=None, help='single image visualization')

    # additional arguments
    parser.add_argument('--dataset-path', type=str, default="imagenet/train/", help='path to the dataset')
    parser.add_argument('--tau', type=float, default=0.2, help='threshold used for producing binary graph')
    parser.add_argument('--num-folder-per-job', type=int, default=1, help='the number of folders each job processes')
    parser.add_argument('--job-index', type=int, default=0, help='the index of the job (for imagenet: in the range of 0 to 1000/args.num_folder_per_job-1)')
    parser.add_argument('--fixed_size', type=int, default=480, help='rescale the input images to a fixed size')
    parser.add_argument('--pretrain_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--N', type=int, default=3, help='the maximum number of pseudo-masks per image')
    parser.add_argument('--cpu', action='store_true', help='use cpu')
    parser.add_argument('--multi-crop', action='store_true', help='run MaskCut on multiple crop scales and merge masks')
    parser.add_argument('--ms-preset', type=str, default='small', choices=['small', 'balanced', 'legacy'], help='bundle of multi-crop defaults; individual flags still override the preset')
    parser.add_argument('--crop-mode', type=str, default='heatmap', choices=['grid', 'heatmap'], help='crop proposal mode: dense grid or DINO feature-contrast heatmap')
    parser.add_argument('--crop-scales', type=str, default='1.0,0.75,0.5', help='comma separated crop scales for multi-crop mode')
    parser.add_argument('--crop-overlap', type=float, default=0.3, help='overlap ratio between adjacent windows in multi-crop mode')
    parser.add_argument('--crop-max-per-scale', type=int, default=0, help='limit number of windows per scale (0 keeps all)')
    parser.add_argument('--merge-iou-thresh', type=float, default=0.5, help='IoU threshold for mask merging in multi-crop mode')
    parser.add_argument('--keep-topk', type=int, default=12, help='max scored crop masks kept after merge; protected full-image masks are always kept (0 keeps all crop masks)')
    parser.add_argument('--min-mask-area-ratio', type=float, default=0.0001, help='min crop-mask area ratio in the original image canvas')
    parser.add_argument('--max-mask-area-ratio', type=float, default=0.02, help='max crop-mask area ratio in the original image canvas; full-image split masks are not capped by this')
    parser.add_argument('--small-first', action='store_true', help='prefer smaller masks first when merging (helps APs)')
    parser.add_argument('--two-stage-crop', action='store_true', help='run full-image MaskCut first and skip crop windows already covered by that foreground')
    parser.add_argument('--two-stage-max-covered-ratio', type=float, default=0.9, help='skip crop windows whose area is covered by full-stage masks at or above this ratio')
    parser.add_argument('--crop-batch-size', type=int, default=8, help='number of crop images per DINO forward pass in multi-crop mode')
    parser.add_argument('--containment-thresh', type=float, default=0.85, help='intersection-over-smaller threshold: connect masks where one covers at least this fraction of the other')
    parser.add_argument('--box-expand-ratio', type=float, default=0.05, help='expand bounding boxes by this fraction when testing adjacency between mask fragments')
    parser.add_argument('--merge-max-aspect-ratio', type=float, default=3.0, help='reject a merged mask if its bounding box aspect ratio exceeds this (catches bad cross-object unions)')
    parser.add_argument('--crop-top-k', type=int, default=0, help='after two-stage coverage filtering, keep only the top-k crop windows ranked by unexplained detail (0 = keep all)')
    parser.add_argument('--heatmap-crop-sizes', type=str, default='0.25,0.35,0.5', help='comma separated heatmap crop sizes; values <=1 are fractions of the shorter image side, values >1 are pixels')
    parser.add_argument('--heatmap-top-k', type=int, default=12, help='number of DINO heatmap crop proposals when crop-top-k is 0')
    parser.add_argument('--heatmap-nms-iou', type=float, default=0.4, help='crop-box NMS IoU for heatmap crop proposals')
    parser.add_argument('--heatmap-percentile', type=float, default=85.0, help='minimum feature-contrast percentile considered for heatmap crop peaks')
    parser.add_argument('--crf-iou-thresh', type=float, default=0.3, help='minimum IoU between raw MaskCut mask and CRF-refined mask for accepting crop proposals')
    parser.add_argument('--primary-output', type=str, default='multiscale', choices=['normal', 'raw_multiscale', 'multiscale', 'combined'], help='which split is written to the unsuffixed JSON/checkpoint in multi-crop mode')
    parser.add_argument('--write-split-outputs', action='store_true', help='legacy flag; split outputs are always written in multi-crop mode')
    parser.add_argument('--log-every', type=int, default=50, help='print aggregate multi-crop stats every this many processed images (0 disables)')

    args = parser.parse_args()
    apply_multiscale_preset(args)
    crop_scales = parse_float_list(args.crop_scales)
    heatmap_crop_sizes = parse_size_list(args.heatmap_crop_sizes)

    if args.pretrain_path is not None:
        url = args.pretrain_path
    if args.vit_arch == 'base' and args.patch_size == 8:
        if args.pretrain_path is None:
            url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        feat_dim = 768
    elif args.vit_arch == 'small' and args.patch_size == 8:
        if args.pretrain_path is None:
            url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        feat_dim = 384

    backbone = dino.ViTFeat(url, feat_dim, args.vit_arch, args.vit_feat, args.patch_size)

    msg = 'Load {} pre-trained feature...'.format(args.vit_arch)
    print (msg)
    backbone.eval()
    if not args.cpu:
        backbone.cuda()

    img_folders = os.listdir(args.dataset_path)

    if args.out_dir is not None and not os.path.exists(args.out_dir) :
        os.mkdir(args.out_dir)

    start_idx = max(args.job_index*args.num_folder_per_job, 0)
    end_idx = min((args.job_index+1)*args.num_folder_per_job, len(img_folders))

    image_id, segmentation_id = 1, 1
    image_names = []
    processed_images = 0
    multicrop_totals = {
        "full_masks": 0,
        "total_windows": 0,
        "skipped_covered": 0,
        "eligible_windows": 0,
        "ranked_windows": 0,
        "crop_windows": 0,
        "crop_candidates": 0,
        "crop_merged_masks": 0,
        "merged_masks": 0,
        "scored_candidates": 0,
    }
    split_outputs = {
        "normal": new_coco_output(),
        "raw_multiscale": new_coco_output(),
        "multiscale": new_coco_output(),
        "combined": new_coco_output(),
    }
    split_image_names = {name: set() for name in split_outputs}
    split_segmentation_ids = {name: 1 for name in split_outputs}
    candidate_debug_records = []
    for img_folder in img_folders[start_idx:end_idx]:
        args.img_dir = os.path.join(args.dataset_path, img_folder)
        if os.path.isdir(os.path.join(args.img_dir, "images")):
            args.img_dir = os.path.join(args.img_dir, "images")
        img_list = sorted(os.listdir(args.img_dir))

        for img_name in tqdm(img_list) :
            # get image path
            img_path = os.path.join(args.img_dir, img_name)
            # get pseudo-masks for each image using MaskCut
            try:
                if args.multi_crop:
                    bipartitions, I_new, multicrop_stats, split_masks, split_debug = maskcut_multicrop(
                        img_path,
                        backbone,
                        args.patch_size,
                        args.tau,
                        N=args.N,
                        fixed_size=args.fixed_size,
                        cpu=args.cpu,
                        crop_scales=crop_scales,
                        crop_overlap=args.crop_overlap,
                        max_windows_per_scale=args.crop_max_per_scale,
                        merge_iou_thresh=args.merge_iou_thresh,
                        keep_topk=args.keep_topk,
                        min_area_ratio=args.min_mask_area_ratio,
                        max_area_ratio=args.max_mask_area_ratio,
                        small_first=args.small_first,
                        two_stage_crop=args.two_stage_crop,
                        two_stage_max_covered_ratio=args.two_stage_max_covered_ratio,
                        crop_batch_size=args.crop_batch_size,
                        containment_thresh=args.containment_thresh,
                        box_expand_ratio=args.box_expand_ratio,
                        merge_max_aspect_ratio=args.merge_max_aspect_ratio,
                        crop_top_k=args.crop_top_k,
                        crop_mode=args.crop_mode,
                        heatmap_crop_sizes=heatmap_crop_sizes,
                        heatmap_top_k=args.heatmap_top_k,
                        heatmap_nms_iou=args.heatmap_nms_iou,
                        heatmap_percentile=args.heatmap_percentile,
                        crf_iou_thresh=args.crf_iou_thresh,
                        return_stats=True,
                        return_splits=True,
                        return_debug=True,
                    )
                else:
                    bipartitions, _, I_new = maskcut(img_path, backbone, args.patch_size, \
                        args.tau, N=args.N, fixed_size=args.fixed_size, cpu=args.cpu)
            except Exception as exc:
                print(f'Skipping {img_name}: {exc}')
                continue
            if args.multi_crop:
                for key, value in multicrop_stats.items():
                    multicrop_totals[key] += value
                processed_images += 1
                bipartitions = split_masks[args.primary_output]
                if args.log_every > 0 and processed_images % args.log_every == 0:
                    print(
                        "Multi-crop stats after {} images: full_masks={}, "
                        "windows={} skipped={} ranked={} crop_candidates={} scored={} crop_merged={} merged={}".format(
                            processed_images,
                            multicrop_totals["full_masks"],
                            multicrop_totals["total_windows"],
                            multicrop_totals["skipped_covered"],
                            multicrop_totals["ranked_windows"],
                            multicrop_totals["crop_candidates"],
                            multicrop_totals["scored_candidates"],
                            multicrop_totals["crop_merged_masks"],
                            multicrop_totals["merged_masks"],
                        )
                    )

            I = Image.open(img_path).convert('RGB')
            width, height = I.size
            image_info = create_image_info(
                image_id, "{}/{}".format(img_folder, img_name), (height, width, 3))
            image_key = image_info["file_name"]
            if args.multi_crop:
                candidate_debug_records.append({
                    "image_id": image_id,
                    "file_name": image_key,
                    "stats": multicrop_stats,
                    "splits": split_debug,
                })
                for split_name, masks in split_masks.items():
                    split_segmentation_ids[split_name] = append_masks_to_output(
                        split_outputs[split_name],
                        masks,
                        image_info,
                        image_key,
                        split_image_names[split_name],
                        split_segmentation_ids[split_name],
                    )
            for idx, bipartition in enumerate(bipartitions):
                if args.multi_crop:
                    pseudo_mask = bipartition.astype(np.bool_)
                else:
                    # post-process pesudo-masks with CRF
                    pseudo_mask = densecrf(np.array(I_new), bipartition)
                    pseudo_mask = ndimage.binary_fill_holes(pseudo_mask>=0.5)

                    # filter out the mask that have a very different pseudo-mask after the CRF
                    mask1 = torch.from_numpy(bipartition)
                    mask2 = torch.from_numpy(pseudo_mask)
                    if not args.cpu: 
                        mask1 = mask1.cuda()
                        mask2 = mask2.cuda()
                    if metric.IoU(mask1, mask2) < 0.5:
                        pseudo_mask = pseudo_mask * -1

                # construct binary pseudo-masks
                pseudo_mask[pseudo_mask < 0] = 0
                pseudo_mask = Image.fromarray(np.uint8(pseudo_mask*255))
                pseudo_mask = np.asarray(pseudo_mask.resize((width, height)))

                # create coco-style image info
                if image_key not in image_names:
                    output["images"].append(image_info)
                    image_names.append(image_key)

                # create coco-style annotation info
                annotation_info = create_annotation_info(
                    segmentation_id, image_id, category_info, pseudo_mask.astype(np.uint8), None)
                if annotation_info is not None:
                    output["annotations"].append(annotation_info)
                    segmentation_id += 1
            image_id += 1
        # incremental checkpoint after each folder
        ckpt_path = os.path.join(args.out_dir, "checkpoint.json")
        with open(ckpt_path, "w") as ckpt_f:
            json.dump(output, ckpt_f)

    # save annotations
    crop_tag = ''
    if args.multi_crop:
        if args.crop_mode == 'heatmap':
            crop_tag = '_heatmap_hs{}_hp{}_hk{}_miou{}'.format(
                args.heatmap_crop_sizes.replace(',', '-'),
                args.heatmap_percentile,
                args.heatmap_top_k,
                args.merge_iou_thresh,
            )
        else:
            crop_tag = '_grid_mc{}_ov{}_miou{}'.format(
                args.crop_scales.replace(',', '-'),
                args.crop_overlap,
                args.merge_iou_thresh,
            )
        crop_tag += '_preset{}'.format(args.ms_preset)
        crop_tag += '_ts{}'.format(args.two_stage_max_covered_ratio)
    if len(img_folders) == args.num_folder_per_job and args.job_index == 0:
        json_name = '{}/imagenet_train_fixsize{}_tau{}_N{}{}.json'.format(args.out_dir, args.fixed_size, args.tau, args.N, crop_tag)
    else:
        json_name = '{}/imagenet_train_fixsize{}_tau{}_N{}{}_{}_{}.json'.format(args.out_dir, args.fixed_size, args.tau, args.N, crop_tag, start_idx, end_idx)
    with open(json_name, 'w') as output_json_file:
        json.dump(output, output_json_file)
    if args.multi_crop:
        base = json_name[:-5] if json_name.endswith(".json") else json_name
        for split_name, split_output in split_outputs.items():
            split_json_name = "{}_{}.json".format(base, split_name)
            with open(split_json_name, "w") as split_json_file:
                json.dump(split_output, split_json_file)
            print(
                "dumping {} ({} images; {} anns.)".format(
                    split_json_name,
                    len(split_output["images"]),
                    len(split_output["annotations"]),
                )
            )
        debug_json_name = "{}_candidate_debug.json".format(base)
        with open(debug_json_name, "w") as debug_json_file:
            json.dump({
                "info": INFO,
                "primary_output": args.primary_output,
                "records": candidate_debug_records,
            }, debug_json_file, indent=2)
        print("dumping {} ({} images)".format(debug_json_name, len(candidate_debug_records)))
    print(f'dumping {json_name}')
    if args.multi_crop and processed_images > 0:
        print(
            "Final multi-crop stats: images={} full_masks={} windows={} "
            "skipped={} ranked={} crop_candidates={} scored={} crop_merged={} merged={}".format(
                processed_images,
                multicrop_totals["full_masks"],
                multicrop_totals["total_windows"],
                multicrop_totals["skipped_covered"],
                multicrop_totals["ranked_windows"],
                multicrop_totals["crop_candidates"],
                multicrop_totals["scored_candidates"],
                multicrop_totals["crop_merged_masks"],
                multicrop_totals["merged_masks"],
            )
        )
    print("Done: {} images; {} anns.".format(len(output['images']), len(output['annotations'])))
