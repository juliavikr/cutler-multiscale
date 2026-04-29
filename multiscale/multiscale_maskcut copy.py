#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import sys
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import re
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
):
    left, top, right, bottom = target_box
    orig_h, orig_w = output_shape

    for bipartition in bipartitions:
        refined = postprocess_crop_mask(crop_resized, bipartition)
        if refined is None:
            continue
        refined = resize_binary_mask(refined, (right - left, bottom - top))
        full_mask = np.zeros((orig_h, orig_w), dtype=np.bool_)
        full_mask[top:bottom, left:right] = np.logical_or(
            full_mask[top:bottom, left:right], refined
        )
        candidates.append(full_mask)


def binary_iou(mask_a, mask_b):
    a = mask_a.astype(np.bool_)
    b = mask_b.astype(np.bool_)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def merge_masks(candidates, merge_iou_thresh, keep_topk, min_area_ratio, max_area_ratio, small_first=True):
    filtered = []
    for m in candidates:
        bm = m.astype(np.bool_)
        area_ratio = float(bm.sum()) / float(bm.shape[0] * bm.shape[1])
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue
        filtered.append((bm, area_ratio))

    if small_first:
        filtered.sort(key=lambda x: x[1])
    else:
        filtered.sort(key=lambda x: x[1], reverse=True)

    kept = []
    for bm, _ in filtered:
        keep = True
        for km in kept:
            if binary_iou(bm, km) > merge_iou_thresh:
                keep = False
                break
        if keep:
            kept.append(bm)
        if keep_topk > 0 and len(kept) >= keep_topk:
            break

    return kept


def postprocess_crop_mask(crop_rgb, bipartition):
    pseudo_mask = densecrf(np.array(crop_rgb), bipartition)
    pseudo_mask = ndimage.binary_fill_holes(pseudo_mask >= 0.5)
    if binary_iou(bipartition > 0, pseudo_mask) < 0.5:
        return None
    return pseudo_mask.astype(np.bool_)


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
    keep_topk=0,
    min_area_ratio=0.0005,
    max_area_ratio=1.0,
    small_first=True,
    two_stage_crop=False,
    two_stage_max_covered_ratio=0.9,
    crop_batch_size=8,
):
    if crop_scales is None:
        crop_scales = [1.0, 0.75, 0.5]

    I = Image.open(img_path).convert("RGB")
    orig_w, orig_h = I.size
    output_shape = (orig_h, orig_w)
    windows = generate_windows(
        image_size=fixed_size,
        crop_scales=crop_scales,
        crop_overlap=crop_overlap,
        max_windows_per_scale=max_windows_per_scale,
    )

    candidates = []
    covered_mask = np.zeros(output_shape, dtype=np.bool_)
    crop_items = []

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
        before_full = len(candidates)
        add_refined_masks_to_candidates(
            candidates,
            full_bipartitions,
            full_resized,
            (0, 0, orig_w, orig_h),
            output_shape,
        )
        if len(candidates) > before_full:
            covered_mask = np.logical_or.reduce(candidates[before_full:])

    for window in windows:
        # In two-stage mode the full image has already been processed once.
        if two_stage_crop and window == (0, 0, fixed_size, fixed_size):
            continue

        # Windows are generated on a normalized fixed-size square canvas. Map
        # them back to the original image, crop there, then resize the crop for
        # inference so smaller objects actually gain effective resolution.
        left, top, right, bottom = project_window_to_original(
            window, fixed_size, orig_w, orig_h
        )

        if two_stage_crop and covered_mask.any():
            crop_area = float((right - left) * (bottom - top))
            covered_ratio = covered_mask[top:bottom, left:right].sum() / crop_area
            if covered_ratio >= two_stage_max_covered_ratio:
                continue

        crop_items.append({
            "crop": I.crop((left, top, right, bottom)),
            "box": (left, top, right, bottom),
        })

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
        add_refined_masks_to_candidates(
            candidates,
            bipartitions,
            crop_resized,
            item["box"],
            output_shape,
        )

    merged = merge_masks(
        candidates=candidates,
        merge_iou_thresh=merge_iou_thresh,
        keep_topk=keep_topk,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        small_first=small_first,
    )
    return merged, I

def resize_binary_mask(array, new_size):
    image = Image.fromarray(array.astype(np.uint8)*255)
    image = image.resize(new_size, PIL.Image.NEAREST)
    return np.asarray(image).astype(np.bool_)

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


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

convert = lambda text: int(text) if text.isdigit() else text.lower()
natrual_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]

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
    parser.add_argument('--crop-scales', type=str, default='1.0,0.75,0.5', help='comma separated crop scales for multi-crop mode')
    parser.add_argument('--crop-overlap', type=float, default=0.3, help='overlap ratio between adjacent windows in multi-crop mode')
    parser.add_argument('--crop-max-per-scale', type=int, default=0, help='limit number of windows per scale (0 keeps all)')
    parser.add_argument('--merge-iou-thresh', type=float, default=0.5, help='IoU threshold for mask merging in multi-crop mode')
    parser.add_argument('--keep-topk', type=int, default=0, help='max masks kept per image after merge (0 keeps all)')
    parser.add_argument('--min-mask-area-ratio', type=float, default=0.0005, help='min mask area ratio in fixed-size canvas')
    parser.add_argument('--max-mask-area-ratio', type=float, default=1.0, help='max mask area ratio in fixed-size canvas')
    parser.add_argument('--small-first', action='store_true', help='prefer smaller masks first when merging (helps APs)')
    parser.add_argument('--two-stage-crop', action='store_true', help='run full-image MaskCut first and skip crop windows already covered by that foreground')
    parser.add_argument('--two-stage-max-covered-ratio', type=float, default=0.9, help='skip crop windows whose area is covered by full-stage masks at or above this ratio')
    parser.add_argument('--crop-batch-size', type=int, default=8, help='number of crop images per DINO forward pass in multi-crop mode')

    args = parser.parse_args()
    crop_scales = parse_float_list(args.crop_scales)

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
    for img_folder in img_folders[start_idx:end_idx]:
        args.img_dir = os.path.join(args.dataset_path, img_folder)
        img_list = sorted(os.listdir(args.img_dir))

        for img_name in tqdm(img_list) :
            # get image path
            img_path = os.path.join(args.img_dir, img_name)
            # get pseudo-masks for each image using MaskCut
            try:
                if args.multi_crop:
                    bipartitions, I_new = maskcut_multicrop(
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
                    )
                else:
                    bipartitions, _, I_new = maskcut(img_path, backbone, args.patch_size, \
                        args.tau, N=args.N, fixed_size=args.fixed_size, cpu=args.cpu)
            except:
                print(f'Skipping {img_name}')
                continue

            I = Image.open(img_path).convert('RGB')
            width, height = I.size
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
                if img_name not in image_names:
                    image_info = create_image_info(
                        image_id, "{}/{}".format(img_folder, img_name), (height, width, 3))
                    output["images"].append(image_info)
                    image_names.append(img_name)           

                # create coco-style annotation info
                annotation_info = create_annotation_info(
                    segmentation_id, image_id, category_info, pseudo_mask.astype(np.uint8), None)
                if annotation_info is not None:
                    output["annotations"].append(annotation_info)
                    segmentation_id += 1
            image_id += 1

    # save annotations
    crop_tag = ''
    if args.multi_crop:
        crop_tag = '_mc{}_ov{}_miou{}'.format(
            args.crop_scales.replace(',', '-'),
            args.crop_overlap,
            args.merge_iou_thresh,
        )
        if args.two_stage_crop:
            crop_tag += '_ts{}'.format(args.two_stage_max_covered_ratio)
    if len(img_folders) == args.num_folder_per_job and args.job_index == 0:
        json_name = '{}/imagenet_train_fixsize{}_tau{}_N{}{}.json'.format(args.out_dir, args.fixed_size, args.tau, args.N, crop_tag)
    else:
        json_name = '{}/imagenet_train_fixsize{}_tau{}_N{}{}_{}_{}.json'.format(args.out_dir, args.fixed_size, args.tau, args.N, crop_tag, start_idx, end_idx)
    with open(json_name, 'w') as output_json_file:
        json.dump(output, output_json_file)
    print(f'dumping {json_name}')
    print("Done: {} images; {} anns.".format(len(output['images']), len(output['annotations'])))
