from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import math
import parse

import pickle
import numpy as np
import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from   matplotlib.ticker import NullLocator

#############################################################################################
## Utility Functions                                                                       ##
#############################################################################################

# Convert a YOLO detection to KITTI pixel coordinates
def cvt_domain(
    det,
    kitti_img_size=416,
    img_h=375,
    img_w=1242,
):
    wl, hu, wr, hd = det[0], det[1], det[2], det[3]
    pad_w = max(img_h - img_w, 0) * (kitti_img_size / max(img_h, img_w))
    pad_h = max(img_w - img_h, 0) * (kitti_img_size / max(img_h, img_w))
    unpad_h = kitti_img_size - pad_h
    unpad_w = kitti_img_size - pad_w
    box_h = int(((hd - hu) / unpad_h) * img_h)
    box_w = int(((wr - wl) / unpad_w) * img_w)
    hu = int(((hu - pad_h // 2) / unpad_h) * img_h)
    wl = int(((wl - pad_w // 2) / unpad_w) * img_w)

    return wl, hu, box_w, box_h
# Returns wl, hu, box_w, box_h

# Compute IoU of two bounding boxes
def iou(
    det1,                   # Detection structure
    det2,                   # Detection structure
    disp1=None,             # Disparity of left detection; optional, used for disparity correction
):
    # Get KITTI domain coordinates
    # wl1, hu1, box_w1, box_h1 = cvt_domain(det1)
    # wr1, hd1 = wl1 + box_w1, hu1 + box_h1
    # wl2, hu2, box_w2, box_h2 = cvt_domain(det2)
    # wr2, hd2 = wl2 + box_w2, hu2 + box_h2
    wl1, hu1, wr1, hd1 = det1[0:4]
    wl2, hu2, wr2, hd2 = det2[0:4]

    # If detection 1 disparity is given, correct for it
    if disp1 is not None:
        wl1, wr1 = max(0, int(wl1-disp1)), max(0, int(wr1-disp1))

    # Get the coordinates of the intersection rectangle
    inter_wl = max([wl1, wl2])
    inter_hu = max([hu1, hu2])
    inter_wr = min([wr1, wr2])
    inter_hd = min([hd1, hd2])

    # Intersection area
    inter_area = max([inter_wr - inter_wl + 1, 0]) * max([inter_hd - inter_hu + 1, 0])

    # Union area
    det1_area = (wr1 - wl1 + 1) * (hd1 - hu1 + 1)
    det2_area = (wr2 - wl2 + 1) * (hd2 - hu2 + 1)
    union_area = det1_area + det2_area - inter_area

    # Compute IoU
    iou = inter_area / (union_area + 1e-16)
    return iou
# Returns iou

# Get KITTI domain disparity for a set of detections
def get_disparity(
    path_l,                 # Path to original left image
    dets_l,                 # Array of detections in left image
    disp_folder,            # Path to disparity folder
):
    # Get disparity image
    path_disp = disp_folder + path_l.split('/')[-1]
    print(path_disp)
    disp_full = cv2.imread(path_disp, cv2.IMREAD_UNCHANGED)

    # Get disparities
    disps_l = []
    for idx_l, det_l in enumerate(dets_l):

        # Get KITTI domain coordinates
        # wl, hu, box_w, box_h = cvt_domain(det_l, img_h=disp_full.shape[0], img_w=disp_full.shape[1])
        wl, hu, wr, hd = [int(k) for k in det_l[0:4]]
        w, h = wr-wl, hd-hu

        if h != 0 and w != 0: # Some detections are real fucking small I guess
            disp_l = np.mean(  disp_full[max(0, hu):min(374, hd), max(0, wl):min(1241, wr), 0])
        else:
            disp_l = 0
        disps_l += [disp_l]

    return disps_l
# Returns disps_l

# Get padding for a set of detections
def get_padding(
    dets_l,                 # Array of detections in left image
    dets_r,                 # Array of detections in right image
    matching,               # Matching dictionary
):
    l2r, r2l = matching

    bboxes = [None for i, _ in enumerate(dets_l)]
    pads_l = [None for i, _ in enumerate(dets_l)]
    pads_r = [None for i, _ in enumerate(dets_r)]

    for idx_l, det_l in enumerate(dets_l):
        idx_r = l2r[idx_l]
        if idx_r is not None:
            det_r = dets_r[idx_r]

            # Get KITTI domain coordinates
            # wl_l, hu_l, box_w_l, box_h_l = cvt_domain(det_l)
            # wr_l, hd_l = wl_l + box_w_l, hu_l + box_h_l
            # wl_r, hu_r, box_w_r, box_h_r = cvt_domain(det_r)
            # wr_r, hd_r = wl_r + box_w_r, hu_r + box_h_r
            wl_l, hu_l, wr_l, hd_l = det_l[0:4]
            wl_r, hu_r, wr_r, hd_r = det_r[0:4]

            # Get bounding box coordinates (camera domain)
            bbox_wl = min(wl_l, wl_r)
            bbox_wr = max(wr_l, wr_r)
            bbox_hd = max(hd_l, hu_r)
            bbox_hu = min(hu_l, hu_r)

            # Compute padding for each image
            pad_wl_l = max(0, wl_l - bbox_wl)
            pad_hu_l = max(0, hu_l - bbox_hu)
            pad_wr_l = max(0, bbox_wr - wr_l)
            pad_hd_l = max(0, bbox_hd - hd_l)

            pad_wl_r = max(0, wl_r - bbox_wl)
            pad_hu_r = max(0, hu_r - bbox_hu)
            pad_wr_r = max(0, bbox_wr - wr_r)
            pad_hd_r = max(0, bbox_hd - hd_r)

            bboxes[idx_l] = [ bbox_wl,  bbox_hu,  bbox_wr,  bbox_hd]
            pads_l[idx_l] = [pad_wl_l, pad_hu_l, pad_wr_l, pad_hd_l]
            pads_r[idx_r] = [pad_wl_r, pad_hu_r, pad_wr_r, pad_hd_r]

    return bboxes, pads_l, pads_r
# Returns bboxes, pads_l, pads_r

# Save image annotated with detections
def plot_image(
    path_l,                 # Path to original left image
    path_r,                 # Path to original right image
    dets_l,                 # Array of detections in left image
    dets_r,                 # Array of detections in right image
    classes,                # Class labels
    disps_l,                # Disparities of left detections
    plot_disparity=False,   # Plot disparity map as third image; optional, default false
    disp_folder=None,       # Path to disparity folder; optional, only needed if plot_disparity is True
    matching=None,          # Matching dictionaries (l2r, r2l); optional, used for coloring
    bboxes=None,            # Bounding boxes; optional, plotted on images and disparity
    path_out=None,          # Output file directory for saving; optional, file not saved if not passed
):
    # Get matplotlib plot
    if not plot_disparity:
        fig, (ax_l, ax_r) = plt.subplots(2, 1)
    else:
        fig, (ax_l, ax_r, ax_d) = plt.subplots(3, 1)
        ax_d.axis('off')
    ax_l.axis('off')
    ax_r.axis('off')

    # Get images (no ketchup, just images, raw images)
    img_l = cv2.imread(path_l, cv2.IMREAD_COLOR)
    img_r = cv2.imread(path_r, cv2.IMREAD_COLOR)
    if plot_disparity and (disp_folder is not None):
        # img_d = np.array(Image.open(disp_folder + path_l.split('/')[-1]))
        img_d = cv2.imread(disp_folder + path_l.split('/')[-1])

    # Plot images
    ax_l.imshow(img_l)
    ax_r.imshow(img_r)
    if plot_disparity and (disp_folder is not None):
        ax_d.imshow(img_d)

    # Get colors
    cmap_l = plt.get_cmap('viridis')
    cmap_r = plt.get_cmap('plasma')
    colors_l = [cmap_l(i) for i in np.linspace(0, 1, len(dets_l))]
    colors_r = [cmap_r(i) for i in np.linspace(0, 1, len(dets_r))]

    # Plot left detections
    for idx_l, det_l in enumerate(dets_l):

        # Get KITTI domain coordinates
        # wl, hu, box_w, box_h = cvt_domain(det_l, img_h=img_l.shape[0], img_w=img_l.shape[1])
        wl, hu, wr, hd = [int(k) for k in det_l[0:4]]

        # Get detection color
        if matching is not None:
            l2r, r2l = matching
            if l2r[idx_l] is not None:
                color = colors_l[idx_l]
            else:
                color = 'r' # Red
        else:
            color = 'white'
            # color = colors_l[idx_l]

        # Plot this detection on the left image
        bbox_l = patches.Rectangle(
            (wl, hu), 
            wr-wl, 
            hd-hu, 
            linewidth=1, 
            edgecolor=color, 
            facecolor='none',
            linestyle='--'
        )
        ax_l.add_patch(bbox_l)

        # # Label this detection on the left image
        # if matching is not None:
        #     l2r, r2l = matching
        #     if l2r[idx_l] is not None:
        #         text = '{}'.format(idx_l)
        #     else:
        #         text = 'X'
        # else:
        #     text = '{}'.format(idx_l)
        # ax_l.text(
        #     wl, 
        #     hu+box_h+30, 
        #     s=text,
        #     color='white' if matching is not None else 'black',
        #     verticalalignment='top',
        #     bbox={'color':color, 'pad':0}
        # )

        # Plot the corresponding bounding box on the left image
        if (matching is not None) and (bboxes is not None):
            l2r, _ = matching
            if l2r[idx_l] is not None:
                bbox_wl, bbox_hu, bbox_wr, bbox_hd = bboxes[idx_l]
                bbox_bbox = patches.Rectangle(
                    (bbox_wl, bbox_hu),
                    bbox_wr-bbox_wl,
                    bbox_hd-bbox_hu,
                    linewidth=2,
                    edgecolor=color,
                    facecolor='none',
                )
                ax_l.add_patch(bbox_bbox)

    # Plot right detections
    for idx_r, det_r in enumerate(dets_r):

        # Get KITTI domain coordinates
        # wl, hu, box_w, box_h = cvt_domain(det_r, img_h=img_l.shape[0], img_w=img_l.shape[1])
        wl, hu, wr, hd = [int(k) for k in det_r[0:4]]

        # Get detection color
        if matching is not None:
            l2r, r2l = matching
            if r2l[idx_r] is not None:
                color = colors_l[r2l[idx_r]]
            else:
                color = 'r' # Red
        else:
            color = 'white'
            # color = colors_r[idx_r]

        # Plot this detection on right image
        bbox_r = patches.Rectangle(
            (wl, hu), 
            wr-wl, 
            hd-hu, 
            linewidth=1, 
            edgecolor=color, 
            facecolor='none',
            linestyle='--',
        )
        ax_r.add_patch(bbox_r)
    
        # # Label this detection on right image
        # if matching is not None:
        #     l2r, r2l = matching
        #     if r2l[idx_r] is not None:
        #         text = '{}'.format(r2l[idx_r])
        #     else:
        #         text = 'X'
        # else:
        #     text = '{}'.format(idx_r)
        # ax_r.text(
        #     wl, 
        #     hu-30, 
        #     s=text,
        #     color='white' if matching is not None else 'black',
        #     verticalalignment='top',
        #     bbox={'color':color, 'pad':0}
        # )

        # Plot the corresponding bounding box on the right image
        if (matching is not None) and (bboxes is not None):
            _, r2l = matching
            if r2l[idx_r] is not None:
                bbox_wl, bbox_hu, bbox_wr, bbox_hd = bboxes[r2l[idx_r]]
                bbox_bbox = patches.Rectangle(
                    (bbox_wl, bbox_hu),
                    bbox_wr-bbox_wl,
                    bbox_hd-bbox_hu,
                    linewidth=2,
                    edgecolor=color,
                    facecolor='none',
                )
                ax_r.add_patch(bbox_bbox)

    # Plot bounding boxes on stereo depth map
    if plot_disparity and (matching is not None) and (bboxes is not None):
        l2r, r2l = matching
        for idx_l, bbox in enumerate(bboxes):
            if bbox is not None:
                color = colors_l[idx_l]
                bbox_wl, bbox_hu, bbox_wr, bbox_hd = bbox
                bbox_bbox = patches.Rectangle(
                    (bbox_wl, bbox_hu),
                    bbox_wr-bbox_wl,
                    bbox_hd-bbox_hu,
                    linewidth=2,
                    edgecolor=color,
                    facecolor='none',
                )
                ax_d.add_patch(bbox_bbox)

    # Prepare and save the figure
    fig.tight_layout()
    if path_out is not None:
        plt.savefig(path_out, bbox_inches='tight', pad_inches=0.0)

    if not plot_disparity:
        return fig, (ax_l, ax_r)
    else:
        return fig, (ax_l, ax_r, ax_d)
    # plt.close()
# Returns fig, axes

# Save detections over a dataset as a video
def save_video(
    paths_l,
    paths_r,
    dets_l,
    dets_r,
    classes,
    disps_l,
    plot_disparity=False,
    disp_folder=None,
    matchings=None,
    bboxes=None,
    path_out=None,
):
    if path_out is None:
        path_out = './stat.mp4'
    if plot_disparity:
        res = (640, 480)
    else:
        res = (610, 407)
    vid = cv2.VideoWriter(
        path_out, 
        cv2.VideoWriter_fourcc(*'mp4v'),
        10,
        res
    )

    for batch_i, (path_l, path_r, det_l, det_r, disp_l) in enumerate(zip(paths_l, paths_r, dets_l, dets_r, disps_l)):
        print("{}/{}".format(batch_i, len(paths_l)))
        matching = matchings[batch_i] if matchings is not None else None
        bbox     = bboxes[batch_i]    if bboxes    is not None else None
        fig, axes = plot_image(
            path_l=path_l,
            path_r=path_r,
            dets_l=det_l,
            dets_r=det_r,
            classes=classes,
            disps_l=disp_l,
            plot_disparity=plot_disparity,
            disp_folder=disp_folder,
            matching=matching,
            bboxes=bbox,
        )

        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())[:,:,:-1]
        vid.write(frame)
        plt.close()
    vid.release()
    cv2.destroyAllWindows()
# Returns nothing

#############################################################################################
## Matching Algorithms                                                                     ##
#############################################################################################

# Perform stereo ROI matching by selecting a single ROI from each frame according to the maximum of some metric
def single_match(
    dets_l,                 # Array of detections in left image
    dets_r,                 # Array of detections in right image
    metric,                 # Metric by which single ROI is chosen in each image; operates on detection structure
):
    l2r = [None for _ in enumerate(dets_l)]
    r2l = [None for _ in enumerate(dets_r)]

    # Compute metric for left ROIs and choose left ROI
    metric_l = np.array([metric(det_l) for det_l in dets_l])
    idx_l = None if len(metric_l) == 0 else np.argmax(metric_l)

    # Compute metric for right ROIs and choose right ROI
    metric_r = np.array([metric(det_r) for det_r in dets_r])
    idx_r = None if len(metric_r) == 0 else np.argmax(metric_r)

    # Update matching dictionaries
    if idx_l is not None:
        l2r[idx_l] = idx_r
    if idx_r is not None:
        r2l[idx_r] = idx_l

    return l2r, r2l
# Returns l2r, r2l

# Match detected ROIs in left and right images 
def match_all(
    dets_l,                 # Array of detections in left
    dets_r,                 # Array of detections in right
    classes,                # List of class labels
    iou_thres,              # Threshold above which ROIs are said to describe same object
    disps_l=None,           # Left detection disparities; optional, used for IoU correction
    class_match=True,       # Require matched detections to match classes; optional, default True
):
    # Useful lambda functions
    wl       = lambda det : det[0]
    hu       = lambda det : det[1]
    wr       = lambda det : det[2]
    hd       = lambda det : det[3]
    conf     = lambda det : det[4]
    cls_conf = lambda det : det[5]
    # If we don't care about classes, just treat everything as a Pedestrian #NUMTOT
    if class_match:
        cls_pred = lambda det : classes[int(det[6])]
    else:
        cls_pred = lambda det : 'Pedestrian'


    # Sort left detections by confidence, from most to least, with indexes
    idxs_l_sorted = sorted([i for i, _ in enumerate(dets_l)], key=lambda i:cls_conf(dets_l[i]), reverse=True)

    # List of remaining indexes to unmatched right detections, for each class
    idxs_r_class = {clas:[i for i, det_r in enumerate(dets_r) if cls_pred(det_r) == clas] for clas in classes}

    # Detection matching dictionaries, idx_l:idx_r and idx_r:idx_l
    l2r = [None for _ in enumerate(dets_l)]
    r2l = [None for _ in enumerate(dets_r)]

    # Iterate over left detections from most to least confident
    for idx_l in idxs_l_sorted:
        det_l = dets_l[idx_l]
        class_l = cls_pred(det_l)
        if disps_l is not None:
            disp_l = disps_l[idx_l]
        else:
            disp_l = None

        # Compute IoU for all remaining unmatched right detections of the same class as the left detection
        rIoU = np.array([iou(det_l, dets_r[idx_r], disp_l) for idx_r in idxs_r_class[class_l]])

        # Only consider IoUs above threshold
        rIoU[rIoU < iou_thres] = 0

        if any(rIoU):
            # Match left detection to right detection with same class and highest IoU
            # Remove right detection from list so it cannot be assigned again
            argmax_idx = np.argmax(rIoU)
            # idx_r = dets_r_classes[cls_pred(det_l)].pop(argmax_idx)
            idx_r = idxs_r_class[class_l].pop(argmax_idx)

            # Record matching
            l2r[idx_l] = idx_r
            r2l[idx_r] = idx_l

    return l2r, r2l
# Returns l2r, r2l

# Perform stereo ROI matching by selecting a single ROI from each frame, with temporal information
def temporal_match(
    dets_l,             # Array of array of detections in left image
    dets_r,             # Array of array of detections in right image
    metric,             # Metric by which single ROI is chosen in each image when temporal metric fails
):
    dist_l1 = lambda pt1, pt2 : sum([     abs(crd1 - crd2)    for crd1, crd2 in zip(pt1, pt2)])
    dist_l2 = lambda pt1, pt2 : sum([math.pow(crd1 - crd2, 2) for crd1, crd2 in zip(pt1, pt2)])

    center_l1 = lambda pt1, pt2 : dist_l1(((pt1[0]+pt1[2])/2, (pt1[1]+pt1[3])/2), ((pt2[0]+pt2[2])/2, (pt2[1]+pt2[3])/2))
    center_l2 = lambda pt1, pt2 : dist_l2(((pt1[0]+pt1[2])/2, (pt1[1]+pt1[3])/2), ((pt2[0]+pt2[2])/2, (pt2[1]+pt2[3])/2))

    matchings = []
    idx_l, idx_r = None, None
    for batch_i, (det_l_n, det_r_n) in enumerate(zip(dets_l, dets_r)):
        # Seed initial frame
        if (idx_l is None) or (idx_r is None):
            l2r_n, r2l_n = single_match(
                dets_l=det_l_n, 
                dets_r=det_r_n, 
                metric=metric
            )
        # Check previous frame
        else:
            l2r_n = [None for _ in enumerate(det_l_n)]
            r2l_n = [None for _ in enumerate(det_r_n)]

            # Get previous frame detection coordinates
            # p_l = cvt_domain(dets_l[batch_i-1][idx_l])
            # p_r = cvt_domain(dets_r[batch_i-1][idx_r])
            p_l = det_l[batch_i-1][idx_l]
            p_r = det_r[batch_i-1][idx_r]

            # Compute distance for all detections in current frame
            dist_l_n = np.array([center_l1(p_l, p_l_n) for p_l_n in det_l_n])
            dist_r_n = np.array([center_l1(p_r, p_r_n) for p_r_n in det_r_n])

            # Get detection minimizing distance metric
            idx_l = None if len(dist_l_n) == 0 else np.argmax(dist_l_n)
            idx_r = None if len(dist_r_n) == 0 else np.argmax(dist_r_n)

            l2r_n[idx_l] = idx_r
            r2l_n[idx_r] = idx_l

        matchings += [(l2r_n, r2l_n)]

    return matchings
# Returns matchings

#############################################################################################
## Evaluation Metrics                                                                      ##
#############################################################################################

# Compute detection imbalance counts over a dataset
def det_imbalance_count(
    dets_l,                 # Array of array of detections in left images
    dets_r,                 # Array of array of detections in right images
):
    ylyr = len([None for det_l, det_r in zip(dets_l, dets_r) if det_l != [] and det_r != []])
    ylnr = len([None for det_l, det_r in zip(dets_l, dets_r) if det_l != [] and det_r == []])
    nlyr = len([None for det_l, det_r in zip(dets_l, dets_r) if det_l == [] and det_r != []])
    nlnr = len([None for det_l, det_r in zip(dets_l, dets_r) if det_l == [] and det_r == []])

    return ylyr, ylnr, nlyr, nlnr
# Returns ylyr, ylnr, nlyr, nlnr

# Compute unmatched detection counts over a dataset
def unmatched_count(
    matchings,              # Array of matchings (l2r, r2l) for each frame
):
    # Matching counts by frame
    match_frame  = 0
    umatch_frame = 0

    # Matching counts by detections
    match_l_det  = 0
    umatch_l_det = 0
    match_r_det  = 0
    umatch_r_det = 0

    for matching in matchings:
        l2r, r2l = matching
        frame = False # Is there a match in this frame?
        for idx_r in l2r:
            if idx_r is not None:
                match_l_det += 1
                frame = True
            else:
                umatch_l_det += 1
        for idx_l in r2l:
            if idx_l is not None:
                match_r_det += 1
            else:
                umatch_r_det += 1
        if frame:
            match_frame += 1
        else:
            umatch_frame += 1

    return (match_frame, umatch_frame), (match_l_det, umatch_l_det), (match_r_det, umatch_r_det)
# Returns (match_frame, umatch_frame), (match_l_det, umatch_l_det), (match_r_det, umatch_r_det)

# Compute Class and IoU Mismatch counts over a dataset with a given matching
def mismatch_count(
    dets_l,                 # Array of array of detections in left images
    dets_r,                 # Array of array of detections in right images
    classes,                # Class labels for the dataset
    matchings,              # Array of matchings (l2r, r2l) for each frame
    disps_l,                # Array of array of average disparities for each left detection
    iou_thres=0.5,          # Threshold for IoU Matching; optional, default 0.5
):
    mcmi = 0    # Match Class, Match IoU
    mcui = 0    # Match Class, Mismatch IoU
    ucmi = 0    # Mismatch Class, Match IoU
    ucui = 0    # Mismatch Class, Mismatch IoU

    for det_l, det_r, matching, disp_l in zip(dets_l, dets_r, matchings, disps_l):
        l2r, r2l = matching
        for idx_l, _ in enumerate(det_l):
            if l2r[idx_l] is not None:
                idx_r = l2r[idx_l]

                # Check Class Matching
                mc = (classes[int(det_l[idx_l][6])] == classes[int(det_r[idx_r][6])])

                # Check IoU Matching
                mi = (iou(det_l[idx_l], det_r[idx_r], disp_l[idx_l]) > iou_thres)

                mcmi += 1 if ((    mc) and (    mi)) else 0
                mcui += 1 if ((    mc) and (not mi)) else 0
                ucmi += 1 if ((not mc) and (    mi)) else 0
                ucui += 1 if ((not mc) and (not mi)) else 0

    return mcmi, mcui, ucmi, ucui
# Returns mcmi, mcui, ucmi, ucui

# Compute padding statistics
def pad_stats(
    dets_l,                 # Array of array of detections in left images
    dets_r,                 # Array of array of detections in right images
    matchings,              # Array of matchings (l2r, r2l) for each frame
    bboxes,                 # Array of array of bounding box for each left detection
    pads_l,                 # Array of array of padding for each left detection
    pads_r,                 # Array of array of padding for each right detection
):
    box_area = lambda box : (box[3] - box[1]) * (box[2] - box[0])

    bbox_det_ratio_l = []   # 1D Array of (BBox Size : Det Size) for all left detections in the dataset
    bbox_det_ratio_r = []   # 1D Array of (BBox Size : Det Size) for all right detections in the dataset
    pads_array_l     = []   # 2D Array of (WL Pad, HU Pad, WR Pad, HD Pad) for all left detections in the dataset
    pads_array_r     = []   # 2D Array of (WL Pad, HU Pad, WR Pad, HD Pad) for all right detections in the dataset

    for batch_i, (det_l, det_r, matching, bbox, pad_l, pad_r) in enumerate(zip(dets_l, dets_r, matchings, bboxes, pads_l, pads_r)):
        l2r, r2l = matching

        for idx_l, _ in enumerate(det_l):
            idx_r = l2r[idx_l]
            if idx_r is not None:
                # _, _, det_w_l, det_h_l = cvt_domain(det_l[idx_l])
                # _, _, det_w_r, det_h_r = cvt_domain(det_r[idx_r])

                bbox_det_ratio_l += [(1.0 * box_area(bbox[idx_l])) / (1.0 * box_area(det_l[idx_l]))]
                bbox_det_ratio_r += [(1.0 * box_area(bbox[idx_l])) / (1.0 * box_area(det_r[idx_r]))]
                pads_array_l     += [pad_l[idx_l]]
                pads_array_r     += [pad_r[idx_r]]

    stats = lambda arr : (np.mean(arr), np.median(arr), np.amax(arr))

    bbox_det_ratio_l_stat = stats(np.array(bbox_det_ratio_l))
    bbox_det_ratio_r_stat = stats(np.array(bbox_det_ratio_r))
    
    pad_wl_l_stat         = stats(np.array(pads_array_l)[:,0])
    pad_hu_l_stat         = stats(np.array(pads_array_l)[:,1])
    pad_wr_l_stat         = stats(np.array(pads_array_l)[:,2])
    pad_hd_l_stat         = stats(np.array(pads_array_l)[:,3])

    pad_wl_r_stat         = stats(np.array(pads_array_r)[:,0])
    pad_hu_r_stat         = stats(np.array(pads_array_r)[:,1])
    pad_wr_r_stat         = stats(np.array(pads_array_r)[:,2])
    pad_hd_r_stat         = stats(np.array(pads_array_r)[:,3])

    return bbox_det_ratio_l_stat, bbox_det_ratio_r_stat, (pad_wl_l_stat, pad_hu_l_stat, pad_wr_l_stat, pad_hd_l_stat), (pad_wl_r_stat, pad_hu_r_stat, pad_wr_r_stat, pad_hd_r_stat)
# Returns bbox_det_ratio_l_stat, bbox_det_ratio_r_stat, (pad_wl_l_stat, pad_hu_l_stat, pad_wr_l_stat, pad_hd_l_stat), (pad_wl_r_stat, pad_hu_r_stat, pad_wr_r_stat, pad_hd_r_stat)

#############################################################################################
## Object Tracking Methods                                                                 ##
#############################################################################################

# Track objects in a segment with object detection keyframes
def object_track(
    paths_l,                # Array of left image paths, assumed sequential
    paths_r,                # Array o right image paths, assumed sequential
    dets_l,                 # Array of array of detections in left images
    dets_r,                 # Array of array of detections in right images
    tracker_type,           # OpenCV Object Tracking type
):

    tracks_l = [[] for frame in range(len(paths_l))]
    tracks_r = [[] for frame in range(len(paths_r))]

    print("&&& Creating Tracker")

    # Create object tracker
    if tracker_type == 'BOOSTING':
        tracker = cv2.legacy.TrackerBoosting_create()
    elif tracker_type == 'MIL':
        tracker = cv2.legacy.TrackerMIL_create()
    elif tracker_type == 'KCF':
        tracker = cv2.legacy.TrackerKCF_create()
    elif tracker_type == 'TLD':
        tracker = cv2.legacy.TrackerTLD_create()
    elif tracker_type == 'MEDIANFLOW':
        tracker = cv2.legacy.TrackerMedianFlow_create()
    elif tracker_type == 'GOTURN':
         tracker = cv2.TrackerGOTURN_create()
    elif tracker_type == 'MOSSE':
        tracker = cv2.legacy.TrackerMOSSE_create()
    elif tracker_type == "CSRT":
        tracker = cv2.legacy.TrackerCSRT_create()

    print("&&& Initializing Tracker")

    # Create the bounding box and initialize the tracker
    wl, hu, wr, hd = dets_l[0][0][0:4]
    tracks_l[0] = dets_l[0]
    w, h = wr-wl, hd-hu
    bbox = (wl, hu, w, h)
    # frame = np.array(Image.open(paths_l[0]))
    frame = cv2.imread(paths_l[0], cv2.IMREAD_COLOR)
    ok = tracker.init(frame, bbox)

    print("&&& Analyzing Video")

    # Iterate over the rest of the video
    for frame_i, (path_l, path_r, det_l, det_r) in enumerate(zip(paths_l, paths_r, dets_l, dets_r)):
        print("{}/{}".format(frame_i, len(dets_l)))
        if frame_i == 0:
            continue

        print("&&& Opening Image")
        # frame = np.array(Image.open(path_l))
        frame = cv2.imread(path_l, cv2.IMREAD_COLOR)

        # Update tracker
        print("&&& Updating Tracker")
        ok, bbox = tracker.update(frame)
        if ok:
            print("&&& Appending Data")
            wl, hu, w, h = bbox
            wr, hd = wl+w, hu+h
            tracks_l[frame_i] += [[wl, hu, wr, hd, 1.0, 1.0, 0]]

    return tracks_l, tracks_r
# Returns tracks_l, tracks_r

#############################################################################################
## Main                                                                                    ##
#############################################################################################

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_file', type=str,   default='dump/obj/all/kittiobj.dump', help='Path to pickle dump file containing results')
    parser.add_argument('--coord_domain', type=str,   default='YOLO',                       help='Spatial coordinate domain; YOLO or KITTI')
    parser.add_argument('--class_path',   type=str,   default='data/kitti.names',           help='Path to class label file')
    parser.add_argument('--matching',     type=str,   default='Single',                     help='Type of stereo detection matching')
    parser.add_argument('--conf_thres',   type=float, default=0.8,                          help='Object confidence threshold')
    parser.add_argument('--iou_thres',    type=float, default=0.5,                          help='IoU threshold for non-maximum suppression and stereo matching')
    parser.add_argument('--program_mode', type=str,   default='Print',                      help='Program Mode')
    parser.add_argument('--disp_folder',  type=str,   default='/afs/eecs.umich.edu/vlsisp/users/erharj/TinyHITNet/result_predict/obj2012_HITNet/', help='Path to disparity images for the dataset')
    parser.add_argument('--tracking',     type=str,   default='BOOSTING',                   help='Tracking algorithm; BOOSTING, MIL, KCF, TLD, MEDIANFLOW, GOTURN, MOSSE, CSRT')
    parser.add_argument('--output_path',  type=str,   default='data/output/',               help='Path for saving output images')
    opt = parser.parse_args()

    # Get classes
    classes = load_classes(opt.class_path)

    # Load results
    with open(opt.results_file, 'rb') as results_file:
        if opt.program_mode == 'Disparity':
            paths_l, paths_r, dets_l, dets_r = pickle.load(results_file)
        else:
            paths_l, paths_r, dets_l, dets_r, disps_l = pickle.load(results_file)

    # Convert YOLO Coordinates to KITTI Coordinates
    if opt.coord_domain == 'YOLO':
        for batch_i, (det_l, det_r) in enumerate(zip(dets_l, dets_r)):
            for idx_l, _ in enumerate(det_l):
                wl, hu, w, h = cvt_domain(det_l[idx_l])
                wr, hd = wl+w, hu+h
                det_l[idx_l][0:4] = [wl, hu, wr, hd]
            for idx_r, _ in enumerate(det_r):
                wl, hu, w, h = cvt_domain(det_r[idx_r])
                wr, hd = wl+w, hu+h
                det_r[idx_r][0:4] = [wl, hu, wr, hd]

    # Compute left detection disparities, if needed
    if opt.program_mode == 'Disparity':
        print(">>> Computing Left Detection Disparities")
        disps_l = []
        for batch_i, (path_l, path_r, det_l, det_r) in enumerate(zip(paths_l, paths_r, dets_l, dets_r)):
            # print(path_l)
            disp_l = get_disparity(
                path_l=path_l,
                dets_l=det_l,
                disp_folder=opt.disp_folder
            )
            disps_l += [disp_l]

    # Compute matchings
    print(">>> Computing Matchings")
    matchings = []
    cls_conf = lambda det : det[5]
    cls_pred = lambda det : classes[int(det[6])]
    det_area = lambda det : (det[3] - det[1]) * (det[2] - det[0])
    if   opt.matching == 'Temporal':
        matchings = temporal_match(
            dets_l=dets_l,
            dets_r=dets_r,
            metric=det_area,
        )
    elif opt.matching == 'Single':
        for batch_i, (det_l, det_r) in enumerate(zip(dets_l, dets_r)):
            matching = single_match(
                dets_l=det_l,
                dets_r=det_r,
                metric=det_area,
            )
            matchings += [matching]
    elif opt.matching == 'All':
        for batch_i, (det_l, det_r, disp_l) in enumerate(zip(dets_l, dets_r, disps_l)):
            matching = match_all(
                dets_l=det_l, 
                dets_r=det_r, 
                classes=classes, 
                iou_thres=opt.iou_thres,
                disps_l=disp_l,
                class_match=True,
            )
            matchings += [matching]

    # Compute bounding boxes and padding
    print(">>> Computing Bounding Boxes")
    bboxes = []
    pads_l = []
    pads_r = []
    for batch_i, (det_l, det_r, matching) in enumerate(zip(dets_l, dets_r, matchings)):
        bbox, pad_l, pad_r = get_padding(
            dets_l=det_l,
            dets_r=det_r,
            matching=matching,
        )
        bboxes += [bbox]
        pads_l += [pad_l]
        pads_r += [pad_r]

    # Run main program
    if   opt.program_mode == 'Disparity':
        print(">>> Performing Disparity")
        with open(opt.results_file, 'wb') as f:
            pickle.dump((paths_l, paths_r, dets_l, dets_r, disps_l), file=f)

    elif opt.program_mode == 'Imbalance':
        # Compute Imbalance
        print(">>> Computing Imbalance")
        ylyr, ylnr, nlyr, nlnr = det_imbalance_count(dets_l, dets_r)
        total = ylyr + ylnr + nlyr + nlnr

        print("+Left +Right: {} \t({}%)".format(ylyr, (100.0*ylyr)/(1.0*total)))
        print("+Left -Right: {} \t({}%)".format(ylnr, (100.0*ylnr)/(1.0*total)))
        print("-Left +Right: {} \t({}%)".format(nlyr, (100.0*nlyr)/(1.0*total)))
        print("-Left -Right: {} \t({}%)".format(nlnr, (100.0*nlnr)/(1.0*total)))

    elif opt.program_mode == 'Unmatched':
        print(">>> Computing Unmatched")
        (match_frame, umatch_frame), (match_l_det, umatch_l_det), (match_r_det, umatch_r_det) = unmatched_count(matchings)
        total_frame = match_frame + umatch_frame
        total_l_det = match_l_det + umatch_l_det
        total_r_det = match_r_det + umatch_r_det

        print("Matched / Unmatched [L DET]: {} / {} ({}%)".format(umatch_l_det, total_l_det, (100.0*umatch_l_det)/(1.0*total_l_det)))
        print("Matched / Unmatched [R DET]: {} / {} ({}%)".format(umatch_r_det, total_r_det, (100.0*umatch_r_det)/(1.0*total_r_det)))
        print("Matched / Unmatched [FRAME]: {} / {} ({}%)".format(umatch_frame, total_frame, (100.0*umatch_frame)/(1.0*total_frame)))

    elif opt.program_mode == 'Mismatch':
        print(">>> Computing Mismatch")
        mcmi, mcui, ucmi, ucui = mismatch_count(
            dets_l=dets_l,
            dets_r=dets_r,
            classes=classes,
            matchings=matchings,
            disps_l=disps_l
        )
        total = mcmi + mcui + ucmi + ucui

        print("Class Match,    IoU Match:    {} / {} ({}%)".format(mcmi, total, (100.0 * mcmi) / (1.0 * total)))
        print("Class Match,    IoU Mismatch: {} / {} ({}%)".format(mcui, total, (100.0 * mcui) / (1.0 * total)))
        print("Class Mismatch, IoU Match:    {} / {} ({}%)".format(ucmi, total, (100.0 * ucmi) / (1.0 * total)))
        print("Class Mismatch, IoU Mismatch: {} / {} ({}%)".format(ucui, total, (100.0 * ucui) / (1.0 * total)))

    elif opt.program_mode == 'Padding':
        print(">>> Computing Padding")
        bbox_det_ratio_l_stat, bbox_det_ratio_r_stat, pad_l_stat, pad_r_stat = pad_stats(
            dets_l=dets_l,
            dets_r=dets_r,
            matchings=matchings,
            bboxes=bboxes,
            pads_l=pads_l,
            pads_r=pads_r,
        )

        print("Bounding Box : Detection Ratio Stats")
        print("       MEAN,      MEDIAN,    MAXIMUM")
        print("Left:  {:9.5f}  {:9.5f}  {:9.5f}".format(*bbox_det_ratio_l_stat))
        print("Right: {:9.5f}  {:9.5f}  {:9.5f}".format(*bbox_det_ratio_r_stat))

        print("Left Padding Stats")
        print("       MEAN,      MEDIAN,    MAXIMUM")
        print("L←:    {:9.5f}  {:9.5f}  {:9.5f}".format(*pad_l_stat[0]))
        print("L↑:    {:9.5f}  {:9.5f}  {:9.5f}".format(*pad_l_stat[1]))
        print("L→:    {:9.5f}  {:9.5f}  {:9.5f}".format(*pad_l_stat[2]))
        print("L↓:    {:9.5f}  {:9.5f}  {:9.5f}".format(*pad_l_stat[3]))

        print("Right Padding Stats")
        print("       MEAN,      MEDIAN,    MAXIMUM")
        print("R←:    {:9.5f}  {:9.5f}  {:9.5f}".format(*pad_r_stat[0]))
        print("R↑:    {:9.5f}  {:9.5f}  {:9.5f}".format(*pad_r_stat[1]))
        print("R→:    {:9.5f}  {:9.5f}  {:9.5f}".format(*pad_r_stat[2]))
        print("R↓:    {:9.5f}  {:9.5f}  {:9.5f}".format(*pad_r_stat[3]))

    elif opt.program_mode == 'Tracking':
        print(">>> Performing Tracking")
        tracks_l, tracks_r = object_track(
            paths_l=paths_l,
            paths_r=paths_r,
            dets_l=dets_l,
            dets_r=dets_r,
            tracker_type=opt.tracking,
        )

        save_video(
            paths_l=paths_l,
            paths_r=paths_r,
            dets_l=tracks_l,
            dets_r=tracks_r,
            classes=classes,
            disps_l=disps_l,
            plot_disparity=True,
            disp_folder=opt.disp_folder,
            # matchings=matchings,
            # bboxes=bboxes,
            path_out=opt.output_path,
        )

    elif opt.program_mode == 'Filter':
        # Filter frames and save as a subset list
        print(">>> Performing Filter")
        save_paths_l = []
        save_paths_r = []
        save_dets_l = []
        save_dets_r = []
        save_disps_l = []

        for batch_i, (path_l, path_r, det_l, det_r, disp_l, matching) in enumerate(zip(paths_l, paths_r, dets_l, dets_r, disps_l, matchings)):
            # condition = True

            # Your condition goes here!

            # # CONDITION - Imbalance
            # condition = (det_l == []) or (det_r == [])

            # # CONDITION - Balance
            # condition = (det_l != []) and (det_r != [])

            # l2r, r2l = matching
            # for idx_l, _ in enumerate(det_l):
            #     if l2r[idx_l] is not None:
            #         idx_r = l2r[idx_l]

            #         # class_l = classes[int(det_l[idx_l][6])]
            #         # class_r = classes[int(det_r[idx_r][6])]
        
            #         # # CONDITION - Class Mismatch
            #         # condition = (class_l != class_r)

            #         # # CONDITION - Class Match
            #         # condition = (class_l == class_r)

            #         iou_l = iou(
            #             det1=det_l[idx_l],
            #             det2=det_r[idx_r],
            #             disp1=disp_l[idx_l],
            #         )

            #         # # CONDITION - IoU Mismatch
            #         # condition = (iou_l < opt.iou_thres)

            #         # # CONDITION - IoU Match
            #         # condition = (iou_l > opt.iou_thres)

            #         break            

            # condition = all([idx_r is None for idx_r in l2r])

            # if condition:
            #     save_paths_l += [path_l]
            #     save_paths_r += [path_r]
            #     save_dets_l += [det_l]
            #     save_dets_r += [det_r]
            #     save_disps_l += [disp_l]

            save_det_l = []
            save_det_r = []
            save_disp_l = []

            for idx_l, _ in enumerate(det_l):
                if classes[int(det_l[idx_l][6])] == 'Car':
                    save_det_l += [det_l[idx_l]]
                    save_disp_l += [disp_l[idx_l]]
            for idx_r, _ in enumerate(det_r):
                if classes[int(det_r[idx_r][6])] == 'Car':
                    save_det_r += [det_r[idx_r]]

            save_paths_l += [path_l]
            save_paths_r += [path_r]
            save_dets_l += [save_det_l]
            save_dets_r += [save_det_r]
            save_disps_l += [save_disp_l]

        with open(opt.output_path, 'wb') as f:
            pickle.dump((save_paths_l, save_paths_r, save_dets_l, save_dets_r, save_disps_l), file=f)

    elif opt.program_mode == 'Print':
        print(">>> Performing Print")
        for batch_i, (path_l, path_r, det_l, det_r, disp_l, matching, bbox) in enumerate(zip(paths_l, paths_r, dets_l, dets_r, disps_l, matchings, bboxes)):
            if batch_i >= 16:
                return

            print("{} (L: {}) (R: {})".format(batch_i, path_l, path_r))

            fig, _ = plot_image(
                path_l=path_l,
                path_r=path_r,
                dets_l=det_l,
                dets_r=det_r,
                classes=classes,
                disps_l=disp_l,
                # plot_disparity=True,
                # disp_folder=opt.disp_folder,
                # matching=matching,
                # bboxes=bbox,
                path_out='{}{}'.format(opt.output_path, path_l.split('/')[-1]),
            )

        plt.close()

    elif opt.program_mode == 'Video':
        print(">>> Performing Video")
        save_video(
            paths_l=paths_l,
            paths_r=paths_r,
            dets_l=dets_l,
            dets_r=dets_r,
            classes=classes,
            disps_l=disps_l,
            plot_disparity=True,
            disp_folder=opt.disp_folder,
            matchings=matchings,
            # bboxes=bboxes,
            path_out=opt.output_path,
        )

    else:
        print("!!! Unrecognized Program Mode: {}".format(opt.program_mode))

if __name__ == "__main__":
    main()