from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import pickle

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

# Main
def main():
	# Parse command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--images_l',     type=str,   default='data/samples_l/',              help='Path to left images')
	parser.add_argument('--images_r',     type=str,   default='data/samples_r/',              help='Path to right images')
	parser.add_argument('--config_path',  type=str,   default='config/yolov3-kitti.cfg',      help='path to model config file')
	parser.add_argument('--weights_path', type=str,   default='weights/yolov3-kitti.weights', help='path to weights file')
	parser.add_argument('--class_path',   type=str,   default='data/kitti.names',             help='path to class label file')
	parser.add_argument('--conf_thres',   type=float, default=0.8,                            help='object confidence threshold')
	parser.add_argument('--nms_thres',    type=float, default=0.4,                            help='iou thresshold for non-maximum suppression')
	parser.add_argument('--batch_size',   type=int,   default=1,                              help='size of the batches')
	parser.add_argument('--n_cpu',        type=int,   default=8,                              help='number of cpu threads to use during batch generation')
	parser.add_argument('--img_size',     type=int,   default=416,                            help='size of each image dimension')
	parser.add_argument('--results_file', type=str,   default='stereo_iou.dump',              help='Path to pickle dump file')
	# parser.add_argument('--output_path', type=str, default='data/output/', help='Path for saving output images')
	opt = parser.parse_args()
	
	# Set up model
	model = Darknet(opt.config_path, img_size=opt.img_size)
	model.load_weights(opt.weights_path)
	model.cuda()
	model.eval()
	
	# Get evaluation data and classes
	dataloader_l = DataLoader(ImageFolder(opt.images_l, img_size=opt.img_size), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
	dataloader_r = DataLoader(ImageFolder(opt.images_r, img_size=opt.img_size), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
	classes = load_classes(opt.class_path)
	
	# Save image paths and detections
	paths_l, paths_r = [], []
	dets_l,  dets_r  = [], []
	
	for batch_i, ((path_l, img_l), (path_r, img_r)) in enumerate(zip(dataloader_l, dataloader_r)):
		print(batch_i)
		print(path_l)
		# Convert format
		img_l = Variable(img_l.type(torch.cuda.FloatTensor))
		img_r = Variable(img_r.type(torch.cuda.FloatTensor))
	
		# Perform inference
		with torch.no_grad():
			det_l = model(img_l)
			det_l = non_max_suppression(det_l, 80, opt.conf_thres, opt.nms_thres)
			det_r = model(img_r)
			det_r = non_max_suppression(det_r, 80, opt.conf_thres, opt.nms_thres)
	
		# Save image paths and detections
		paths_l.extend(path_l)
		paths_r.extend(path_r)
		dets_l.extend(det_l)
		dets_r.extend(det_r)

	dets_l_py = []
	for det_l in dets_l:
		if det_l is None:
			dets_l_py.extend([[]])
		else:
			dets_l_py.extend([det_l.tolist()])
	dets_r_py = []
	for det_r in dets_r:
		if det_r is None:
			dets_r_py.extend([[]])
		else:
			dets_r_py.extend([det_r.tolist()])

	# Take a dump
	with open(opt.results_file, 'wb') as f:
		pickle.dump((paths_l, paths_r, dets_l_py, dets_r_py), file=f)

	# # Perform post-processing
	# for batch_i, (path_l, path_r, det_l, det_r) in enumerate(zip(paths_l, paths_r, dets_l, dets_r)):
	# 	# print("{} (L: {}) (R: {})".format(batch_i, path_l, path_r))
		
	# 	l2r, r2l = match_dets(det_l, det_r, classes, opt.nms_thres)
	# 	save_image(path_l, path_r, det_l, det_r, l2r, r2l, classes, path_out=opt.output_path+"{:06d}.png".format(batch_i))


# Main
if __name__ == "__main__":
	main()