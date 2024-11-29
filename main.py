import cv2
import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
from utils import *
from student_feature_matching import match_features
from student_HoG import get_features
from student_LoG import get_interest_points
from IPython.core.debugger import set_trace
import argparse


parser = argparse.ArgumentParser(description="dlut-cv21-homework1-args")
parser.add_argument('--mode', type=str, default='debug')

args = parser.parse_args()
assert args.mode in ['debug','eval']

#=========================================set up======================================
# Notre Dame
image1 = load_image('../data/Notre Dame/1.jpg')
image2 = load_image('../data/Notre Dame/4191453057_c86028ce1f_o.jpg')
eval_file = '../data/Notre Dame/Notre_Dame_match_ground_truth.pkl'

#for reduction the calculation consuming, we set a scale_factor to downsample the image.
#DO NOT CHANGE THIS FACTOR
scale_factor = 0.5
image1 = cv2.resize(image1, (0, 0), fx=scale_factor, fy=scale_factor)
image2 = cv2.resize(image2, (0, 0), fx=scale_factor, fy=scale_factor)
image1_bw = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
image2_bw = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)


#================================LoG corner detector==============================
x1, y1 = get_interest_points(image1_bw)
x2, y2 = get_interest_points(image2_bw)

# x1, y1, x2, y2 = cheat_interest_points(
#                         eval_file, scale_factor)

# Visualize the interest points
if args.mode != 'eval' :
    c1 = show_interest_points(image1, x1, y1)
    c2 = show_interest_points(image2, x2, y2)
    plt.figure(); plt.imshow(c1); plt.show()
    plt.figure(); plt.imshow(c2); plt.show()
print('{:d} corners in image 1, {:d} corners in image 2'.format(len(x1), len(x2)))


#================================HoG feature encoding==============================
image1_features = get_features(image1_bw, x1, y1)
image2_features = get_features(image2_bw, x2, y2)

assert image1_features.shape[1] <= 256
assert image2_features.shape[1] <= 256

matches = match_features(image1_features, image2_features, x1, y1, x2, y2)
print('{:d} matches from {:d} corners'.format(len(matches), len(x1)))

#================================visualization=================================
if args.mode != 'eval' :
    num_pts_to_visualize = 100
    c1 = show_correspondence_circles(image1, image2,
                        x1[matches[:num_pts_to_visualize, 0]], y1[matches[:num_pts_to_visualize, 0]],
                        x2[matches[:num_pts_to_visualize, 1]], y2[matches[:num_pts_to_visualize, 1]])
    plt.figure(); plt.imshow(c1); plt.show()
    plt.savefig('../results/vis_circles.jpg', dpi=1000)
    c2 = show_correspondence_lines(image1, image2,
                        x1[matches[:num_pts_to_visualize, 0]], y1[matches[:num_pts_to_visualize, 0]],
                        x2[matches[:num_pts_to_visualize, 1]], y2[matches[:num_pts_to_visualize, 1]])

    plt.figure(); plt.imshow(c2); plt.show()
    plt.savefig('../results/vis_lines.jpg', dpi=1000)


#================================evaluation=================================
evaluate_script(image1,image2,
    eval_file, scale_factor, x1, x2, y1, y2, matches,mode=args.mode)
