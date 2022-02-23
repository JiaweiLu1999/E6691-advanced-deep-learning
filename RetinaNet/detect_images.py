import torchvision
import torch
import argparse
import cv2
import detect_utils
import numpy as np
from PIL import Image

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to input image/video')
parser.add_argument('-m', '--min-size', dest='min_size', default=800, 
                    help='minimum input size for the RetinaNet network')
parser.add_argument('-t', '--threshold', default=0.6, type=float,
                    help='minimum confidence score for detection')
args = vars(parser.parse_args())
print('USING:')
print(f"Minimum image size: {args['min_size']}")
print(f"Confidence threshold: {args['threshold']}")