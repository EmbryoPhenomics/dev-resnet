import vuba
import cv2
import numpy as np
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from statsmodels.nonparametric.smoothers_lowess import lowess
import wget

from dev_resnet import DevResNet

# Parameters -------------------------------
# Event you have trained you model on
events = ['pre_gastrula', 'gastrula', 'trocophore', 'veliger', 'eye', 'heart', 'crawling', 'radula', 'hatch', 'dead']

# Input shape that you have trained Dev-ResNet with, if you followed the same
# format of as presented in the paper it will be (12,128,128,1)
input_shape = (12,128,128,1)

# File path to video to analyse
# Note this is for an individual timepoint so it must be a single timepoint
use_example_video = True

# Local path to video on your system if use_example_video=False
# Note that this must be filtered to the same field of view used in training
# I.e. if you trained on video of only the egg, then you need to filter 
# video to just the egg before using the script.
video_path = '/path/to/video'

# Path to model weights (produced from training)
# Can be a URL or local file path, only specify one or the other
weights_path = None
weights_path_url = 'https://github.com/EmbryoPhenomics/dev-resnet/releases/download/v0.1/Dev-Resnet_lymnaea.h5'

# ----------------------------------------

def lowess_smooth(x, frac=0.025):
    x = np.asarray(x)
    return lowess(x, [i for i in range(len(x))], it=0, frac=frac)[:,1]

# Initiate 3D model
# Only requires 12 frames but at an increment of 10
# i.e. 120 frames, but every 10th
model = DevResNet(
    input_shape=input_shape, 
    n_classes=len(events), 
    pretrained_weights=weights_path,
    pretrained_weights_url=weights_path_url)

if use_example_video:
    # Downloading this example video will take a while as it is 4Gb in size.
    file = wget.download('https://zenodo.org/record/8214689/files/example_video.avi')
    video = vuba.Video(file)

    # Read, filter and resize video frames
    frames = video.read(start=0, stop=0+120, step=10, grayscale=True, low_memory=False).ndarray
    frames = np.expand_dims(frames, -1)
    frames = tf.image.resize_with_pad(frames, input_shape[1], input_shape[2])
else:
    video = vuba.Video(video_path)

    # Read, filter and resize video frames
    frames = video.read(start=0, stop=0+120, step=10, grayscale=True, low_memory=False).ndarray

    frames = np.expand_dims(frames, -1)
    frames = tf.image.resize_with_pad(frames, input_shape[1], input_shape[2])

# Expand for first dimension for a single batch
frames = np.expand_dims(frames, 0)

# Perform inference across the sequence of timepoint videos
results = model.predict(frames)

# Unlist nested results list
results = results[0]

# Print results
print('Predicted event:', events[np.argmax(results)], round(np.max(results), 2), '\n')

for e,r in zip(events, results):
    print(e, round(r, 2))