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

def lowess_smooth(x, frac=0.025):
    x = np.asarray(x)
    return lowess(x, [i for i in range(len(x))], it=0, frac=frac)[:,1]

events = ['pre_gastrula', 'gastrula', 'trocophore', 'veliger', 'eye', 'heart', 'crawling', 'radula', 'hatch', 'dead']

# Initiate 3D model
# Only requires 12 frames but at an increment of 10
# i.e. 120 frames, but every 10th
model = DevResNet((12,128,128,1), n_classes=len(events), pretrained_weights=True)

# Lymnaea video and bbox for this video
# Downloading this example video will take a while as it is 4Gb in size.
file = wget.download('https://zenodo.org/record/8214689/files/example_video.avi')
video = vuba.Video(file)
x1,x2,y1,y2 = (230, 520, 490, 780)

# Read, filter and resize video frames
frames = []
for ind in tqdm(range(0, len(video), 600)):
  frame = video.read(start=ind, stop=ind+120, step=10, grayscale=True, low_memory=False).ndarray
  frame = frame[:, x1:x2, y1:y2, ...] # limit to egg only
  frame = np.expand_dims(frame, -1)
  frame = tf.image.resize_with_pad(frame, 128, 128)
  frames.append(frame)

frames = np.asarray(frames)

# Perform inference across the sequence of timepoint videos
results = model.predict(frames)

# Smooth and visualise results
for e,r in zip(events, results.T):
  plt.plot(lowess_smooth(r, 0.01), label=e)

plt.legend(loc='lower right')
plt.show()