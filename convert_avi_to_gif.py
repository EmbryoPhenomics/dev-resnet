# Script for converting AVI video files to GIF format for
# use with training of Dev-ResNet

import imageio
import vuba
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
import re
import os

# Annotation DataFrame produced using create_training_video.py
data = pd.read_csv('./annotations.csv')
files = list(data.out_file)

# Function for converting AVI files to GIF format
def convert_to_gif(fn):
	if os.path.exists(re.sub('.avi', '.gif', fn)):
		return

	# Note here that a temporal stride of 10 frames is
	# used for training Dev-ResNet. If you would like 
	# a different temporal stride for training Dev-ResNet
	# for your application, adjust the keyword argument 'step'
	# below:
	video = vuba.Video(fn)
	try:
		frames = video.read(0, 128, step=10, grayscale=False)
		frames = list(frames)
	except IndexError:
		print(len(video))
		print(fn)
		raise

	imageio.mimsave(re.sub('.avi', '.gif', fn), frames)
	video.close()

with mp.Pool(processes=10) as pool:
	list(tqdm(pool.imap(convert_to_gif, files), total=len(files)))

# Export new CSV with filenames for GIF videos
data['out_file'] = [re.sub('.avi', '.gif', fn) for fn in data.out_file]
data.to_csv('./annotations_gif.csv')