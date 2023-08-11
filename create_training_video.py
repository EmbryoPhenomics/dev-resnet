import vuba
import glob
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import multiprocessing as mp

# Parameters ------------------------------------
# Note here we have structured this script for our experimental format, where timelapse videos of embryos are split into 
# folders by treatment, e.g. 20C, 22.5C and 25C

# All videos are in AVI format, encoded with MJPG, though because this script uses OpenCV there should not be issues
# with other formats
source_dir = '/path/to/video/' # Directory where folders of timelapse video are located
out_dir = '/path/to/training_video/' # Folder where you would like training video to be exported

# Because our timelapse videos were not filtered to just the eggs, we add in here our bounding box measurements 
# for each embryo for limiting to just the egg for training.
boxes = pd.read_csv('/path/to/egg_boxes.csv')

# Path to manual annotations of developmental events for creating training data
# Note this is in the format of | Temperature | Replicate | Event | Time |
# If you have a different structure for your annotations you will need to adjust the parsing of 
# this file below
dev_events = pd.read_csv('/path/to/annotations.csv')

# Number of cores for parallel processing
cores = 10
# -----------------------------------------------

# Create output annotations CSV file, where each hourly timepoint video file has a given event
annotations = dict(temp=[], replicate=[], source_file=[], out_file=[], single_event=[])

# Because our experiment includes treatments at different temperatures, we first iterate across this axis
for t,temp in dev_events.groupby('temp'):

	# Then we iterate across each embryo replicate per treatment (that has been annotated)
	for r,replicate in temp.groupby('replicate'):
		print(t, r)

		# For our own file parsing
		if t % round(t) == 0:
			t = int(t)

		# Initiate video instance
		video = vuba.Video(f'{source_dir}/{t}C/{r}.avi')

		# Create array of manually identified event times
		event_times = np.asarray(list(replicate.timepoint))

		# Add in a descriptor for the developmental period prior
		# to the first event, in this case we use Pre-Gastrula
		event_times = np.asarray([0] + list(replicate.timepoint))
		events_rep = ['pre_gastrula'] + list(replicate.event)

		# Main function for exporting hourly video files 
		# args is simply the frame index of a given timepoint to export
		# from the timelapse videos
		def export(args):
			frame_index = args
			timepoint = round(frame_index / 600)

			# To check if video already exists
			out_file = f'{out_dir}/{t}C_{r}_{timepoint}.avi'
			out_video = vuba.Video(f'{out_dir}/{t}C_{r}_{timepoint}.avi')
			out_len = len(out_video)
			out_video.close()

			if os.path.exists(f'{out_dir}/{t}C_{r}_{timepoint}.avi') and out_len >= 128:
				return
			else:
				at_box = boxes[(boxes.temp == f'{t}C') & (boxes.replicate == r) & (boxes.timepoint == timepoint+1)]
				x1,y1,x2,y2 = list(at_box.x1)[0],list(at_box.y1)[0],list(at_box.x2)[0],list(at_box.y2)[0]

				video = vuba.Video(f'{source_dir}/{t}C/{r}.avi') # Video instance for reading in training video frames
				writer = vuba.Writer(out_file, video, resolution=(x2-x1, y2-y1), codec='FFV1') # Lossless output of training video

				# Read in the first 128 frames at the given frame index
				for frame in video.read(frame_index, frame_index+128, grayscale=False):
					# We resize frames here for our bounding box filtering, omit if you do not have bounding boxes.
					frame = cv2.resize(frame, (512, 512))
					frame = frame[y1:y2, x1:x2, ...]

					# Export frame
					writer.write(frame)

				writer.close()
				video.close()

		# Perform export of video files across multiple cores.
		with mp.Pool(processes=cores) as pool:
			args = list(range(0, len(video), 600))
			list(tqdm(pool.imap(export, args), total=len(args)))

		# Iterate through each timepoint and assign events to a given 
		# video files. Event labels only change when a subsequent event occurs
		current = event_times[0]
		for frame_index in tqdm(range(0, len(video), 600)):
			timepoint = round(frame_index / 600)
			diff = np.abs(timepoint - event_times)

			if 0 in diff:
				current = event_times[np.argmin(diff)]	

			at = np.argmin(np.abs(timepoint - event_times))
			out_file = f'{out_dir}/{t}C_{r}_{timepoint}.avi'
			annotations['temp'].append(t)
			annotations['replicate'].append(r)
			annotations['source_file'].append(f'{source_dir}/{t}C/{r}.avi')
			annotations['out_file'].append(out_file)
			annotations['single_event'].append(events_rep[event_times.tolist().index(current)])

		video.close()

annotations_df = pd.DataFrame(annotations)
annotations_df.to_csv('./annotations_original_new.csv')
