import vuba
import glob
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import multiprocessing as mp
import imageio

# Parameters ------------------------------------
# Note here we have structured this script for our experimental format, where timelapse videos of embryos are split into 
# folders by treatment, e.g. 20C, 22.5C and 25C

# All videos are in AVI format, encoded with MJPG, though because this script uses OpenCV there should not be issues
# with other formats
source_dir = '/run/media/z/fast1/dev-resnet_video' # Directory where folders of timelapse video are located
out_dir = '/run/media/z/fast1/dev-resnet_video/timepoint_videos' # Folder where you would like training video to be exported
source_video_file_extension = '.avi' # Change to the file format of your videos, e.g. .mp4 or .avi

# Because our timelapse videos were not filtered to just the eggs, we add in here our bounding box measurements 
# for each embryo for limiting to just the egg for training.
boxes = pd.read_csv('/run/media/z/fast1/dev-resnet_video/egg_boxes_constant_treatments.csv')
resize_before_filter = dict(check=True, shape=(512,512)) # Whether to resize before filtering to just region of interest

# Path to manual annotations of developmental events for creating training data
# Note this is in the format of | Temperature | Replicate | Event | Time |
# If you have a different structure for your annotations you will need to adjust the parsing of 
# this file below
dev_events = pd.read_csv('./developmental_events_new.csv')
treatment_column_name = 'temp' # Change this to the treatment column name you specified  

# The developmental events you have recorded in the CSV file above, in chronological order
events = ['pre_gastrula', 'gastrula', 'trocophore', 'veliger', 'eye', 'heart', 'crawling', 'radula', 'hatch', 'dead']

# Number of cores for parallel processing
cores = 14

# Temporal stride for each timepoint video
step = 3

# Timepoint length (frames)
timepoint_len = 600 # Number of video frames associated with each timepoint, 30sec x 20fps = 600 frames.
timepoint_len_export = 128 # Number of frames to export in output GIFs

# The name of the output CSV containing annotations
output_csv = f'./annotations_new_{step}s.csv'
# -----------------------------------------------

# Create output annotations CSV file, where each hourly timepoint video file has a given event
annotations = dict(temp=[], replicate=[], source_file=[], out_file=[], single_event=[])
for e in events:
	annotations[e] = []

# Because our experiment includes treatments at different temperatures, we first iterate across this axis
for t,temp in dev_events.groupby(treatment_column_name):

	# Then we iterate across each embryo replicate per treatment (that has been annotated)
	for r,replicate in temp.groupby('replicate'):
		print(t, r)

		# For our own file parsing
		if t % round(t) == 0:
			t = int(t)

		# Initiate video instance
		video = vuba.Video(f'{source_dir}/{t}C/{r}{source_video_file_extension}')

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
			timepoint = round(frame_index / timepoint_len)

			# To check if video already exists
			out_file = f'{out_dir}/{t}C_{r}_{timepoint}hpf_{step}s.gif'
			out_video = vuba.Video(out_file)
			out_len = len(out_video)
			out_video.close()

			if os.path.exists(out_file) and out_len >= int(timepoint_len_export / step):
				return
			else:
				at_box = boxes[(boxes.temp == f'{t}C') & (boxes.replicate == r) & (boxes.timepoint == timepoint+1)]
				x1,y1,x2,y2 = list(at_box.x1)[0],list(at_box.y1)[0],list(at_box.x2)[0],list(at_box.y2)[0]

				video = vuba.Video(f'{source_dir}/{t}C/{r}{source_video_file_extension}') # Video instance for reading in training video frames

				export_frames = []
				for frame in video.read(frame_index, frame_index+timepoint_len_export, step=step, grayscale=False):
					if resize_before_filter['check']:
						frame = cv2.resize(frame, resize_before_filter['shape'])

					frame = frame[y1:y2, x1:x2, ...]

					export_frames.append(frame)

				video.close()
				imageio.mimsave(out_file, export_frames)

		# Perform export of video files across multiple cores.
		with mp.Pool(processes=cores) as pool:
			args = list(range(0, len(video), timepoint_len))
			list(tqdm(pool.imap(export, args), total=len(args)))

		# Iterate through each timepoint and assign events to a given 
		# video files. Event labels only change when a subsequent event occurs
		current = event_times[0]
		dont_continue = {e: True for e in events}
		for frame_index in tqdm(range(0, len(video), timepoint_len)):
			timepoint = round(frame_index / timepoint_len)
			diff = np.abs(timepoint - event_times)

			if 0 in diff:
				current = event_times[np.argmin(diff)]	

			at = np.argmin(np.abs(timepoint - event_times))
			out_file = f'{out_dir}/{t}C_{r}_{timepoint}hpf_{step}s.gif'
			annotations['temp'].append(t)
			annotations['replicate'].append(r)
			annotations['source_file'].append(f'{source_dir}/{t}/{r}.avi')
			annotations['out_file'].append(out_file)
			annotations['single_event'].append(events_rep[event_times.tolist().index(current)])

			for e in events:
				if e == events_rep[event_times.tolist().index(current)]:
					annotations[e].append(1)

					if e != 'pre_gastrula':
						dont_continue[e] = False
				else:
					if dont_continue[e]:
						annotations[e].append(0)
					else:
						annotations[e].append(1)

		video.close()

annotations_df = pd.DataFrame(annotations)
annotations_df.to_csv(output_csv)
