import imageio
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import re
from tqdm import tqdm
import vuba
import pandas as pd
import os
import multiprocessing as mp
import imageio
import random
import time

# Parameters -----------------------
limit = 3000 # For removing class imbalances, all individual classes will be augmented until this limit is reached
ann = pd.read_csv('./annotations_train_3s.csv') # Annotation training dataset produced from 'split_dataset.py'
im_dir = '/run/media/z/fast1/dev-resnet_video/timepoint_videos' # Output folder for exporting augmenting GIF files
output_file = './annotations_train_3s_aug.csv' # Output file for annotations of augmented files
check_class_imbalances = False # Check class imbalances present within training data
# ----------------------------------

if check_class_imbalances:
    print('Event imbalances:')
    for e, event in ann.groupby('single_event'):
        print(f'Event: {e}', f'No. samples: {len(event)}')
else:
    # Check for existing augmented files
    print('Removing existing files...')
    aug_files = glob.glob(f'{im_dir}/*_3s_*.gif')
    for f in tqdm(aug_files):
        os.remove(f)

    ia.seed(1)

    seq = iaa.Sequential([
        iaa.Sometimes(0.5, iaa.Fliplr()),
        iaa.Sometimes(0.5, iaa.Flipud()),
        iaa.OneOf((
            iaa.Multiply((0.5, 2)),
            iaa.Salt(0.025),
            iaa.Pepper(0.025)
        )),
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5)))
    ], random_order=True)

    def process(arg):
        i, fn = arg

        ann_aug_batch = {e:[] for e in ann.keys() if 'Unnamed' not in e}

        frames = imageio.mimread(fn)

        for j in range(1):
            seq_det = seq.to_deterministic()
            aug_frames = [seq_det(image=im) for im in frames]

            out_fn = str.split(fn, '/')[-1]
            out_fn = os.path.join(im_dir, re.sub('.gif', f'_{int(time.time())}.gif', out_fn))

            imageio.mimsave(out_fn, aug_frames)

            ann_aug_batch['out_file'].append(out_fn)

            for k in ann_aug_batch.keys():
                if k == 'out_file':
                    continue

                ann_aug_batch[k].append(ann[k][i])  

        return ann_aug_batch


    ann_aug = {e:[] for e in ann.keys() if 'Unnamed' not in e}

    for e,event in ann.groupby('single_event'):
        print('Upsampling for ', e, '.... ')
        for k,i in event.items():
            if 'Unnamed' in k:
                continue

            for f in i:
                ann_aug[k].append(f)

        if len(event) < limit:
            counter = len(event)

            event = event.sample(frac=1).reset_index(drop=True)
            print(len(event), limit, limit - len(event))
            for T in tqdm(range(limit - len(event))):
                if counter > limit:
                    break

                file = random.sample(list(event.out_file), 1)[0]
                index = list(ann.out_file).index(file)

                arg = (index, file)
                ret = process(arg)

                for k,i in ret.items():
                    for f in i:
                        ann_aug[k].append(f)            

                counter += 1


    ann_aug = pd.DataFrame(ann_aug)
    ann_aug.to_csv(output_file)