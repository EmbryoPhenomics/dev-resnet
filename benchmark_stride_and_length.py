import glob
import vuba
import cv2
import numpy as np
import re
from tensorflow import keras
import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import multiprocessing as mp
from tqdm import tqdm
from typing import Tuple
import atexit
import time
import os
import ujson
import math
import seaborn as sns
from mpl_toolkits.axes_grid1 import ImageGrid

from dev_resnet import DevResNet

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

input_shapes = {
    10: [(12,128,128,1), (24,128,128,1), (48,128,128,1)],
    5: [(24,128,128,1)],
    3: [(40,128,128,1)]
}

# Parameters ----------------------------------------------------------
batch_size = 32
epochs = 50
model_save_dir = './trained_models'
model_name = 'Dev-ResNet_lymnaea'
events = ['pre_gastrula', 'gastrula', 'trocophore', 'veliger', 'eye', 'heart', 'crawling', 'radula', 'hatch', 'dead']
# ---------------------------------------------------------------------

results = dict(step=[], length=[], acc=[], seed=[])

for step in input_shapes.keys():
	for input_shape in input_shapes[step]:		

		# Dataset pipeline -------------------------
	    def read_data(fn, label):
	        gif = tf.io.read_file(fn)
	        gif = tf.image.decode_gif(gif)
	        gif = tf.image.resize_with_pad(gif, 128, 128)
	        gif = tf.image.rgb_to_grayscale(gif)
	        gif = gif[:input_shape[0], ...]
	        return gif, label
	
	    def dataset(images, labels, batch_size): 
	        data = tf.data.Dataset.from_tensor_slices((images, labels))
	        data = data.map(read_data, num_parallel_calls=tf.data.AUTOTUNE)
	        data = data.batch(batch_size, drop_remainder=True)
	        return data
	
	    annotations_train = pd.read_csv(f'./annotations_train_{step}s_aug.csv')
	    annotations_train.single_event[annotations_train.single_event == 'spinning'] = 'pre_gastrula'
	    annotations_train = annotations_train.sample(frac=1).reset_index(drop=True)
	    annotations_train['categorical'] = [events.index(e) for e in annotations_train.single_event]
	
	    annotations_val = pd.read_csv(f'./annotations_val_{step}s.csv')
	    annotations_val.single_event[annotations_val.single_event == 'spinning'] = 'pre_gastrula'
	    annotations_val = annotations_val.sample(frac=1).reset_index(drop=True)
	    annotations_val['categorical'] = [events.index(e) for e in annotations_val.single_event]
	
	    annotations_test = pd.read_csv(f'./annotations_test_{step}s.csv')
	    annotations_test.single_event[annotations_test.single_event == 'spinning'] = 'pre_gastrula'
	    annotations_test['categorical'] = [events.index(e) for e in annotations_test.single_event]
	
	    # Training data pipeline
	    train_files = list(annotations_train.out_file)
	    train_labels = list(annotations_train.categorical)
	
	    val_files = list(annotations_val.out_file)
	    val_labels = list(annotations_val.categorical)
	
	    # Test data pipeline
	    test_files = list(annotations_test.out_file)
	    test_labels = list(annotations_test.categorical)
	
	    train_data = dataset(train_files, train_labels, batch_size)
	    val_data = dataset(val_files, val_labels, batch_size)   
	    test_data = dataset(test_files, test_labels, batch_size)
	
	    for b in train_data:
	        images, labels = b
	        print(images.shape)
	        print(labels)
	        break
	
	
	    # Train and evaluate with three different seeds for computing metrics --------------------
	    for i in range(3):
	        np.random.seed(i)
	        tf.random.set_seed(i)
	        
	        model = DevResNet(input_shape, n_classes=len(events))
	
	        model.compile(
	            optimizer=keras.optimizers.Adam(learning_rate=0.000001),
	            loss='sparse_categorical_crossentropy',
	            metrics=['accuracy']
	        )
	
	        class EvaluateCallback(keras.callbacks.Callback):
	            def __init__(self):
	                super().__init__()
	                self.loss = []
	                self.accuracy = []
	
	            def on_epoch_end(self, epoch, log=None):
	                loss, acc = self.model.evaluate(test_data, verbose=0)
	                print('-', 'test_loss:', round(loss, 4), 'test_accuracy:', round(acc, 4))
	                self.loss.append(loss)
	                self.accuracy.append(acc)
	
	        evaluate_callback = EvaluateCallback()
	        callbacks = [
	            keras.callbacks.ModelCheckpoint(
	                filepath=f'{model_save_dir}/Dev-Resnet_lymnaea_{step}s_{i}_{input_shape[0]}.h5',
	                save_best_only=True,
	                monitor='val_accuracy',
	                save_weights_only=True
	            ),
	            evaluate_callback
	        ]
	
	        start = time.time()
	        history = model.fit(
	            train_data,
	            epochs=epochs, 
	            callbacks=callbacks,
	            validation_data=val_data)        
	        end = time.time()

			# Evaluate each model iteration ------------------------------
	        print('[INFO] Evaluating model.')
	        model.load_weights(f'{model_save_dir}/Dev-Resnet_lymnaea_{step}s_{i}_{input_shape[0]}.h5')
	        test_loss, test_accuracy = model.evaluate(test_data)
	
	        results['step'].append(step)
	        results['length'].append(input_shape[0])
	        results['acc'].append(test_accuracy)
	        results['seed'].append(i)
	        
	        del model
	        keras.backend.clear_session()

results = pd.DataFrame(results)
results.to_csv('./dev-resnet_model_comparisons.csv')
