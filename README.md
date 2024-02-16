# Dev-ResNet

## Introduction

![Alt Text](https://github.com/EmbryoPhenomics/dev-resnet/blob/main/assets/model_schematic.png)

A novel network for automating the detection of developmental events, specifically applied here to embryonic stages of the aquatic snail, *Lymnaea stagnalis*. We supply training, evaluation and inference code so that others can extend and apply this model to a species of their choosing. 

**Inference example of Dev-ResNet**

This is an example showcase of Dev-ResNet and it's capabilities for detecting developmental events, shown here for the entire embryonic development of the great pond snail, *Lymnaea stagnalis*. 

![Alt Text](https://github.com/EmbryoPhenomics/dev-resnet/blob/main/assets/inference.gif)

Note to achieve these results, Dev-ResNet was trained on the following dataset: https://zenodo.org/record/8214975.

## Guide

*If you encounter an issue with any of these steps, please open an issue on the GitHub repository: https://github.com/EmbryoPhenomics/dev-resnet/issues*

For training and testing Dev-ResNet on your own video, please follow the following steps:

### 1. Clone or download this repository

You can clone this repository using `git` in the terminal using the following command: 

```bash
git clone https://github.com/EmbryoPhenomics/dev-resnet.git
```

Or simply download and unzip the source code in the latest release in the `releases` section on the GitHub repository: https://github.com/EmbryoPhenomics/dev-resnet/releases

*Complete all of the following steps in your local copy of this repository, e.g. installation of python requirements.*

### 2. Setting up a Python environment

We recommend using [MiniConda](https://docs.conda.io/projects/miniconda/en/latest/) as your python environment for easy installation of the key deep learning dependencies for *Dev-ResNet*: [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/). To set up MiniConda please follow these steps:

**Install Python if you don't have it**

For installers for Windows, MacOs and Linux, please head over to the following website: https://www.python.org/downloads/. These installers will walk you through the installation process for your system. 

**Install MiniConda**

To download and install MiniConda, head over to the following documentation: https://docs.conda.io/projects/miniconda/en/latest/. There will be both installers which will walk you through the installation process, but also command line instructions at the bottom of the page if you prefer installing via the terminal.

**Setting up a MiniConda environment**

Once the installation step above has completed, open a terminal window and run the following commands:

Create the Conda environment:
```bash
conda create -n tf python=3.9 
```
*Note the Python version needs to be between 3.9 and 3.11, see https://www.tensorflow.org/install/pip#software_requirements. We have specified to 3.9 here*

Once completed you can enter the environment like so:
```bash
conda activate tf
```

and exit the environment like this: 
```bash
conda deactivate tf
```

To install the latest version of `TensorFlow` and `Keras`, please enter your Conda environment and follow the steps here: https://www.tensorflow.org/install/pip.

Once you have completed installation of TensoFlow/Keras, you will need to install the rest of the dependencies of `Dev-ResNet` using the following command in your conda environment:
```bash
conda activate tf # Enter the environment
pip3 install -r requirements.txt
```

### 3. Manual identification of event timings

Provided you have timelapse videos corresponding to a period of development, and that you can view the video at each timepoint in this timelapse, simply record the timepoint at which the onset of a given developmental event occurs. Repeat until you have identified the timings of all events of interest, and also for a representative number of individuals. For compatibility with subsequent steps, we recommend recording developmental events and their timings in the following CSV format:

| timepoint | temp | replicate | event |
|:---:|:---:|:---:|:---:|
| 1 | 20C | A_A1 | gastrula 
| ... | ... | ... | ...

Here we use `temp` as our treatment name, given that we trained Dev-ResNet on *L. stagnalis* embryos incubated at a range of constant temperature treatments. However, you can change this column name to better describe the treatments you exposed your animals to.

**Note** Dev-ResNet can only be trained on video samples, not images. If you have timelapse recordings but only one image captured at each timepoint, then unfortunately you cannot train Dev-ResNet on such imaging data. However, you can train popular image classification models such as ResNet-50. For a guide on how to train such 2D-CNNs please refer to the following guide: https://keras.io/examples/vision/image_classification_from_scratch/

### 4. Create training video (`create_training_video.py`)

To perform this step, you will need to structure your training video folder into the following format:

	.../training_video
		.../treatment1
			.../replicate1.avi
			.../replicate2.avi
		.../treatment2

Where `treatment` and `replicate` can be replaced with the treatment and replicate naming schemes you have used, respectively. Don't worry if you video format is `.mp4` or something else other than `.avi`, it will still work with the code.

**Note** These naming schemes should be used throughout, i.e. the same naming scheme is present in your folder structure as in the developmental event CSV recorded above.

Once you have your video in this folder structure, please view the `create_training_video.py` python script and edit the parameters with your own values, such as annotation filenames and training video filepaths, and then run this script to generate GIF sample files for training and evaluation.

An annotation CSV file will be generated from this script which will have the following columns:

- **treatment**: The treatment of a given sample. *Note this will differ depending on the treatment name you have specified in the first step above.*
- **replicate**: Replicate or individual ID for a given sample.
- **source_file**: File path and name of timelapse video from which the GIF sample has been generated.
- **out_file**: File path and name of GIF sample file generated.
- **single_event**: The event classification corresponding to the GIF sample, determined by the event timings you have identifed in the first step above.

### 5. Split dataset into training, validation and testing datasets (`split_dataset.py`)

After running the `create_training_video.py` script in the preceding step, a CSV containing annotations corresponding to your training video will be generated. In this step, we will be splitting this dataset into training, validation and testing datasets for subsequent steps. To do this, please run the `split_dataset.py` script with your own parameters.

### 6. Image augmentation (`image_augmentation.gif`)

In this step we will perform image augmentation on only the training video, to introduce variation in the training dataset and reduce the chances of overfitting in the training process. 

The augmentation script that we have included performs selective data augmentation to remove class imbalances in the training dataset, i.e. if you have a disproportionate number of samples allocated to one developmental event classification.

Simply specify the parameters in the script `image_augmentation.py` according to your system. Note that the parameter `limit` should be specified as equal or greater than the class with the most samples. You can check the class imbalances present within your training data by specifying the `check_class_imbalances` parameter as `True`, and then specifying it as `False` once you have finished checking.

### 7. Training and evaluation (`train.ipynb`)

We have included a guided training notebook for use with [Jupyter Notebook](https://jupyter.org/). To install Jupyter Notebook on your system please use the following commands in the terminal:

```bash
conda activate tf # Enter you conda environment

pip install jupyterlab # Insall Jupyter
```

Once installed, you can launch a 'Jupyter Lab' using the following command in your conda environment:

```bash
jupyter lab
```

This will launch a browser session including a web server for running code in Jupyter Notebooks, which are interactive coding environments. 

Each 'Jupyter Lab' session is only limited to the folder in which you run the above command - so if you would like to launch a Jupyter Lab in your local copy of this repository, simply launch a terminal window in that folder and type the above command. Alternatively, you can navigate via the terminal to your folder like so:

```bash
cd /path/to/dev-resnet # Full file path of where dev-resnet is located
conda activate tf # Activate conda environment if it isn't alread
jupyter lab # Launch Jupyter Lab in the Dev-ResNet folder
```

Once you have installed and launched Jupyter Lab on your system, you can open the `train.ipynb` jupyter notebook that we include in this repository. Simply follow this notebook and run the code based on your parameters for your system.

### 8. Inference (`inference.py`)

Finally, once you have successfully trained Dev-ResNet on your dataset, head over to the `inference.py` script in this repository and specifiy the required parameters in the script. Note that this script is only for analysing a single timepoint, not a multi-timepoint timelapse video. 

## Contributing

We welcome feedback and pull requests regarding any issues with this source code, for issues head over to https://github.com/EmbryoPhenomics/dev-resnet/issues and for pull requests https://github.com/EmbryoPhenomics/dev-resnet/pulls.
